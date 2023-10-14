import time
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
from pydrake.geometry.optimization import HPolyhedron
from pydrake.solvers import MathematicalProgram, Solve
from pydrake.symbolic import Expression

from admm_gcs.gcs import GCS, Edge, VertexId, path_to_edges
from admm_gcs.tools import add_noise, calc_polytope_centroid


class QpBase(NamedTuple):
    prog: MathematicalProgram
    x_eu: npt.NDArray
    x_ev: npt.NDArray
    s_eu: npt.NDArray
    s_ev: npt.NDArray
    l_e: Expression


class EdgeVar(NamedTuple):
    xu: npt.NDArray[np.float64]
    xv: npt.NDArray[np.float64]


@dataclass
class AdmmParameters:
    rho: float = 1.0


# TODO(bernhardpg): Replace with Bindings
def _sq_eucl_dist(xu, xv):
    diff = xu - xv
    return diff.T.dot(diff)


class MultiblockADMMSolver:
    def __init__(self, gcs: GCS, params: AdmmParameters):
        self.gcs = gcs

        self.local_vars = {}  # one for each edge
        self.cone_vars = {}  # one for each edge
        self.slack_cone_vars = {}  # one for each edge
        self.price_vars = {}  # one for each edge
        self.cone_price_vars = {}  # one for each edge
        self.consensus_vars = {}  # one for each vertex
        self.discrete_vars = {}  # one for each edge

        self.params = params
        self.rho = params.rho

        self.individual_qp_solve_times = []
        self.local_solve_times = []
        self.discrete_solve_times = []

        self.iteration = 0

        self.programs = {e: self._create_program_for_edge(
            e) for e in self.gcs.edges}

    def initialize(self) -> None:
        """
        Initialize variables
        """

        for edge in self.gcs.edges:
            u, v = edge
            poly_u = self.gcs.vertices[u]
            poly_v = self.gcs.vertices[v]

            self.local_vars[edge] = EdgeVar(
                xu=add_noise(calc_polytope_centroid(poly_u)),
                xv=add_noise(calc_polytope_centroid(poly_v)),
            )

        for edge in self.gcs.edges:
            u, v = edge
            poly_u = self.gcs.vertices[u]
            poly_v = self.gcs.vertices[v]

            X_u = self.gcs.h_polyhedrons[u]
            X_v = self.gcs.h_polyhedrons[v]

            x_eu = self.local_vars[edge].xu
            x_ev = self.local_vars[edge].xv

            s_e = EdgeVar(
                xu=X_u.A().dot(x_eu) - X_u.b(),
                xv=X_v.A().dot(x_ev) - X_v.b(),
            )
            self.cone_vars[edge] = s_e
            self.slack_cone_vars[edge] = EdgeVar(
                xu=np.zeros(X_u.b().shape), xv=np.zeros(X_v.b().shape)
            )
            self.cone_price_vars[edge] = s_e

        for vertex_id in self.gcs.vertices:
            self.consensus_vars[vertex_id] = np.zeros((2,))

        self.update_discrete()

        for e in self.gcs.edges:
            violation_e = self._calc_consensus_violation(e)
            self.price_vars[e] = violation_e

    def _create_program_for_edge(self, edge: Edge) -> QpBase:
        prog = MathematicalProgram()
        x_eu = prog.NewContinuousVariables(2, "x_eu")
        x_ev = prog.NewContinuousVariables(2, "x_ev")

        u, v = edge
        X_u = self.gcs.h_polyhedrons[u]
        X_v = self.gcs.h_polyhedrons[v]

        s_eu = prog.NewContinuousVariables(X_u.b().size, "s_eu")
        s_ev = prog.NewContinuousVariables(X_v.b().size, "s_ev")

        A_u_bar = np.hstack((X_u.A(), -np.eye(X_u.b().size)))
        A_v_bar = np.hstack((X_v.A(), -np.eye(X_v.b().size)))

        prog.AddLinearEqualityConstraint(
            A_u_bar, X_u.b(), np.concatenate((x_eu, s_eu)))
        prog.AddLinearEqualityConstraint(
            A_v_bar, X_v.b(), np.concatenate((x_ev, s_ev)))

        l_e = _sq_eucl_dist(x_eu, x_ev)

        u, v = edge
        return QpBase(prog, x_eu, x_ev, s_eu, s_ev, l_e)

    def _update_local_for_edge(self, edge: Edge) -> Tuple[EdgeVar, EdgeVar]:
        """
        Solves a small QP to update the local variables. This can
        be parallelized.
        """

        u, v = edge
        x_u = self.consensus_vars[u]
        x_v = self.consensus_vars[v]

        u_eu = self.price_vars[edge].xu
        u_ev = self.price_vars[edge].xv

        y_e = self.discrete_vars[edge]

        prog, x_eu, x_ev, s_eu, s_ev, l_e = self.programs[edge]

        u_e = np.concatenate((u_eu, u_ev))
        # TODO duplicate code
        G_e_1 = np.concatenate((x_eu - x_u, x_ev - x_v)) + u_e

        s_e = np.concatenate((s_eu, s_ev))
        s_eu_slack = self.slack_cone_vars[edge].xu
        s_ev_slack = self.slack_cone_vars[edge].xv

        s_e_slack = np.concatenate((s_eu_slack, s_ev_slack))

        w_eu = self.cone_price_vars[edge].xu
        w_ev = self.cone_price_vars[edge].xv

        w_e = np.concatenate((w_eu, w_ev))

        G_e_2 = s_e - s_e_slack + w_e
        cost_expr = y_e * l_e + (self.rho / 2) * (
            G_e_1.T.dot(G_e_1) + G_e_2.T.dot(G_e_2)
        )
        cost = prog.AddQuadraticCost(cost_expr)

        start = time.time()
        result = Solve(prog)
        self.individual_qp_solve_times.append(time.time() - start)

        assert result.is_success()

        # NOTE: Most of solve time seems to be solver overhead
        # Clean up prog so we don't have to rebuild it
        prog.RemoveCost(cost)  # type: ignore

        x_e_sol = EdgeVar(xu=result.GetSolution(x_eu),
                          xv=result.GetSolution(x_ev))
        s_e_sol = EdgeVar(xu=result.GetSolution(s_eu),
                          xv=result.GetSolution(s_ev))

        return x_e_sol, s_e_sol

    def update_local(self):
        """
        Update local variables (e.g., using projections onto convex sets).
        """
        start = time.time()
        for edge in self.gcs.edges:
            x_e_sol, s_e_sol = self._update_local_for_edge(edge)
            self.local_vars[edge] = x_e_sol
            self.cone_vars[edge] = s_e_sol

        self.local_solve_times.append(time.time() - start)

    def update_cone(self):
        """
        Update cone variablesusing projections onto cones.
        """
        for edge in self.gcs.edges:
            w_eu = self.cone_price_vars[edge].xu
            w_ev = self.cone_price_vars[edge].xv

            s_eu = self.cone_vars[edge].xu
            s_ev = self.cone_vars[edge].xv

            s_eu_slack = np.maximum(w_eu + s_eu, 0)
            s_ev_slack = np.maximum(w_ev + s_ev, 0)

            self.slack_cone_vars[edge] = EdgeVar(xu=s_eu_slack, xv=s_ev_slack)

    def _calc_x_mean_for_vertex(self, vertex_id: VertexId) -> npt.NDArray[np.float64]:
        local_vars_for_v = []
        for (u, v), var in self.local_vars.items():
            if u == vertex_id:
                local_vars_for_v.append(var.xu)
            elif v == vertex_id:
                local_vars_for_v.append(var.xv)

        x_mean = np.mean(local_vars_for_v, axis=0)

        return x_mean

    def _calc_u_mean_for_vertex(self, vertex_id: VertexId) -> npt.NDArray[np.float64]:
        price_vars_for_v = []
        for (u, v), var in self.price_vars.items():
            if u == vertex_id:
                price_vars_for_v.append(var.xu)
            elif v == vertex_id:
                price_vars_for_v.append(var.xv)

        u_mean = np.mean(price_vars_for_v, axis=0)

        return u_mean

    def update_consensus(self) -> None:
        """
        Update consensus variables.
        """
        for vertex_id in self.gcs.vertices:
            x_mean = self._calc_x_mean_for_vertex(vertex_id)
            u_mean = self._calc_u_mean_for_vertex(vertex_id)

            self.consensus_vars[vertex_id] = x_mean + u_mean

    def update_discrete(self, show_graph=False) -> None:
        """
        Update the discrete variables representing which vertices are in the shortest path.
        """
        assert self.gcs.source is not None
        assert self.gcs.target is not None

        start = time.time()
        path = solve_discrete_spp(
            self.local_vars, self.gcs.source, self.gcs.target, show_graph=show_graph
        )
        self.discrete_solve_times.append(time.time() - start)

        edges_on_sp = path_to_edges(path)

        for e in self.gcs.edges:
            if e in edges_on_sp:
                self.discrete_vars[e] = 1
            else:
                self.discrete_vars[e] = 0

        self.path = path

    def _calc_consensus_violation(self, edge: Edge) -> EdgeVar:
        x_e = self.local_vars[edge]
        x_eu = x_e.xu
        x_ev = x_e.xv

        u, v = edge
        x_u = self.consensus_vars[u]
        x_v = self.consensus_vars[v]

        violation = EdgeVar(xu=x_eu - x_u, xv=x_ev - x_v)
        return violation

    def update_prices(self) -> None:
        """
        Updates the dual variables associated with the constraints that the
        local variables (x_eu, x_ev) are equal to the global variables
        (x_u, x_v)
        """
        # Local vars
        for e in self.gcs.edges:
            violation = self._calc_consensus_violation(e)
            u = self.price_vars[e]
            u_next = EdgeVar(xu=u.xu + violation.xu, xv=u.xv + violation.xv)
            self.price_vars[e] = u_next

        # Cone vars
        for edge in self.gcs.edges:
            s_eu = self.cone_vars[edge].xu
            s_ev = self.cone_vars[edge].xv

            s_eu_slack = self.slack_cone_vars[edge].xu
            s_ev_slack = self.slack_cone_vars[edge].xv

            violation = EdgeVar(s_eu - s_eu_slack, s_ev - s_ev_slack)
            w_e = self.cone_price_vars[edge]
            w_e_next = EdgeVar(xu=w_e.xu + violation.xu,
                               xv=w_e.xv + violation.xv)
            self.cone_price_vars[edge] = w_e_next

    def _step(self) -> None:
        self.update_local()
        self.update_consensus()
        self.update_discrete()
        self.update_prices()

        self.iteration += 1
        # TODO(bernhardpg): Update rho here

    def solve(self):
        """
        Solve the optimization problem using multi-block ADMM.
        """

        N = 10
        for it in range(N):
            self._step()


def solve_discrete_spp(
    local_vars: Dict[Edge, EdgeVar],
    source: VertexId,
    target: VertexId,
    show_graph=False,
) -> List[VertexId]:
    G = nx.DiGraph()

    for edge, var in local_vars.items():
        u, v = edge
        w = _sq_eucl_dist(var.xu, var.xv)
        G.add_edge(u, v, weight=w)

    if show_graph:
        plt.figure()
        pos = nx.spring_layout(G)  # positions for all nodes
        nx.draw(G, pos, with_labels=True)
        labels = nx.get_edge_attributes(G, "weight")
        labels = {
            (k[0], k[1]): f"{v:.2f}" for k, v in labels.items()
        }  # Formatting to two decimal places
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.show()
        breakpoint()

    path = nx.shortest_path(G, source=source, target=target, weight="weight")
    return path
