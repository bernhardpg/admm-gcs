import time
from dataclasses import dataclass
from typing import Dict, List, NamedTuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
from pydrake.geometry.optimization import HPolyhedron
from pydrake.solvers import MathematicalProgram, Solve
from pydrake.symbolic import Expression

from admm_gcs.non_convex_admm.gcs import GCS, Edge, VertexId, path_to_edges
from admm_gcs.tools import add_noise, calc_polytope_centroid


class QpBase(NamedTuple):
    prog: MathematicalProgram
    x_eu: npt.NDArray
    x_ev: npt.NDArray
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
        self.price_vars = {}  # one for each edge
        self.consensus_vars = {}  # one for each vertex
        self.discrete_vars = {}  # one for each edge
        self.params = params
        self.rho = params.rho

        self.individual_qp_solve_times = []
        self.local_solve_times = []
        self.discrete_solve_times = []

        self.iteration = 0

        self.programs = {e: self._create_program_for_edge(e) for e in self.gcs.edges}

    def initialize(self) -> None:
        """
        Initialize local and consensus variables.
        """

        for edge in self.gcs.edges:
            u, v = edge
            poly_u = self.gcs.vertices[u]
            poly_v = self.gcs.vertices[v]

            self.local_vars[edge] = EdgeVar(
                xu=add_noise(calc_polytope_centroid(poly_u)),
                xv=add_noise(calc_polytope_centroid(poly_v)),
            )

        for vertex_id in self.gcs.vertices:
            # self.consensus_vars[vertex_id] = self._calc_x_mean_for_vertex(
            #     vertex_id)
            self.consensus_vars[vertex_id] = np.zeros((2,))

        self.update_discrete()

        for e in self.gcs.edges:
            violation_e = self._calc_consensus_violation(e)
            self.price_vars[e] = violation_e
            # self.price_vars[e] = EdgeVar(xu=np.zeros((2,)), xv=np.zeros((2,)))

    def _create_program_for_edge(self, edge: Edge) -> QpBase:
        prog = MathematicalProgram()
        x_eu = prog.NewContinuousVariables(2, "x_eu")
        x_ev = prog.NewContinuousVariables(2, "x_ev")

        l_e = _sq_eucl_dist(x_eu, x_ev)

        u, v = edge
        X_u = self.gcs.h_polyhedrons[u]
        X_v = self.gcs.h_polyhedrons[v]

        prog.AddLinearConstraint(
            X_u.A(), np.full(X_u.b().shape, -np.inf), X_u.b(), x_eu
        )
        prog.AddLinearConstraint(
            X_v.A(), np.full(X_v.b().shape, -np.inf), X_v.b(), x_ev
        )
        return QpBase(prog, x_eu, x_ev, l_e)

    def _update_local_for_edge(self, edge: Edge) -> EdgeVar:
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

        prog, x_eu, x_ev, l_e = self.programs[edge]

        u_e = np.concatenate((u_eu, u_ev))
        # TODO duplicate code
        G_e = np.concatenate((x_eu - x_u, x_ev - x_v)) + u_e
        cost_expr = y_e * l_e + (self.rho / 2) * G_e.T.dot(G_e)
        cost = prog.AddQuadraticCost(cost_expr)

        start = time.time()
        result = Solve(prog)
        self.individual_qp_solve_times.append(time.time() - start)

        assert result.is_success()

        # NOTE: Most of solve time seems to be solver overhead
        # Clean up prog so we don't have to rebuild it
        prog.RemoveCost(cost)  # type: ignore

        return EdgeVar(xu=result.GetSolution(x_eu), xv=result.GetSolution(x_ev))

    def update_local(self):
        """
        Update local variables (e.g., using projections onto convex sets).
        """
        start = time.time()
        for edge in self.gcs.edges:
            self.local_vars[edge] = self._update_local_for_edge(edge)

        self.local_solve_times.append(time.time() - start)

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
        for e in self.gcs.edges:
            violation = self._calc_consensus_violation(e)
            u = self.price_vars[e]
            u_next = EdgeVar(xu=u.xu + violation.xu, xv=u.xv + violation.xv)
            self.price_vars[e] = u_next

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
