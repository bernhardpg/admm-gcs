import copy
from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeVar

import numpy as np
import numpy.typing as npt
from pydrake.math import eq
from pydrake.solvers import MathematicalProgram, MathematicalProgramResult, MosekSolver

from admm_gcs.non_convex_admm.gcs import GCS, Edge, VertexId
from admm_gcs.tools import add_noise, calc_polytope_centroid, squared_eucl_norm

T = TypeVar("T", bound=Any)


@dataclass
class EdgeVars(Generic[T]):
    """
    e = (u, v)
    """

    z_u: npt.NDArray[T]
    z_v: npt.NDArray[T]
    y: T

    @classmethod
    def make_zero(cls, ambient_dim: int):
        z_u = np.zeros((ambient_dim,))
        z_v = np.zeros((ambient_dim,))
        y = 0

        return cls(z_u, z_v, y)

    def u_stacked(self) -> npt.NDArray[T]:
        return np.concatenate((self.z_u, [self.y]))

    def v_stacked(self) -> npt.NDArray[T]:
        return np.concatenate((self.z_v, [self.y]))

    def eval_result(self, result: MathematicalProgramResult) -> "EdgeVars":
        z_u = result.GetSolution(self.z_u)
        z_v = result.GetSolution(self.z_v)
        y = result.GetSolution(self.y)
        return EdgeVars(z_u, z_v, y)

    @property
    def flow_treshold(self) -> float:
        return 1e-3

    @property
    def x_u(self) -> Optional[npt.NDArray[T]]:
        if self.y > self.flow_treshold:
            return self.z_u / self.y  # type: ignore
        else:
            return None

    @property
    def x_v(self) -> Optional[npt.NDArray[T]]:
        if self.y > self.flow_treshold:
            return self.z_v / self.y  # type: ignore
        else:
            return None


@dataclass
class AdmmParameters:
    rho: float = 1.0
    sigma: float = 1.0
    max_iterations: int = 50
    store_iterations: bool = False
    init_consensus_w_noise: bool = False


class AdmmSolver:
    def __init__(self, gcs: GCS, params: AdmmParameters):
        self.iteration = 0
        self.params = params
        self.graph = gcs

        self.rho = params.rho
        self.sigma = params.sigma

        self.dim = 2  # TODO

    def initialize(self) -> None:
        self.local_vars = {
            v: {
                e: EdgeVars.make_zero(ambient_dim=2)
                for e in self.graph.edges_per_vertex[v]
            }
            for v in self.graph.vertices
        }

        if self.params.init_consensus_w_noise:
            self.consensus_vars = {
                (u, v): EdgeVars(
                    z_u=add_noise(calc_polytope_centroid(self.graph.vertices[u])),
                    z_v=add_noise(calc_polytope_centroid(self.graph.vertices[v])),
                    y=0.0,
                )
                for (u, v) in self.graph.edges
            }
        else:
            self.consensus_vars = {
                e: EdgeVars.make_zero(ambient_dim=self.dim) for e in self.graph.edges
            }

        self.price_vars = {
            v: {
                e: EdgeVars.make_zero(ambient_dim=2)
                for e in self.graph.edges_per_vertex[v]
            }
            for v in self.graph.vertices
        }

        if self.params.store_iterations:
            self.local_vars_history = []
            self.consensus_vars_history = []
            self.price_vars_history = []

    def _update_local_for_vertex(self, vertex: VertexId) -> None:
        prog = MathematicalProgram()
        edges = self.graph.edges_per_vertex[vertex]

        local_vars = {
            e: EdgeVars(
                prog.NewContinuousVariables(self.dim, f"z_{e}_u"),
                prog.NewContinuousVariables(self.dim, f"z_{e}_v"),
                prog.NewContinuousVariables(1, f"y_{e}").item(),
            )
            for e in edges
        }

        # Flow variables are nonnegative
        for e in edges:
            y_e = local_vars[e].y
            prog.AddLinearConstraint(y_e >= 0)

        price_vars = self.price_vars[vertex]

        # Add consensus cost
        for e in edges:
            z_ve_u = local_vars[e].z_u
            z_ve_v = local_vars[e].z_v
            y_ve = local_vars[e].y

            z_e_u = self.consensus_vars[e].z_u
            z_e_v = self.consensus_vars[e].z_v
            y_e = self.consensus_vars[e].y

            lam_ve_u = price_vars[e].z_u
            lam_ve_v = price_vars[e].z_v
            mu_ve = price_vars[e].y

            z_e_u_err = squared_eucl_norm(z_ve_u - z_e_u + lam_ve_u)
            z_e_v_err = squared_eucl_norm(z_ve_v - z_e_v + lam_ve_v)
            y_e_err = (y_ve - y_e + mu_ve) ** 2

            prog.AddQuadraticCost(
                (self.rho / 2) * z_e_u_err
                + (self.rho / 2) * z_e_v_err
                + (self.sigma / 2) * y_e_err,
                is_convex=True,
            )

        # Flow constraints
        incoming_edges = self.graph.incoming_edges_per_vertex[vertex]
        outgoing_edges = self.graph.outgoing_edges_per_vertex[vertex]

        outgoing_flows = sum([local_vars[e].y for e in outgoing_edges], start=0)
        incoming_flows = sum([local_vars[e].y for e in incoming_edges], start=0)

        if vertex == self.graph.source:
            prog.AddLinearEqualityConstraint(outgoing_flows == 1)  # type: ignore
        elif vertex == self.graph.target:
            prog.AddLinearEqualityConstraint(incoming_flows == 1)  # type: ignore
        else:
            # Degree constraint
            prog.AddLinearConstraint(incoming_flows <= 1)  # type: ignore

            # Preservation of flow
            prog.AddLinearConstraint(outgoing_flows == incoming_flows)  # type: ignore

            # Spatial flow constraints
            incoming_spatial_flows = sum(
                [local_vars[e].z_v for e in incoming_edges], start=np.zeros((self.dim,))
            )
            outgoing_spatial_flows = sum(
                [local_vars[e].z_u for e in outgoing_edges], start=np.zeros((self.dim,))
            )
            prog.AddLinearConstraint(eq(outgoing_spatial_flows, incoming_spatial_flows))

        # Perspective set containment
        for e in edges:
            u, v = e
            X_u = self.graph.h_polyhedrons[u]
            X_v = self.graph.h_polyhedrons[v]

            A_u = np.hstack((X_u.A(), -X_u.b().reshape((-1, 1))))
            x_u = local_vars[e].u_stacked()
            prog.AddLinearConstraint(
                A_u, np.full(A_u.shape[0], -np.inf), np.zeros(A_u.shape[0]), x_u
            )

            A_v = np.hstack((X_v.A(), -X_v.b().reshape((-1, 1))))
            x_v = local_vars[e].v_stacked()
            prog.AddLinearConstraint(
                A_v, np.full(A_v.shape[0], -np.inf), np.zeros(A_v.shape[0]), x_v
            )

        # Perspective cost
        # (Euclidean Distance)
        # (Note that the perspective of the euclidean distance is just the eucl distance itself as long
        # as the scaling variable is nonnegative)
        for e in edges:
            u, v = e
            s = prog.NewContinuousVariables(1, f"s_{e}").item()
            prog.AddLinearCost(s)

            z_u = local_vars[e].z_u
            z_v = local_vars[e].z_v
            prog.AddLorentzConeConstraint(np.concatenate([[s], z_u - z_v]))

        solver = MosekSolver()
        result = solver.Solve(prog)  # type: ignore

        assert result.is_success()

        self.local_vars[vertex] = {
            edge: vars.eval_result(result) for edge, vars in local_vars.items()
        }

    def update_local(self) -> None:
        for v in self.graph.vertices:
            self._update_local_for_vertex(v)

    def _update_consensus_for_edge(self, e: Edge) -> None:
        u, v = e

        # Update z_e_u
        z_ue_u = self.local_vars[u][e].z_u
        lam_ue_u = self.price_vars[u][e].z_u
        const_1 = z_ue_u + lam_ue_u

        z_ve_u = self.local_vars[v][e].z_u
        lam_ve_u = self.price_vars[v][e].z_u
        const_2 = z_ve_u + lam_ve_u

        z_e_u = 0.5 * (const_1 + const_2)

        # Update z_e_v
        z_ue_v = self.local_vars[u][e].z_v
        lam_ue_v = self.price_vars[u][e].z_v
        const_1 = z_ue_v + lam_ue_v

        z_ve_v = self.local_vars[v][e].z_v
        lam_ve_v = self.price_vars[v][e].z_v
        const_2 = z_ve_v + lam_ve_v

        z_e_v = 0.5 * (const_1 + const_2)

        # Update y_e
        y_ue = self.local_vars[u][e].y
        mu_ue = self.price_vars[u][e].y
        const_1 = y_ue + mu_ue

        y_ve = self.local_vars[v][e].y
        mu_ve = self.price_vars[v][e].y
        const_2 = y_ve + mu_ve

        y_e = 0.5 * (const_1 + const_2)

        self.consensus_vars[e] = EdgeVars(z_e_u, z_e_v, y_e)

    def update_consensus(self) -> None:
        for e in self.graph.edges:
            self._update_consensus_for_edge(e)

    def update_prices(self) -> None:
        for v in self.graph.vertices:
            for e in self.graph.edges_per_vertex[v]:
                lam_ve_u = self.price_vars[v][e].z_u
                lam_ve_v = self.price_vars[v][e].z_v
                mu_ve = self.price_vars[v][e].y

                z_ve_u = self.local_vars[v][e].z_u
                z_ve_v = self.local_vars[v][e].z_v
                y_ve = self.local_vars[v][e].y

                z_e_u = self.consensus_vars[e].z_u
                z_e_v = self.consensus_vars[e].z_v
                y_e = self.consensus_vars[e].y

                lam_ve_u_next = lam_ve_u + z_ve_u - z_e_u
                lam_ve_v_next = lam_ve_v + z_ve_v - z_e_v
                mu_ve_next = mu_ve + y_ve - y_e

                self.price_vars[v][e] = EdgeVars(
                    lam_ve_u_next, lam_ve_v_next, mu_ve_next
                )

    def _step(self) -> None:
        self.update_local()
        self.update_consensus()
        self.update_prices()

        print(self.iteration)

        self.iteration += 1
        # TODO(bernhardpg): Update rho here

    def solve(self) -> None:
        for it in range(self.params.max_iterations):
            if self.params.store_iterations:
                self.local_vars_history.append(copy.deepcopy(self.local_vars))
                self.consensus_vars_history.append(copy.deepcopy(self.consensus_vars))
                self.price_vars_history.append(copy.deepcopy(self.price_vars))

            self._step()
