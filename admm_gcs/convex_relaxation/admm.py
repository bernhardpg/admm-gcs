from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np
import numpy.typing as npt
from pydrake.math import eq
from pydrake.solvers import MathematicalProgram, MosekSolver

from admm_gcs.non_convex_admm.gcs import GCS, VertexId
from admm_gcs.test_cases import create_test_graph
from admm_gcs.tools import squared_eucl_distance, squared_eucl_norm
from admm_gcs.visualize import plot_gcs_graph

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


@dataclass
class AdmmParameters:
    rho: float = 1.0
    max_iterations: int = 50


class AdmmSolver:
    def __init__(self, gcs: GCS, params: AdmmParameters):
        self.iteration = 0
        self.params = params
        self.graph = gcs

        self.dim = 2  # TODO

    def initialize(self) -> None:
        self.local_vars = {
            v: {
                e: EdgeVars.make_zero(ambient_dim=2)
                for e in self.graph.edges_per_vertex[v]
            }
            for v in self.graph.vertices
        }

        self.consensus_vars = {
            e: EdgeVars.make_zero(ambient_dim=2) for e in self.graph.edges
        }

        self.price_vars = {
            v: {
                e: EdgeVars.make_zero(ambient_dim=2)
                for e in self.graph.edges_per_vertex[v]
            }
            for v in self.graph.vertices
        }

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

            prog.AddQuadraticCost(z_e_u_err + z_e_v_err + y_e_err, is_convex=True)

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

        solver = MosekSolver()
        result = solver.Solve(prog)  # type: ignore

        assert result.is_success()

    def update_local(self) -> None:
        for v in self.graph.vertices:
            self._update_local_for_vertex(v)

    def update_consensus(self) -> None:
        ...

    def update_prices(self) -> None:
        ...

    def _step(self) -> None:
        self.update_local()
        self.update_consensus()
        self.update_prices()

        self.iteration += 1
        # TODO(bernhardpg): Update rho here

    def solve(self):
        """
        Solve the optimization problem using multi-block ADMM.
        """

        for it in range(self.params.max_iterations):
            self._step()


def main() -> None:
    gcs = create_test_graph()

    plot_gcs_graph(gcs.vertices, gcs.edges, gcs.source, gcs.target, save_to_file=True)

    params = AdmmParameters()
    solver = AdmmSolver(gcs, params)

    solver.initialize()
    solver.update_local()


if __name__ == "__main__":
    main()
