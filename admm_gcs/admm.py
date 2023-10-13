from typing import List, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pydrake.geometry.optimization import VPolytope

from admm_gcs.gcs import GCS, Edge, VertexId
from admm_gcs.test_cases import create_test_graph, create_test_polytopes
from admm_gcs.tools import add_noise, calc_polytope_centroid
from admm_gcs.visualize import plot_polytopes_with_edges


class EdgeVar(NamedTuple):
    xu: npt.NDArray[np.float64]
    xv: npt.NDArray[np.float64]


class MultiblockADMMSolver:
    def __init__(self, gcs: GCS):
        self.gcs = gcs
        self.local_vars = {}
        self.consensus_vars = {}
        self.discrete_vars = []

    def initialize(self):
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

        for vertex_id, polytope in self.gcs.vertices.items():
            self.consensus_vars[vertex_id] = calc_polytope_centroid(polytope)

    def update_local(self):
        """
        Update local variables (e.g., using projections onto convex sets).
        """

    def update_consensus(self):
        """
        Update consensus variables.
        """

    def update_discrete(self):
        """
        Update the discrete variable representing which vertices are in the shortest path.
        """

    def solve(self):
        """
        Solve the optimization problem using multi-block ADMM.
        """

    def get_local_vars_flattened(self) -> List[npt.NDArray[np.float64]]:
        return sum([[var.xu, var.xv] for var in self.local_vars.values()], [])

    def get_consensus_vars_flattened(self) -> List[npt.NDArray[np.float64]]:
        return [x for x in self.consensus_vars.values()]


def plot_admm_graph(
    admm: MultiblockADMMSolver,
):
    plot_polytopes_with_edges(
        admm.gcs.vertices,
        admm.gcs.edges,
        admm.get_local_vars_flattened(),
        admm.get_consensus_vars_flattened(),
    )


def main():
    gcs = create_test_graph()
    # gcs.plot()

    admm = MultiblockADMMSolver(gcs)

    admm.initialize()
    plot_admm_graph(admm)


if __name__ == "__main__":
    main()
