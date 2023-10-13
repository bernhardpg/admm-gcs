from typing import Dict, NamedTuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
from pydrake.geometry.optimization import VPolytope

from admm_gcs.gcs import GCS, Edge, VertexId
from admm_gcs.test_cases import create_test_graph, create_test_polytopes
from admm_gcs.tools import add_noise, calc_polytope_centroid


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

        assert self.gcs.source is not None
        assert self.gcs.target is not None

        self.discrete_vars = solve_discrete_spp(
            self.local_vars, self.gcs.source, self.gcs.target
        )

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


def solve_discrete_spp(
    local_vars: Dict[Edge, EdgeVar], source: VertexId, target: VertexId
) -> None:
    G = nx.Graph()

    def _sq_eucl_dist(xu, xv) -> float:
        diff = xu - xv
        return diff.T.dot(diff).item()

    for edge, var in local_vars.items():
        u, v = edge
        w = _sq_eucl_dist(var.xu, var.xv)
        G.add_edge(u, v, weight=w)

    path = nx.shortest_path(G, source=source, target=target)
    return path
