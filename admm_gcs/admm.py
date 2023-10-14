from dataclasses import dataclass
from typing import Dict, List, NamedTuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
from pydrake.geometry.optimization import VPolytope

from admm_gcs.gcs import GCS, Edge, VertexId, path_to_edges
from admm_gcs.test_cases import create_test_graph, create_test_polytopes
from admm_gcs.tools import add_noise, calc_polytope_centroid


class EdgeVar(NamedTuple):
    xu: npt.NDArray[np.float64]
    xv: npt.NDArray[np.float64]


@dataclass
class AdmmParameters:
    rho: float = 1.0


class MultiblockADMMSolver:
    def __init__(self, gcs: GCS, params: AdmmParameters):
        self.gcs = gcs
        self.local_vars = {}  # one for each edge
        self.price_vars = {}  # one for each edge
        self.consensus_vars = {}  # one for each vertex
        self.discrete_vars = {}  # one for each edge

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
            self.consensus_vars[vertex_id] = self._calc_x_mean_for_vertex(
                vertex_id)

        self.update_discrete()

        for e in self.gcs.edges:
            violation_e = self._calc_consensus_violation(e)
            self.price_vars[e] = violation_e

    def update_local(self):
        """
        Update local variables (e.g., using projections onto convex sets).
        """

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

    def update_discrete(self) -> None:
        """
        Update the discrete variables representing which vertices are in the shortest path.
        """
        assert self.gcs.source is not None
        assert self.gcs.target is not None

        path = solve_discrete_spp(
            self.local_vars, self.gcs.source, self.gcs.target)
        edges_on_sp = path_to_edges(path)

        for e in self.gcs.vertices:
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
            self.price_vars[e].xu += violation.xu
            self.price_vars[e].xv += violation.xv

    def solve(self):
        """
        Solve the optimization problem using multi-block ADMM.
        """


def solve_discrete_spp(
    local_vars: Dict[Edge, EdgeVar], source: VertexId, target: VertexId
) -> List[VertexId]:
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
