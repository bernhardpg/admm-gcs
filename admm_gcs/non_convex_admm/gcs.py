from functools import cached_property
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from pydrake.geometry.optimization import HPolyhedron, VPolytope

VertexId = int
Vertex = VPolytope
Edge = Tuple[VertexId, VertexId]
Path = List[VertexId]


class GCS:
    def __init__(self):
        self.vertices = {}
        self.edges = []
        self.target = None
        self.source = None

    def add_vertex(self, id: VertexId, convex_set: VPolytope):
        """
        Add a vertex with an associated convex set.
        """
        self.vertices[id] = convex_set

    def add_edge(self, vertex_a_id: VertexId, vertex_b_id: VertexId):
        """
        Add an edge with an associated cost function.
        """
        self.edges.append((vertex_a_id, vertex_b_id))

    def set_source(self, source_id: VertexId) -> None:
        self.source = source_id

    def set_target(self, target_id: VertexId) -> None:
        self.target = target_id

    @cached_property
    def h_polyhedrons(self) -> Dict[VertexId, HPolyhedron]:
        return {id: HPolyhedron(self.vertices[id]) for id in self.vertices}

    def get_incoming_edges(self, target: VertexId) -> List[Edge]:
        incoming_edges = [(u, v) for (u, v) in self.edges if (v == target)]
        return incoming_edges

    def get_outgoing_edges(self, target: VertexId) -> List[Edge]:
        outgoing_edges = [(u, v) for (u, v) in self.edges if (u == target)]
        return outgoing_edges

    @cached_property
    def edges_per_vertex(self) -> Dict[VertexId, List[Edge]]:
        return {
            target: self.get_incoming_edges(target) + self.get_outgoing_edges(target)
            for target in self.vertices
        }


def path_to_edges(path: Path) -> List[Edge]:
    edges = [(u, v) for u, v in zip(path[:-1], path[1:])]
    return edges


def edges_to_path(edges: List[Edge]) -> Path:
    vertices = [u for u, _ in edges] + [edges[-1][1]]
    return vertices
