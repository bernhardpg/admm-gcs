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
        if (vertex_a_id, vertex_b_id) in self.edges:
            raise RuntimeError(
                f"Edge {(vertex_a_id, vertex_b_id)} already in the edge list!"
            )
        self.edges.append((vertex_a_id, vertex_b_id))

    def set_source(self, source_id: VertexId) -> None:
        self.source = source_id

    def set_target(self, target_id: VertexId) -> None:
        self.target = target_id

    @cached_property
    def h_polyhedrons(self) -> Dict[VertexId, HPolyhedron]:
        return {id: HPolyhedron(self.vertices[id]) for id in self.vertices}

    @cached_property
    def incoming_edges_per_vertex(self) -> Dict[VertexId, List[Edge]]:
        incoming_edges = {
            target: [(u, v) for (u, v) in self.edges if (v == target)]
            for target in self.vertices
        }
        return incoming_edges

    @cached_property
    def outgoing_edges_per_vertex(self) -> Dict[VertexId, List[Edge]]:
        outgoing_edges = {
            target: [(u, v) for (u, v) in self.edges if (u == target)]
            for target in self.vertices
        }
        return outgoing_edges

    @cached_property
    def edges_per_vertex(self) -> Dict[VertexId, List[Edge]]:
        return {
            target: self.incoming_edges_per_vertex[target]
            + self.outgoing_edges_per_vertex[target]
            for target in self.vertices
        }


def path_to_edges(path: Path) -> List[Edge]:
    edges = [(u, v) for u, v in zip(path[:-1], path[1:])]
    return edges


def edges_to_path(edges: List[Edge]) -> Path:
    vertices = [u for u, _ in edges] + [edges[-1][1]]
    return vertices
