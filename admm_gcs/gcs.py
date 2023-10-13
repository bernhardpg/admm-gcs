from typing import Tuple

import numpy as np
import numpy.typing as npt
from pydrake.geometry.optimization import VPolytope

Vertex = VPolytope
Edge = Tuple[int, int]

VertexId = int


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
