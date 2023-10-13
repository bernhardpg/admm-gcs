from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from pydrake.geometry.optimization import VPolytope

from admm_gcs.visualize import plot_polytopes_with_edges

Vertex = VPolytope
Edge = Tuple[int, int]


@dataclass
class GCS:
    vertices: List[Vertex]
    edges: List[Edge]
    source_idx: Optional[int] = None
    target_idx: Optional[int] = None

    def plot(self) -> None:
        plot_polytopes_with_edges(self.vertices, self.edges)

    def set_source(self, idx: int) -> None:
        self.source_idx = idx

    def set_target(self, idx: int) -> None:
        self.target_idx = idx
