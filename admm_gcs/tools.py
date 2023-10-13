import numpy as np
import numpy.typing as npt
from pydrake.geometry.optimization import VPolytope


def calc_polytope_centroid(polytope: VPolytope) -> npt.NDArray[np.float64]:
    centroid = np.mean(polytope.vertices(), axis=0)
    return centroid
