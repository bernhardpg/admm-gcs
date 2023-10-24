from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt
from pydrake.geometry.optimization import VPolytope

T = TypeVar("T", bound=Any)


def squared_eucl_norm(v: npt.NDArray[T]) -> T:
    return v.dot(v)


def squared_eucl_distance(v1: npt.NDArray[T], v2: npt.NDArray[T]) -> T:
    dist = v1 - v2
    return dist.T.dot(dist)


def calc_polytope_centroid(polytope: VPolytope) -> npt.NDArray[np.float64]:
    centroid = np.mean(polytope.vertices(), axis=1)
    return centroid


def add_noise(vec: npt.NDArray[np.float64], scale=0.1) -> npt.NDArray[np.float64]:
    return vec + np.random.normal(loc=0, scale=scale, size=vec.shape)
