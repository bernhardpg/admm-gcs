from typing import List

import numpy as np
import numpy.typing as npt
from pydrake.geometry.optimization import VPolytope

from admm_gcs.gcs import GCS


def get_vertices(idx: int) -> npt.NDArray[np.float64]:
    """
    Returns a polytope given some idx. Function intended for constructing tests.
    """
    # Vertices for various polytopes
    vertices_list = [
        np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),  # Square
        np.array([[2, 2], [3, 2], [2.5, 3]]),  # Triangle
        np.array([[4, 4], [5, 4], [5.5, 5], [5, 6], [4, 6]]),  # Pentagon
        np.array([[7, 7], [8, 7], [8.5, 7.5], [8, 8], [7, 8], [6.5, 7.5]]),  # Hexagon
        np.array(
            [
                [10, 10],
                [11, 10],
                [11.5, 10.5],
                [11, 11],
                [10.5, 11.5],
                [10, 11],
                [9.5, 10.5],
            ]
        ),  # Heptagon
    ]
    return vertices_list[idx % len(vertices_list)]


def create_centered_polytopes(desired_centroids: List[List]):
    num_polytopes = len(desired_centroids)

    vertices_list = [get_vertices(idx) for idx in range(num_polytopes)]
    positioned_polytopes = []
    for vertices, desired_centroid in zip(vertices_list, desired_centroids):
        # Compute the centroid of the polytope
        original_centroid = np.mean(vertices, axis=0)

        # Compute the translation vector from the original centroid to the desired centroid
        translation_vector = np.array(desired_centroid) - original_centroid

        # Translate the vertices to the desired centroid
        translated_vertices = vertices + translation_vector

        # Create a VPolytope with the translated vertices and add it to the list
        polytope = VPolytope(translated_vertices)
        positioned_polytopes.append(polytope)

    return positioned_polytopes


def create_test_polytopes() -> List[VPolytope]:
    # Example usage:
    desired_centroids = [
        [1, 0],
        [2, 2],
        [2, -2],
        [4, 2],
        [5, -2],
        [7, 2],
        [7, -2],
        [9, 0],
    ]
    polytopes = create_centered_polytopes(desired_centroids)
    return polytopes


def create_test_graph() -> GCS:
    polytopes = create_test_polytopes()
    edges = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 7)]

    gcs = GCS()
    for idx, v in enumerate(polytopes):
        gcs.add_vertex(idx, v)

    for u, v in edges:
        gcs.add_edge(u, v)

    gcs.set_source(0)
    gcs.set_source(len(polytopes))

    return gcs
