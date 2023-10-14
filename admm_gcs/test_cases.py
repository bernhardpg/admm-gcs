import random
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
        np.array([[7, 7], [8, 7], [8.5, 7.5], [8, 8],
                 [7, 8], [6.5, 7.5]]),  # Hexagon
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
        polytope = VPolytope(translated_vertices.T)
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
        [7, 0],
        [3, 0.5],
        [5, -0.5],
        [6, 0.5],
        [3, -1],
    ]
    polytopes = create_centered_polytopes(desired_centroids)
    return polytopes


def create_test_graph() -> GCS:
    polytopes = create_test_polytopes()
    edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
        (5, 7),
        (6, 7),
        (5, 8),
        (8, 7),
        (1, 9),
        (9, 10),
        (10, 8),
        (11, 8),
        (10, 8),
        (3, 11),
        (9, 11),
        (2, 12),
        (12, 4),
        (0, 12),
    ]

    gcs = GCS()
    for idx, v in enumerate(polytopes):
        gcs.add_vertex(idx, v)

    for u, v in edges:
        gcs.add_edge(u, v)

    gcs.set_source(0)
    gcs.set_target(7)

    return gcs


def generate_random_points(n_points, x_min, x_max, y_min, y_max) -> List[List[float]]:
    points = []
    for _ in range(n_points):
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        point = [x, y]
        points.append(point)
    return points


def create_random_polytopes(num_polytopes: int) -> VPolytope:
    x_min, x_max = 0, 10  # Min and max for x coordinate
    y_min, y_max = 0, 10  # Min and max for y coordinate
    points = generate_random_points(num_polytopes, x_min, x_max, y_min, y_max)
    polytopes = create_centered_polytopes(points)
    return polytopes


def generate_random_edges(N, p):
    """
    Generate a graph with N vertices, where each possible edge is
    included with probability p.

    Parameters:
    - N (int): Number of vertices.
    - p (float): Probability of edge creation.
    """
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if random.uniform(0, 1) < p:
                edges.append((i, j))
    return edges


def generate_random_gcs(seed: int = 123):
    random.seed(seed)

    # Example usage:
    N = 25  # Number of vertices
    p = 0.23  # Probability of edge creation

    polytopes = create_random_polytopes(N)
    edges = generate_random_edges(len(polytopes), p)

    # Add edges to vertices that are not connected
    for v in range(N):
        if not any([v in edge for edge in edges]):
            u_1 = random.choice(edges)[0]
            edges.append((v, u_1))
            u_2 = random.choice(edges)[1]
            edges.append((u_2, v))

    source = random.randint(1, N)
    # find a connected target
    v = source
    length_from_source = 0
    while True:
        edges_from_v = [e for e in edges if e[0] == v]
        if len(edges_from_v) == 0:
            break
        next_v = random.choice(edges_from_v)[1]
        if next_v == source:
            break
        else:
            length_from_source += 1
            v = next_v

    print(f"Length from source to target: {length_from_source}")
    breakpoint()
    target = v

    gcs = GCS()
    for idx, v in enumerate(polytopes):
        gcs.add_vertex(idx, v)

    for u, v in edges:
        gcs.add_edge(u, v)

    gcs.set_source(source)
    gcs.set_target(target)

    return gcs
