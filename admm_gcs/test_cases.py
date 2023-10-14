import random
from typing import List

import networkx as nx
import numpy as np
import numpy.typing as npt
from pydrake.geometry.optimization import VPolytope

from admm_gcs.gcs import GCS
from admm_gcs.tools import calc_polytope_centroid


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


def is_point_too_close(point, points, min_distance):
    for existing_point in points:
        distance = np.sqrt(
            np.sum((np.array(point) - np.array(existing_point)) ** 2))
        if distance < min_distance:
            return True  # Point is too close to another point
    return False  # Point is not too close to any other point


def generate_random_points(
    num_points, x_min, x_max, y_min, y_max, min_distance
) -> List[List[float]]:
    points = []
    while len(points) < num_points:
        new_point = (random.uniform(x_min, x_max),
                     random.uniform(x_min, y_max))
        if not points or not is_point_too_close(new_point, points, min_distance):
            points.append(new_point)
    return points


def create_random_polytopes(num_polytopes: int, min_distance: float) -> VPolytope:
    x_min, x_max = 0, 15  # Min and max for x coordinate
    y_min, y_max = 0, 15  # Min and max for y coordinate
    points = generate_random_points(
        num_polytopes, x_min, x_max, y_min, y_max, min_distance
    )
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
    N = 30  # Number of vertices
    min_distance = 0.8

    polytopes = create_random_polytopes(N, min_distance)
    source = random.randint(1, N - 1)

    def _dist(u_id, v_id) -> float:
        source_centroid = calc_polytope_centroid(polytopes[u_id])
        target_centroid = calc_polytope_centroid(polytopes[v_id])
        dist = np.linalg.norm(source_centroid - target_centroid)
        return dist  # type: ignore

    edges = []
    min_dist = 4.0
    for u in range(N):
        for v in range(N):
            if u == v:
                continue
            if _dist(u, v) < min_dist:
                edges.append((u, v))
                edges.append((v, u))

    G = nx.DiGraph()
    G.add_edges_from(edges)

    # Find the largest connected component
    largest_cc = max(nx.strongly_connected_components(G), key=len)

    # Subgraph with the largest connected component
    G_connected = G.subgraph(largest_cc).copy()

    vertices = list(G_connected.nodes)
    edges = list(G_connected.edges)

    print("Created random edges...")
    print(f"Num edges: {len(edges)}")

    target = source

    TRES = 8.5
    while _dist(source, target) < TRES:
        target = random.choice(vertices)

    # # Create edges by sampling paths from source
    # edges = []
    # num_paths = 20
    # edge_tres = 2.5
    # for _ in range(num_paths):
    #     path_length = random.randint(1, round(N / 3))
    #     if len(edges) == 0:
    #         v = source
    #     else:
    #         v = random.choice(edges)[1]
    #     next_v = v
    #     for idx in range(path_length):
    #         while next_v == v or _dist(v, next_v) > edge_tres:
    #             next_v = random.randint(1, N - 1)
    #
    #         edge = (v, next_v)
    #         if edge not in edges:
    #             edges.append(edge)
    #             print(f"dist: {_dist(v, next_v)}")
    #         v = next_v

    # find a connected target
    # length_from_source = 0
    # while length_from_source < 3:
    #     v = source
    #     length_from_source = 0
    #     while True:
    #         edges_from_v = [e for e in edges if e[0] == v]
    #         if len(edges_from_v) == 0:
    #             break
    #         next_v = random.choice(edges_from_v)[1]
    #         if next_v == source:
    #             break
    #         else:
    #             length_from_source += 1
    #             v = next_v
    # target = v
    # print(f"Length from source to target: {length_from_source}")

    # Add edges to vertices that are not connected
    # for v in range(N):
    #     if not any([v in edge for edge in edges]):
    #         u_1 = random.choice(edges)[0]
    #         edges.append((v, u_1))
    #         u_2 = random.choice(edges)[1]
    #         edges.append((u_2, v))

    gcs = GCS()
    for id in vertices:
        gcs.add_vertex(id, polytopes[id])

    for u, v in edges:
        gcs.add_edge(u, v)

    gcs.set_source(source)
    gcs.set_target(target)

    return gcs
