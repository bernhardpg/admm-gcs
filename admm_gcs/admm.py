from typing import List

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pydrake.geometry.optimization import VPolytope


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


def plot_polytope(polytope: VPolytope, ax=None, **kwargs):
    """
    plots a 2d polytope using matplotlib.

    parameters:
        polytope (VPolytope): polytope to plot.
        ax (matplotlib.axes._axes.axes, optional): axes to plot on.
            if None, use current axes.
        **kwargs: additional arguments passed to plt.polygon.
    """
    # extract vertices from the polytope and plot.
    vertices = np.asarray(polytope.vertices())
    if vertices.shape[1] == 2:  # 2d polytope
        if ax is None:
            ax = plt.gca()
        poly = plt.Polygon(vertices, **kwargs)
        ax.add_patch(poly)
    else:
        raise ValueError("only 2d polytopes are supported for plotting.")


def main():
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

    # visualize polytopes using matplotlib.
    fig, ax = plt.subplots()
    for poly in polytopes:
        plot_polytope(poly, ax, edgecolor="k", facecolor=np.random.rand(3), alpha=0.5)

    ax.set_aspect("equal", "box")  # type: ignore
    ax.set_xlim([0, 10])  # type: ignore
    ax.set_ylim([-5, 5])  # type: ignore
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("2d polytopes visualization")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
