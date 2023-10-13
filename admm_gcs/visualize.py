from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pydrake.geometry.optimization import VPolytope

from admm_gcs.tools import calc_polytope_centroid


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


def plot_polytopes_with_edges(
    polytopes: Dict[int, VPolytope],
    edges: List[Tuple[int, int]],
    grey_points: Optional[List[np.ndarray]] = None,
    red_points: Optional[List[np.ndarray]] = None,
):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot each polytope
    for polytope in polytopes.values():
        plot_polytope(polytope, ax, edgecolor="k", facecolor="c", alpha=0.2)

    # Plot grey points if provided
    if grey_points is not None:
        grey_x = [point[0] for point in grey_points]
        grey_y = [point[1] for point in grey_points]
        plt.scatter(grey_x, grey_y, color="grey", label="Grey Points")

    # Plot red points if provided
    if red_points is not None:
        red_x = [point[0] for point in red_points]
        red_y = [point[1] for point in red_points]
        plt.scatter(red_x, red_y, color="red", label="Red Points")

    # Draw arrows for edges
    for u, v in edges:
        # Compute the centroids of the two polytopes
        centroid_u = calc_polytope_centroid(polytopes[u])
        centroid_v = calc_polytope_centroid(polytopes[v])

        # Draw an arrow from centroid of polytope u to centroid of polytope v
        plt.arrow(
            centroid_u[0],
            centroid_u[1],
            centroid_v[0] - centroid_u[0],
            centroid_v[1] - centroid_u[1],
            shape="full",
            color="black",
            linewidth=1,
            length_includes_head=True,
            head_width=0.1,
            alpha=0.3,
        )

    # Set axis properties and show the plot
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Polytopes Visualization with Edges")
    plt.legend()
    plt.axis("equal")
    plt.show()
