from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pydrake.geometry.optimization import VPolytope

from admm_gcs.admm import EdgeVar
from admm_gcs.gcs import Edge, VertexId
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


def plot_admm_solution(
    polytopes: Dict[VertexId, VPolytope],
    edges: List[Tuple[VertexId, VertexId]],
    local_vars: Optional[Dict[Edge, EdgeVar]] = None,
    consensus_vars: Optional[Dict[VertexId, np.ndarray]] = None,
    path: Optional[List[int]] = None,
):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot each polytope
    for polytope in polytopes.values():
        plot_polytope(polytope, ax, edgecolor="k", facecolor="c", alpha=0.2)

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
            alpha=0.5,
        )

    # Plot grey points if provided
    if local_vars is not None:
        grey_x = [point[0] for var in local_vars.values() for point in var]
        grey_y = [point[1] for var in local_vars.values() for point in var]
        plt.scatter(grey_x, grey_y, color="red")

    # Plot red points if provided
    if consensus_vars is not None:
        red_x = [point[0] for point in consensus_vars.values()]
        red_y = [point[1] for point in consensus_vars.values()]
        plt.scatter(red_x, red_y, color="grey")

    if path is not None:
        assert local_vars is not None

        for idx_curr, idx_next in zip(path[:-1], path[1:]):
            edge = (idx_curr, idx_next)
            local_var = local_vars[edge]

            # Draw an arrow from decision variables
            plt.arrow(
                local_var.xu[0],
                local_var.xu[1],
                local_var.xv[0] - local_var.xu[0],
                local_var.xv[1] - local_var.xu[1],
                shape="full",
                color="red",
                linewidth=2,
                length_includes_head=True,
                head_width=0.1,
                alpha=1.0,
            )

    # Set axis properties and show the plot
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Polytopes Visualization with Edges")
    plt.legend()
    plt.axis("equal")
    plt.show()
