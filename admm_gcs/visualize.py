from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pydrake.geometry.optimization import VPolytope

from admm_gcs.admm import EdgeVar
from admm_gcs.gcs import Edge, VertexId
from admm_gcs.tools import calc_polytope_centroid


def plot_polytope(polytope: VPolytope, ax=None, name: Optional[str] = None, **kwargs):
    """
    plots a 2d polytope using matplotlib.

    parameters:
        polytope (VPolytope): polytope to plot.
        ax (matplotlib.axes._axes.axes, optional): axes to plot on.
            if None, use current axes.
        **kwargs: additional arguments passed to plt.polygon.
    """
    # extract vertices from the polytope and plot.
    vertices = np.asarray(polytope.vertices().T)
    if vertices.shape[1] == 2:  # 2d polytope
        if ax is None:
            ax = plt.gca()
        poly = plt.Polygon(vertices, **kwargs)
        ax.add_patch(poly)

        # If a name is provided, display it above the polytope
        if name is not None:
            # Compute the centroid of the polytope
            centroid = np.mean(vertices, axis=0)
            ax.text(centroid[0], centroid[1], name, ha="center", va="bottom")

    else:
        raise ValueError("only 2d polytopes are supported for plotting.")


def _set_lims(polytopes, ax, buffer=0.1) -> None:
    # Extract all vertices
    all_vertices = np.vstack([poly.vertices().T for poly in polytopes])

    # Find min and max coordinates along x and y
    min_x, min_y = np.min(all_vertices, axis=0)
    max_x, max_y = np.max(all_vertices, axis=0)

    # Apply buffer
    x_buffer = buffer * (max_x - min_x)
    y_buffer = buffer * (max_y - min_y)

    # Set axis limits
    ax.set_xlim(min_x - x_buffer, max_x + x_buffer)
    ax.set_ylim(min_y - y_buffer, max_y + y_buffer)


def plot_gcs_graph(
    polytopes: Dict[VertexId, VPolytope],
    edges: List[Tuple[VertexId, VertexId]],
    source: Optional[VertexId] = None,
    target: Optional[VertexId] = None,
    save_to_file: bool = True,
):
    fig, ax = plt.subplots()

    # Plot each polytope
    for id, polytope in polytopes.items():
        if id == source:
            facecolor = "green"
            name = str(id) + " (s)"
        elif id == target:
            facecolor = "orange"
            name = str(id) + " (t)"
        else:
            facecolor = "c"
            name = str(id)
        plot_polytope(
            polytope, ax, name=name, edgecolor="k", facecolor=facecolor, alpha=0.2
        )

    # Draw arrows for edges
    for u, v in edges:
        # Compute the centroids of the two polytopes
        centroid_u = calc_polytope_centroid(polytopes[u])
        centroid_v = calc_polytope_centroid(polytopes[v])

        # Draw an arrow from centroid of polytope u to centroid of polytope v
        ax.arrow(
            centroid_u[0],
            centroid_u[1],
            centroid_v[0] - centroid_u[0],
            centroid_v[1] - centroid_u[1],
            shape="full",
            color="black",
            linewidth=1,
            length_includes_head=True,
            head_width=0.1,
            alpha=0.1,
        )

    # Set axis properties and show the plot
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Polytopes Visualization with Edges")
    ax.axis("equal")

    _set_lims(polytopes.values(), ax)
    if save_to_file:
        plt.savefig("gcs.png", dpi=300)
    else:
        plt.show()


def plot_admm_solution(
    ax: plt.Axes,  # Add ax parameter here
    polytopes: Dict[VertexId, VPolytope],
    edges: List[Tuple[VertexId, VertexId]],
    source: Optional[VertexId] = None,
    target: Optional[VertexId] = None,
    local_vars: Optional[Dict[Edge, EdgeVar]] = None,
    consensus_vars: Optional[Dict[VertexId, np.ndarray]] = None,
    path: Optional[List[int]] = None,
):
    ax.clear()  # Clear previous plots on ax

    # Plot each polytope
    for id, polytope in polytopes.items():
        if id == source:
            facecolor = "green"
            name = str(id) + " (s)"
        elif id == target:
            facecolor = "orange"
            name = str(id) + " (t)"
        else:
            facecolor = "c"
            name = str(id)
        plot_polytope(
            polytope, ax, name=name, edgecolor="k", facecolor=facecolor, alpha=0.2
        )

    # Draw arrows for edges
    for u, v in edges:
        # Compute the centroids of the two polytopes
        centroid_u = calc_polytope_centroid(polytopes[u])
        centroid_v = calc_polytope_centroid(polytopes[v])

        # Draw an arrow from centroid of polytope u to centroid of polytope v
        ax.arrow(
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

    # if local_vars is not None:
    #     grey_x = [point[0] for var in local_vars.values() for point in var]
    #     grey_y = [point[1] for var in local_vars.values() for point in var]
    #     ax.scatter(grey_x, grey_y, color="red")

    if consensus_vars is not None:
        red_x = [point[0] for point in consensus_vars.values()]
        red_y = [point[1] for point in consensus_vars.values()]
        ax.scatter(red_x, red_y, color="grey")

    if path is not None:
        assert local_vars is not None

        for idx_curr, idx_next in zip(path[:-1], path[1:]):
            edge = (idx_curr, idx_next)
            local_var = local_vars[edge]

            # Draw an arrow from decision variables
            ax.arrow(
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
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Polytopes Visualization with Edges")
    ax.axis("equal")

    _set_lims(polytopes.values(), ax)
