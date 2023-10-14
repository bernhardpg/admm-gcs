import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.animation import FuncAnimation
from pydrake.geometry.optimization import VPolytope

from admm_gcs.admm import AdmmParameters, MultiblockADMMSolver
from admm_gcs.gcs import GCS, Edge, VertexId, edges_to_path
from admm_gcs.test_cases import create_test_graph, create_test_polytopes
from admm_gcs.tools import add_noise, calc_polytope_centroid
from admm_gcs.visualize import plot_admm_solution


def plot_admm_graph(
    admm: MultiblockADMMSolver,
):
    plot_admm_solution(
        admm.gcs.vertices,
        admm.gcs.edges,
        admm.local_vars,
        admm.consensus_vars,
        admm.path,
    )


def main():
    gcs = create_test_graph()
    # gcs.plot()

    params = AdmmParameters()
    admm = MultiblockADMMSolver(gcs, params)

    admm.initialize()

    # Define a plot update function
    def update(frame):
        # Execute a step of the ADMM algorithm
        admm._step()

        # Ensure axes `ax` is cleared, not the whole figure
        ax.clear()

        # Update the plot using the same `ax`
        plot_admm_solution(
            ax,
            admm.gcs.vertices,
            admm.gcs.edges,
            admm.local_vars,
            admm.consensus_vars,
            admm.path,
        )

    # Prepare your figure and axes outside the update function
    fig, ax = plt.subplots(figsize=(10, 10))

    # Set up the animation on the `fig` created above
    ani = FuncAnimation(fig, update, frames=np.arange(
        0, 10), interval=2000, blit=False)

    # Save the animation
    ani.save("animation.mp4", writer="ffmpeg", fps=1)


if __name__ == "__main__":
    main()
