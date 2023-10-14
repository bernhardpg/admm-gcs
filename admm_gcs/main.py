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

    params = AdmmParameters(rho=1)
    admm = MultiblockADMMSolver(gcs, params)

    admm.initialize()

    # Define a plot update function
    def update(frame):
        # Ensure axes `ax` is cleared, not the whole figure
        ax.clear()  # type: ignore

        # Update the plot using the same `ax`
        plot_admm_solution(
            ax,  # type: ignore
            admm.gcs.vertices,
            admm.gcs.edges,
            admm.local_vars,
            admm.consensus_vars,
            admm.path,
        )

        # if frame % 3 == 0:
        #     admm.update_local()
        # if frame % 3 == 1:
        #     admm.update_consensus()
        # if frame % 3 == 2:
        #     admm.update_discrete()
        #     admm.update_prices()

        admm._step()

    # Prepare your figure and axes outside the update function
    fig, ax = plt.subplots(figsize=(10, 10))

    step_ms = 1000
    # Set up the animation on the `fig` created above
    ani = FuncAnimation(
        fig, update, frames=np.arange(0, 150), interval=step_ms, blit=False
    )

    # Save the animation
    ani.save("animation.mp4", writer="ffmpeg", fps=30)

    print(f"Local solve time (mean): {np.mean(admm.local_solve_times)} s")
    print(f"Local solve time (max): {np.max(admm.local_solve_times)} s")
    print(f"Local solve time (min): {np.min(admm.local_solve_times)} s")

    print(f"SPP solve time (mean): {np.mean(admm.discrete_solve_times)} s")
    print(f"SPP solve time (max): {np.max(admm.discrete_solve_times)} s")
    print(f"SPP solve time (min): {np.min(admm.discrete_solve_times)} s")

    print(
        f"Individual QP time (mean): {np.mean(admm.individual_qp_solve_times)} s")
    print(
        f"Individual QP time (max): {np.max(admm.individual_qp_solve_times)} s")
    print(
        f"Individual QP time (min): {np.min(admm.individual_qp_solve_times)} s")


if __name__ == "__main__":
    main()
