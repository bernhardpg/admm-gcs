import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from admm_gcs.convex_relaxation.admm import AdmmParameters, AdmmSolver
from admm_gcs.test_cases import RandomGcsParams, create_test_graph, generate_random_gcs
from admm_gcs.visualize import plot_gcs_graph, plot_gcs_relaxation


def main() -> None:
    # gcs = create_test_graph()

    gcs = generate_random_gcs(RandomGcsParams(num_vertices=15, seed=3, target_dist=8.0))
    plot_gcs_graph(gcs.vertices, gcs.edges, gcs.source, gcs.target, save_to_file=True)
    print("Saved gcs graph to file")

    params = AdmmParameters(
        rho=10.0,
        sigma=10.0,
        max_iterations=60,
        store_iterations=True,
        init_consensus_w_noise=True,
    )
    solver = AdmmSolver(gcs, params)

    solver.initialize()
    solver.solve()

    num_steps = solver.params.max_iterations

    animate = True
    if animate:
        # Define a plot update function
        def update(frame):
            # Ensure axes `ax` is cleared, not the whole figure
            ax.clear()  # type: ignore

            # Update the plot using the same `ax`
            plot_gcs_relaxation(
                ax,  # type: ignore
                gcs,
                solver.consensus_vars_history[frame],
            )

        # Prepare your figure and axes outside the update function
        fig, ax = plt.subplots(figsize=(10, 10))

        step_ms = 1000 / 3
        # Set up the animation on the `fig` created above
        ani = FuncAnimation(
            fig, update, frames=np.arange(0, num_steps), interval=step_ms, blit=False
        )

        # Save the animation
        ani.save("animation.mp4", writer="ffmpeg", fps=30)


if __name__ == "__main__":
    main()
