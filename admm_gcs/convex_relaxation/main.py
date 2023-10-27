import matplotlib.pyplot as plt
import numpy as np
import pydrake.geometry.optimization as opt
from matplotlib.animation import FuncAnimation
from pydrake.solvers import Binding, L2NormCost
from pydrake.symbolic import DecomposeLinearExpressions

from admm_gcs.convex_relaxation.admm import AdmmParameters, AdmmSolver, EdgeVars
from admm_gcs.non_convex_admm.gcs import GCS
from admm_gcs.test_cases import RandomGcsParams, create_test_graph, generate_random_gcs
from admm_gcs.visualize import plot_gcs_graph, plot_gcs_relaxation


def solve_with_drake(custom_gcs: GCS) -> None:
    gcs = opt.GraphOfConvexSets()

    vertices = {id: gcs.AddVertex(v, str(id)) for id, v in custom_gcs.vertices.items()}
    edges = {
        (u, v): gcs.AddEdge(vertices[u], vertices[v], str((u, v)))
        for u, v in custom_gcs.edges
    }

    for (u, v), e in edges.items():
        dist = e.xu() - e.xv()
        x = np.concatenate((e.xu(), e.xv()))
        A = DecomposeLinearExpressions(dist, x)

        cost = L2NormCost(A, np.zeros(e.xu().shape))
        e.AddCost(Binding[L2NormCost](cost, x))

    options = opt.GraphOfConvexSetsOptions()
    options.max_rounded_paths = 0
    options.convex_relaxation = True
    options.max_rounding_trials = 0
    options.preprocessing = False

    result = gcs.SolveShortestPath(
        vertices[custom_gcs.source], vertices[custom_gcs.target], options
    )
    assert result.is_success()

    edge_vars = {
        (u, v): EdgeVars(
            result.GetSolution(e.xu()),
            result.GetSolution(e.xv()),
            result.GetSolution(e.phi()),
        )
        for (u, v), e in edges.items()
    }

    return edge_vars


def main() -> None:
    gcs = create_test_graph()

    # gcs = generate_random_gcs(RandomGcsParams(num_vertices=15, seed=3, target_dist=8.0))
    # plot_gcs_graph(gcs.vertices, gcs.edges, gcs.source, gcs.target, save_to_file=True)
    print("Saved gcs graph to file")

    true_vars = solve_with_drake(gcs)

    params = AdmmParameters(
        rho=0.10,
        sigma=0.10,
        max_iterations=300,
        store_iterations=True,
        init_consensus_w_noise=False,
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
                true_vars,
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
