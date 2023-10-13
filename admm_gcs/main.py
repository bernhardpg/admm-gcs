import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pydrake.geometry.optimization import VPolytope

from admm_gcs.admm import MultiblockADMMSolver
from admm_gcs.gcs import GCS, Edge, VertexId
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
        admm.discrete_vars,
    )


def main():
    gcs = create_test_graph()
    # gcs.plot()

    admm = MultiblockADMMSolver(gcs)

    admm.initialize()
    plot_admm_graph(admm)


if __name__ == "__main__":
    main()
