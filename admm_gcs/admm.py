import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pydrake.geometry.optimization import VPolytope

from admm_gcs.gcs import GCS
from admm_gcs.test_cases import create_test_graph, create_test_polytopes


class MultiblockADMMSolver:
    def __init__(self, gcs: GCS):
        self.gcs = gcs
        self.local_vars = {}
        self.consensus_vars = {}
        self.discrete_vars = []

    def initialize(self):
        """
        Initialize local and consensus variables.
        """

    def update_local(self):
        """
        Update local variables (e.g., using projections onto convex sets).
        """

    def update_consensus(self):
        """
        Update consensus variables.
        """

    def update_discrete(self):
        """
        Update the discrete variable representing which vertices are in the shortest path.
        """

    def solve(self):
        """
        Solve the optimization problem using multi-block ADMM.
        """


def main():
    gcs = create_test_graph()
    gcs.plot()

    admm = MultiblockADMMSolver(gcs)


if __name__ == "__main__":
    main()
