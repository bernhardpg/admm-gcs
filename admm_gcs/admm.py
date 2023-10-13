import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pydrake.geometry.optimization import VPolytope

from admm_gcs.test_cases import create_test_graph, create_test_polytopes


def main():
    gcs = create_test_graph()
    gcs.plot()


if __name__ == "__main__":
    main()
