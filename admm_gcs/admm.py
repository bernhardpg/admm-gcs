import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pydrake.geometry.optimization import VPolytope

from admm_gcs.test_cases import create_test_polytopes
from admm_gcs.visualize import plot_polytope


def main():
    polytopes = create_test_polytopes()

    # visualize polytopes using matplotlib.
    fig, ax = plt.subplots()
    for poly in polytopes:
        plot_polytope(poly, ax, edgecolor="k", facecolor=np.random.rand(3), alpha=0.5)

    ax.set_aspect("equal", "box")  # type: ignore
    ax.set_xlim([0, 10])  # type: ignore
    ax.set_ylim([-5, 5])  # type: ignore
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("2d polytopes visualization")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
