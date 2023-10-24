from dataclasses import dataclass

from non_convex_admm.gcs import GCS

from admm_gcs.test_cases import create_test_graph


@dataclass
class AdmmParameters:
    rho: float = 1.0
    max_iterations: int = 50


class AdmmSolver:
    def __init__(self, gcs: GCS, params: AdmmParameters):
        self.iteration = 0
        self.params = params

    def initialize(self) -> None:
        ...

    def update_local(self) -> None:
        ...

    def update_consensus(self) -> None:
        ...

    def update_prices(self) -> None:
        ...

    def _step(self) -> None:
        self.update_local()
        self.update_consensus()
        self.update_prices()

        self.iteration += 1
        # TODO(bernhardpg): Update rho here

    def solve(self):
        """
        Solve the optimization problem using multi-block ADMM.
        """

        for it in range(self.params.max_iterations):
            self._step()


def main() -> None:
    G = create_test_graph()
    breakpoint()


if __name__ == "__main__":
    main()
