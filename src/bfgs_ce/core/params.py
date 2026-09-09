from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CounterexampleParams:
    r"""Parameters for the C^∞ counterexample function and line search.

    Attributes:
        R: Gradient norm on the orbit (Paper: \|g_k\| = R, default 1.0).
        c1: Armijo parameter in (0, 1).
        c2: Wolfe / Goldstein curvature parameter in (c1, 1).
        n_steps: Number of segments / steps to construct.
        target_rho: Target Armijo ratio \rho^* \in (c1, c2), defaults to midpoint (c1 + c2) / 2.
    """

    R: float = 1.0
    c1: float = 0.1
    c2: float = 0.9
    n_steps: int = 40
    target_rho: float | None = None

    def __post_init__(self) -> None:
        if self.R <= 0.0:
            raise ValueError("Gradient norm R must be strictly positive.")
        if not (0.0 < self.c1 < self.c2 < 1.0):
            raise ValueError(f"Line search parameters must satisfy 0 < c1 < c2 < 1; got c1={self.c1}, c2={self.c2}.")
        if self.n_steps < 2:
            raise ValueError("n_steps must be at least 2.")
        if self.target_rho is not None and not (self.c1 < self.target_rho < self.c2):
            raise ValueError(f"target_rho must be in (c1, c2); got {self.target_rho}.")

    @property
    def h(self) -> float:
        """Step length, normalized to 1.0 (Paper §3: every step has \\|s_k\\| = 1)."""
        return 1.0

    @property
    def delta(self) -> float:
        r"""Tube half-width (Paper §5, Lemma "Parameter calibration", eq:delta-armijo).

        \delta = \min(1/20, c1 / 6, (1 - c2) / 6).
        """
        return min(1.0 / 20.0, self.c1 / 6.0, (1.0 - self.c2) / 6.0)

    @property
    def r_inner(self) -> float:
        r"""Inner radius r_\circ = 3\delta / 5 (Paper §4.3).

        In the paper r_\circ is the plateau radius of the vertex bump V_k, the plateau
        half-width of the transverse window W, and the ramp width of the axial window A_k;
        here it is the radius of the inner tube where f \equiv F with no cutoff applied.
        """
        return 3.0 * self.delta / 5.0

    @property
    def rho_star(self) -> float:
        """Default target Armijo ratio: midpoint of (c1, c2)."""
        if self.target_rho is not None:
            return self.target_rho
        return 0.5 * (self.c1 + self.c2)
