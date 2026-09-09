from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bfgs_ce.methods.base import QuasiNewtonMethod


class PowellDampedBFGS(QuasiNewtonMethod):
    r"""Powell's damped BFGS algorithm for non-convex optimization (Powell 1978; Paper §2, class C).

    Ensures positive definiteness by damping y_k towards B_k s_k whenever s_k^T y_k < 0.2 s_k^T B_k s_k.
    Belongs to Class \mathcal{C} whenever g_{k+1} \perp s_k.
    """

    def __init__(self, damping_factor: float = 0.2, name: str = "Powell-Damped-BFGS") -> None:
        super().__init__(name=name)
        self.damping_factor = damping_factor
        self.H: NDArray[np.float64] | None = None

    def reset(self) -> None:
        self.H = None

    def compute_direction(self, x: NDArray[np.float64], g: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.H is None:
            self.H = np.eye(len(g), dtype=np.float64)
            return -g
        return -self.H @ g

    def update(self, s: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        dim = len(s)
        if self.H is None:
            self.H = np.eye(dim, dtype=np.float64)

        # Invert H to get B s = H^{-1} s
        # For 2D, direct 2x2 inverse is exact and fast
        B = np.linalg.inv(self.H)
        Bs = B @ s
        sBs = float(s @ Bs)
        sy = float(s @ y)

        if sy >= self.damping_factor * sBs:
            theta = 1.0
            r = y
        else:
            theta = ((1.0 - self.damping_factor) * sBs) / (sBs - sy)
            r = theta * y + (1.0 - theta) * Bs

        sr = float(s @ r)
        if sr <= 1e-14:
            return

        rho = 1.0 / sr
        I = np.eye(dim, dtype=np.float64)
        V = I - rho * np.outer(s, r)
        self.H = V @ self.H @ V.T + rho * np.outer(s, s)
