from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bfgs_ce.methods.base import QuasiNewtonMethod


class BFGS(QuasiNewtonMethod):
    r"""Standard full-memory BFGS updating explicit inverse Hessian approximation H_k (Paper §2, class C)."""

    def __init__(self, name: str = "BFGS") -> None:
        super().__init__(name=name)
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

        ys = float(y @ s)
        if ys <= 1e-14:
            return

        rho = 1.0 / ys
        I = np.eye(dim, dtype=np.float64)
        V = I - rho * np.outer(s, y)
        self.H = V @ self.H @ V.T + rho * np.outer(s, s)
