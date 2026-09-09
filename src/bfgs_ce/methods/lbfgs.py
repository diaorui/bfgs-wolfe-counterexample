from __future__ import annotations

from collections import deque
import numpy as np
from numpy.typing import NDArray

from bfgs_ce.methods.base import QuasiNewtonMethod


class LBFGS(QuasiNewtonMethod):
    r"""General Limited-memory BFGS (L-BFGS) supporting any memory m >= 1 (Paper §2, class C).

    For m = 1, this coincides with memoryless BFGS (L-BFGS-1).
    For any m >= 1, the newest secant pair ensures membership in Class \mathcal{C} whenever g_{k+1} \perp s_k.
    """

    def __init__(self, m: int | None = 5, scale_h0: bool = True, name: str | None = None) -> None:
        name = name or (f"L-BFGS-{m}" if m is not None else "L-BFGS-inf")
        super().__init__(name=name)
        self.m = m
        self.scale_h0 = scale_h0
        self.history: deque[tuple[NDArray[np.float64], NDArray[np.float64], float]] = (
            deque(maxlen=m) if m is not None else deque()
        )

    def reset(self) -> None:
        self.history.clear()

    def compute_direction(self, x: NDArray[np.float64], g: NDArray[np.float64]) -> NDArray[np.float64]:
        if len(self.history) == 0:
            # Steepest descent step d_0 = -g_0
            return -g

        # Two-loop recursion (Nocedal & Wright Algorithm 7.4)
        q = g.copy()
        alphas = []

        for s_i, y_i, rho_i in reversed(self.history):
            alpha_i = rho_i * float(s_i @ q)
            alphas.append(alpha_i)
            q -= alpha_i * y_i

        # Base matrix H_0 = gamma * I
        s_last, y_last, _ = self.history[-1]
        y_dot_y = float(y_last @ y_last)
        if self.scale_h0 and y_dot_y > 1e-14:
            gamma = float(s_last @ y_last) / y_dot_y
        else:
            gamma = 1.0

        r = gamma * q

        for (s_i, y_i, rho_i), alpha_i in zip(self.history, reversed(alphas)):
            beta = rho_i * float(y_i @ r)
            r += s_i * (alpha_i - beta)

        return -r

    def update(self, s: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        ys = float(y @ s)
        if ys > 1e-14:
            rho = 1.0 / ys
            self.history.append((s.copy(), y.copy(), rho))
