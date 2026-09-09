from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bfgs_ce.methods.base import QuasiNewtonMethod


class PerryShanno(QuasiNewtonMethod):
    r"""Perry (1976) and Shanno (1978) memoryless quasi-Newton conjugate direction update (Paper §2, class C).

    Computes search direction without matrix storage using self-scaled secant projection:
    d_{k+1} = -\gamma_k g_{k+1} + ( <s_k, g_{k+1}> / <s_k, y_k> + \gamma_k <y_k, g_{k+1}> / <s_k, y_k> ) s_k
              - \gamma_k ( <s_k, g_{k+1}> / <s_k, y_k> ) y_k.
    When g_{k+1} \perp s_k, d_{k+1} satisfies eq:classC and aligns with the counterexample orbit.
    """

    def __init__(self, self_scaling: bool = True, name: str | None = None) -> None:
        name = name or ("Perry-Memoryless" if self_scaling else "Shanno-Memoryless")
        super().__init__(name=name)
        self.self_scaling = self_scaling
        self.last_s: NDArray[np.float64] | None = None
        self.last_y: NDArray[np.float64] | None = None

    def reset(self) -> None:
        self.last_s = None
        self.last_y = None

    def compute_direction(self, x: NDArray[np.float64], g: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.last_s is None or self.last_y is None:
            return -g

        s = self.last_s
        y = self.last_y
        sy = float(s @ y)
        yy = float(y @ y)

        if sy <= 1e-14 or yy <= 1e-14:
            return -g

        gamma = (sy / yy) if self.self_scaling else 1.0

        sg = float(s @ g)
        yg = float(y @ g)

        # Shanno (1978) eq (15) / Perry (1976)
        coeff_s = (sg / sy) + gamma * (yg / sy)
        coeff_y = gamma * (sg / sy)

        d = -gamma * g + coeff_s * s - coeff_y * y
        return d

    def update(self, s: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        self.last_s = s.copy()
        self.last_y = y.copy()
