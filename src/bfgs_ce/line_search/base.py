from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LineSearchResult:
    """Result of a 1D line search along ray x + t * d."""

    step_size: float             # Step length t
    step_vector: NDArray[np.float64] # Step s = t * d
    x_next: NDArray[np.float64]  # Landed point x_{k+1} = x_k + s
    f_next: float                # Function value f(x_{k+1})
    grad_next: NDArray[np.float64] # Gradient \nabla f(x_{k+1})
    armijo_ratio: float          # \rho = (f_{next} - f_curr) / <grad_curr, s>
    curvature_ratio: float       # |<grad_next, d>| / |<grad_curr, d>|
    success: bool                # Whether the line search condition was strictly met
