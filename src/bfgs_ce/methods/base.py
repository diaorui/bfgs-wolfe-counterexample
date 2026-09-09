from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class QuasiNewtonMethod(ABC):
    """Abstract base class for quasi-Newton optimization methods in Class C."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def reset(self) -> None:
        """Reset internal history and state."""
        pass

    @abstractmethod
    def compute_direction(self, x: NDArray[np.float64], g: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the search direction d_k given current position and gradient."""
        pass

    @abstractmethod
    def update(self, s: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Update inverse Hessian approximation / history with step s and gradient change y."""
        pass
