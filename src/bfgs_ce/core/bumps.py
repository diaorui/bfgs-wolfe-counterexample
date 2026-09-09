r"""Joining bump \Gamma(\tau; \alpha, M) and its exact analytic integrals (Paper §4.1, eq:Lambdaminus).

\Gamma(\tau; \alpha, M) = \alpha(1 - S(\tau; \delta, 2\delta)) + M(S(\tau; \delta, 2\delta) - S(\tau; 2\delta, 3\delta)).
Supported on [0, 3\delta], with \Gamma \equiv \alpha on [0, \delta] and \Gamma \equiv 0 on [3\delta, \infty).
Total mass: \int_0^{3\delta} \Gamma(\tau; \alpha, M) d\tau = 1.5 \alpha \delta + M \delta.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray

from bfgs_ce.core.smootherstep import (
    JS_ONE,
    S_interval,
    Sp_interval,
    double_int_S_interval,
    int_S_interval,
)


def gamma_bump(t: ArrayLike, alpha: float, M: float, delta: float) -> NDArray[np.float64]:
    r"""Evaluate \Gamma(\tau; \alpha, M) on R."""
    t_arr = np.asarray(t, dtype=np.float64)
    res = np.zeros_like(t_arr)

    # Regime 1: tau <= delta: constant alpha
    m1 = t_arr <= delta
    res[m1] = alpha

    # Regime 2: delta < tau <= 2*delta: blend from alpha to M
    m2 = (t_arr > delta) & (t_arr <= 2.0 * delta)
    s2 = S_interval(t_arr[m2], delta, 2.0 * delta)
    res[m2] = alpha * (1.0 - s2) + M * s2

    # Regime 3: 2*delta < tau <= 3*delta: blend from M to 0
    m3 = (t_arr > 2.0 * delta) & (t_arr <= 3.0 * delta)
    s3 = S_interval(t_arr[m3], 2.0 * delta, 3.0 * delta)
    res[m3] = M * (1.0 - s3)

    # Regime 4: tau > 3*delta: identically 0
    return res if isinstance(t, (np.ndarray, list, tuple)) else float(res)


def gamma_bump_deriv(t: ArrayLike, alpha: float, M: float, delta: float) -> NDArray[np.float64]:
    r"""First derivative d/d\tau \Gamma(\tau; \alpha, M)."""
    t_arr = np.asarray(t, dtype=np.float64)
    res = np.zeros_like(t_arr)

    m2 = (t_arr > delta) & (t_arr <= 2.0 * delta)
    sp2 = Sp_interval(t_arr[m2], delta, 2.0 * delta)
    res[m2] = (M - alpha) * sp2

    m3 = (t_arr > 2.0 * delta) & (t_arr <= 3.0 * delta)
    sp3 = Sp_interval(t_arr[m3], 2.0 * delta, 3.0 * delta)
    res[m3] = -M * sp3

    return res if isinstance(t, (np.ndarray, list, tuple)) else float(res)


def int_gamma_bump(t: ArrayLike, alpha: float, M: float, delta: float) -> NDArray[np.float64]:
    r"""First antiderivative G(\tau) = \int_0^\tau \Gamma(u; \alpha, M) du."""
    t_arr = np.asarray(t, dtype=np.float64)
    res = np.zeros_like(t_arr)

    # tau <= delta: alpha * tau
    m1 = t_arr <= delta
    res[m1] = alpha * t_arr[m1]

    # delta < tau <= 2*delta
    m2 = (t_arr > delta) & (t_arr <= 2.0 * delta)
    u2 = t_arr[m2]
    res[m2] = alpha * delta + alpha * (u2 - delta) + (M - alpha) * int_S_interval(u2, delta, 2.0 * delta)

    # 2*delta < tau <= 3*delta
    m3 = (t_arr > 2.0 * delta) & (t_arr <= 3.0 * delta)
    u3 = t_arr[m3]
    G_2d = 2.0 * alpha * delta + (M - alpha) * 0.5 * delta
    res[m3] = G_2d + M * (u3 - 2.0 * delta) - M * int_S_interval(u3, 2.0 * delta, 3.0 * delta)

    # tau > 3*delta
    m4 = t_arr > 3.0 * delta
    res[m4] = 1.5 * alpha * delta + M * delta

    return res if isinstance(t, (np.ndarray, list, tuple)) else float(res)


def double_int_gamma_bump(t: ArrayLike, alpha: float, M: float, delta: float) -> NDArray[np.float64]:
    r"""Second antiderivative H(\tau) = \int_0^\tau \int_0^u \Gamma(v; \alpha, M) dv du."""
    t_arr = np.asarray(t, dtype=np.float64)
    res = np.zeros_like(t_arr)

    H_d = 0.5 * alpha * (delta ** 2)
    G_d = alpha * delta

    G_2d = 2.0 * alpha * delta + (M - alpha) * 0.5 * delta
    H_2d = H_d + G_d * delta + 0.5 * alpha * (delta ** 2) + (M - alpha) * (delta ** 2) * JS_ONE

    G_3d = 1.5 * alpha * delta + M * delta
    H_3d = H_2d + G_2d * delta + 0.5 * M * (delta ** 2) - M * (delta ** 2) * JS_ONE

    m1 = t_arr <= delta
    res[m1] = 0.5 * alpha * (t_arr[m1] ** 2)

    m2 = (t_arr > delta) & (t_arr <= 2.0 * delta)
    u2 = t_arr[m2]
    res[m2] = H_d + G_d * (u2 - delta) + 0.5 * alpha * ((u2 - delta) ** 2) + (M - alpha) * double_int_S_interval(u2, delta, 2.0 * delta)

    m3 = (t_arr > 2.0 * delta) & (t_arr <= 3.0 * delta)
    u3 = t_arr[m3]
    res[m3] = H_2d + G_2d * (u3 - 2.0 * delta) + 0.5 * M * ((u3 - 2.0 * delta) ** 2) - M * double_int_S_interval(u3, 2.0 * delta, 3.0 * delta)

    m4 = t_arr > 3.0 * delta
    res[m4] = H_3d + G_3d * (t_arr[m4] - 3.0 * delta)

    return res if isinstance(t, (np.ndarray, list, tuple)) else float(res)


@dataclass(frozen=True)
class SegmentProfile:
    r"""Complete axial profile a(\tau), a'(\tau), a''(\tau) for segment k (Paper §4.1)."""

    k: int
    alpha_k: float
    alpha_next: float
    M_k: float
    lambda_k: float
    delta: float
    eta_k: float
    f_k: float

    @property
    def M_left(self) -> float:
        return self.lambda_k * self.M_k

    @property
    def M_right(self) -> float:
        return (1.0 - self.lambda_k) * self.M_k

    def a_pp(self, tau: ArrayLike) -> NDArray[np.float64]:
        r"""Axial curvature a''(\tau) = \Gamma(\tau; \alpha_k, \lambda_k M_k) + \Gamma(1-\tau; \alpha_{k+1}, (1-\lambda_k)M_k)."""
        tau_arr = np.asarray(tau, dtype=np.float64)
        left = gamma_bump(tau_arr, self.alpha_k, self.M_left, self.delta)
        right = gamma_bump(1.0 - tau_arr, self.alpha_next, self.M_right, self.delta)
        return left + right

    def a_p(self, tau: ArrayLike) -> NDArray[np.float64]:
        r"""Axial derivative a'(\tau) = \eta_k + \int_0^\tau a''(u) du."""
        tau_arr = np.asarray(tau, dtype=np.float64)
        int_left = int_gamma_bump(tau_arr, self.alpha_k, self.M_left, self.delta)
        tot_right = 1.5 * self.alpha_next * self.delta + self.M_right * self.delta
        int_right = tot_right - int_gamma_bump(1.0 - tau_arr, self.alpha_next, self.M_right, self.delta)
        return self.eta_k + int_left + int_right

    def a(self, tau: ArrayLike) -> NDArray[np.float64]:
        r"""Axial profile a(\tau) = f_k + \int_0^\tau a'(u) du."""
        tau_arr = np.asarray(tau, dtype=np.float64)
        dint_left = double_int_gamma_bump(tau_arr, self.alpha_k, self.M_left, self.delta)
        tot_right = 1.5 * self.alpha_next * self.delta + self.M_right * self.delta
        H_1_right = double_int_gamma_bump(np.array([1.0]), self.alpha_next, self.M_right, self.delta)[0]
        dint_right = tot_right * tau_arr - (H_1_right - double_int_gamma_bump(1.0 - tau_arr, self.alpha_next, self.M_right, self.delta))
        return self.f_k + self.eta_k * tau_arr + dint_left + dint_right

    @property
    def f_next(self) -> float:
        r"""Arrival value f_{k+1} = a(1)."""
        return float(self.a(np.array([1.0]))[0])

    @property
    def armijo_ratio(self) -> float:
        r"""Actual Armijo ratio \rho_k = (f_{k+1} - f_k) / \eta_k."""
        return (self.f_next - self.f_k) / self.eta_k
