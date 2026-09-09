r"""Paper transition S(u) = χ(u) / (χ(u) + χ(1-u)), χ(t) = exp(-1/t) 1_{t>0}.

S ∈ C^∞(R), S ≡ 0 on (-∞, 0], S ≡ 1 on [1, ∞), S(u) + S(1-u) = 1, ∫_0^1 S = 1/2.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import cumulative_trapezoid


def _maybe_float(u: ArrayLike, res: NDArray[np.float64]):
    if np.ndim(np.asarray(u)) == 0:
        return float(np.reshape(res, ()))
    return res


def S(u: ArrayLike) -> NDArray[np.float64] | float:
    u_arr = np.asarray(u, dtype=np.float64)
    out = np.zeros(np.shape(u_arr), dtype=np.float64)
    out[u_arr >= 1.0] = 1.0
    mid = (u_arr > 0.0) & (u_arr < 1.0)
    if np.any(mid):
        um = u_arr[mid]
        w = (1.0 - 2.0 * um) / (um * (1.0 - um))
        w = np.clip(w, -60.0, 60.0)
        out[mid] = 1.0 / (1.0 + np.exp(w))
    return _maybe_float(u, out)


def Sp(u: ArrayLike) -> NDArray[np.float64] | float:
    u_arr = np.asarray(u, dtype=np.float64)
    out = np.zeros(np.shape(u_arr), dtype=np.float64)
    mid = (u_arr > 1e-12) & (u_arr < 1.0 - 1e-12)
    if np.any(mid):
        um = np.asarray(u_arr[mid], dtype=np.float64)
        s = np.asarray(S(um), dtype=np.float64)
        q = 1.0 / um**2 + 1.0 / (1.0 - um) ** 2
        val = s * (1.0 - s) * q
        out[mid] = np.where(np.isfinite(val), val, 0.0)
    return _maybe_float(u, out)


def Spp(u: ArrayLike) -> NDArray[np.float64] | float:
    u_arr = np.asarray(u, dtype=np.float64)
    out = np.zeros(np.shape(u_arr), dtype=np.float64)
    mid = (u_arr > 1e-8) & (u_arr < 1.0 - 1e-8)
    if np.any(mid):
        um = np.asarray(u_arr[mid], dtype=np.float64)
        s = np.asarray(S(um), dtype=np.float64)
        q = 1.0 / um**2 + 1.0 / (1.0 - um) ** 2
        qp = -2.0 / um**3 + 2.0 / (1.0 - um) ** 3
        val = s * (1.0 - s) * ((1.0 - 2.0 * s) * q**2 + qp)
        out[mid] = np.where(np.isfinite(val), val, 0.0)
    return _maybe_float(u, out)


def _integral_tables(n: int = 20001) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    ug = np.linspace(0.0, 1.0, n)
    sg = np.asarray(S(ug), dtype=np.float64)
    ig = cumulative_trapezoid(sg, ug, initial=0.0)
    if ig[-1] > 0.0:
        ig *= 0.5 / ig[-1]
    jg = cumulative_trapezoid(ig, ug, initial=0.0)
    return ug, ig, jg


_UG, _IG, _JG = _integral_tables()
JS_ONE = float(_JG[-1])


def I_S(u: ArrayLike) -> NDArray[np.float64] | float:
    u_arr = np.asarray(u, dtype=np.float64)
    out = np.empty(np.shape(u_arr), dtype=np.float64)
    lo = u_arr <= 0.0
    hi = u_arr >= 1.0
    mid = ~lo & ~hi
    out[lo] = 0.0
    out[hi] = 0.5 + (u_arr[hi] - 1.0)
    if np.any(mid):
        out[mid] = np.interp(u_arr[mid], _UG, _IG)
    return _maybe_float(u, out)


def J_S(u: ArrayLike) -> NDArray[np.float64] | float:
    u_arr = np.asarray(u, dtype=np.float64)
    out = np.empty(np.shape(u_arr), dtype=np.float64)
    lo = u_arr <= 0.0
    hi = u_arr >= 1.0
    mid = ~lo & ~hi
    out[lo] = 0.0
    du = u_arr[hi] - 1.0
    out[hi] = JS_ONE + 0.5 * du + 0.5 * du**2
    if np.any(mid):
        out[mid] = np.interp(u_arr[mid], _UG, _JG)
    return _maybe_float(u, out)


def S_interval(t: ArrayLike, a: float, b: float) -> NDArray[np.float64] | float:
    return S((np.asarray(t, dtype=np.float64) - a) / (b - a))


def Sp_interval(t: ArrayLike, a: float, b: float) -> NDArray[np.float64] | float:
    return Sp((np.asarray(t, dtype=np.float64) - a) / (b - a)) / (b - a)


def Spp_interval(t: ArrayLike, a: float, b: float) -> NDArray[np.float64] | float:
    return Spp((np.asarray(t, dtype=np.float64) - a) / (b - a)) / ((b - a) ** 2)


def int_S_interval(t: ArrayLike, a: float, b: float) -> NDArray[np.float64] | float:
    w = b - a
    return w * I_S((np.asarray(t, dtype=np.float64) - a) / w)


def double_int_S_interval(t: ArrayLike, a: float, b: float) -> NDArray[np.float64] | float:
    w = b - a
    return (w ** 2) * J_S((np.asarray(t, dtype=np.float64) - a) / w)
