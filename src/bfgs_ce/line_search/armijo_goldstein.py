from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bfgs_ce.core.function import CounterexampleFunction
from bfgs_ce.line_search.base import LineSearchResult
from bfgs_ce.line_search.exact import find_first_local_minimizer


def armijo_line_search(
    fun: CounterexampleFunction,
    x: NDArray[np.float64],
    d: NDArray[np.float64],
    g: NDArray[np.float64] | None = None,
    c1: float | None = None,
    tol: float = 1e-12,
) -> LineSearchResult:
    r"""Find and verify Armijo step along d (Paper §5, Lemma "Parameter calibration").

    Condition: f(x + t * d) <= f(x) + c1 * t * <g, d> <=> \rho >= c1.
    """
    c1 = c1 if c1 is not None else fun.params.c1

    if g is None:
        f_curr, g = fun.evaluate(x)
    else:
        f_curr, _ = fun.evaluate(x)

    gd = float(g @ d)
    if gd >= 0.0:
        raise ValueError("Search direction is not a descent direction!")

    # Autonomous 1D search for the first local minimizer satisfying Armijo condition
    t = find_first_local_minimizer(fun, x, d, g=g, tol=tol)

    s = t * d
    x_next = x + s
    f_next, grad_next = fun.evaluate(x_next)

    eta = float(g @ s)
    armijo_rho = (f_next - f_curr) / eta if eta != 0.0 else 0.0
    armijo_ok = armijo_rho >= c1 - 1e-12
    curv_ratio = abs(float(grad_next @ d)) / abs(gd) if gd != 0.0 else 0.0

    return LineSearchResult(
        step_size=t,
        step_vector=s,
        x_next=x_next,
        f_next=f_next,
        grad_next=grad_next,
        armijo_ratio=armijo_rho,
        curvature_ratio=curv_ratio,
        success=bool(armijo_ok),
    )


def goldstein_line_search(
    fun: CounterexampleFunction,
    x: NDArray[np.float64],
    d: NDArray[np.float64],
    g: NDArray[np.float64] | None = None,
    c1: float | None = None,
    c2: float | None = None,
    tol: float = 1e-12,
) -> LineSearchResult:
    r"""Find and verify Goldstein step along d (Paper §5, Lemma "Parameter calibration").

    Condition: c1 <= (f(x + t * d) - f(x)) / (t * <g, d>) <= c2 <=> c1 <= \rho <= c2.
    """
    c1 = c1 if c1 is not None else fun.params.c1
    c2 = c2 if c2 is not None else fun.params.c2

    if g is None:
        f_curr, g = fun.evaluate(x)
    else:
        f_curr, _ = fun.evaluate(x)

    gd = float(g @ d)
    if gd >= 0.0:
        raise ValueError("Search direction is not a descent direction!")

    # Autonomous 1D search for the first local minimizer satisfying Goldstein condition
    t = find_first_local_minimizer(fun, x, d, g=g, tol=tol)

    s = t * d
    x_next = x + s
    f_next, grad_next = fun.evaluate(x_next)

    eta = float(g @ s)
    armijo_rho = (f_next - f_curr) / eta if eta != 0.0 else 0.0
    goldstein_ok = (c1 - 1e-12 <= armijo_rho <= c2 + 1e-12)
    curv_ratio = abs(float(grad_next @ d)) / abs(gd) if gd != 0.0 else 0.0

    return LineSearchResult(
        step_size=t,
        step_vector=s,
        x_next=x_next,
        f_next=f_next,
        grad_next=grad_next,
        armijo_ratio=armijo_rho,
        curvature_ratio=curv_ratio,
        success=bool(goldstein_ok),
    )
