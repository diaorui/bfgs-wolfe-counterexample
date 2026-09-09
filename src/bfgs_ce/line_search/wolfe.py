from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bfgs_ce.core.function import CounterexampleFunction
from bfgs_ce.line_search.base import LineSearchResult
from bfgs_ce.line_search.exact import find_first_local_minimizer


def strong_wolfe_line_search(
    fun: CounterexampleFunction,
    x: NDArray[np.float64],
    d: NDArray[np.float64],
    g: NDArray[np.float64] | None = None,
    c1: float | None = None,
    c2: float | None = None,
    tol: float = 1e-12,
) -> LineSearchResult:
    r"""Find and verify Strong Wolfe step along d (Paper §5, Lemma "Line-search conditions along the orbit").

    Conditions:
    1. Armijo: f(x + t * d) <= f(x) + c1 * t * <g, d>
    2. Strong curvature: |<grad(x + t * d), d>| <= c2 * |<g, d>|

    Parameters:
    -----------
    fun : CounterexampleFunction
        The counterexample objective function.
    x : NDArray[np.float64]
        Current iterate x_k.
    d : NDArray[np.float64]
        Descent direction d_k.
    g : NDArray[np.float64] | None
        Current gradient g_k. If None, evaluated from fun.
    c1, c2 : float | None
        Wolfe parameters (defaults to fun.params.c1, fun.params.c2).
    tol : float
        Tolerance for 1D root-finding.
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

    # Autonomous 1D search for the first local minimizer satisfying Wolfe conditions
    t = find_first_local_minimizer(fun, x, d, g=g, tol=tol)

    s = t * d
    x_next = x + s
    f_next, grad_next = fun.evaluate(x_next)

    eta = float(g @ s)
    armijo_rho = (f_next - f_curr) / eta if eta != 0.0 else 0.0

    # Armijo check: \rho >= c1
    armijo_ok = armijo_rho >= c1 - 1e-12

    # Strong curvature check: |<grad_next, d>| <= c2 * |<g, d>|
    curv_lhs = abs(float(grad_next @ d))
    curv_rhs = c2 * abs(gd)
    curv_ratio = curv_lhs / abs(gd) if gd != 0.0 else 0.0
    curv_ok = curv_lhs <= curv_rhs + 1e-12

    return LineSearchResult(
        step_size=t,
        step_vector=s,
        x_next=x_next,
        f_next=f_next,
        grad_next=grad_next,
        armijo_ratio=armijo_rho,
        curvature_ratio=curv_ratio,
        success=bool(armijo_ok and curv_ok),
    )


def weak_wolfe_line_search(
    fun: CounterexampleFunction,
    x: NDArray[np.float64],
    d: NDArray[np.float64],
    g: NDArray[np.float64] | None = None,
    c1: float | None = None,
    c2: float | None = None,
    tol: float = 1e-12,
) -> LineSearchResult:
    r"""Find and verify Weak Wolfe step along d (Paper §5, Lemma "Line-search conditions along the orbit").

    Conditions:
    1. Armijo: f(x + t * d) <= f(x) + c1 * t * <g, d>
    2. Weak curvature: <grad(x + t * d), d> >= c2 * <g, d>
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

    # Autonomous 1D search for the first local minimizer satisfying Weak Wolfe conditions
    t = find_first_local_minimizer(fun, x, d, g=g, tol=tol)

    s = t * d
    x_next = x + s
    f_next, grad_next = fun.evaluate(x_next)

    eta = float(g @ s)
    armijo_rho = (f_next - f_curr) / eta if eta != 0.0 else 0.0

    armijo_ok = armijo_rho >= c1 - 1e-12
    curv_lhs = float(grad_next @ d)
    curv_rhs = c2 * gd
    curv_ratio = abs(curv_lhs) / abs(gd) if gd != 0.0 else 0.0
    curv_ok = curv_lhs >= curv_rhs - 1e-12

    return LineSearchResult(
        step_size=t,
        step_vector=s,
        x_next=x_next,
        f_next=f_next,
        grad_next=grad_next,
        armijo_ratio=armijo_rho,
        curvature_ratio=curv_ratio,
        success=bool(armijo_ok and curv_ok),
    )
