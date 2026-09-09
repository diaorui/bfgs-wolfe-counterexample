from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from bfgs_ce.core.function import CounterexampleFunction
from bfgs_ce.line_search.base import LineSearchResult


def find_first_local_minimizer(
    fun: CounterexampleFunction,
    x: NDArray[np.float64],
    d: NDArray[np.float64],
    g: NDArray[np.float64] | None = None,
    tol: float = 1e-12,
    max_steps: int = 1000,
) -> float:
    r"""Find the first local minimizer t* > 0 of \varphi(t) = f(x + t * d) on t >= 0 (Paper §5, Lemma "Line-search conditions along the orbit").

    This performs an autonomous 1D root-finding search along the descent ray without hardcoding
    any theoretical step size. Starting from t = 0 (where \varphi'(0) = <g, d> < 0), it advances
    adaptively until the directional derivative \varphi'(t) changes sign from negative to positive.
    The first root is then solved to machine precision using Brent's method (brentq).
    """
    norm_d = float(np.linalg.norm(d))
    if norm_d == 0.0:
        raise ValueError("Search direction has zero norm!")

    if g is None:
        _, g = fun.evaluate(x)

    gd = float(g @ d)
    if gd >= 0.0:
        raise ValueError("Search direction is not a descent direction!")

    delta = fun.params.delta
    # The forward walk must not step over the window past the vertex on which \varphi' > 0.
    # The paper guarantees that window has length r_\circ = 3*delta/5 (Paper §5, Lemma
    # "Line-search conditions along the orbit"); 0.25*delta below happens to sit under it,
    # but is not derived from it.
    h_fine = 0.25 * delta / norm_d
    h_coarse = 0.05 / norm_d
    abs_gd = abs(gd)

    def dphi(t: float) -> float:
        _, grad_t = fun.evaluate(x + t * d)
        return float(grad_t @ d)

    t = 0.0
    t_prev = 0.0

    for _ in range(max_steps):
        val_t = dphi(t)
        ratio = val_t / abs_gd
        if ratio >= 0.0:
            # First local minimizer bracketed in [t_prev, t]
            return float(brentq(dphi, t_prev, t, xtol=tol))

        t_prev = t
        # Adaptive step size: coarse on flat gradient interior, fine near root
        h = h_coarse if ratio < -0.1 else h_fine
        t += h

    raise RuntimeError("Failed to bracket first local minimizer within maximum steps!")


def exact_line_search(
    fun: CounterexampleFunction,
    x: NDArray[np.float64],
    d: NDArray[np.float64],
    g: NDArray[np.float64] | None = None,
    tol: float = 1e-12,
) -> LineSearchResult:
    r"""Find the first local minimizer step along d via 1D root-finding (Paper §5, Lemma "Line-search conditions along the orbit").

    Parameters:
    -----------
    fun : CounterexampleFunction
        The counterexample objective function.
    x : NDArray[np.float64]
        Current iterate x_k.
    d : NDArray[np.float64]
        Descent direction d_k.
    g : NDArray[np.float64] | None
        Current gradient g_k = \nabla f(x_k). If None, evaluated from fun.
    tol : float
        Tolerance for 1D root-finding.
    """
    if g is None:
        f_curr, g = fun.evaluate(x)
    else:
        f_curr, _ = fun.evaluate(x)

    gd = float(g @ d)
    if gd >= 0.0:
        raise ValueError("Search direction is not a descent direction!")

    # Autonomous 1D root-finding along ray x + t*d
    t_star = find_first_local_minimizer(fun, x, d, g=g, tol=tol)

    s = t_star * d
    x_next = x + s
    f_next, grad_next = fun.evaluate(x_next)

    eta = float(g @ s)
    armijo_rho = (f_next - f_curr) / eta if eta != 0.0 else 0.0
    curv_ratio = abs(float(grad_next @ d)) / abs(gd)

    return LineSearchResult(
        step_size=t_star,
        step_vector=s,
        x_next=x_next,
        f_next=f_next,
        grad_next=grad_next,
        armijo_ratio=armijo_rho,
        curvature_ratio=curv_ratio,
        success=True,
    )
