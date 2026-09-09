from __future__ import annotations

import numpy as np

from bfgs_ce.core.function import CounterexampleFunction
from bfgs_ce.line_search.armijo_goldstein import armijo_line_search, goldstein_line_search
from bfgs_ce.line_search.exact import exact_line_search
from bfgs_ce.line_search.wolfe import strong_wolfe_line_search, weak_wolfe_line_search


def test_line_search_conditions_on_all_segments() -> None:
    """Paper §5, Lemmas "Parameter calibration" and "Line-search conditions along the orbit":
    every designed step satisfies all standard line search rules."""
    fun = CounterexampleFunction()
    orbit = fun.orbit
    p = fun.params

    for k in range(len(orbit.ss)):
        x_k = orbit.xs[k]
        d_k = orbit.ds[k]
        g_k = orbit.gs[k]

        # 1. Exact line search (autonomous 1D root-finding on \varphi'(t) = 0)
        res_exact = exact_line_search(fun, x_k, d_k, g_k)
        assert res_exact.success
        assert abs(float(res_exact.grad_next @ orbit.ss[k])) < 1e-10
        if k <= 15:
            assert abs(np.linalg.norm(res_exact.step_vector) - 1.0) < 1e-10
        else:
            assert abs(np.linalg.norm(res_exact.step_vector) - 1.0) < 1e-4

        # 2. Strong Wolfe line search (autonomous 1D search)
        res_sw = strong_wolfe_line_search(fun, x_k, d_k, g_k, c1=p.c1, c2=p.c2)
        assert res_sw.success
        assert p.c1 <= res_sw.armijo_ratio <= p.c2

        # 3. Weak Wolfe line search (autonomous 1D search)
        res_ww = weak_wolfe_line_search(fun, x_k, d_k, g_k, c1=p.c1, c2=p.c2)
        assert res_ww.success

        # 4. Armijo line search (autonomous 1D search)
        res_armijo = armijo_line_search(fun, x_k, d_k, g_k, c1=p.c1)
        assert res_armijo.success

        # 5. Goldstein line search (autonomous 1D search)
        res_goldstein = goldstein_line_search(fun, x_k, d_k, g_k, c1=p.c1, c2=p.c2)
        assert res_goldstein.success
