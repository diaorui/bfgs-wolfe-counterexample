from __future__ import annotations

import numpy as np
import pytest

from bfgs_ce.core.function import CounterexampleFunction
from bfgs_ce.line_search.wolfe import strong_wolfe_line_search
from bfgs_ce.methods.lbfgs import LBFGS
from bfgs_ce.methods.perry_shanno import PerryShanno


@pytest.mark.parametrize("method_name, opt_cls, kwargs", [
    ("L-BFGS-1", LBFGS, {"m": 1}),
    ("L-BFGS-5", LBFGS, {"m": 5}),
    ("Perry", PerryShanno, {"self_scaling": True}),
    ("Shanno", PerryShanno, {"self_scaling": False}),
])
def test_end_to_end_escape_to_infinity(method_name: str, opt_cls: type, kwargs: dict) -> None:
    r"""Paper main theorem: Class C methods generate an unbounded sequence with \|g_k\| = R."""
    fun = CounterexampleFunction()
    p = fun.params
    N = 30

    opt = opt_cls(**kwargs)
    x = fun.orbit.xs[0].copy()

    for k in range(N):
        val, g = fun.evaluate(x)
        # Gradient norm never vanishes
        assert abs(np.linalg.norm(g) - p.R) < 1e-10

        # Method computes descent direction
        d = opt.compute_direction(x, g)

        # Line search accepts designed unit step
        res = strong_wolfe_line_search(fun, x, d, g)
        assert res.success

        # Update optimizer state
        opt.update(res.step_vector, res.grad_next - g)
        x = res.x_next

    # Unbounded trajectory: distance from origin grows proportionally to N
    final_distance = np.linalg.norm(x)
    assert final_distance > 0.9 * N
