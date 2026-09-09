from __future__ import annotations

import numpy as np

from bfgs_ce.core.function import CounterexampleFunction
from bfgs_ce.core.smootherstep import I_S, JS_ONE, J_S, S, Sp, Spp


def test_S_paper_identities() -> None:
    u = np.linspace(0.0, 1.0, 401)
    s = np.asarray(S(u), dtype=np.float64)
    assert np.all(s >= 0.0) and np.all(s <= 1.0)
    assert float(S(-1.0)) == 0.0
    assert float(S(0.0)) == 0.0
    assert float(S(1.0)) == 1.0
    assert float(S(2.0)) == 1.0
    assert np.allclose(s + np.asarray(S(1.0 - u), dtype=np.float64), 1.0, atol=1e-12)
    assert abs(float(I_S(1.0)) - 0.5) < 1e-12
    assert abs(float(J_S(1.0)) - JS_ONE) < 1e-14
    assert abs(float(Sp(0.0))) < 1e-15
    assert abs(float(Sp(1.0))) < 1e-15
    assert abs(float(Spp(0.0))) < 1e-15
    assert abs(float(Spp(1.0))) < 1e-15
    assert float(Sp(0.5)) > 0.0


def test_Sp_matches_finite_difference() -> None:
    u = 0.4
    h = 1e-6
    fd = (float(S(u + h)) - float(S(u - h))) / (2.0 * h)
    assert abs(fd - float(Sp(u))) < 1e-6


def test_vertex_gradient_and_value_exactness() -> None:
    fun = CounterexampleFunction()
    orbit = fun.orbit

    for k in range(len(orbit.xs)):
        x_k = orbit.xs[k]
        val, grad = fun.evaluate(x_k)
        expected_g = orbit.gs[k]
        expected_f = fun.fs[k]

        assert abs(val - expected_f) < 1e-12, f"Value error at vertex {k}"
        assert np.linalg.norm(grad - expected_g) < 1e-12, f"Gradient error at vertex {k}"


def test_collar_c1_continuity() -> None:
    fun = CounterexampleFunction()
    orbit = fun.orbit
    delta = fun.params.delta
    eps = 1e-6

    k = 2
    x_interface = orbit.xs[k] + delta * orbit.ss[k]
    val_mid, grad_mid = fun.evaluate(x_interface)

    x_left = orbit.xs[k] + (delta - eps) * orbit.ss[k]
    val_left, grad_left = fun.evaluate(x_left)

    x_right = orbit.xs[k] + (delta + eps) * orbit.ss[k]
    val_right, grad_right = fun.evaluate(x_right)

    assert abs(val_left - val_mid) < 2 * eps
    assert abs(val_right - val_mid) < 2 * eps
    assert np.linalg.norm(grad_left - grad_mid) < 10 * eps
    assert np.linalg.norm(grad_right - grad_mid) < 10 * eps
