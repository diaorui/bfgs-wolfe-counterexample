from __future__ import annotations

import numpy as np

from bfgs_ce.core.orbit import AlgebraicOrbit
from bfgs_ce.core.params import CounterexampleParams


def test_orbit_algebraic_invariants() -> None:
    """Paper §3, Lemma "Properties of the orbit"."""
    p = CounterexampleParams(n_steps=40)
    orbit = AlgebraicOrbit.generate(p)
    N = p.n_steps

    assert len(orbit.ss) == N
    assert len(orbit.xs) == N + 1
    assert len(orbit.gs) == N + 1

    for k in range(N):
        assert abs(np.linalg.norm(orbit.gs[k]) - p.R) < 1e-12
        assert abs(np.linalg.norm(orbit.ss[k]) - 1.0) < 1e-12
        assert abs(float(orbit.ss[k] @ orbit.gs[k + 1])) < 1e-12
        assert float(orbit.ds[k] @ orbit.gs[k]) < 0.0
        assert float(orbit.ss[k] @ orbit.ys[k]) > 0.0
        expected_cos = np.sin(np.pi / (2.0 ** (k + 1)))
        assert abs(orbit.cos_thetas[k] - expected_cos) < 1e-12


def test_heading_and_turning_angles() -> None:
    r"""Paper §3, Lemma "Secant decomposition and turn contraction": |ψ_k| = π/2^{k+1}."""
    orbit = AlgebraicOrbit.generate(CounterexampleParams(n_steps=20))
    for k in range(1, len(orbit.ss)):
        # Angle from s_{k-1} to s_k
        s_prev = orbit.ss[k - 1]
        s_curr = orbit.ss[k]
        dot_val = float(np.clip(s_prev @ s_curr, -1.0, 1.0))
        angle = np.arccos(dot_val)
        expected_angle = np.pi / (2.0 ** (k + 1))
        assert np.isclose(angle, expected_angle, rtol=1e-4, atol=1e-10)
