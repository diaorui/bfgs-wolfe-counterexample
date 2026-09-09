from __future__ import annotations

import numpy as np

from bfgs_ce.core.orbit import AlgebraicOrbit
from bfgs_ce.core.params import CounterexampleParams
from bfgs_ce.methods.bfgs import BFGS
from bfgs_ce.methods.lbfgs import LBFGS
from bfgs_ce.methods.perry_shanno import PerryShanno
from bfgs_ce.methods.powell_damped import PowellDampedBFGS


def test_class_c_parallel_directions() -> None:
    r"""Paper §3, Lemma "Heading lock": every Class C method has d_k parallel to s_k."""
    orbit = AlgebraicOrbit.generate(CounterexampleParams(n_steps=15))

    methods = [
        LBFGS(m=1, name="L-BFGS-1"),
        LBFGS(m=3, name="L-BFGS-3"),
        LBFGS(m=5, name="L-BFGS-5"),
        BFGS(),
        PowellDampedBFGS(),
        PerryShanno(self_scaling=True, name="Perry"),
        PerryShanno(self_scaling=False, name="Shanno"),
    ]

    for method in methods:
        method.reset()
        for k in range(len(orbit.ss)):
            x_k = orbit.xs[k]
            g_k = orbit.gs[k]
            d_k = method.compute_direction(x_k, g_k)

            norm_d = np.linalg.norm(d_k)
            assert norm_d > 1e-14
            cos_sim = float((d_k / norm_d) @ orbit.ss[k])

            # Directions must be collinear and pointing in the descent direction (cos_sim = 1.0)
            assert abs(cos_sim - 1.0) < 1e-11, f"{method.name} failed at step {k}: cos_sim={cos_sim}"

            # Update method with the exact step taken
            method.update(orbit.ss[k], orbit.ys[k])
