from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bfgs_ce.core.function import CounterexampleFunction
from bfgs_ce.core.params import CounterexampleParams
from bfgs_ce.high_precision.mpmath_orbit import verify_orbit_high_precision
from bfgs_ce.line_search.armijo_goldstein import armijo_line_search, goldstein_line_search
from bfgs_ce.line_search.exact import exact_line_search
from bfgs_ce.line_search.wolfe import strong_wolfe_line_search, weak_wolfe_line_search
from bfgs_ce.methods.bfgs import BFGS
from bfgs_ce.methods.lbfgs import LBFGS
from bfgs_ce.methods.perry_shanno import PerryShanno
from bfgs_ce.methods.powell_damped import PowellDampedBFGS


def main() -> None:
    print("================================================================================")
    print("     COUNTEREXAMPLES FOR BFGS-TYPE METHODS: NUMERICAL VERIFICATION              ")
    print("================================================================================\n")

    p = CounterexampleParams(c1=0.1, c2=0.9, n_steps=40)
    fun = CounterexampleFunction(p)
    orbit = fun.orbit

    print(f"Parameters: c1 = {p.c1}, c2 = {p.c2}, delta = {p.delta:.6f}, steps = {p.n_steps}")
    print(f"Theoretical target Armijo ratio: rho* = {p.rho_star}\n")

    print("--- 1. End-to-end run (L-BFGS m=5, strong Wolfe; paper Table) ---")
    opt = LBFGS(m=5)
    x = orbit.xs[0].copy()

    records = []

    for k in range(p.n_steps):
        val, g = fun.evaluate(x)
        dist = float(np.linalg.norm(x))
        gnorm = float(np.linalg.norm(g))

        d = opt.compute_direction(x, g)
        ls_res = strong_wolfe_line_search(fun, x, d, g)

        records.append({
            "k": k,
            "x": x.copy(),
            "dist": dist,
            "val": val,
            "gnorm": gnorm,
            "rho": ls_res.armijo_ratio,
            "curv": ls_res.curvature_ratio,
        })

        opt.update(ls_res.step_vector, ls_res.grad_next - g)
        x = ls_res.x_next

    # Final point
    val_end, g_end = fun.evaluate(x)
    records.append({
        "k": p.n_steps,
        "x": x.copy(),
        "dist": float(np.linalg.norm(x)),
        "val": val_end,
        "gnorm": float(np.linalg.norm(g_end)),
        "rho": None,
        "curv": None,
    })

    # Print sample steps
    sample_indices = [0, 1, 2, 5, 10, 20, 30, 40]
    print(f"{'k':>3} | {'x_k':>24} | {'||x_k||':>8} | {'f(x_k)':>10} | {'||g_k||':>8} | {'rho_k':>8} | {'|<g+,d>/<g,d>|':>14}")
    print("-" * 92)
    for idx in sample_indices:
        rec = records[idx]
        xk_str = f"[{rec['x'][0]:8.4f}, {rec['x'][1]:8.4f}]"
        rho_str = f"{rec['rho']:8.4f}" if rec['rho'] is not None else "     N/A"
        curv_str = f"{rec['curv']:14.2e}" if rec['curv'] is not None else "           N/A"
        print(f"{rec['k']:3d} | {xk_str:>24} | {rec['dist']:8.4f} | {rec['val']:10.4f} | {rec['gnorm']:8.4f} | {rho_str} | {curv_str}")

    e2e_ok = True
    for rec in records[:-1]:
        if abs(rec["gnorm"] - p.R) > 1e-8:
            e2e_ok = False
        if rec["rho"] is not None and abs(rec["rho"] - p.rho_star) > 1e-4:
            e2e_ok = False

    print("\n--- 2. Class C heading lock ---")
    methods = [
        LBFGS(m=1, name="L-BFGS-1"),
        LBFGS(m=3, name="L-BFGS-3"),
        LBFGS(m=5, name="L-BFGS-5"),
        BFGS(),
        PowellDampedBFGS(),
        PerryShanno(self_scaling=True, name="Perry"),
        PerryShanno(self_scaling=False, name="Shanno"),
    ]

    class_c_ok = True
    for m in methods:
        m.reset()
        max_cos_err = 0.0
        for k in range(p.n_steps):
            xk = orbit.xs[k]
            gk = orbit.gs[k]
            dk = m.compute_direction(xk, gk)
            unit_d = dk / np.linalg.norm(dk)
            cos_sim = float(unit_d @ orbit.ss[k])
            err = abs(cos_sim - 1.0)
            if err > max_cos_err:
                max_cos_err = err
            m.update(orbit.ss[k], orbit.ys[k])
        if max_cos_err >= 1e-10:
            class_c_ok = False
        print(f"  * {m.name:<22}: max |cos(angle(d_k, s_k)) - 1| = {max_cos_err:.2e}")

    print("\n--- 3. Line Search Criteria Satisfaction ---")
    all_exact_ok = True
    all_sw_ok = True
    all_ww_ok = True
    all_armijo_ok = True
    all_goldstein_ok = True

    for k in range(p.n_steps):
        xk, dk, gk = orbit.xs[k], orbit.ds[k], orbit.gs[k]
        all_exact_ok &= exact_line_search(fun, xk, dk, gk).success
        all_sw_ok &= strong_wolfe_line_search(fun, xk, dk, gk).success
        all_ww_ok &= weak_wolfe_line_search(fun, xk, dk, gk).success
        all_armijo_ok &= armijo_line_search(fun, xk, dk, gk).success
        all_goldstein_ok &= goldstein_line_search(fun, xk, dk, gk).success

    print(f"  * Exact / First Local Minimizer: {'PASS' if all_exact_ok else 'FAIL'}")
    print(f"  * Strong Wolfe (c1={p.c1}, c2={p.c2}): {'PASS' if all_sw_ok else 'FAIL'}")
    print(f"  * Weak Wolfe (c1={p.c1}, c2={p.c2}):   {'PASS' if all_ww_ok else 'FAIL'}")
    print(f"  * Armijo (c1={p.c1}):                  {'PASS' if all_armijo_ok else 'FAIL'}")
    print(f"  * Goldstein (c1={p.c1}, c2={p.c2}):    {'PASS' if all_goldstein_ok else 'FAIL'}")

    print("\n--- 4. Arbitrary-precision (mpmath) long-horizon orbit ---")
    hp_ok = True
    for steps, dps in [(100, 80), (1000, 350), (10000, 3100)]:
        res = verify_orbit_high_precision(n_steps=steps, dps=dps)
        if res["max_grad_err"] >= 1e-20 or res["max_ortho_err"] >= 1e-20:
            hp_ok = False
        print(f"  * Steps: {res['n_steps']:>5d} | Precision: {res['dps']:>4d} dps | Dist: {res['final_dist']:8.2f} | Max ||g|| err: {res['max_grad_err']:.2e} | Max <s,g+> err: {res['max_ortho_err']:.2e}")

    ok = e2e_ok and class_c_ok and all_exact_ok and all_sw_ok and all_ww_ok and all_armijo_ok and all_goldstein_ok and hp_ok
    print("\n================================================================================")
    if ok:
        print("                    ALL NUMERICAL VERIFICATIONS CONFIRMED                       ")
        print("================================================================================")
        return
    print("                    SOME NUMERICAL VERIFICATIONS FAILED                         ")
    print("================================================================================")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
