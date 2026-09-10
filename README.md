# Counterexamples for BFGS-type methods under arbitrary strong Wolfe constants

Companion Python package for

> Rui Diao, *Counterexamples for BFGS-type methods under arbitrary strong Wolfe constants*, [arXiv:2609.09686](https://arxiv.org/abs/2609.09686), 2026.

Repository: https://github.com/diaorui/bfgs-wolfe-counterexample

It implements the discrete orbit, Class $\mathcal{C}$ methods (full-memory BFGS, L-BFGS of any memory $m \ge 1$, Powell's damped BFGS, Perry, and Shanno), standard line searches (first local minimizer, strong/weak Wolfe, Armijo, Goldstein), and high-precision orbit checks via `mpmath`. On the polyline the interpolant matches the paper. Off the polyline the global cutoff $\zeta$ is replaced by a radial profile; every check in this package is evaluated on the orbit, where the two agree.

---

## 1. Layout

```
.
├── src/bfgs_ce/
│   ├── core/
│   │   ├── params.py          # R, c1, c2, delta = min(1/20, c1/6, (1-c2)/6)
│   │   ├── orbit.py           # Discrete algebraic orbit (x_k, s_k, g_k, y_k, d_k)
│   │   ├── smootherstep.py    # C^∞ transition S(u) = χ(u)/(χ(u)+χ(1-u))
│   │   ├── bumps.py           # Axial bumps Γ and antiderivatives
│   │   └── function.py        # Interpolant f(x), ∇f(x)
│   ├── methods/
│   │   ├── base.py
│   │   ├── bfgs.py
│   │   ├── lbfgs.py
│   │   ├── powell_damped.py
│   │   └── perry_shanno.py
│   ├── line_search/
│   │   ├── base.py
│   │   ├── exact.py
│   │   ├── wolfe.py
│   │   └── armijo_goldstein.py
│   └── high_precision/
│       └── mpmath_orbit.py
├── tests/
├── scripts/
│   └── run_verification.py
└── figures/
    ├── plot_orbit.py
    ├── plot_tube.py
    ├── plot_spike.py
    └── style.py
```

---

## 2. Installation

Python ≥ 3.9. For tests, install the `dev` extra:

```bash
pip install -e ".[dev]"
```

---

## 3. Tests

```bash
pytest -v
```

- `test_orbit_algebra.py`: orbit invariants $\|g_k\|\equiv R$, $\langle s_k, g_{k+1}\rangle=0$, $\cos\theta_k=\sin(\pi/2^{k+1})$.
- `test_class_c_invariance.py`: Class $\mathcal{C}$ directions $d_k \parallel s_k$.
- `test_smoothness.py`: paper $S$ identities and collar matching of $f$ and $\nabla f$.
- `test_line_search_criteria.py`: exact, strong/weak Wolfe, Armijo, Goldstein.
- `test_end_to_end_solver.py`: iterates escape with $\|g_k\|=R$.

---

## 4. Numerical checks and figures

```bash
python scripts/run_verification.py
```

The last block runs an `mpmath` orbit with up to 10,000 steps and 3,100 decimal digits; that part is slow.

```bash
python figures/plot_orbit.py   # -> figures/orbit.pdf
python figures/plot_tube.py    # -> figures/tube.pdf
python figures/plot_spike.py   # -> figures/spike.pdf
```

PDF/PNG outputs are generated locally and gitignored.

---

## 5. License

MIT. See `LICENSE`.
