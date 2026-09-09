from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from bfgs_ce.core.params import CounterexampleParams


def rot90_ccw(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Counter-clockwise quarter turn J(a, b) = (-b, a)."""
    return np.array([-v[1], v[0]], dtype=np.float64)


@dataclass
class AlgebraicOrbit:
    """Discrete algebraic orbit (x_k, s_k, g_k, y_k, d_k) of Paper §3 (eq:orbit).

    Only the directions are re-derived rather than read off eq:orbit: at each step d_k is
    taken on the class-C ray, the unique ray in y_{k-1}^\\perp with <d, g_k> < 0 whose
    existence and uniqueness are Paper §2, Lemma "Conjugate descent ray". The explicit
    2-D representative d_k = <y_{k-1}, g_k> s_{k-1} - <s_{k-1}, y_{k-1}> g_k used below
    does not appear in the paper; it is one vector on that ray, and s_k is its unit vector.
    The gradients are the orbit's own prescription g_{k+1} = (-1)^{k+1} J s_k from
    eq:orbit, with the sign selected by <g_{k+1}, g_k> < 0 instead of by parity.
    """

    params: CounterexampleParams
    xs: NDArray[np.float64]       # Vertices x_k, shape (N+1, 2)
    ss: NDArray[np.float64]       # Unit step directions s_k, shape (N, 2)
    gs: NDArray[np.float64]       # Gradients g_k, shape (N+1, 2)
    ys: NDArray[np.float64]       # Differences y_k = g_{k+1} - g_k, shape (N, 2)
    ds: NDArray[np.float64]       # Unnormalized search directions d_k, shape (N, 2)
    nus: NDArray[np.float64]      # Normal vectors \nu_k = J s_k, shape (N, 2)
    thetas: NDArray[np.float64]   # Heading angles \theta_k, shape (N,)
    cos_thetas: NDArray[np.float64] # \cos\theta_k, shape (N,)
    alphas: NDArray[np.float64]   # Curvature alphas \alpha_k, shape (N+1,)
    etas: NDArray[np.float64]     # Inner products \eta_k = <g_k, s_k>, shape (N,)
    Ms: NDArray[np.float64]       # Height budgets M_k, shape (N,)

    @classmethod
    def generate(cls, params: CounterexampleParams | None = None) -> AlgebraicOrbit:
        """Generate the discrete orbit recursively up to params.n_steps."""
        p = params or CounterexampleParams()
        N = p.n_steps
        R = p.R
        delta = p.delta

        xs = np.zeros((N + 1, 2), dtype=np.float64)
        ss = np.zeros((N, 2), dtype=np.float64)
        gs = np.zeros((N + 1, 2), dtype=np.float64)
        ys = np.zeros((N, 2), dtype=np.float64)
        ds = np.zeros((N, 2), dtype=np.float64)
        nus = np.zeros((N, 2), dtype=np.float64)
        thetas = np.zeros(N, dtype=np.float64)
        cos_thetas = np.zeros(N, dtype=np.float64)
        alphas = np.zeros(N + 1, dtype=np.float64)
        etas = np.zeros(N, dtype=np.float64)
        Ms = np.zeros(N, dtype=np.float64)

        # Base step k = 0 (Paper §3, eq:orbit)
        xs[0] = np.array([0.0, 0.0])
        ss[0] = np.array([1.0, 0.0])
        gs[0] = np.array([-R, 0.0])
        gs[1] = np.array([0.0, -R])
        ys[0] = gs[1] - gs[0]
        ds[0] = -gs[0]  # = (R, 0)
        nus[0] = rot90_ccw(ss[0])
        xs[1] = xs[0] + ss[0]

        cos_thetas[0] = float(-gs[0] @ ss[0]) / R  # = 1.0
        thetas[0] = np.arccos(np.clip(cos_thetas[0], -1.0, 1.0))
        etas[0] = float(gs[0] @ ss[0])             # = -R
        alphas[0] = 0.0                            # Definition alpha_0 = 0

        # Recursive steps k = 1, ..., N - 1 (Paper eq:orbit)
        for k in range(1, N):
            # d_k = <y_{k-1}, g_k> s_{k-1} - <s_{k-1}, y_{k-1}> g_k
            coeff_s = float(ys[k - 1] @ gs[k])
            coeff_g = float(ss[k - 1] @ ys[k - 1])
            d_k = coeff_s * ss[k - 1] - coeff_g * gs[k]
            ds[k] = d_k

            norm_d = np.linalg.norm(d_k)
            if norm_d == 0.0:
                raise ZeroDivisionError(f"Direction d_{k} vanished!")
            s_k = d_k / norm_d
            ss[k] = s_k
            nus[k] = rot90_ccw(s_k)
            xs[k + 1] = xs[k] + s_k

            # g_{k+1} = -sign(<J s_k, g_k>) * R * (J s_k)
            js = rot90_ccw(s_k)
            sign_js = np.sign(float(js @ gs[k]))
            if sign_js == 0.0:
                sign_js = 1.0
            g_next = -sign_js * R * js
            gs[k + 1] = g_next
            ys[k] = g_next - gs[k]

            # Scalars
            cos_th = float(-gs[k] @ s_k) / R
            cos_thetas[k] = cos_th
            thetas[k] = np.arccos(np.clip(cos_th, -1.0, 1.0))
            etas[k] = float(gs[k] @ s_k)
            alphas[k] = cos_thetas[k - 1]  # alpha_k = cos\theta_{k-1}

        alphas[N] = cos_thetas[N - 1]

        # Height budgets M_k = alpha_{k+1} / delta - 1.5 * (alpha_k + alpha_{k+1})
        for k in range(N):
            Ms[k] = alphas[k + 1] / delta - 1.5 * (alphas[k] + alphas[k + 1])

        return cls(
            params=p,
            xs=xs,
            ss=ss,
            gs=gs,
            ys=ys,
            ds=ds,
            nus=nus,
            thetas=thetas,
            cos_thetas=cos_thetas,
            alphas=alphas,
            etas=etas,
            Ms=Ms,
        )
