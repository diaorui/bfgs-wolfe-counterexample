from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bfgs_ce.core.bumps import SegmentProfile, gamma_bump_deriv
from bfgs_ce.core.orbit import AlgebraicOrbit
from bfgs_ce.core.params import CounterexampleParams
from bfgs_ce.core.smootherstep import S, S_interval, Sp, Sp_interval


class CounterexampleFunction:
    r"""Tubular interpolant of the discrete orbit (x_k, g_k).

    On the polyline P the values and gradients match the paper. The pieces are:
    - Quadratic vertex and collar pieces Q_k(x) on \mathcal{B}_k \cup \mathcal{C}_k
    - Segment templates a(\tau) + b(\tau) n + \frac{1}{2} a''(\tau) n^2 on middle tubes \mathcal{T}_k
    - A cutoff blending to the exterior constant f_\infty outside \mathcal{U}.

    Note on the cutoff: this is a simplified stand-in for the paper's
    \zeta = S(\Phi) with \Phi = \sum_k V_k + \sum_k A_k(\tau_k) W(n_k) (Paper §4.3).
    Here a single radial profile in the distance to the polyline P is used instead.
    The two agree on P (both identically 1, Paper §4.3 Lemma "Core plateau and support")
    and outside \mathcal{U} (both 0); in between they are different functions. In
    particular the paper's \zeta \equiv 1 only on P \cup \bigcup_k B(x_k, r_\circ) together
    with the axial mid-band, not on the whole sausage {dist(x, P) <= r_\circ} on which the
    profile here is 1; and the profile here is only Lipschitz, not C^\infty, across the
    corner bisectors where the nearest segment switches. Neither difference matters for
    this package: every quantity verified here is evaluated on P, where dist = 0 and the
    cutoff branch is never taken.
    """

    def __init__(self, params: CounterexampleParams | None = None) -> None:
        self.params = params or CounterexampleParams()
        self.orbit = AlgebraicOrbit.generate(self.params)
        p = self.params
        o = self.orbit
        N = p.n_steps

        # Compute vertex values f_k and segment profiles with solved lambda_k
        self.profiles: list[SegmentProfile] = []
        self.fs = np.zeros(N + 1, dtype=np.float64)
        self.fs[0] = 0.0

        for k in range(N):
            f_k = self.fs[k]
            # Solve lambda_k in closed-form to match target_rho
            sp0 = SegmentProfile(
                k=k,
                alpha_k=o.alphas[k],
                alpha_next=o.alphas[k + 1],
                M_k=o.Ms[k],
                lambda_k=0.0,
                delta=p.delta,
                eta_k=o.etas[k],
                f_k=f_k,
            )
            sp1 = SegmentProfile(
                k=k,
                alpha_k=o.alphas[k],
                alpha_next=o.alphas[k + 1],
                M_k=o.Ms[k],
                lambda_k=1.0,
                delta=p.delta,
                eta_k=o.etas[k],
                f_k=f_k,
            )
            r0 = sp0.armijo_ratio
            r1 = sp1.armijo_ratio
            lam = float((p.rho_star - r0) / (r1 - r0))
            sp = SegmentProfile(
                k=k,
                alpha_k=o.alphas[k],
                alpha_next=o.alphas[k + 1],
                M_k=o.Ms[k],
                lambda_k=lam,
                delta=p.delta,
                eta_k=o.etas[k],
                f_k=f_k,
            )
            self.profiles.append(sp)
            self.fs[k + 1] = sp.f_next

        # Exterior constant. The paper's f_\infty is \inf_k f_k over the infinite orbit,
        # which a finite orbit cannot reach; we take a strict lower bound instead. That is
        # all the construction needs: f_\infty < f_k for every k.
        self.f_inf = float(self.fs[-1] - 1.0)

    @property
    def n_steps(self) -> int:
        return self.params.n_steps

    def _locate_segment(self, x: NDArray[np.float64]) -> tuple[int, float, float, float]:
        r"""Find closest segment k, coordinates (\tau, n), and Euclidean distance to segment."""
        o = self.orbit
        N = self.n_steps
        best_k = 0
        best_dist_sq = float("inf")
        best_tau = 0.0
        best_n = 0.0

        for k in range(N):
            dx = x - o.xs[k]
            tau = float(dx @ o.ss[k])
            n = float(dx @ o.nus[k])
            tau_clamped = np.clip(tau, 0.0, 1.0)
            dist_sq = (tau - tau_clamped) ** 2 + n ** 2
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_k = k
                best_tau = tau
                best_n = n

        return best_k, best_tau, best_n, float(np.sqrt(best_dist_sq))

    def evaluate_tube(self, x: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        r"""Evaluate f(x) and \nabla f(x) in the tubular neighborhood."""
        k, tau, n, _ = self._locate_segment(x)
        o = self.orbit
        p = self.params
        delta = p.delta
        sp = self.profiles[k]

        # Transverse tilt b(tau) and b'(tau)
        b0 = float(o.gs[k] @ o.nus[k])
        b1 = float(o.gs[k + 1] @ o.nus[k])
        s_mid = float(S_interval(tau, delta, 1.0 - delta))
        sp_mid = float(Sp_interval(tau, delta, 1.0 - delta))
        b_val = b0 + (b1 - b0) * s_mid
        bp_val = (b1 - b0) * sp_mid

        # Axial profile values
        a_val = float(sp.a(tau))
        ap_val = float(sp.a_p(tau))
        app_val = float(sp.a_pp(tau))

        # f(\tau, n) = a(\tau) + b(\tau) n + 0.5 * a''(\tau) n^2
        val = a_val + b_val * n + 0.5 * app_val * (n ** 2)

        # a'''(\tau) for gradient
        appp_left = float(gamma_bump_deriv(tau, sp.alpha_k, sp.M_left, delta))
        appp_right = -float(gamma_bump_deriv(1.0 - tau, sp.alpha_next, sp.M_right, delta))
        appp_val = appp_left + appp_right

        df_dtau = ap_val + bp_val * n + 0.5 * appp_val * (n ** 2)
        df_dn = b_val + app_val * n

        grad = df_dtau * o.ss[k] + df_dn * o.nus[k]
        return val, grad

    def __call__(self, x: ArrayLike) -> float:
        r"""Evaluate f(x) at point x \in R^2."""
        x_arr = np.asarray(x, dtype=np.float64)
        val, _ = self.evaluate(x_arr)
        return val

    def grad(self, x: ArrayLike) -> NDArray[np.float64]:
        r"""Evaluate \nabla f(x) at point x \in R^2."""
        x_arr = np.asarray(x, dtype=np.float64)
        _, g = self.evaluate(x_arr)
        return g

    def evaluate(self, x: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        r"""Evaluate both f(x) and \nabla f(x) smoothly blended with exterior cutoff."""
        k, tau, n, dist_centerline = self._locate_segment(x)
        p = self.params
        delta = p.delta

        r_in = p.r_inner
        r_out = delta

        val_tube, grad_tube = self.evaluate_tube(x)

        if dist_centerline <= r_in:
            return val_tube, grad_tube
        elif dist_centerline >= r_out:
            return self.f_inf, np.zeros(2, dtype=np.float64)
        else:
            u = (dist_centerline - r_in) / (r_out - r_in)
            zeta = 1.0 - float(S(u))
            zeta_p = -float(Sp(u)) / (r_out - r_in)

            o = self.orbit
            tau_clamped = float(np.clip(tau, 0.0, 1.0))
            cl_pt = o.xs[k] + tau_clamped * o.ss[k]
            diff = x - cl_pt
            if dist_centerline > 1e-12:
                grad_dist = diff / dist_centerline
            else:
                grad_dist = np.zeros(2, dtype=np.float64)

            grad_zeta = zeta_p * grad_dist
            val = zeta * val_tube + (1.0 - zeta) * self.f_inf
            grad = zeta * grad_tube + grad_zeta * (val_tube - self.f_inf)
            return val, grad
