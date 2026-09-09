from __future__ import annotations

import mpmath as mp


def verify_orbit_high_precision(n_steps: int = 100, dps: int = 80) -> dict[str, float]:
    r"""Verify the discrete algebraic orbit using arbitrary-precision arithmetic (Paper §6).

    Verifies that the convergence failure (\|g_k\| \equiv 1, \|x_k\| \to \infty) holds to hundreds
    of steps without floating point roundoff degradation.
    """
    mp.mp.dps = dps

    def rot90(v):
        return [-v[1], v[0]]

    def dot(u, v):
        return u[0] * v[0] + u[1] * v[1]

    def norm(u):
        return mp.sqrt(dot(u, u))

    # Base step k = 0
    x_0 = [mp.mpf(0), mp.mpf(0)]
    s_0 = [mp.mpf(1), mp.mpf(0)]
    g_0 = [mp.mpf(-1), mp.mpf(0)]
    g_1 = [mp.mpf(0), mp.mpf(-1)]
    y_0 = [g_1[0] - g_0[0], g_1[1] - g_0[1]]
    d_0 = [-g_0[0], -g_0[1]]

    xs = [x_0, [x_0[0] + s_0[0], x_0[1] + s_0[1]]]
    gs = [g_0, g_1]
    ss = [s_0]
    ys = [y_0]
    ds = [d_0]

    max_grad_err = mp.mpf(0)
    max_ortho_err = mp.mpf(0)

    for k in range(1, n_steps):
        s_prev = ss[k - 1]
        y_prev = ys[k - 1]
        g_curr = gs[k]

        coeff_s = dot(y_prev, g_curr)
        coeff_g = dot(s_prev, y_prev)
        d_k = [coeff_s * s_prev[0] - coeff_g * g_curr[0], coeff_s * s_prev[1] - coeff_g * g_curr[1]]
        ds.append(d_k)

        nd = norm(d_k)
        s_k = [d_k[0] / nd, d_k[1] / nd]
        ss.append(s_k)

        x_next = [xs[k][0] + s_k[0], xs[k][1] + s_k[1]]
        xs.append(x_next)

        js = rot90(s_k)
        js_dot_g = dot(js, g_curr)
        sign_js = mp.mpf(1) if js_dot_g >= 0 else mp.mpf(-1)
        g_next = [-sign_js * js[0], -sign_js * js[1]]
        gs.append(g_next)

        y_k = [g_next[0] - g_curr[0], g_next[1] - g_curr[1]]
        ys.append(y_k)

        # Invariants check
        ng = abs(norm(g_next) - 1)
        if ng > max_grad_err:
            max_grad_err = ng

        ortho = abs(dot(s_k, g_next))
        if ortho > max_ortho_err:
            max_ortho_err = ortho

    final_dist = float(norm(xs[-1]))

    return {
        "n_steps": n_steps,
        "dps": dps,
        "final_dist": final_dist,
        "max_grad_err": float(max_grad_err),
        "max_ortho_err": float(max_ortho_err),
    }
