r"""Figure 3: Axial-curvature two-bump profile a''(\tau) on the middle tube T_k.

Plots the two-bump curvature profile:
  a''(\tau) = \Gamma(\tau; \alpha_k, \lambda_k M_k) + \Gamma(1-\tau; \alpha_{k+1}, (1-\lambda_k)M_k)
which places bumps of height \lambda_k M_k and (1-\lambda_k) M_k on [\delta, 3\delta] and [1-3\delta, 1-\delta],
sandwiching a flat zero-curvature region a'' \equiv 0 on [3\delta, 1-3\delta].
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add src to sys.path
src_dir = Path(__file__).resolve().parents[1] / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from bfgs_ce.core.bumps import gamma_bump


def main() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "font.family": "serif",
        "pdf.fonttype": 42,
    })

    r = 2.0 * np.cos(np.pi / 8.0)
    delta = 1.0 / 20.0
    M = 1.0 / delta - 1.5 * (r + 1.0)
    t = 0.4
    M_L, M_R = t * M, (1.0 - t) * M
    tau = np.linspace(0.0, 1.0, 4000)

    # Axial profile a''(\tau) = Gamma(\tau; r, M_L, delta) + Gamma(1-\tau; 1.0, M_R, delta)
    A = gamma_bump(tau, r, M_L, delta) + gamma_bump(1.0 - tau, 1.0, M_R, delta)

    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    ax.plot(tau, A, color="#1a1a1a", lw=1.4)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.2, max(M_L, M_R) * 1.25)
    ax.set_xticks([0.0, delta, 3.0 * delta, 1.0 - 3.0 * delta, 1.0 - delta, 1.0])
    ax.set_xticklabels(
        [r"$0$", r"$\delta$", r"$3\delta$", r"$1-3\delta$", r"$1-\delta$", r"$1$"]
    )
    ax.set_yticks([])
    ax.set_xlabel(r"$\tau$", fontsize=11)
    ax.text(2.0 * delta, M_L * 1.06, r"$\lambda_k M_k$", ha="center", va="bottom", fontsize=10)
    ax.text(1.0 - 2.0 * delta, M_R * 1.04, r"$(1-\lambda_k)M_k$", ha="center", va="bottom", fontsize=10)
    ax.text(0.5 * delta, r + 0.35, r"$\alpha_k$", ha="center", va="bottom", fontsize=10)
    ax.text(1.0 - 0.5 * delta, 1.0 + 0.35, r"$\alpha_{k+1}$", ha="center", va="bottom", fontsize=10)
    ax.text(0.5, 0.25, r"$a''(\tau) \equiv 0$", ha="center", va="bottom", fontsize=9, color="#555555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()

    dirs = [Path(__file__).resolve().parent]
    extra = os.environ.get("PAPER_FIG_DIR")
    if extra:
        dirs.append(Path(extra))
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "spike.pdf", bbox_inches="tight")
        fig.savefig(d / "spike.png", bbox_inches="tight", dpi=300)
        print(f"Saved {d / 'spike.pdf'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
