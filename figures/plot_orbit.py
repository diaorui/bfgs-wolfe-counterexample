"""Figure 1: The discrete orbit trajectory and gradient vectors.

Plots the polyline x_{k+1} = x_k + s_k for the first twelve steps, with unit gradients
g_k = \nabla f(x_k) at vertices x_1, ..., x_6 and the limiting search direction s_*.
Generated from AlgebraicOrbit so the figure is numerically identical to the paper.
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

from bfgs_ce.core.orbit import AlgebraicOrbit
from bfgs_ce.core.params import CounterexampleParams

INK = "#1a1a1a"
GRAY = "#8a8a8a"
ACCENT = "#c0392b"


def main() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "font.family": "serif",
        "axes.linewidth": 0.8,
        "axes.edgecolor": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    })

    p = CounterexampleParams(n_steps=12)
    orbit = AlgebraicOrbit.generate(p)
    xs = orbit.xs
    gs = orbit.gs

    # Limiting search direction s_* = (cos(pi/6), sin(pi/6)) = (sqrt(3)/2, 1/2)
    e_star = np.array([np.sqrt(3.0) / 2.0, 0.5])

    fig, ax = plt.subplots(figsize=(6.2, 3.8))

    ax.plot(xs[:, 0], xs[:, 1], "-", color=INK, linewidth=1.2, zorder=2)
    ax.plot(xs[:, 0], xs[:, 1], "o", color=INK, markersize=3.5, zorder=3)

    label_offsets = {0: (-4, 8), 1: (-6, 10), 2: (2, -11)}
    for k, offset in label_offsets.items():
        ax.annotate(
            f"$x_{{{k}}}$",
            xs[k],
            textcoords="offset points",
            xytext=offset,
            fontsize=10,
        )

    g_scale = 1.0  # Unit norm ||g_k|| = 1.0
    arrow_ends = []
    for k in range(1, 7):
        start = xs[k]
        end = xs[k] + g_scale * gs[k]
        arrow_ends.append(end)
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "-|>", "color": ACCENT, "linewidth": 1.1, "mutation_scale": 11},
            zorder=4,
        )

    padding_points = np.vstack([
        np.array(arrow_ends),
        [xs[1, 0], xs[1, 1] - g_scale * 1.3],
    ])
    ax.plot(padding_points[:, 0], padding_points[:, 1], "none")

    tail = xs[-1]
    front = tail + 1.6 * e_star
    ax.plot([tail[0], front[0]], [tail[1], front[1]], "--", color=GRAY, linewidth=0.9, zorder=1)
    ax.annotate(
        r"$s_*$",
        tail + 0.9 * e_star,
        textcoords="offset points",
        xytext=(-6, 8),
        fontsize=10,
        color=INK,
    )

    ax.set_xlabel("$x^{(1)}$")
    ax.set_ylabel("$x^{(2)}$")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    dirs = [Path(__file__).resolve().parent]
    extra = os.environ.get("PAPER_FIG_DIR")
    if extra:
        dirs.append(Path(extra))
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "orbit.pdf", bbox_inches="tight")
        fig.savefig(d / "orbit.png", bbox_inches="tight", dpi=300)
        print(f"Saved {d / 'orbit.pdf'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
