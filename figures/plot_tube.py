"""Figure 2: Decomposition of the tubular neighborhood U along segment k.

Plots the constituent pieces of U:
  - Vertex discs B_k, B_{k+1} of radius delta
  - Meeting collar pairs C_k, C_{k+1} of length delta and width 2*delta
  - Middle tubes T_k = (delta, 1-delta) x (-delta, delta) (with flanking tubes T_{k-1}, T_{k+1} continuing open-ended)
  - Geometric breakpoint coordinates delta, 1-2*delta, 1, and transverse width 2*delta.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

INK = "#1a252f"
GRAY = "#7f8c8d"
TUBE_FILL = "#ebf5fb"     # soft pastel blue
COLLAR_FILL = "#fef9e7"   # soft pastel warm amber
DISC_FILL = "#fdedec"     # soft pastel rose
DISC_EDGE = "#c0392b"     # muted crimson
TUBE_TEXT = "#1b4f72"     # deep blue
COLLAR_TEXT = "#7d6608"   # deep amber


def strip_quad(origin: np.ndarray, direction: np.ndarray, normal: np.ndarray,
               t_start: float, t_end: float, w: float) -> np.ndarray:
    """Return 4 corner vertices [bottom_start, bottom_end, top_end, top_start] of a rectangular strip."""
    p0 = origin + t_start * direction - w * normal
    p1 = origin + t_end * direction - w * normal
    p2 = origin + t_end * direction + w * normal
    p3 = origin + t_start * direction + w * normal
    return np.array([p0, p1, p2, p3])


def main() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(8.0, 3.2))

    # Geometry setup
    x_k = np.array([0.0, 0.0])
    x_kp1 = np.array([1.0, 0.0])

    psi_k = np.radians(22.0)
    psi_kp1 = np.radians(18.0)

    s_km1 = np.array([np.cos(-psi_k), np.sin(-psi_k)])
    nu_km1 = np.array([-s_km1[1], s_km1[0]])

    s_k = np.array([1.0, 0.0])
    nu_k = np.array([0.0, 1.0])

    s_kp1 = np.array([np.cos(psi_kp1), np.sin(psi_kp1)])
    nu_kp1 = np.array([-s_kp1[1], s_kp1[0]])

    delta = 0.16
    L_flank = 0.34

    # 1. Shaded regions
    p_T_km1 = strip_quad(x_k, -s_km1, nu_km1, delta, delta + L_flank, delta)
    ax.fill(p_T_km1[:, 0], p_T_km1[:, 1], facecolor=TUBE_FILL, edgecolor="none", zorder=2)

    p_T_kp1 = strip_quad(x_kp1, s_kp1, nu_kp1, delta, delta + L_flank, delta)
    ax.fill(p_T_kp1[:, 0], p_T_kp1[:, 1], facecolor=TUBE_FILL, edgecolor="none", zorder=2)

    p_T_k = strip_quad(x_k, s_k, nu_k, delta, 1.0 - delta, delta)
    ax.fill(p_T_k[:, 0], p_T_k[:, 1], facecolor=TUBE_FILL, edgecolor="none", zorder=2)

    p_C_km1_in = strip_quad(x_k, -s_km1, nu_km1, 0, delta, delta)
    ax.fill(p_C_km1_in[:, 0], p_C_km1_in[:, 1], facecolor=COLLAR_FILL, edgecolor="none", zorder=3)

    p_C_k_out = strip_quad(x_k, s_k, nu_k, 0, delta, delta)
    ax.fill(p_C_k_out[:, 0], p_C_k_out[:, 1], facecolor=COLLAR_FILL, edgecolor="none", zorder=3)

    p_C_k_in = strip_quad(x_k, s_k, nu_k, 1.0 - delta, 1.0, delta)
    ax.fill(p_C_k_in[:, 0], p_C_k_in[:, 1], facecolor=COLLAR_FILL, edgecolor="none", zorder=3)

    p_C_kp1_out = strip_quad(x_kp1, s_kp1, nu_kp1, 0, delta, delta)
    ax.fill(p_C_kp1_out[:, 0], p_C_kp1_out[:, 1], facecolor=COLLAR_FILL, edgecolor="none", zorder=3)

    # 2. Boundary rails (open at outer ends)
    ax.plot([p_T_km1[2, 0], p_C_km1_in[3, 0]], [p_T_km1[2, 1], p_C_km1_in[3, 1]], color=INK, lw=1.1, zorder=4)
    ax.plot([p_T_km1[1, 0], p_C_km1_in[0, 0]], [p_T_km1[1, 1], p_C_km1_in[0, 1]], color=INK, lw=1.1, zorder=4)

    ax.plot([0.0, 1.0], [delta, delta], color=INK, lw=1.1, zorder=4)
    ax.plot([0.0, 1.0], [-delta, -delta], color=INK, lw=1.1, zorder=4)

    ax.plot([p_C_kp1_out[3, 0], p_T_kp1[2, 0]], [p_C_kp1_out[3, 1], p_T_kp1[2, 1]], color=INK, lw=1.1, zorder=4)
    ax.plot([p_C_kp1_out[0, 0], p_T_kp1[1, 0]], [p_C_kp1_out[0, 1], p_T_kp1[1, 1]], color=INK, lw=1.1, zorder=4)

    # 3. Seam interfaces at distance delta from vertices
    ax.plot([delta, delta], [-delta, delta], color=INK, lw=1.1, zorder=5)
    ax.plot([1.0 - delta, 1.0 - delta], [-delta, delta], color=INK, lw=1.1, zorder=5)

    ax.plot([p_T_km1[0, 0], p_T_km1[3, 0]], [p_T_km1[0, 1], p_T_km1[3, 1]], color=INK, lw=1.1, zorder=5)
    ax.plot([p_T_kp1[0, 0], p_T_kp1[3, 0]], [p_T_kp1[0, 1], p_T_kp1[3, 1]], color=INK, lw=1.1, zorder=5)

    # 4. Vertex discs B_k and B_{k+1}
    disc_k = Circle(x_k, delta, facecolor=DISC_FILL, edgecolor=DISC_EDGE, lw=1.3, linestyle="--", alpha=0.85, zorder=6)
    ax.add_patch(disc_k)

    disc_kp1 = Circle(x_kp1, delta, facecolor=DISC_FILL, edgecolor=DISC_EDGE, lw=1.3, linestyle="--", alpha=0.85, zorder=6)
    ax.add_patch(disc_kp1)

    # 5. Vertices and labels directly below circle centers
    ax.plot(*x_k, "o", color=INK, markersize=4.5, zorder=10)
    ax.text(x_k[0], x_k[1] - 0.022, r"$x_k$", fontsize=11.5, ha="center", va="top", fontweight="bold", zorder=11)

    ax.plot(*x_kp1, "o", color=INK, markersize=4.5, zorder=10)
    ax.text(x_kp1[0], x_kp1[1] - 0.022, r"$x_{k+1}$", fontsize=11.5, ha="center", va="top", fontweight="bold", zorder=11)

    # 6. Region labels
    ax.text(0.44, 0.0, r"$\mathcal{T}_k = (\delta, 1-\delta) \times (-\delta, \delta)$",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color=TUBE_TEXT, zorder=9)

    pos_T_km1 = x_k - (delta + 0.52 * L_flank) * s_km1
    ax.text(pos_T_km1[0], pos_T_km1[1],
            r"$\mathcal{T}_{k-1}$", ha="center", va="center", fontsize=11, fontweight="bold", color=TUBE_TEXT, zorder=9)

    pos_T_kp1 = x_kp1 + (delta + 0.52 * L_flank) * s_kp1
    ax.text(pos_T_kp1[0], pos_T_kp1[1],
            r"$\mathcal{T}_{k+1}$", ha="center", va="center", fontsize=11, fontweight="bold", color=TUBE_TEXT, zorder=9)

    ax.text(x_k[0], x_k[1] + 0.065, r"$\mathcal{B}_k$", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=DISC_EDGE, zorder=9)
    ax.text(x_kp1[0], x_kp1[1] + 0.065, r"$\mathcal{B}_{k+1}$", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=DISC_EDGE, zorder=9)

    # Meeting Collar Callouts: branching arrows to both constituent collars
    c_k_pos = np.array([0.02, 0.27])
    ax.text(c_k_pos[0], c_k_pos[1], r"$\mathcal{C}_k$ (collars)",
            fontsize=10, fontweight="bold", color=COLLAR_TEXT, ha="center", va="bottom", zorder=9)
    ax.annotate("", xy=(0.5 * delta, delta), xytext=(c_k_pos[0] + 0.03, c_k_pos[1] - 0.01),
                arrowprops=dict(arrowstyle="->", color=COLLAR_TEXT, lw=0.8), zorder=9)
    tgt_km1 = x_k - 0.5 * delta * s_km1 + delta * nu_km1
    ax.annotate("", xy=(tgt_km1[0], tgt_km1[1]), xytext=(c_k_pos[0] - 0.03, c_k_pos[1] - 0.01),
                arrowprops=dict(arrowstyle="->", color=COLLAR_TEXT, lw=0.8), zorder=9)

    c_kp1_pos = np.array([0.98, 0.27])
    ax.text(c_kp1_pos[0], c_kp1_pos[1], r"$\mathcal{C}_{k+1}$ (collars)",
            fontsize=10, fontweight="bold", color=COLLAR_TEXT, ha="center", va="bottom", zorder=9)
    ax.annotate("", xy=(1.0 - 0.5 * delta, delta), xytext=(c_kp1_pos[0] - 0.03, c_kp1_pos[1] - 0.01),
                arrowprops=dict(arrowstyle="->", color=COLLAR_TEXT, lw=0.8), zorder=9)
    tgt_kp1 = x_kp1 + 0.5 * delta * s_kp1 + delta * nu_kp1
    ax.annotate("", xy=(tgt_kp1[0], tgt_kp1[1]), xytext=(c_kp1_pos[0] + 0.03, c_kp1_pos[1] - 0.01),
                arrowprops=dict(arrowstyle="->", color=COLLAR_TEXT, lw=0.8), zorder=9)

    # 7. Dimension brackets below
    by = -delta - 0.04
    ax.plot([0, delta], [by, by], color=INK, lw=0.8, zorder=7)
    ax.plot([0, 0], [by - 0.012, by + 0.012], color=INK, lw=0.8, zorder=7)
    ax.plot([delta, delta], [by - 0.012, by + 0.012], color=INK, lw=0.8, zorder=7)
    ax.text(0.5 * delta, by - 0.018, r"$\delta$", ha="center", va="top", fontsize=9.5, zorder=8)

    ax.plot([delta, 1.0 - delta], [by, by], color=GRAY, lw=0.8, zorder=7)
    ax.plot([delta, delta], [by - 0.012, by + 0.012], color=GRAY, lw=0.8, zorder=7)
    ax.plot([1.0 - delta, 1.0 - delta], [by - 0.012, by + 0.012], color=GRAY, lw=0.8, zorder=7)
    ax.text(0.5, by - 0.018, r"$1 - 2\delta$", ha="center", va="top", fontsize=9.5, color=GRAY, zorder=8)

    ax.plot([1.0 - delta, 1.0], [by, by], color=INK, lw=0.8, zorder=7)
    ax.plot([1.0 - delta, 1.0 - delta], [by - 0.012, by + 0.012], color=INK, lw=0.8, zorder=7)
    ax.plot([1.0, 1.0], [by - 0.012, by + 0.012], color=INK, lw=0.8, zorder=7)
    ax.text(1.0 - 0.5 * delta, by - 0.018, r"$\delta$", ha="center", va="top", fontsize=9.5, zorder=8)

    by2 = by - 0.08
    ax.plot([0, 1.0], [by2, by2], color=GRAY, lw=0.8, zorder=7)
    ax.plot([0, 0], [by2 - 0.012, by2 + 0.012], color=GRAY, lw=0.8, zorder=7)
    ax.plot([1.0, 1.0], [by2 - 0.012, by2 + 0.012], color=GRAY, lw=0.8, zorder=7)
    ax.text(0.5, by2 - 0.018, r"$\|s_k\| = 1$", ha="center", va="top", fontsize=9.5, color=GRAY, zorder=8)

    # Transverse width bracket at tau = 0.71
    ax.annotate("", xy=(0.71, delta), xytext=(0.71, -delta),
                arrowprops=dict(arrowstyle="<->", color=GRAY, lw=0.8), zorder=7)
    ax.text(0.73, 0.0, r"$2\delta$", ha="left", va="center", fontsize=9.5, color=GRAY, zorder=8)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(-0.35, 0.40)

    fig.tight_layout()

    dirs = [Path(__file__).resolve().parent]
    extra = os.environ.get("PAPER_FIG_DIR")
    if extra:
        dirs.append(Path(extra))
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "tube.pdf", bbox_inches="tight")
        fig.savefig(d / "tube.png", bbox_inches="tight", dpi=300)
        print(f"Saved {d / 'tube.pdf'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
