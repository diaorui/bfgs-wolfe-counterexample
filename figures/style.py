"""Shared matplotlib styling for the paper's figures.

Every figure here is generated from the same tested companion code that backs the
paper's formulas, so a figure is never hand-drawn or independently re-parameterized.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt

LOCAL_FIGURES_DIR = Path(__file__).resolve().parent

INK = "#1a1a1a"
GRAY = "#8a8a8a"
ACCENT = "#c0392b"
FILL = "#ececec"


def apply_style() -> None:
    plt.rcParams.update(
        {
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
        }
    )


def save_pdf_png(fig: plt.Figure, stem: str) -> None:
    dirs = [LOCAL_FIGURES_DIR]
    extra = os.environ.get("PAPER_FIG_DIR")
    if extra:
        dirs.append(Path(extra))
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(d / f"{stem}.png", bbox_inches="tight", dpi=300)
        print(f"Saved {d / f'{stem}.pdf'}")
