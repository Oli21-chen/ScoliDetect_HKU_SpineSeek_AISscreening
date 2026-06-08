"""
Matplotlib defaults aligned with Nature-family figure guidance.

Use for evaluation / interpretability figures (e.g. ``run_test_visualization.py``).
"""

from __future__ import annotations

from typing import Any, Dict

import matplotlib.pyplot as plt

# Single-column ~89 mm; double ~183 mm. Figures here target ~180 mm width when 3 panels sit in one row.
_MM_PER_INCH = 25.4

NATURE_MPL_RCPARAMS: Dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.titleweight": "normal",
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "lines.solid_capstyle": "round",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.45,
    "ytick.minor.width": 0.45,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "legend.frameon": False,
}


def apply_nature_journal_mpl_style() -> None:
    """Apply global rcParams (safe to call once per process before saving figures)."""
    plt.rcParams.update(NATURE_MPL_RCPARAMS)


def style_axis_nature(ax) -> None:
    """Remove top/right spines; outward ticks (redundant if rcParams already set)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def mm_to_inch(mm: float) -> float:
    return float(mm) / _MM_PER_INCH


def add_panel_label(ax, label: str, *, x: float = -0.12, y: float = 1.02) -> None:
    """Bold panel letter outside axes (Nature-style a, b, c)."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
