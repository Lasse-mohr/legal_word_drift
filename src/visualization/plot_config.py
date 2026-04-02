"""Matplotlib plot configuration with Tol color schemes.

Provides a consistent, minimal visual style for all plots:
white background, no grid, no top/right spines, Tol color palettes.

Usage:
    from src.visualization.plot_config import apply_plot_style, get_categorical_colors
    apply_plot_style()
    colors = get_categorical_colors(n)
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from tol_colors import colorsets, colormaps


def apply_plot_style() -> None:
    """Configure matplotlib globally for minimal, clean plots."""
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "legend.frameon": False,
    })


def get_categorical_colors(n: int) -> list[str]:
    """Return a list of n colors from the appropriate Tol colorset.

    <=3: high-contrast (black, blue, red)
    <=7: bright
    >7:  cycle through bright
    """
    if n <= 3:
        # high_contrast includes white; skip it for plot lines
        pool = [c for c in colorsets["high_contrast"] if c != "#FFFFFF"]
    else:
        pool = list(colorsets["bright"])

    if n <= len(pool):
        return pool[:n]
    # cycle
    return [pool[i % len(pool)] for i in range(n)]


def get_sequential_cmap() -> mpl.colors.Colormap:
    """Return the YlOrBr sequential colormap from tol_colors."""
    return colormaps["YlOrBr"]


def remove_extra_spines(ax: plt.Axes) -> None:
    """Remove top and right spines from an Axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
