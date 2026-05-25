"""Paginated grid figures of within-group distance distributions.

Originally lifted from ``scripts/bert/21_pilot_polysemy.py`` and generalised:
``groups`` is any ``{group_name: [item_keys]}`` mapping (categories, EuroVoc
domains, frequency bins, …). Each subplot shows a histogram + KDE of one
distance vector with a small text annotation drawn from the per-item stats.
"""
from __future__ import annotations

import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.plot_config import (
    SEQUENTIAL_PALETTES,
    apply_plot_style,
    remove_extra_spines,
)

GRID_ROWS = 5
GRID_COLS = 5


def _draw_subplot(
    ax,
    item: str,
    group: str,
    dists: np.ndarray,
    s: dict,
    hist_color: str,
    kde_color: str,
    *,
    xlabel: str,
    pval_key: str,
) -> None:
    ax.hist(
        dists, bins=60, density=True,
        color=hist_color, edgecolor="white", linewidth=0.3, alpha=0.85,
    )
    try:
        from scipy.stats import gaussian_kde
        xs = np.linspace(dists.min(), dists.max(), 200)
        ax.plot(xs, gaussian_kde(dists)(xs), color=kde_color, linewidth=1.3)
    except Exception:
        pass

    sil = s.get("silhouette_k2", float("nan"))
    pval = s.get(pval_key, float("nan"))
    parts = [f"sil₂={sil:.2f}"]
    if not np.isnan(pval):
        parts.append(f"dip p={pval:.1e}")
    ax.set_title(f"{item}\n{'  '.join(parts)}", fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("density", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.text(
        0.02, 0.95, group[:8],
        transform=ax.transAxes, ha="left", va="top",
        fontsize=7, color="#555",
    )
    ax.text(
        0.98, 0.95, f"N={int(s['n_usages'])}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8, color="#222",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5),
    )
    remove_extra_spines(ax)


def plot_distance_grids(
    distances: dict[str, np.ndarray],
    stats: dict[str, dict],
    groups: dict[str, list[str]],
    out_dir: str,
    tag: str,
    title_prefix: str,
    *,
    kind: str = "pairwise",
    pval_key: str = "dip_pvalue",
    xlabel: str = "cos dist",
    palette: str = "blues",
    filename_prefix: str = "distance_grid",
) -> list[str]:
    """Render paginated 5×5 grids, one subplot per item.

    Items are ordered by ``groups`` iteration order, then by their order
    within each group's list. Items missing from ``distances`` are skipped.
    Returns the list of figure paths written.
    """
    apply_plot_style()
    pal = SEQUENTIAL_PALETTES[palette]
    hist_color, kde_color = pal[2], pal[5]

    ordered: list[tuple[str, str]] = []
    for group_name, items in groups.items():
        for item in items:
            if item in distances:
                ordered.append((group_name, item))
    if not ordered:
        raise RuntimeError("No items with distances to plot")

    per_page = GRID_ROWS * GRID_COLS
    n_pages = (len(ordered) + per_page - 1) // per_page
    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []

    for page in range(n_pages):
        chunk = ordered[page * per_page : (page + 1) * per_page]
        fig, axes = plt.subplots(
            GRID_ROWS, GRID_COLS,
            figsize=(2.9 * GRID_COLS, 2.3 * GRID_ROWS),
            sharex=False,
        )
        axes = np.atleast_2d(axes).reshape(GRID_ROWS, GRID_COLS)

        for i, (group, item) in enumerate(chunk):
            ax = axes[i // GRID_COLS, i % GRID_COLS]
            _draw_subplot(
                ax, item, group, distances[item], stats[item],
                hist_color, kde_color,
                xlabel=xlabel, pval_key=pval_key,
            )
        for j in range(len(chunk), per_page):
            axes[j // GRID_COLS, j % GRID_COLS].axis("off")

        fig.suptitle(
            f"{title_prefix}  (page {page + 1}/{n_pages})",
            fontsize=13, y=1.00,
        )
        fig.tight_layout()
        out_path = os.path.join(
            out_dir, f"{filename_prefix}_{tag}_{kind}_p{page + 1:02d}.png"
        )
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)
    return written


def plot_pairwise_by_year_grid(
    distances: dict[str, dict[int, tuple[np.ndarray, int]]],
    words: Sequence[str],
    years: Sequence[int],
    out_dir: str,
    *,
    words_per_page: int = 8,
    nbins: int = 50,
    palette: str = "blues",
    filename_prefix: str = "pairwise_by_year",
    title: str = "Per-year pairwise cosine distance distributions",
) -> list[str]:
    """Years × words grid of within-year distance histograms, paginated by words.

    For each ``(word, year)`` cell, ``distances[word][year]`` should be
    ``(dists, n)`` — a 1-D array of pairwise distances and the original
    sample count. Missing cells render as a grey panel. xlim is shared
    per-word across years (0.1%/99.9% quantiles + 2% pad). Returns figure
    paths.
    """
    apply_plot_style()
    pal = SEQUENTIAL_PALETTES[palette]
    hist_color, kde_color = pal[2], pal[5]

    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []

    xlims: dict[str, tuple[float, float]] = {}
    for word in words:
        all_d = [d for (d, _n) in distances.get(word, {}).values()]
        if all_d:
            flat = np.concatenate(all_d)
            lo = float(np.quantile(flat, 0.001))
            hi = float(np.quantile(flat, 0.999))
            span = hi - lo if hi > lo else 1.0
            xlims[word] = (lo - 0.02 * span, hi + 0.02 * span)

    n_pages = (len(words) + words_per_page - 1) // words_per_page
    n_rows = len(years)

    for page in range(n_pages):
        chunk = list(words[page * words_per_page : (page + 1) * words_per_page])
        n_cols = len(chunk)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(1.8 * n_cols, 0.65 * n_rows),
            squeeze=False,
            sharex="col",
        )

        for ci, word in enumerate(chunk):
            word_d = distances.get(word, {})
            axes[0, ci].set_title(word, fontsize=10, pad=6)
            for ri, year in enumerate(years):
                ax = axes[ri, ci]
                entry = word_d.get(year)
                if ci == 0:
                    ax.set_ylabel(
                        str(year), rotation=0, fontsize=8,
                        ha="right", va="center", labelpad=12,
                    )
                ax.set_yticks([])
                if ri != n_rows - 1:
                    ax.tick_params(axis="x", labelbottom=False)
                else:
                    ax.tick_params(axis="x", labelsize=7)

                if entry is None:
                    ax.set_facecolor("#f7f7f7")
                    ax.set_xticks([])
                    remove_extra_spines(ax)
                    continue

                dists, n = entry
                ax.hist(
                    dists, bins=nbins, density=True,
                    color=hist_color, edgecolor="white", linewidth=0.2,
                    alpha=0.85,
                )
                try:
                    from scipy.stats import gaussian_kde
                    xs = np.linspace(dists.min(), dists.max(), 200)
                    ax.plot(
                        xs, gaussian_kde(dists)(xs),
                        color=kde_color, linewidth=1.0,
                    )
                except Exception:
                    pass
                ax.text(
                    0.97, 0.88, f"n={n}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=6, color="#555",
                )
                if word in xlims:
                    ax.set_xlim(*xlims[word])
                remove_extra_spines(ax)

            axes[-1, ci].set_xlabel("cos dist", fontsize=8)

        fig.suptitle(
            f"{title}  (page {page + 1}/{n_pages})",
            fontsize=12, y=1.00,
        )
        fig.tight_layout()
        out_path = os.path.join(
            out_dir, f"{filename_prefix}_p{page + 1:02d}.png"
        )
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)

    return written
