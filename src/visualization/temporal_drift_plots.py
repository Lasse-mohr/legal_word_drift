"""Visualisations for temporal drift metrics.

Distribution-rich plots designed for exploratory inspection of which
legal terms drift in meaning and when. Two families of plots:

A. Centroid-trajectory plots — step + cumulative drift panels, 2D PCA
   trajectories of yearly centroids, a global per-year strip plot of
   drift across all words, and a histogram of total drift.
B. Cross-period APD plots — per-word (Y x Y) heatmaps with shared
   colour scales, per-word marginal "distance from all other years"
   curves, and a histogram of drift excess across all words.

Reuses ``get_sequential_cmap`` and ``remove_extra_spines`` from
``src/visualization/plot_config.py``.
"""
from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.visualization.plot_config import (
    apply_plot_style,
    get_heatmap_cmap,
    get_sequential_cmap,
    remove_extra_spines,
)

logger = logging.getLogger(__name__)


# ── Centroid trajectory ──────────────────────────────────────────────────


def plot_centroid_trajectory_panel(
    ax: plt.Axes,
    word: str,
    drift_rows: pd.DataFrame,
) -> None:
    """Step + cumulative drift for one word on a single Axes.

    Bars: per-year step distance to the previous year.
    Line on a twin axis: cumulative drift.
    The year of maximum step is highlighted.
    """
    df = drift_rows.sort_values("year")
    years = df["year"].to_numpy()
    steps = df["step_distance"].to_numpy()
    cum = df["cumulative_drift"].to_numpy()

    valid = ~np.isnan(steps)
    if valid.sum() == 0:
        ax.set_title(word, fontsize=9)
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="grey")
        return

    bar_colors = ["#888888"] * len(years)
    max_i = int(np.nanargmax(steps))
    bar_colors[max_i] = "#CC3311"

    ax.bar(years, np.nan_to_num(steps, nan=0.0),
           color=bar_colors, width=0.8, alpha=0.85, zorder=2)
    ax.set_ylabel("step", fontsize=8, color="#555555")
    ax.tick_params(axis="y", labelsize=7, colors="#555555")
    ax.tick_params(axis="x", labelsize=7)

    ax2 = ax.twinx()
    ax2.plot(years, cum, color="#0077BB", linewidth=1.4, zorder=3)
    ax2.set_ylabel("cumulative", fontsize=8, color="#0077BB")
    ax2.tick_params(axis="y", labelsize=7, colors="#0077BB")
    ax2.spines["top"].set_visible(False)

    ax.set_title(
        f"{word}  (max@{int(years[max_i])})",
        fontsize=9, fontweight="bold",
    )
    remove_extra_spines(ax)


def plot_centroid_trajectory_grid(
    drift_df: pd.DataFrame,
    words: list[str],
    save_path: str,
    ncols: int = 5,
) -> plt.Figure:
    """Small-multiples grid of step+cumulative drift panels."""
    apply_plot_style()
    n = len(words)
    nrows = max(1, (n + ncols - 1) // ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 3.2, nrows * 2.4),
    )
    axes = np.array(axes).reshape(nrows, ncols)

    for i, word in enumerate(words):
        row, col = divmod(i, ncols)
        sub = drift_df[drift_df["word"] == word]
        plot_centroid_trajectory_panel(axes[row, col], word, sub)

    for i in range(n, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved {save_path}")
    return fig


def plot_centroid_pca_trajectory(
    ax: plt.Axes,
    word: str,
    years: list[int],
    centroids: np.ndarray,
    label_every: int = 5,
) -> None:
    """2D PCA of one word's yearly centroids, drawn as a connected line.

    The line is coloured by year (sequential cmap), with start/end
    markers and year labels every ``label_every`` years.
    """
    if centroids.shape[0] < 3:
        ax.set_title(word, fontsize=9)
        ax.text(0.5, 0.5, "too few years", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="grey")
        return

    pca = PCA(n_components=2)
    coords = pca.fit_transform(centroids.astype(np.float32))

    cmap = get_sequential_cmap()
    n = len(years)
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    ax.plot(coords[:, 0], coords[:, 1],
            color="#333333", alpha=0.3, linewidth=1, zorder=2)
    for i, color in enumerate(colors):
        ax.scatter(coords[i, 0], coords[i, 1],
                   c=[color], s=30, zorder=3,
                   edgecolors="white", linewidths=0.4)

    # Start/end labels
    ax.annotate(str(years[0]), coords[0],
                fontsize=6, fontweight="bold", color=colors[0],
                xytext=(-8, -8), textcoords="offset points")
    ax.annotate(str(years[-1]), coords[-1],
                fontsize=6, fontweight="bold", color=colors[-1],
                xytext=(4, 4), textcoords="offset points")
    # Periodic year labels
    for i in range(label_every, n - 1, label_every):
        ax.annotate(str(years[i]), coords[i],
                    fontsize=5, color="#666666",
                    xytext=(3, 3), textcoords="offset points")

    var = pca.explained_variance_ratio_
    ax.set_title(
        f"{word}  ({var[0]*100:.0f}+{var[1]*100:.0f}%)",
        fontsize=9, fontweight="bold",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    remove_extra_spines(ax)


def plot_centroid_pca_grid(
    centroids_dict: dict[str, np.ndarray],
    years_dict: dict[str, list[int]],
    words: list[str],
    save_path: str,
    ncols: int = 5,
) -> plt.Figure:
    """Small multiples of per-word PCA centroid trajectories."""
    apply_plot_style()
    n = len(words)
    nrows = max(1, (n + ncols - 1) // ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 3.0, nrows * 3.0),
    )
    axes = np.array(axes).reshape(nrows, ncols)

    for i, word in enumerate(words):
        row, col = divmod(i, ncols)
        if word not in centroids_dict:
            axes[row, col].set_visible(False)
            continue
        plot_centroid_pca_trajectory(
            axes[row, col], word, years_dict[word], centroids_dict[word],
        )

    for i in range(n, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved {save_path}")
    return fig


def plot_global_year_strip(
    drift_df: pd.DataFrame,
    save_path: str,
) -> plt.Figure:
    """For each year, a strip of step distances across all words.

    Reveals system-wide shifts: a year where many words moved at once is
    likely a corpus / drafting-style change rather than per-word semantic
    drift.
    """
    apply_plot_style()
    df = drift_df.dropna(subset=["step_distance"])
    years = sorted(df["year"].unique())

    fig, ax = plt.subplots(figsize=(max(8, 0.32 * len(years)), 4.5))

    rng = np.random.default_rng(0)
    for y in years:
        vals = df.loc[df["year"] == y, "step_distance"].to_numpy()
        if vals.size == 0:
            continue
        jitter = rng.uniform(-0.25, 0.25, size=vals.size)
        ax.scatter(np.full_like(vals, y, dtype=float) + jitter, vals,
                   s=4, alpha=0.25, color="#555555")

    # Per-year mean and 90th percentile overlays
    means = [df.loc[df["year"] == y, "step_distance"].mean() for y in years]
    p90 = [df.loc[df["year"] == y, "step_distance"].quantile(0.90) for y in years]
    ax.plot(years, means, color="#CC3311", linewidth=1.6,
            marker="o", markersize=3.5, label="mean")
    ax.plot(years, p90, color="#0077BB", linewidth=1.2,
            marker="o", markersize=2.5, linestyle="--", label="90th pct")

    ax.set_xlabel("year")
    ax.set_ylabel("step distance to previous year")
    ax.set_title(
        "Per-year centroid step across all words "
        "(spikes = system-wide shifts)"
    )
    ax.legend(loc="upper right")
    remove_extra_spines(ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved {save_path}")
    return fig


def plot_total_drift_distribution(
    ranking_df: pd.DataFrame,
    save_path: str,
    top_k_label: int = 15,
) -> plt.Figure:
    """Histograms of total_drift and end_vs_start with outliers labelled."""
    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, col, title in zip(
        axes,
        ["total_drift", "end_vs_start"],
        ["Cumulative drift (sum of step distances)",
         "Net displacement (start vs end centroid)"],
    ):
        vals = ranking_df[col].dropna().to_numpy()
        ax.hist(vals, bins=60, color="#888888", alpha=0.8)
        ax.set_yscale("log")
        ax.set_xlabel(col)
        ax.set_ylabel("words")
        ax.set_title(title, fontsize=10)
        remove_extra_spines(ax)

        # Annotate top outliers
        top = ranking_df.sort_values(col, ascending=False).head(top_k_label)
        ymin, ymax = ax.get_ylim()
        for _, row in top.iterrows():
            ax.axvline(row[col], color="#CC3311", alpha=0.25, linewidth=0.6)
            ax.text(row[col], ymax * 0.7, str(row["word"]),
                    fontsize=6, rotation=90, va="top", ha="right",
                    color="#CC3311")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved {save_path}")
    return fig


# ── Cross-period APD ──────────────────────────────────────────────────────


def plot_cross_period_heatmap(
    ax: plt.Axes,
    word: str,
    years: list[int],
    matrix: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap=None,
) -> "plt.cm.ScalarMappable":
    """Heatmap of one word's (Y x Y) cross-period APD matrix."""
    Y = len(years)
    if cmap is None:
        cmap = get_heatmap_cmap()
    im = ax.imshow(
        matrix, cmap=cmap, origin="lower", aspect="equal",
        vmin=vmin, vmax=vmax,
    )

    # Year ticks: every 5 years
    tick_idx = [i for i, y in enumerate(years) if y % 5 == 0]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(years[i]) for i in tick_idx],
                       fontsize=6, rotation=45)
    ax.set_yticks(tick_idx)
    ax.set_yticklabels([str(years[i]) for i in tick_idx], fontsize=6)

    ax.set_title(word, fontsize=9, fontweight="bold")
    return im


def plot_cross_period_grid(
    matrices_dict: dict[str, dict],
    words: list[str],
    save_path: str,
    ncols: int = 5,
) -> plt.Figure:
    """Small multiples of cross-period heatmaps with one shared colourbar."""
    apply_plot_style()
    n = len(words)
    nrows = max(1, (n + ncols - 1) // ncols)

    # Compute shared vmin/vmax across the grid
    available = [w for w in words if w in matrices_dict]
    if not available:
        logger.warning("No words available for cross-period grid")
        fig, _ = plt.subplots(figsize=(4, 3))
        return fig

    all_vals = np.concatenate([
        matrices_dict[w]["matrix"].flatten() for w in available
    ])
    vmin = float(np.nanmin(all_vals))
    vmax = float(np.nanmax(all_vals))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 2.6, nrows * 2.6),
    )
    axes = np.array(axes).reshape(nrows, ncols)

    last_im = None
    for i, word in enumerate(words):
        row, col = divmod(i, ncols)
        if word not in matrices_dict:
            axes[row, col].set_visible(False)
            continue
        info = matrices_dict[word]
        last_im = plot_cross_period_heatmap(
            axes[row, col], word, info["years"], info["matrix"],
            vmin=vmin, vmax=vmax,
        )

    for i in range(n, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)

    fig.tight_layout(rect=(0, 0, 0.93, 1))
    if last_im is not None:
        cax = fig.add_axes((0.94, 0.15, 0.012, 0.7))
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.set_label("cosine distance", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved {save_path}")
    return fig


def plot_cross_period_marginals(
    ax: plt.Axes,
    word: str,
    years: list[int],
    matrix: np.ndarray,
) -> None:
    """Per-year row mean of the cross-period matrix (excluding diagonal).

    The peak year is the candidate change-point.
    """
    Y = len(years)
    if Y < 3:
        ax.set_title(word, fontsize=9)
        return
    off = matrix.copy()
    np.fill_diagonal(off, np.nan)
    row_means = np.nanmean(off, axis=1)

    ax.plot(years, row_means, color="#0077BB",
            linewidth=1.4, marker="o", markersize=3)
    peak_idx = int(np.argmax(row_means))
    ax.axvline(years[peak_idx], color="#CC3311",
               linewidth=1.0, alpha=0.6)
    ax.scatter(years[peak_idx], row_means[peak_idx],
               color="#CC3311", s=30, zorder=4)
    ax.set_title(
        f"{word}  (peak {years[peak_idx]})",
        fontsize=9, fontweight="bold",
    )
    ax.tick_params(axis="both", labelsize=7)
    ax.set_ylabel("mean dist", fontsize=7)
    remove_extra_spines(ax)


def plot_cross_period_marginals_grid(
    matrices_dict: dict[str, dict],
    words: list[str],
    save_path: str,
    ncols: int = 5,
) -> plt.Figure:
    """Small multiples of per-year row-mean curves."""
    apply_plot_style()
    n = len(words)
    nrows = max(1, (n + ncols - 1) // ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 3.0, nrows * 2.2),
    )
    axes = np.array(axes).reshape(nrows, ncols)

    for i, word in enumerate(words):
        row, col = divmod(i, ncols)
        if word not in matrices_dict:
            axes[row, col].set_visible(False)
            continue
        info = matrices_dict[word]
        plot_cross_period_marginals(
            axes[row, col], word, info["years"], info["matrix"],
        )

    for i in range(n, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved {save_path}")
    return fig


def plot_drift_excess_distribution(
    ranking_df: pd.DataFrame,
    save_path: str,
    top_k_label: int = 15,
) -> plt.Figure:
    """Histograms of drift_excess and drift_ratio across all words."""
    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, col, title in zip(
        axes,
        ["drift_excess", "drift_ratio"],
        ["Cross-year excess over within-year baseline (additive)",
         "Cross-year / within-year (multiplicative)"],
    ):
        vals = ranking_df[col].dropna().to_numpy()
        ax.hist(vals, bins=60, color="#888888", alpha=0.8)
        ax.set_yscale("log")
        ax.set_xlabel(col)
        ax.set_ylabel("words")
        ax.set_title(title, fontsize=10)
        remove_extra_spines(ax)

        top = ranking_df.sort_values(col, ascending=False).head(top_k_label)
        _, ymax = ax.get_ylim()
        for _, row in top.iterrows():
            ax.axvline(row[col], color="#CC3311", alpha=0.25, linewidth=0.6)
            ax.text(row[col], ymax * 0.7, str(row["word"]),
                    fontsize=6, rotation=90, va="top", ha="right",
                    color="#CC3311")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved {save_path}")
    return fig


# ── Single-word combined panels ──────────────────────────────────────────


def plot_centroid_word_detail(
    word: str,
    drift_df: pd.DataFrame,
    centroids: np.ndarray,
    years: list[int],
    save_path: str,
) -> plt.Figure:
    """Per-word figure: step+cumulative panel beside PCA trajectory."""
    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    sub = drift_df[drift_df["word"] == word]
    plot_centroid_trajectory_panel(axes[0], word, sub)
    plot_centroid_pca_trajectory(axes[1], word, years, centroids)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved {save_path}")
    return fig


def plot_cross_period_word_detail(
    word: str,
    years: list[int],
    matrix: np.ndarray,
    save_path: str,
) -> plt.Figure:
    """Per-word figure: heatmap beside its row-mean marginal curve."""
    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    im = plot_cross_period_heatmap(axes[0], word, years, matrix)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    plot_cross_period_marginals(axes[1], word, years, matrix)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved {save_path}")
    return fig
