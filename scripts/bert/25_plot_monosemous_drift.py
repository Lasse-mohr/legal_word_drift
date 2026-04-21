"""Render four figures for the 100 monosemous words from script 24.

For each of the four analyses we produce ``--n-figs`` (default 4) PNGs,
each a 5x5 grid of words (25 per figure):

  fig1: cross-period mean pairwise-distance heatmap
  fig2: within-year smoothed pairwise-distance density (line per year)
  fig3: cross-period variance heatmap
  fig4: consecutive-year smoothed pairwise-distance density (line per pair)

Inputs (from script 24):
  data/models/bert/monosemous_drift.npz
  data/results/metrics/monosemous_selection.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import BERT_DIR, FIGURES_DIR, METRICS_DIR, setup_logging
from src.visualization.plot_config import apply_plot_style


GRID_ROWS = 5
GRID_COLS = 5


def load_arrays(npz_path: str) -> dict:
    data = np.load(npz_path, allow_pickle=False)
    out: dict = {"mean": {}, "var": {}, "within_kde": {}, "consec_kde": {}}
    for key in data.files:
        if key == "grid::":
            out["grid"] = data[key]
        elif key == "years::":
            out["years"] = data[key]
        else:
            prefix, word = key.split("::", 1)
            out[prefix][word] = data[key]
    data.close()
    return out


def shared_limits(mats: list[np.ndarray]) -> tuple[float, float]:
    vals = np.concatenate([m[~np.isnan(m)].ravel() for m in mats if m.size])
    if vals.size == 0:
        return 0.0, 1.0
    return float(np.nanpercentile(vals, 2)), float(np.nanpercentile(vals, 98))


def chunk(words: list[str], size: int) -> list[list[str]]:
    return [words[i : i + size] for i in range(0, len(words), size)]


def heatmap_grid(
    mats: dict[str, np.ndarray],
    words: list[str],
    years: np.ndarray,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    save_path: str,
) -> None:
    fig, axes = plt.subplots(
        GRID_ROWS, GRID_COLS, figsize=(GRID_COLS * 2.6, GRID_ROWS * 2.6),
        sharex=True, sharey=True,
    )
    tick_idx = [i for i, y in enumerate(years) if int(y) % 5 == 0]
    tick_labels = [str(int(years[i])) for i in tick_idx]
    for i, word in enumerate(words):
        r, c = divmod(i, GRID_COLS)
        ax = axes[r, c]
        m = mats.get(word)
        if m is None:
            ax.set_visible(False)
            continue
        ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower",
                  aspect="equal", interpolation="nearest")
        ax.set_title(word, fontsize=9)
        if c == 0:
            ax.set_yticks(tick_idx)
            ax.set_yticklabels(tick_labels, fontsize=6)
        if r == GRID_ROWS - 1:
            ax.set_xticks(tick_idx)
            ax.set_xticklabels(tick_labels, fontsize=6, rotation=90)
    for j in range(len(words), GRID_ROWS * GRID_COLS):
        r, c = divmod(j, GRID_COLS)
        axes[r, c].set_visible(False)

    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes((0.90, 0.12, 0.02, 0.76))
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def density_grid(
    kdes: dict[str, np.ndarray],
    words: list[str],
    grid: np.ndarray,
    line_years: np.ndarray,
    title: str,
    cmap_name: str,
    save_path: str,
) -> None:
    fig, axes = plt.subplots(
        GRID_ROWS, GRID_COLS, figsize=(GRID_COLS * 2.6, GRID_ROWS * 2.4),
        sharex=True,
    )
    cmap = plt.get_cmap(cmap_name)
    n_lines = len(line_years)
    norm = Normalize(vmin=float(line_years[0]), vmax=float(line_years[-1]))

    for i, word in enumerate(words):
        r, c = divmod(i, GRID_COLS)
        ax = axes[r, c]
        kde = kdes.get(word)
        if kde is None:
            ax.set_visible(False)
            continue
        for k in range(n_lines):
            row = kde[k]
            if np.all(np.isnan(row)):
                continue
            ax.plot(grid, row, color=cmap(norm(float(line_years[k]))),
                    linewidth=0.7, alpha=0.85)
        ax.set_title(word, fontsize=9)
        ax.tick_params(labelsize=6)
        if r != GRID_ROWS - 1:
            ax.set_xlabel("")
        if c == 0:
            ax.set_ylabel("density", fontsize=7)
    for j in range(len(words), GRID_ROWS * GRID_COLS):
        r, c = divmod(j, GRID_COLS)
        axes[r, c].set_visible(False)

    for ax in axes[GRID_ROWS - 1]:
        if ax.get_visible():
            ax.set_xlabel("cosine distance", fontsize=7)

    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes((0.90, 0.12, 0.02, 0.76))
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("year", fontsize=8)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def order_words(selection: list[dict]) -> list[str]:
    """Order by frequency decile, then alphabetically within decile."""
    df_sorted = sorted(
        selection,
        key=lambda r: (int(r.get("decile", 0)), r["word"]),
    )
    return [r["word"] for r in df_sorted]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-figs", type=int, default=4,
                        help="Split words into this many figures of 25.")
    args = parser.parse_args()

    setup_logging("25_plot_monosemous_drift")
    logger = logging.getLogger(__name__)
    apply_plot_style()

    npz_path = os.path.join(BERT_DIR, "monosemous_drift.npz")
    sel_path = os.path.join(METRICS_DIR, "monosemous_selection.json")

    arr = load_arrays(npz_path)
    years = arr["years"]
    grid = arr["grid"]
    logger.info(
        f"Loaded {len(arr['mean'])} words × {len(years)} years "
        f"(grid size {grid.size})"
    )

    with open(sel_path) as f:
        selection = json.load(f)
    word_order = [w for w in order_words(selection) if w in arr["mean"]]

    chunks = chunk(word_order, GRID_ROWS * GRID_COLS)
    if args.n_figs and args.n_figs < len(chunks):
        chunks = chunks[: args.n_figs]

    out_dir = os.path.join(FIGURES_DIR, "monosemous_drift")
    os.makedirs(out_dir, exist_ok=True)

    # Shared color limits across all chunks.
    mean_vmin, mean_vmax = shared_limits(list(arr["mean"].values()))
    var_vmin, var_vmax = shared_limits(list(arr["var"].values()))

    consec_years = years[:-1]

    for idx, ws in enumerate(chunks, start=1):
        tag = f"{idx:02d}"
        heatmap_grid(
            arr["mean"], ws, years,
            title=f"Cross-period mean pairwise distance — set {tag}",
            cmap="magma", vmin=mean_vmin, vmax=mean_vmax,
            save_path=os.path.join(out_dir, f"fig1_cross_period_mean_{tag}.png"),
        )
        density_grid(
            arr["within_kde"], ws, grid, years,
            title=f"Within-year pairwise-distance density — set {tag}",
            cmap_name="viridis",
            save_path=os.path.join(out_dir, f"fig2_within_year_kde_{tag}.png"),
        )
        heatmap_grid(
            arr["var"], ws, years,
            title=f"Cross-period pairwise-distance variance — set {tag}",
            cmap="cividis", vmin=var_vmin, vmax=var_vmax,
            save_path=os.path.join(out_dir, f"fig3_cross_period_var_{tag}.png"),
        )
        density_grid(
            arr["consec_kde"], ws, grid, consec_years,
            title=f"Consecutive-year pairwise-distance density — set {tag}",
            cmap_name="viridis",
            save_path=os.path.join(out_dir, f"fig4_consec_year_kde_{tag}.png"),
        )
        logger.info(f"Wrote 4 PNGs for set {tag} ({len(ws)} words)")


if __name__ == "__main__":
    main()
