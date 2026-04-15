"""Scatter of quantile-ratio vs path-efficiency for single-big-shift words.

Combines two cheap per-word signals:
  - quantile ratio q100/q90 (and q100/q80): a single dominant yearly shift
    dwarfs the rest (high ratio).
  - path efficiency end_vs_start / total_drift: the word went somewhere and
    stayed (ratio ≈ 1), rather than jumping out and back (ratio ≈ 0).

A word with a single lasting shift should score high on both. Heatmap-grid
rendering is commented out while we iterate on the combined score.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import (
    FIGURES_DIR,
    METRICS_DIR,
    setup_logging,
)
from src.visualization.plot_config import apply_plot_style, get_categorical_colors


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    n = num.astype(float)
    d = den.astype(float)
    return (n / d.where(d > 1e-10)).astype(float)


def compute_step_ratios(centroid_path: str, min_steps: int) -> pd.DataFrame:
    """Per-word ratio of 2nd-largest to largest yearly step distance."""
    drift_df = pd.read_parquet(centroid_path)
    rows = []
    for word, grp in drift_df.groupby("word"):
        steps = grp["step_distance"].dropna().to_numpy(dtype=np.float64)
        if steps.size < min_steps or steps.size < 2:
            continue
        sorted_steps = np.sort(steps)
        step1 = float(sorted_steps[-1])
        step2 = float(sorted_steps[-2])
        ratio = step2 / step1 if step1 > 1e-10 else float("nan")
        rows.append(
            {"word": word, "step1": step1, "step2": step2, "step2_step1": ratio}
        )
    return pd.DataFrame(rows)


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    x_thresh: float,
    y_thresh: float,
    save_path: str,
    title: str,
    max_labels: int = 30,
) -> list[str]:
    """Scatter of two metrics; label words above both per-axis thresholds."""
    colors = get_categorical_colors(2)
    fig, ax = plt.subplots(figsize=(9, 7))

    high_mask = (df[x_col] >= x_thresh) & (df[y_col] >= y_thresh)
    other = df[~high_mask]
    highs = df[high_mask].copy()

    ax.scatter(
        other[x_col], other[y_col],
        s=8, color=colors[0], alpha=0.25, linewidths=0, label="other",
    )
    ax.scatter(
        highs[x_col], highs[y_col],
        s=18, color=colors[1], alpha=0.9, linewidths=0,
        label=f"high on both (n={len(highs)})",
    )

    ax.axvline(x_thresh, color="grey", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.axhline(y_thresh, color="grey", linewidth=0.7, linestyle="--", alpha=0.6)

    # Label top words by product of (per-axis percentile ranks) within the
    # high-on-both region.
    if len(highs) > 0:
        highs["score"] = highs[x_col].rank(pct=True) * highs[y_col].rank(pct=True)
        to_label = highs.nlargest(min(max_labels, len(highs)), "score")
        for _, row in to_label.iterrows():
            ax.annotate(
                row["word"],
                (row[x_col], row[y_col]),
                fontsize=7, alpha=0.85,
                xytext=(3, 2), textcoords="offset points",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_label + "  (log)")
    ax.set_ylabel(y_label + "  (log)")
    ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if not len(highs):
        return []
    highs["score"] = highs[x_col].rank(pct=True) * highs[y_col].rank(pct=True)
    return highs.sort_values("score", ascending=False)["word"].tolist()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scatter quantile-ratio vs path-efficiency"
    )
    parser.add_argument("--min-steps", type=int, default=10)
    parser.add_argument(
        "--eff-quantile", type=float, default=0.90,
        help="Per-axis percentile threshold for path efficiency",
    )
    parser.add_argument(
        "--ratio-quantile", type=float, default=0.90,
        help="Per-axis percentile threshold for quantile ratio",
    )
    # parser.add_argument("--top-k", type=int, default=50)
    # parser.add_argument("--ncols", type=int, default=5)
    args = parser.parse_args()

    setup_logging("20_plot_quantile_ratio_heatmaps")
    logger = logging.getLogger(__name__)
    apply_plot_style()

    quantiles_df = pd.read_parquet(
        os.path.join(METRICS_DIR, "shift_quantiles.parquet")
    )
    df = quantiles_df[quantiles_df["n_steps"] >= args.min_steps].copy()
    logger.info(
        f"shift_quantiles: {len(quantiles_df)} → {len(df)} after "
        f"n_steps >= {args.min_steps}"
    )

    step_df = compute_step_ratios(
        os.path.join(METRICS_DIR, "centroid_drift.parquet"),
        min_steps=args.min_steps,
    )
    df = df.merge(step_df, on="word", how="left")
    logger.info(f"Merged step1/step2 for {step_df['word'].nunique()} words")

    df["r_q100_q90"] = safe_ratio(df["q100"], df["q90"])
    df["r_q100_q80"] = safe_ratio(df["q100"], df["q80"])

    # Path efficiency: net displacement / cumulative path length. A word that
    # moves and stays scores ≈ 1; an oscillator that returns home scores ≈ 0.
    ranking_df = pd.read_parquet(
        os.path.join(METRICS_DIR, "centroid_drift_ranking.parquet")
    )
    ranking_df = ranking_df[["word", "total_drift", "end_vs_start"]].copy()
    ranking_df["path_efficiency"] = safe_ratio(
        ranking_df["end_vs_start"], ranking_df["total_drift"]
    )
    df = df.merge(ranking_df, on="word", how="left")
    logger.info(
        f"Path efficiency available for "
        f"{df['path_efficiency'].notna().sum()} / {len(df)} words"
    )

    out_dir = os.path.join(FIGURES_DIR, "cross_period_apd")
    os.makedirs(out_dir, exist_ok=True)

    scatter_df = df.dropna(subset=["path_efficiency", "r_q100_q90", "r_q100_q80"])
    # Log scale requires strictly positive values on both axes.
    scatter_df = scatter_df[
        (scatter_df["path_efficiency"] > 0)
        & (scatter_df["r_q100_q90"] > 0)
        & (scatter_df["r_q100_q80"] > 0)
    ]
    eff_thresh = float(scatter_df["path_efficiency"].quantile(args.eff_quantile))

    from importlib import import_module
    sys.path.insert(0, os.path.dirname(__file__))
    _script18 = import_module("18_plot_cross_period_apd")
    load_matrices = _script18.load_matrices
    from src.visualization.temporal_drift_plots import plot_cross_period_grid
    from src.utils.config import BERT_CROSS_PERIOD_APD_PATH

    matrices = load_matrices(BERT_CROSS_PERIOD_APD_PATH)
    logger.info(f"Loaded {len(matrices)} cross-period matrices")

    top_k_heatmap = 30

    for ratio_col, label in [
        ("r_q100_q90", "q100 / q90"),
        ("r_q100_q80", "q100 / q80"),
    ]:
        ratio_thresh = float(scatter_df[ratio_col].quantile(args.ratio_quantile))
        save_path = os.path.join(
            out_dir, f"scatter_path_eff_vs_{ratio_col}.png"
        )
        highs = plot_scatter(
            scatter_df,
            x_col="path_efficiency",
            y_col=ratio_col,
            x_label="path efficiency  (end_vs_start / total_drift)",
            y_label=f"quantile ratio  ({label})",
            x_thresh=eff_thresh,
            y_thresh=ratio_thresh,
            save_path=save_path,
            title=(
                f"Single-lasting-shift candidates: "
                f"path efficiency vs {label}\n"
                f"(thresholds at {int(args.eff_quantile * 100)}th / "
                f"{int(args.ratio_quantile * 100)}th percentile)"
            ),
        )
        logger.info(
            f"[{ratio_col}] eff_thresh={eff_thresh:.3f} "
            f"ratio_thresh={ratio_thresh:.3f} → {len(highs)} words above both"
        )
        if highs:
            logger.info(f"[{ratio_col}] top candidates: {highs[:30]}")
        logger.info(f"Saved {save_path}")

        top_words = [w for w in highs[:top_k_heatmap] if w in matrices]
        missing = [w for w in highs[:top_k_heatmap] if w not in matrices]
        if missing:
            logger.warning(
                f"[{ratio_col}] {len(missing)} of top-{top_k_heatmap} missing "
                f"from matrices: {missing[:10]}"
            )
        if top_words:
            heatmap_path = os.path.join(
                out_dir,
                f"heatmap_grid_top{top_k_heatmap}_{ratio_col}_x_path_eff.png",
            )
            plot_cross_period_grid(
                matrices, top_words, save_path=heatmap_path, ncols=5,
            )
            logger.info(f"Saved {heatmap_path}")


if __name__ == "__main__":
    main()
