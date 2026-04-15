"""Visualise centroid-trajectory drift.

Loads the parquet outputs of script 15 plus the per-word centroids NPZ
and produces:
  - small-multiples grid of step + cumulative drift per year
  - small-multiples grid of 2D PCA centroid trajectories
  - global per-year strip plot of step distances across all words
  - histogram of total drift across all words

Optional ``--words`` flag produces a one-per-word detail figure.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import (
    BERT_CENTROIDS_PATH,
    FIGURES_DIR,
    METRICS_DIR,
    setup_logging,
)
from src.visualization.temporal_drift_plots import (
    plot_centroid_pca_grid,
    plot_centroid_trajectory_grid,
    plot_centroid_word_detail,
    plot_global_year_strip,
    plot_total_drift_distribution,
)


def load_centroids(path: str) -> tuple[dict[str, np.ndarray], dict[str, list[int]]]:
    """Load centroids NPZ written by script 15."""
    data = np.load(path, allow_pickle=False)
    centroids: dict[str, np.ndarray] = {}
    years: dict[str, list[int]] = {}
    for key in data.files:
        if key.startswith("w::"):
            centroids[key[3:]] = data[key]
        elif key.startswith("y::"):
            years[key[3:]] = data[key].tolist()
    data.close()
    return centroids, years


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot centroid-trajectory drift")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--ncols", type=int, default=5)
    parser.add_argument(
        "--words", type=str, default="",
        help="Comma-separated list of words to also render as detail figures",
    )
    parser.add_argument(
        "--rank-by", type=str, default="total_drift",
        choices=["total_drift", "end_vs_start", "max_step"],
    )
    args = parser.parse_args()

    setup_logging("16_plot_centroid_drift")
    logger = logging.getLogger(__name__)

    drift_df = pd.read_parquet(os.path.join(METRICS_DIR, "centroid_drift.parquet"))
    ranking_df = pd.read_parquet(
        os.path.join(METRICS_DIR, "centroid_drift_ranking.parquet")
    )
    centroids, years_dict = load_centroids(BERT_CENTROIDS_PATH)
    logger.info(
        f"Loaded {len(drift_df)} drift rows, {len(ranking_df)} ranked words, "
        f"{len(centroids)} centroid trajectories"
    )

    out_dir = os.path.join(FIGURES_DIR, "centroid_drift")
    os.makedirs(out_dir, exist_ok=True)

    top_words = (
        ranking_df.sort_values(args.rank_by, ascending=False)
        .head(args.top_k)["word"]
        .tolist()
    )
    logger.info(f"Top-{args.top_k} by {args.rank_by}: {top_words[:10]} ...")

    plot_centroid_trajectory_grid(
        drift_df, top_words,
        save_path=os.path.join(out_dir, f"step_grid_top{args.top_k}.png"),
        ncols=args.ncols,
    )
    plot_centroid_pca_grid(
        centroids, years_dict, top_words,
        save_path=os.path.join(out_dir, f"pca_grid_top{args.top_k}.png"),
        ncols=args.ncols,
    )
    plot_global_year_strip(
        drift_df,
        save_path=os.path.join(out_dir, "year_strip.png"),
    )
    plot_total_drift_distribution(
        ranking_df,
        save_path=os.path.join(out_dir, "total_drift_hist.png"),
    )

    explicit = [w.strip() for w in args.words.split(",") if w.strip()]
    for word in explicit:
        if word not in centroids:
            logger.warning(f"Word not found in centroids: {word}")
            continue
        plot_centroid_word_detail(
            word, drift_df, centroids[word], years_dict[word],
            save_path=os.path.join(out_dir, f"{word}.png"),
        )


if __name__ == "__main__":
    main()
