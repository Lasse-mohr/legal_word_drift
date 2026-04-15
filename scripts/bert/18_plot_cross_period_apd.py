"""Visualise cross-period APD matrices.

Loads the matrices NPZ and ranking parquet from script 17 and produces:
  - small-multiples grid of (Y x Y) heatmaps with shared colourbar
  - small-multiples grid of per-year row-mean curves (peak = candidate
    change-point)
  - histogram of drift_excess and drift_ratio across all words

Optional ``--words`` flag produces a one-per-word detail figure
combining the heatmap and its row-mean curve.
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
    BERT_CROSS_PERIOD_APD_PATH,
    FIGURES_DIR,
    METRICS_DIR,
    setup_logging,
)
from src.visualization.temporal_drift_plots import (
    plot_cross_period_grid,
    plot_cross_period_marginals_grid,
    plot_cross_period_word_detail,
    plot_drift_excess_distribution,
)


def load_matrices(path: str) -> dict[str, dict]:
    """Load cross-period APD matrices NPZ written by script 17."""
    data = np.load(path, allow_pickle=False)
    out: dict[str, dict] = {}
    for key in data.files:
        if key.startswith("w::"):
            word = key[3:]
            out.setdefault(word, {})["matrix"] = data[key]
        elif key.startswith("y::"):
            word = key[3:]
            out.setdefault(word, {})["years"] = data[key].tolist()
    data.close()
    # Drop incomplete entries
    return {w: v for w, v in out.items() if "matrix" in v and "years" in v}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot cross-period APD")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--ncols", type=int, default=5)
    parser.add_argument(
        "--words", type=str, default="",
        help="Comma-separated list of words to also render as detail figures",
    )
    parser.add_argument(
        "--rank-by", type=str, default="drift_excess",
        choices=["drift_excess", "drift_ratio", "max_off_diag"],
    )
    args = parser.parse_args()

    setup_logging("18_plot_cross_period_apd")
    logger = logging.getLogger(__name__)

    ranking_df = pd.read_parquet(
        os.path.join(METRICS_DIR, "cross_period_apd_ranking.parquet")
    )
    matrices = load_matrices(BERT_CROSS_PERIOD_APD_PATH)
    logger.info(
        f"Loaded {len(ranking_df)} ranked words, {len(matrices)} matrices"
    )

    out_dir = os.path.join(FIGURES_DIR, "cross_period_apd")
    os.makedirs(out_dir, exist_ok=True)

    top_words = (
        ranking_df.sort_values(args.rank_by, ascending=False)
        .head(args.top_k)["word"]
        .tolist()
    )
    logger.info(f"Top-{args.top_k} by {args.rank_by}: {top_words[:10]} ...")

    plot_cross_period_grid(
        matrices, top_words,
        save_path=os.path.join(out_dir, f"heatmap_grid_top{args.top_k}.png"),
        ncols=args.ncols,
    )
    plot_cross_period_marginals_grid(
        matrices, top_words,
        save_path=os.path.join(out_dir, f"marginals_grid_top{args.top_k}.png"),
        ncols=args.ncols,
    )
    plot_drift_excess_distribution(
        ranking_df,
        save_path=os.path.join(out_dir, "drift_excess_hist.png"),
    )

    explicit = [w.strip() for w in args.words.split(",") if w.strip()]
    for word in explicit:
        if word not in matrices:
            logger.warning(f"Word not found in matrices: {word}")
            continue
        info = matrices[word]
        plot_cross_period_word_detail(
            word, info["years"], info["matrix"],
            save_path=os.path.join(out_dir, f"{word}.png"),
        )


if __name__ == "__main__":
    main()
