"""Compute centroid-trajectory drift from per-year BERT embeddings.

For each word with sufficient coverage, reduces each year's
contextualised embeddings to a single 768-d centroid and computes a
year-by-year drift signal plus summary scores.

Outputs:
  data/results/metrics/centroid_drift.parquet            (long-form)
  data/results/metrics/centroid_drift_ranking.parquet
  data/models/bert/centroids.npz                         (per-word centroids)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.metrics.temporal_drift import compute_centroid_drift_table
from src.utils.config import (
    BERT_CENTROIDS_PATH,
    BERT_EMBEDDINGS_DIR,
    METRICS_DIR,
    setup_logging,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute centroid-trajectory drift from BERT embeddings"
    )
    parser.add_argument("--start", type=int, default=1990)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--min-usages", type=int, default=10)
    parser.add_argument("--min-years", type=int, default=20)
    parser.add_argument("--anchor-start", type=int, default=1990)
    parser.add_argument("--anchor-end", type=int, default=1994)
    args = parser.parse_args()

    setup_logging("15_compute_centroid_drift")
    logger = logging.getLogger(__name__)

    years = range(args.start, args.end + 1)

    drift_df, ranking_df, centroids, word_years = compute_centroid_drift_table(
        BERT_EMBEDDINGS_DIR,
        years,
        words=None,
        min_usages=args.min_usages,
        min_years=args.min_years,
        anchor_window=(args.anchor_start, args.anchor_end),
    )

    if drift_df.empty:
        logger.error("No drift data computed — check that embeddings exist")
        return

    os.makedirs(METRICS_DIR, exist_ok=True)
    drift_path = os.path.join(METRICS_DIR, "centroid_drift.parquet")
    drift_df.to_parquet(drift_path, index=False)
    logger.info(
        f"Saved {len(drift_df)} word-year rows to {drift_path}"
    )

    ranking_path = os.path.join(METRICS_DIR, "centroid_drift_ranking.parquet")
    ranking_df.to_parquet(ranking_path, index=False)
    logger.info(
        f"Saved ranking ({len(ranking_df)} words) to {ranking_path}"
    )

    # Save centroids NPZ. Use w:: prefix for the centroid arrays and
    # y:: prefix for the parallel year arrays so the plot script can
    # reconstruct (years, centroids) per word.
    arrays: dict[str, np.ndarray] = {}
    for word, mat in centroids.items():
        arrays[f"w::{word}"] = mat.astype(np.float32)
        arrays[f"y::{word}"] = np.asarray(word_years[word], dtype=np.int32)
    os.makedirs(os.path.dirname(BERT_CENTROIDS_PATH), exist_ok=True)
    np.savez_compressed(BERT_CENTROIDS_PATH, **arrays)
    logger.info(
        f"Saved centroids for {len(centroids)} words to {BERT_CENTROIDS_PATH}"
    )

    logger.info("\nTop 20 by total_drift:")
    for _, row in ranking_df.head(20).iterrows():
        logger.info(
            f"  {row['word']:25s}  total={row['total_drift']:.4f}  "
            f"max_step={row['max_step']:.4f}@{int(row['max_step_year'])}  "
            f"end_vs_start={row['end_vs_start']:.4f}  "
            f"years={int(row['n_years'])}"
        )

    logger.info("\nTop 20 by end_vs_start:")
    for _, row in ranking_df.sort_values(
        "end_vs_start", ascending=False
    ).head(20).iterrows():
        logger.info(
            f"  {row['word']:25s}  end_vs_start={row['end_vs_start']:.4f}  "
            f"total={row['total_drift']:.4f}  years={int(row['n_years'])}"
        )


if __name__ == "__main__":
    main()
