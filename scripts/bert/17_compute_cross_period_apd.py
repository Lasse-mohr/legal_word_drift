"""Compute cross-period APD matrices from per-year BERT embeddings.

For each word with sufficient coverage, computes a (Y x Y) matrix whose
entry (a, b) is the mean cosine distance between embeddings sampled in
year a and embeddings sampled in year b. Diagonal entries are the
within-year APD; off-diagonal block structure reveals regime shifts.

Outputs:
  data/models/bert/cross_period_apd.npz                   (per-word matrices)
  data/results/metrics/cross_period_apd_ranking.parquet
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.metrics.temporal_drift import compute_cross_period_table
from src.utils.config import (
    BERT_CROSS_PERIOD_APD_PATH,
    BERT_EMBEDDINGS_DIR,
    METRICS_DIR,
    setup_logging,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute cross-period APD matrices from BERT embeddings"
    )
    parser.add_argument("--start", type=int, default=1990)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--min-usages", type=int, default=10)
    parser.add_argument("--min-years", type=int, default=20)
    parser.add_argument("--max-per-year", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging("17_compute_cross_period_apd")
    logger = logging.getLogger(__name__)

    years = range(args.start, args.end + 1)

    matrices, ranking_df = compute_cross_period_table(
        BERT_EMBEDDINGS_DIR,
        years,
        words=None,
        min_usages=args.min_usages,
        min_years=args.min_years,
        max_per_year=args.max_per_year,
        seed=args.seed,
    )

    if ranking_df.empty:
        logger.error("No matrices computed — check that embeddings exist")
        return

    os.makedirs(METRICS_DIR, exist_ok=True)
    ranking_path = os.path.join(METRICS_DIR, "cross_period_apd_ranking.parquet")
    ranking_df.to_parquet(ranking_path, index=False)
    logger.info(
        f"Saved ranking ({len(ranking_df)} words) to {ranking_path}"
    )

    # Save matrices NPZ. Use w:: prefix for matrices and y:: prefix for
    # the corresponding year arrays.
    arrays: dict[str, np.ndarray] = {}
    for word, info in matrices.items():
        arrays[f"w::{word}"] = info["matrix"].astype(np.float32)
        arrays[f"y::{word}"] = np.asarray(info["years"], dtype=np.int32)
    os.makedirs(os.path.dirname(BERT_CROSS_PERIOD_APD_PATH), exist_ok=True)
    np.savez_compressed(BERT_CROSS_PERIOD_APD_PATH, **arrays)
    logger.info(
        f"Saved matrices for {len(matrices)} words to {BERT_CROSS_PERIOD_APD_PATH}"
    )

    logger.info("\nTop 20 by drift_excess:")
    for _, row in ranking_df.head(20).iterrows():
        logger.info(
            f"  {row['word']:25s}  excess={row['drift_excess']:.4f}  "
            f"ratio={row['drift_ratio']:.3f}  "
            f"diag={row['mean_diag']:.4f}  off={row['mean_off_diag']:.4f}  "
            f"peak={int(row['peak_year'])}"
        )

    logger.info("\nTop 20 by drift_ratio:")
    for _, row in ranking_df.sort_values(
        "drift_ratio", ascending=False
    ).head(20).iterrows():
        logger.info(
            f"  {row['word']:25s}  ratio={row['drift_ratio']:.3f}  "
            f"excess={row['drift_excess']:.4f}  peak={int(row['peak_year'])}"
        )


if __name__ == "__main__":
    main()
