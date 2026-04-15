"""Compute polysemy metrics from stored contextualized embeddings.

Loads per-year NPZ embedding files, computes APD and self-similarity
for each word-year, and produces a polysemy ranking.

Output:
  data/results/metrics/polysemy.parquet
  data/results/metrics/polysemy_ranking.parquet
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.metrics.polysemy import compute_polysemy_metrics, compute_polysemy_ranking
from src.utils.config import BERT_EMBEDDINGS_DIR, METRICS_DIR, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute polysemy metrics from BERT embeddings")
    parser.add_argument("--start", type=int, default=1990)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--min-usages", type=int, default=10,
                        help="Minimum usages per word-year for inclusion")
    parser.add_argument("--min-years", type=int, default=20,
                        help="Minimum years for polysemy ranking")
    args = parser.parse_args()

    setup_logging("14_compute_polysemy")
    logger = logging.getLogger(__name__)

    years = range(args.start, args.end + 1)

    # Compute per word-year metrics
    logger.info("Computing polysemy metrics...")
    metrics_df = compute_polysemy_metrics(
        BERT_EMBEDDINGS_DIR, years, min_usages=args.min_usages
    )

    if metrics_df.empty:
        logger.error("No metrics computed — check that embeddings exist")
        return

    # Save per word-year metrics
    os.makedirs(METRICS_DIR, exist_ok=True)
    metrics_path = os.path.join(METRICS_DIR, "polysemy.parquet")
    metrics_df.to_parquet(metrics_path, index=False)
    logger.info(f"Saved {len(metrics_df)} word-year observations to {metrics_path}")

    # Compute ranking
    ranking_df = compute_polysemy_ranking(metrics_df, min_years=args.min_years)
    ranking_path = os.path.join(METRICS_DIR, "polysemy_ranking.parquet")
    ranking_df.to_parquet(ranking_path, index=False)
    logger.info(f"Saved ranking ({len(ranking_df)} words) to {ranking_path}")

    # Print top and bottom 20
    logger.info("\nTop 20 most polysemous words (highest APD):")
    for _, row in ranking_df.head(20).iterrows():
        logger.info(
            f"  {row['word']:30s}  APD={row['mean_apd']:.4f}  "
            f"self_sim={row['mean_self_similarity']:.4f}  "
            f"years={row['n_years']}"
        )

    logger.info("\nTop 20 most monosemous words (lowest APD):")
    for _, row in ranking_df.tail(20).iterrows():
        logger.info(
            f"  {row['word']:30s}  APD={row['mean_apd']:.4f}  "
            f"self_sim={row['mean_self_similarity']:.4f}  "
            f"years={row['n_years']}"
        )

    # Frequency correlation check
    try:
        from scipy.stats import spearmanr
        corr, pval = spearmanr(ranking_df["mean_apd"], ranking_df["mean_n_usages"])
        logger.info(f"\nAPD-frequency Spearman correlation: r={corr:.3f}, p={pval:.2e}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
