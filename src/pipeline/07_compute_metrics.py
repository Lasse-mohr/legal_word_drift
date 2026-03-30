"""Compute all drift metrics on aligned embeddings.

Loads aligned KeyedVectors, computes shift, dispersion, and clustering
metrics across all time-slice pairs, and saves results as parquet.

Usage:
    python -m src.pipeline.07_compute_metrics
"""
from __future__ import annotations

import argparse
import logging
import os

import pandas as pd
from tqdm import tqdm

from src.embeddings.alignment import load_aligned
from src.embeddings.vocabulary import load_vocab
from src.metrics.shift import compute_all_shifts
from src.metrics.dispersion import compute_all_dispersion
from src.metrics.clustering import compute_all_domain_coherence
from src.utils.config import ALIGNED_DIR, VOCAB_DIR, METRICS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def compute_pairwise_shifts(
    aligned_kvs: dict,
    v_analysis: list[str],
    k: int = 25,
) -> pd.DataFrame:
    """Compute shift metrics between all consecutive time-slice pairs."""
    labels = sorted(aligned_kvs.keys())
    rows = []

    for i in tqdm(range(len(labels) - 1), desc="Pairwise shifts", unit="pair"):
        l1, l2 = labels[i], labels[i + 1]
        shifts = compute_all_shifts(aligned_kvs[l1], aligned_kvs[l2], v_analysis, k=k)
        for word, metrics in shifts.items():
            rows.append({
                "word": word,
                "t1": l1,
                "t2": l2,
                **metrics,
            })

    return pd.DataFrame(rows)


def compute_dispersion_timeseries(
    aligned_kvs: dict,
    v_analysis: list[str],
    k: int = 25,
) -> pd.DataFrame:
    """Compute dispersion metrics for each word at each time slice."""
    rows = []
    for label in tqdm(sorted(aligned_kvs.keys()), desc="Dispersion", unit="slice"):
        kv = aligned_kvs[label]
        dispersions = compute_all_dispersion(kv, v_analysis, k=k)
        for word, metrics in dispersions.items():
            rows.append({
                "word": word,
                "time_slice": label,
                **metrics,
            })

    return pd.DataFrame(rows)


def compute_coherence_timeseries(aligned_kvs: dict) -> pd.DataFrame:
    """Compute domain coherence at each time slice."""
    rows = []
    for label in tqdm(sorted(aligned_kvs.keys()), desc="Coherence", unit="slice"):
        coherence = compute_all_domain_coherence(aligned_kvs[label])
        for domain, score in coherence.items():
            rows.append({
                "domain": domain,
                "time_slice": label,
                "coherence": score,
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Compute drift metrics")
    parser.add_argument("--k", type=int, default=25, help="k for k-NN metrics")
    args = parser.parse_args()

    os.makedirs(METRICS_DIR, exist_ok=True)

    # Load aligned models and vocabulary
    logger.info("Loading aligned models...")
    aligned_kvs = load_aligned(ALIGNED_DIR)
    logger.info(f"Loaded {len(aligned_kvs)} aligned models")

    v_analysis_path = os.path.join(VOCAB_DIR, "v_analysis.json")
    v_analysis = load_vocab(v_analysis_path)
    logger.info(f"V_analysis: {len(v_analysis)} words")

    # Pairwise shifts
    logger.info("Computing pairwise shifts...")
    shifts_df = compute_pairwise_shifts(aligned_kvs, v_analysis, k=args.k)
    shifts_df.to_parquet(os.path.join(METRICS_DIR, "shifts.parquet"), index=False)
    logger.info(f"  Shifts: {len(shifts_df)} rows")

    # Dispersion time series
    logger.info("Computing dispersion time series...")
    dispersion_df = compute_dispersion_timeseries(aligned_kvs, v_analysis, k=args.k)
    dispersion_df.to_parquet(os.path.join(METRICS_DIR, "dispersion.parquet"), index=False)
    logger.info(f"  Dispersion: {len(dispersion_df)} rows")

    # Domain coherence time series
    logger.info("Computing domain coherence...")
    coherence_df = compute_coherence_timeseries(aligned_kvs)
    coherence_df.to_parquet(os.path.join(METRICS_DIR, "coherence.parquet"), index=False)
    logger.info(f"  Coherence: {len(coherence_df)} rows")

    logger.info("Done. Results saved to " + METRICS_DIR)


if __name__ == "__main__":
    main()
