"""Compute frequency-adjusted drift metrics.

Normalizes word drift by frequency-rank-percentile group mean, following
Hamilton et al. (2016). Produces frequency_drift.parquet and a
diagnostic plot comparing raw vs adjusted drift distributions.

Depends on:
    - Word2Vec models in data/models/word2vec/windows/
    - shifts.parquet from 07_compute_metrics
    - v_analysis.json from 06_align_embeddings

Produces:
    - data/results/metrics/frequency_drift.parquet
    - data/results/metrics/word_frequencies.parquet
    - data/results/figures/drift_distributions.png

Usage:
    python -m src.pipeline.07b_frequency_adjusted_drift [--min-percentile 6]
"""
from __future__ import annotations

import argparse
import glob
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from gensim.models import Word2Vec

from src.metrics.frequency import compute_frequency_adjusted_drift
from src.embeddings.vocabulary import load_vocab
from src.visualization.plot_config import apply_plot_style, get_categorical_colors
from src.utils.config import W2V_MODELS_DIR, VOCAB_DIR, METRICS_DIR, FIGURES_DIR, setup_logging

setup_logging("07b_frequency_adjusted_drift")
logger = logging.getLogger(__name__)


def load_window_models(models_dir: str) -> dict[str, Word2Vec]:
    """Load all sliding-window Word2Vec models from disk."""
    windows_dir = os.path.join(models_dir, "windows")
    models = {}
    for path in sorted(glob.glob(os.path.join(windows_dir, "*.model"))):
        label = os.path.splitext(os.path.basename(path))[0]
        logger.info(f"Loading model {label}")
        models[label] = Word2Vec.load(path)
    logger.info(f"Loaded {len(models)} window models")
    return models


def plot_drift_distributions(
    adjusted_df: pd.DataFrame,
    save_path: str,
) -> None:
    """Plot side-by-side histograms of raw vs frequency-adjusted drift."""
    colors = get_categorical_colors(2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(adjusted_df["total_drift"], bins=50, color=colors[0], alpha=0.8,
             edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Total drift (L1 norm of cosine shifts)")
    ax1.set_ylabel("Number of words")
    ax1.set_title("Raw drift distribution")

    ax2.hist(adjusted_df["adjusted_drift"], bins=50, color=colors[1], alpha=0.8,
             edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Frequency-adjusted drift")
    ax2.set_ylabel("Number of words")
    ax2.set_title("Frequency-adjusted drift distribution")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved drift distributions to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute frequency-adjusted drift metrics"
    )
    parser.add_argument(
        "--min-percentile", type=int, default=6,
        help="Minimum frequency percentile to include (default: 6, filters bottom 5%%)"
    )
    args = parser.parse_args()

    apply_plot_style()
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Load inputs
    models = load_window_models(W2V_MODELS_DIR)
    v_analysis = load_vocab(os.path.join(VOCAB_DIR, "v_analysis.json"))
    shifts_df = pd.read_parquet(os.path.join(METRICS_DIR, "shifts.parquet"))

    logger.info(f"Computing frequency-adjusted drift for {len(v_analysis)} words...")
    adjusted_df, freq_df = compute_frequency_adjusted_drift(
        models, shifts_df, v_analysis,
        min_percentile=args.min_percentile,
    )

    # Save outputs
    freq_path = os.path.join(METRICS_DIR, "word_frequencies.parquet")
    freq_df.to_parquet(freq_path, index=False)
    logger.info(f"Saved word frequencies to {freq_path}")

    drift_path = os.path.join(METRICS_DIR, "frequency_drift.parquet")
    adjusted_df.to_parquet(drift_path, index=False)
    logger.info(f"Saved frequency-adjusted drift to {drift_path}")

    # Log top drifters
    top = adjusted_df.nlargest(20, "adjusted_drift")[["word", "total_drift", "freq_percentile", "adjusted_drift"]]
    logger.info(f"Top 20 frequency-adjusted drifters:\n{top.to_string(index=False)}")

    # Plot distributions
    plot_drift_distributions(
        adjusted_df,
        save_path=os.path.join(FIGURES_DIR, "drift_distributions.png"),
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
