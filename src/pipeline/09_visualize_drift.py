"""Generate Hamilton-style drift visualizations for top drifting words.

For each top-drifting word (by frequency-adjusted drift), computes a 2D
visualization where background neighbor words are fixed at their modern
positions and the target word moves through time.

Depends on:
    - Aligned embeddings in data/models/aligned/
    - frequency_drift.parquet from 07b_frequency_adjusted_drift

Produces:
    - data/results/figures/drift_plots/drift_grid_{NNN}.png

Usage:
    python -m src.pipeline.09_visualize_drift [--m 200] [--k 10] [--method pca]
"""
from __future__ import annotations

import argparse
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.embeddings.alignment import load_aligned
from src.visualization.drift_plots import compute_word_trajectory, plot_drift_grid
from src.visualization.plot_config import apply_plot_style
from src.utils.config import ALIGNED_DIR, METRICS_DIR, FIGURES_DIR, setup_logging

setup_logging("09_visualize_drift")
logger = logging.getLogger(__name__)


def get_top_adjusted_drifters(freq_drift_path: str, m: int = 200) -> list[str]:
    """Get top m words by frequency-adjusted drift."""
    df = pd.read_parquet(freq_drift_path)
    return (
        df.sort_values("adjusted_drift", ascending=False)
        .head(m)["word"]
        .tolist()
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate Hamilton-style drift visualizations"
    )
    parser.add_argument("--m", type=int, default=200, help="Number of top drifters to plot")
    parser.add_argument("--k", type=int, default=10, help="k for neighbor union")
    parser.add_argument("--k-prime", type=int, default=5, help="k' for position initialization")
    parser.add_argument("--k-label", type=int, default=3, help="k for labeling nearest neighbors")
    parser.add_argument("--ncols", type=int, default=5, help="Columns in grid plot")
    parser.add_argument("--batch-size", type=int, default=25, help="Words per grid image")
    parser.add_argument("--method", type=str, default="pca",
                        help="Dimensionality reduction method: pca or umap")
    args = parser.parse_args()

    method = args.method.lower()
    if method not in ("pca", "umap"):
        parser.error(f"Unknown method '{args.method}'. Use 'pca' or 'umap'.")

    apply_plot_style()

    # Load data
    logger.info("Loading aligned embeddings...")
    aligned_kvs = load_aligned(ALIGNED_DIR)
    time_labels = sorted(aligned_kvs.keys())

    freq_drift_path = os.path.join(METRICS_DIR, "frequency_drift.parquet")
    top_words = get_top_adjusted_drifters(freq_drift_path, m=args.m)
    logger.info(f"Computing trajectories for {len(top_words)} words using {method.upper()}...")

    # Compute trajectories
    trajectories = []
    for i, word in enumerate(top_words):
        if (i + 1) % 20 == 0 or i == 0:
            logger.info(f"  [{i+1}/{len(top_words)}] {word}")
        traj = compute_word_trajectory(
            word, aligned_kvs, k=args.k, k_prime=args.k_prime,
            k_label=args.k_label, method=method,
        )
        trajectories.append(traj)

    # Plot in batches
    drift_dir = os.path.join(FIGURES_DIR, "drift_plots")
    os.makedirs(drift_dir, exist_ok=True)

    for batch_start in range(0, len(trajectories), args.batch_size):
        batch = trajectories[batch_start:batch_start + args.batch_size]
        batch_idx = batch_start // args.batch_size
        save_path = os.path.join(drift_dir, f"drift_grid_{batch_idx:03d}.png")
        plot_drift_grid(batch, time_labels, ncols=args.ncols, save_path=save_path)
        plt.close("all")

    logger.info(f"Saved {len(trajectories)} word plots to {drift_dir}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
