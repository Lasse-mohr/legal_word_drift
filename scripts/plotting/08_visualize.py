"""Generate prototype visualizations from aligned embeddings and metrics.

Produces:
1. PCA trajectory plots for top-drifting words
2. Domain coherence over time
3. Cumulative shift time series for selected words

Usage:
    python -m src.pipeline.08_visualize
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
from src.embeddings.vocabulary import load_vocab
from src.visualization.embedding_plots import (
    stack_embeddings,
    reduce_pca,
    plot_trajectories,
)
from src.visualization.plot_config import apply_plot_style, get_categorical_colors
from src.utils.config import ALIGNED_DIR, VOCAB_DIR, METRICS_DIR, FIGURES_DIR, setup_logging

setup_logging("08_visualize")
logger = logging.getLogger(__name__)


def find_top_drifters(shifts_path: str, n: int = 15) -> list[str]:
    """Find the n words with the highest cumulative combined shift."""
    df = pd.read_parquet(shifts_path)
    top = (
        df.groupby("word")["combined_shift"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
    )
    logger.info(f"Top {n} drifters:\n{top}")
    return top.index.tolist()


def plot_shift_timeseries(
    shifts_path: str,
    words: list[str],
    save_path: str,
) -> None:
    """Plot cumulative cosine shift over time for selected words."""
    df = pd.read_parquet(shifts_path)
    df = df[df["word"].isin(words)]

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = get_categorical_colors(len(words))

    for word, color in zip(words, colors):
        wdf = df[df["word"] == word].sort_values("t2")
        if wdf.empty:
            continue
        cumulative = wdf["cosine_shift"].cumsum()
        # Use midpoint of each pair as x
        x = [
            (int(t1.split("_")[0][1:]) + int(t2.split("_")[1])) / 2
            for t1, t2 in zip(wdf["t1"], wdf["t2"])
        ]
        ax.plot(x, cumulative, label=word, color=color, linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Year (window midpoint)")
    ax.set_ylabel("Cumulative cosine shift")
    ax.set_title("Cumulative semantic shift over time")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved shift timeseries to {save_path}")


def plot_domain_coherence(
    coherence_path: str,
    save_path: str,
) -> None:
    """Plot domain coherence over time."""
    df = pd.read_parquet(coherence_path)

    domains = sorted(df["domain"].unique())
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = get_categorical_colors(len(domains))

    for domain, color in zip(domains, colors):
        ddf = df[df["domain"] == domain].sort_values("time_slice")
        x = [
            (int(ts.split("_")[0][1:]) + int(ts.split("_")[1])) / 2
            for ts in ddf["time_slice"]
        ]
        ax.plot(x, ddf["coherence"], label=domain, color=color, linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Year (window midpoint)")
    ax.set_ylabel("Intra-domain coherence (mean cosine sim)")
    ax.set_title("Legal domain coherence over time")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved domain coherence to {save_path}")


def plot_dispersion_timeseries(
    dispersion_path: str,
    words: list[str],
    save_path: str,
) -> None:
    """Plot k-NN dispersion over time for selected words."""
    df = pd.read_parquet(dispersion_path)
    df = df[df["word"].isin(words)]

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = get_categorical_colors(len(words))

    for word, color in zip(words, colors):
        wdf = df[df["word"] == word].sort_values("time_slice")
        if wdf.empty:
            continue
        x = [
            (int(ts.split("_")[0][1:]) + int(ts.split("_")[1])) / 2
            for ts in wdf["time_slice"]
        ]
        ax.plot(x, wdf["dispersion"], label=word, color=color, linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Year (window midpoint)")
    ax.set_ylabel("k-NN dispersion (mean cosine distance to neighbors)")
    ax.set_title("Semantic dispersion over time")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved dispersion timeseries to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations")
    parser.add_argument("--top-n", type=int, default=12, help="Number of top drifters to plot")
    args = parser.parse_args()

    apply_plot_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    shifts_path = os.path.join(METRICS_DIR, "shifts.parquet")
    dispersion_path = os.path.join(METRICS_DIR, "dispersion.parquet")
    coherence_path = os.path.join(METRICS_DIR, "coherence.parquet")

    # 1. Find top drifters
    logger.info("Finding top drifters...")
    top_words = find_top_drifters(shifts_path, n=args.top_n)

    # 2. PCA trajectory plot for top drifters
    logger.info("Generating PCA trajectory plot...")
    aligned_kvs = load_aligned(ALIGNED_DIR)
    matrix, word_labels, time_labels = stack_embeddings(aligned_kvs, top_words)
    if matrix.shape[0] > 0:
        coords_2d = reduce_pca(matrix)
        plot_trajectories(
            coords_2d, word_labels, time_labels,
            words_to_plot=top_words,
            method="PCA",
            save_path=os.path.join(FIGURES_DIR, "trajectories_pca.png"),
        )

    # 3. Cumulative shift time series
    logger.info("Generating shift timeseries...")
    plot_shift_timeseries(
        shifts_path, top_words,
        save_path=os.path.join(FIGURES_DIR, "shift_timeseries.png"),
    )

    # 4. Domain coherence
    logger.info("Generating domain coherence plot...")
    plot_domain_coherence(
        coherence_path,
        save_path=os.path.join(FIGURES_DIR, "domain_coherence.png"),
    )

    # 5. Dispersion time series for top drifters
    logger.info("Generating dispersion timeseries...")
    plot_dispersion_timeseries(
        dispersion_path, top_words,
        save_path=os.path.join(FIGURES_DIR, "dispersion_timeseries.png"),
    )

    logger.info(f"All figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
