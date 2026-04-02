"""Frequency-adjusted drift metrics.

Computes word frequency ranks, assigns percentile groups, and normalizes
drift by mean drift within each frequency-rank-percentile group. This
accounts for the frequency-drift correlation identified by Hamilton et al.
(2016) without assuming a specific mechanism.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

logger = logging.getLogger(__name__)


def compute_word_frequencies(
    models: dict[str, Word2Vec],
    words: Sequence[str],
) -> pd.DataFrame:
    """Extract word frequency counts from Word2Vec models.

    Returns DataFrame with columns: word, time_slice, count.
    """
    rows = []
    for label, model in sorted(models.items()):
        wv = model.wv
        for word in words:
            if word in wv:
                try:
                    count = int(wv.get_vecattr(word, "count"))
                except KeyError:
                    count = 0
                rows.append({"word": word, "time_slice": label, "count": count})
    return pd.DataFrame(rows)


def compute_mean_frequency(freq_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean frequency per word across all time slices.

    Returns DataFrame with columns: word, mean_count.
    """
    return (
        freq_df.groupby("word")["count"]
        .mean()
        .reset_index()
        .rename(columns={"count": "mean_count"})
    )


def assign_frequency_percentiles(
    mean_freq_df: pd.DataFrame,
    n_bins: int = 100,
) -> pd.DataFrame:
    """Rank words by mean frequency and assign percentile groups 1-100.

    Returns DataFrame with columns: word, mean_count, freq_rank, freq_percentile.
    """
    df = mean_freq_df.copy()
    df["freq_rank"] = df["mean_count"].rank(method="average", ascending=True)
    df["freq_percentile"] = pd.qcut(
        df["freq_rank"], q=n_bins, labels=False, duplicates="drop"
    ) + 1  # 1-indexed
    return df


def compute_total_drift(
    shifts_df: pd.DataFrame,
    metric: str = "cosine_shift",
) -> pd.DataFrame:
    """Compute total drift per word as sum of pairwise shifts (L1 norm).

    Returns DataFrame with columns: word, total_drift.
    """
    return (
        shifts_df.groupby("word")[metric]
        .sum()
        .reset_index()
        .rename(columns={metric: "total_drift"})
    )


def normalize_drift_by_frequency(
    drift_df: pd.DataFrame,
    percentile_df: pd.DataFrame,
    min_percentile: int = 6,
) -> pd.DataFrame:
    """Normalize each word's drift by the mean drift of its frequency-percentile group.

    Filters out rare words (freq_percentile <= min_percentile - 1).

    Returns DataFrame with columns: word, total_drift, mean_count,
    freq_percentile, group_mean_drift, adjusted_drift.
    """
    df = drift_df.merge(percentile_df[["word", "mean_count", "freq_percentile"]], on="word")

    # Filter out rare words
    df = df[df["freq_percentile"] >= min_percentile].copy()

    # Mean drift per percentile group
    group_means = df.groupby("freq_percentile")["total_drift"].mean()
    df["group_mean_drift"] = df["freq_percentile"].map(group_means)

    # Normalize
    df["adjusted_drift"] = df["total_drift"] / df["group_mean_drift"]

    logger.info(
        f"Frequency-adjusted drift: {len(df)} words "
        f"(filtered {min_percentile - 1} lowest percentiles)"
    )
    return df


def compute_frequency_adjusted_drift(
    models: dict[str, Word2Vec],
    shifts_df: pd.DataFrame,
    v_analysis: Sequence[str],
    min_percentile: int = 6,
    n_bins: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full pipeline: frequencies -> percentiles -> normalize drift.

    Returns (adjusted_drift_df, word_frequencies_df).
    """
    freq_df = compute_word_frequencies(models, v_analysis)
    mean_freq = compute_mean_frequency(freq_df)
    percentile_df = assign_frequency_percentiles(mean_freq, n_bins=n_bins)
    drift_df = compute_total_drift(shifts_df)
    adjusted_df = normalize_drift_by_frequency(drift_df, percentile_df, min_percentile)
    return adjusted_df, freq_df
