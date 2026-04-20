"""Diagnose per-word-per-year occurrence counts in the BERT usage index.

Reads data/models/bert/usage_index/{year}.jsonl (the full index, before
sampling) and reports the count distribution we could draw from if we
re-ran embedding extraction with a higher --n-usages cap.

Two views matter for the dip-test power question:
  (a) Per-(word, year) count distribution — how many word-years have at
      least N occurrences available?
  (b) Per-word "floor" across all years it appears in — if we want a word
      to qualify as monosemous in *every* year, what is its worst-year
      count? Fraction of words whose floor >= N tells us how many
      candidates survive a strict "all years have >= N usages" rule.

Outputs:
  data/results/metrics/usage_count_diagnostic.csv
  data/results/figures/usage_count_diagnostic.png
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import (
    BERT_USAGE_INDEX_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    setup_logging,
)

THRESHOLDS = [50, 100, 200, 300, 500, 1000]


def count_usages_per_word(path: str) -> Counter:
    """Count rows per word in a usage index JSONL (one row = one usage)."""
    counts: Counter = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word = json.loads(line)["word"]
            counts[word] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=1990)
    parser.add_argument("--end", type=int, default=2025)
    args = parser.parse_args()

    setup_logging("22_usage_count_diagnostic")
    logger = logging.getLogger(__name__)

    # word -> {year -> count}
    word_year_counts: dict[str, dict[int, int]] = defaultdict(dict)

    for year in range(args.start, args.end + 1):
        path = os.path.join(BERT_USAGE_INDEX_DIR, f"{year}.jsonl")
        if not os.path.exists(path):
            logger.warning(f"Missing {path}")
            continue
        logger.info(f"Reading {year}...")
        counts = count_usages_per_word(path)
        for w, c in counts.items():
            word_year_counts[w][year] = c

    if not word_year_counts:
        logger.error("No usage indexes found.")
        return

    # Build the long-form (word, year, count) table (word-years with >= 1 usage only).
    rows = [
        {"word": w, "year": y, "count": c}
        for w, yc in word_year_counts.items()
        for y, c in yc.items()
    ]
    df = pd.DataFrame(rows)

    # (a) Per-(word, year) summary.
    logger.info(f"Total word-years with >=1 usage: {len(df):,}")
    logger.info(f"Unique words: {df['word'].nunique():,}")
    logger.info(f"Years covered: {df['year'].min()}-{df['year'].max()}")

    quantiles = df["count"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
    logger.info(f"Per-(word,year) count quantiles: {quantiles}")

    wy_frac = {t: float((df["count"] >= t).mean()) for t in THRESHOLDS}
    logger.info(f"Fraction of word-years with count >= threshold: {wy_frac}")

    # (b) Per-word floor (min count across years the word appears in).
    floor = df.groupby("word")["count"].min().rename("floor_count")
    years_present = df.groupby("word")["year"].nunique().rename("n_years")
    per_word = pd.concat([floor, years_present], axis=1).reset_index()

    word_frac = {
        t: float((per_word["floor_count"] >= t).mean()) for t in THRESHOLDS
    }
    logger.info(
        f"Fraction of words whose *every-year* count >= threshold: {word_frac}"
    )

    # Also: restricted to words present in all 36 years (strict monosemy cohort).
    n_total_years = args.end - args.start + 1
    full_cohort = per_word[per_word["n_years"] == n_total_years]
    full_frac = {
        t: float((full_cohort["floor_count"] >= t).mean()) for t in THRESHOLDS
    }
    logger.info(
        f"Among words present in all {n_total_years} years "
        f"({len(full_cohort)} words), fraction with floor >= threshold: "
        f"{full_frac}"
    )

    # Save CSVs.
    os.makedirs(METRICS_DIR, exist_ok=True)
    per_word_path = os.path.join(METRICS_DIR, "usage_count_diagnostic_per_word.csv")
    per_word.to_csv(per_word_path, index=False)
    logger.info(f"Wrote {per_word_path}")

    summary_rows = []
    for t in THRESHOLDS:
        summary_rows.append({
            "threshold": t,
            "frac_word_years_ge": wy_frac[t],
            "frac_words_all_years_ge": word_frac[t],
            "frac_full_cohort_all_years_ge": full_frac[t],
        })
    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(METRICS_DIR, "usage_count_diagnostic_summary.csv")
    summary.to_csv(summary_path, index=False)
    logger.info(f"Wrote {summary_path}")

    # Figures.
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (0,0) Per-(word,year) count histogram, log x.
    ax = axes[0, 0]
    vals = df["count"].clip(lower=1).values
    bins = np.logspace(0, np.log10(vals.max()), 60)
    ax.hist(vals, bins=bins, color="steelblue", edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("Usages per (word, year)")
    ax.set_ylabel("# word-years")
    ax.set_title("Per-(word, year) occurrence counts")
    for t in THRESHOLDS:
        ax.axvline(t, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.text(t, ax.get_ylim()[1] * 0.95, str(t),
                rotation=90, va="top", ha="right",
                color="red", fontsize=7, alpha=0.7)

    # (0,1) Fraction of word-years >= threshold.
    ax = axes[0, 1]
    ax.plot(THRESHOLDS, [wy_frac[t] for t in THRESHOLDS], "o-",
            label="word-years", color="steelblue")
    ax.plot(THRESHOLDS, [word_frac[t] for t in THRESHOLDS], "s-",
            label="words (floor across all years present)", color="darkorange")
    ax.plot(THRESHOLDS, [full_frac[t] for t in THRESHOLDS], "^-",
            label=f"words present in all {n_total_years} years (floor)",
            color="darkgreen")
    ax.set_xscale("log")
    ax.set_xlabel("Threshold N")
    ax.set_ylabel("Fraction >= N")
    ax.set_title("Survival under a minimum-N filter")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    # (1,0) Per-word floor histogram.
    ax = axes[1, 0]
    fvals = per_word["floor_count"].clip(lower=1).values
    bins = np.logspace(0, np.log10(fvals.max()), 60)
    ax.hist(fvals, bins=bins, color="darkorange", edgecolor="white",
            label="all words in index")
    if len(full_cohort):
        ax.hist(full_cohort["floor_count"].clip(lower=1).values, bins=bins,
                color="darkgreen", edgecolor="white", alpha=0.7,
                label=f"present in all {n_total_years} years")
    ax.set_xscale("log")
    ax.set_xlabel("Min count across years the word appears in")
    ax.set_ylabel("# words")
    ax.set_title("Per-word floor (worst-year count)")
    for t in THRESHOLDS:
        ax.axvline(t, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.legend(fontsize=8)

    # (1,1) Fraction of word-years >= threshold, per year.
    ax = axes[1, 1]
    for t in [100, 200, 300]:
        by_year = df.groupby("year")["count"].apply(lambda x: (x >= t).mean())
        ax.plot(by_year.index, by_year.values, "o-", label=f">= {t}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Fraction of words in index")
    ax.set_title("Per-year survival under threshold")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "usage_count_diagnostic.png")
    fig.savefig(fig_path, dpi=150)
    logger.info(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
