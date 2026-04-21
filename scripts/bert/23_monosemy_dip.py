"""Classify words as monosemous via Hartigan's dip test per (word, year).

For every year a word appears in the BERT embeddings:
  - If n_usages >= --min-usages, compute the upper-triangle pairwise cosine
    distances and run diptest.diptest(), recording (dip, p_value).
  - Otherwise record NaN and flag the year as untested.

A word is flagged ``is_monosemous`` when the fraction of its non-zero-count
years that (a) meet the min_usages floor and (b) pass the dip test
(p > --p-threshold) exceeds 0.5. This is the simple first pass; we can
tighten the criterion later.

Outputs:
  data/results/metrics/dip_by_word_year.csv
  data/results/metrics/monosemy_summary.csv
  data/results/metrics/monosemous_words.json
  data/results/figures/monosemy/dip_over_time_examples.png
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import diptest
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.metrics.polysemy import pairwise_cosine_distances
from src.metrics.temporal_drift import load_per_year_embeddings
from src.utils.config import (
    BERT_EMBEDDINGS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    setup_logging,
)
from src.visualization.plot_config import apply_plot_style, get_categorical_colors


def dip_per_word_year(
    embeddings_dir: str,
    years: range,
    min_usages: int,
    min_year_coverage: int,
) -> pd.DataFrame:
    """Run dip test on every (word, year) that has at least one embedding."""
    logger = logging.getLogger(__name__)

    per_word = load_per_year_embeddings(
        embeddings_dir, list(years), words=None, min_usages=1
    )
    logger.info(f"Loaded embeddings for {len(per_word)} words")

    rows = []
    for word, per_year in per_word.items():
        if len(per_year) < min_year_coverage:
            continue
        for year in sorted(per_year):
            embs = per_year[year]
            n = int(embs.shape[0])
            if n >= min_usages:
                dists = pairwise_cosine_distances(embs)
                dip, pval = diptest.diptest(dists)
                rows.append({
                    "word": word, "year": int(year), "n_usages": n,
                    "dip": float(dip), "p_value": float(pval),
                    "tested": True,
                })
            else:
                rows.append({
                    "word": word, "year": int(year), "n_usages": n,
                    "dip": float("nan"), "p_value": float("nan"),
                    "tested": False,
                })
    return pd.DataFrame(rows)


def summarize_monosemy(
    df: pd.DataFrame, p_threshold: float
) -> pd.DataFrame:
    """Per-word summary; flag monosemous = > 50% of present years pass."""
    rows = []
    for word, g in df.groupby("word"):
        n_present = len(g)
        tested = g[g["tested"]]
        n_tested = len(tested)
        n_passing = int((tested["p_value"] > p_threshold).sum())
        passing_rows = tested[tested["p_value"] > p_threshold]
        frac_passing = n_passing / n_present if n_present else 0.0
        rows.append({
            "word": word,
            "n_years_present": n_present,
            "n_years_tested": n_tested,
            "n_years_passing": n_passing,
            "frac_passing": frac_passing,
            "min_p_tested": (
                float(tested["p_value"].min()) if n_tested else float("nan")
            ),
            "max_dip_tested": (
                float(tested["dip"].max()) if n_tested else float("nan")
            ),
            "mean_p_passing": (
                float(passing_rows["p_value"].mean())
                if n_passing else float("nan")
            ),
            "is_monosemous": frac_passing > 0.5,
        })
    return pd.DataFrame(rows).sort_values("frac_passing", ascending=False)


def plot_dip_over_time(
    df: pd.DataFrame,
    words: list[str],
    p_threshold: float,
    save_path: str,
) -> None:
    """For a handful of example words, plot dip p-value vs year (log y)."""
    logger = logging.getLogger(__name__)
    apply_plot_style()

    words = [w for w in words if not df[df["word"] == w].empty]
    if not words:
        logger.warning("No example words found in dip dataframe; skipping plot.")
        return

    colors = get_categorical_colors(len(words))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    ax_p, ax_d = axes

    for word, color in zip(words, colors):
        g = df[(df["word"] == word) & df["tested"]].sort_values("year")
        if g.empty:
            continue
        ax_p.plot(g["year"], g["p_value"].clip(lower=1e-4),
                  "o-", color=color, label=word, markersize=4, linewidth=1.2)
        ax_d.plot(g["year"], g["dip"], "o-", color=color, label=word,
                  markersize=4, linewidth=1.2)

    for ref, style, lbl in [(0.1, "--", "p=0.1"), (0.05, ":", "p=0.05")]:
        ax_p.axhline(ref, color="#888888", linestyle=style, linewidth=0.9)
        ax_p.text(ax_p.get_xlim()[1], ref, f"  {lbl}",
                  color="#666666", fontsize=8, va="center", ha="left")

    ax_p.axhline(p_threshold, color="#CC3311", linestyle="-", linewidth=0.8,
                 alpha=0.7)
    ax_p.set_yscale("log")
    ax_p.set_ylabel("dip p-value (log)")
    ax_p.set_xlabel("year")
    ax_p.set_title("Dip test p-value over time")
    ax_p.legend(fontsize=8, loc="best")

    ax_d.set_ylabel("dip statistic")
    ax_d.set_xlabel("year")
    ax_d.set_title("Dip statistic over time")
    ax_d.legend(fontsize=8, loc="best")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Wrote {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=1990)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--min-usages", type=int, default=100,
                        help="Per-year usage floor for dip testing.")
    parser.add_argument("--min-year-coverage", type=int, default=10,
                        help="Word must appear (n>=1) in at least this many years.")
    parser.add_argument("--p-threshold", type=float, default=0.1,
                        help="Dip p-value threshold (pass if p > threshold).")
    parser.add_argument("--example-words", type=str,
                        default="market,court,article,regulation,measure",
                        help="Comma-separated words for the exploration figure.")
    args = parser.parse_args()

    setup_logging("23_monosemy_dip")
    logger = logging.getLogger(__name__)

    years = range(args.start, args.end + 1)

    logger.info(
        f"Dip testing with min_usages={args.min_usages}, "
        f"min_year_coverage={args.min_year_coverage}, "
        f"p_threshold={args.p_threshold}"
    )

    df = dip_per_word_year(
        BERT_EMBEDDINGS_DIR, years,
        min_usages=args.min_usages,
        min_year_coverage=args.min_year_coverage,
    )
    if df.empty:
        logger.error("No dip rows produced; aborting.")
        return

    os.makedirs(METRICS_DIR, exist_ok=True)
    dip_path = os.path.join(METRICS_DIR, "dip_by_word_year.csv")
    df.to_csv(dip_path, index=False)
    logger.info(f"Wrote {dip_path} ({len(df):,} rows)")

    summary = summarize_monosemy(df, p_threshold=args.p_threshold)
    summary_path = os.path.join(METRICS_DIR, "monosemy_summary.csv")
    summary.to_csv(summary_path, index=False)
    logger.info(f"Wrote {summary_path}")

    mono_words = summary.loc[summary["is_monosemous"], "word"].tolist()
    mono_path = os.path.join(METRICS_DIR, "monosemous_words.json")
    with open(mono_path, "w") as f:
        json.dump(mono_words, f, indent=2)
    logger.info(
        f"Wrote {mono_path} ({len(mono_words)} monosemous words out of "
        f"{len(summary)} eligible)"
    )

    # Warn about words where many present years were untestable.
    low_coverage = summary[summary["n_years_tested"] < summary["n_years_present"] * 0.5]
    if len(low_coverage):
        logger.warning(
            f"{len(low_coverage)} words have <50% of their present years "
            f"tested (n_usages < {args.min_usages} in the rest). "
            f"Example: {low_coverage['word'].head(5).tolist()}"
        )

    example_words = [w.strip() for w in args.example_words.split(",") if w.strip()]
    fig_path = os.path.join(FIGURES_DIR, "monosemy", "dip_over_time_examples.png")
    plot_dip_over_time(df, example_words, args.p_threshold, fig_path)


if __name__ == "__main__":
    main()
