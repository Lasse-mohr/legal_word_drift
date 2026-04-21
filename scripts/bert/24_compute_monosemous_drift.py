"""Compute per-word drift arrays for 100 monosemous words (figures 1-4).

Reads the monosemous cohort produced by script 23, picks 100 words
stratified by **true corpus frequency** (row counts in the usage index,
pre-sampling), and for each selected word computes:

  mean[a, b]  = mean pairwise cosine distance between years a and b
  var[a, b]   = variance of those distances
  within_kde  = smoothed density of within-year pairwise distances per year
  consec_kde  = smoothed density of cross-year pairwise distances for (y, y+1)

Years where the word has fewer than --min-usages embeddings in the NPZ
are treated as missing: the corresponding matrix rows/cols and KDE rows
are NaN. This keeps every word on the same 36-year grid so the plotting
script can align subplots with shared ticks.

Outputs:
  data/models/bert/monosemous_drift.npz
  data/results/metrics/monosemous_selection.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.metrics.polysemy import pairwise_cosine_distances
from src.metrics.temporal_drift import (
    cross_period_mean_var_matrix,
    load_per_year_embeddings,
)
from src.utils.config import (
    BERT_DIR,
    BERT_EMBEDDINGS_DIR,
    BERT_USAGE_INDEX_DIR,
    METRICS_DIR,
    setup_logging,
)


def corpus_frequency(
    usage_index_dir: str, years: range, words: set[str]
) -> dict[str, int]:
    """Total row count across all years' usage-index JSONL files."""
    logger = logging.getLogger(__name__)
    counts: dict[str, int] = {w: 0 for w in words}
    for year in years:
        path = os.path.join(usage_index_dir, f"{year}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                w = json.loads(line)["word"]
                if w in counts:
                    counts[w] += 1
    logger.info(f"Tallied corpus frequency for {len(counts)} words")
    return counts


def stratified_sample(
    freq: dict[str, int], n_words: int, n_strata: int, seed: int
) -> list[dict]:
    """Sample n_words with even coverage across n_strata frequency deciles."""
    logger = logging.getLogger(__name__)
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "word": list(freq.keys()),
        "freq": list(freq.values()),
    }).query("freq > 0").sort_values("freq").reset_index(drop=True)

    if len(df) == 0:
        return []

    df["decile"] = pd.qcut(
        df["freq"], q=n_strata, labels=False, duplicates="drop"
    )
    per_bin = max(1, n_words // n_strata)
    picks = []
    for _, g in df.groupby("decile"):
        if len(g) <= per_bin:
            picks.extend(g.to_dict("records"))
        else:
            idx = rng.choice(len(g), size=per_bin, replace=False)
            picks.extend(g.iloc[idx].to_dict("records"))

    logger.info(
        f"Stratified pick: {len(picks)} words across {df['decile'].nunique()} deciles"
    )
    return picks


def kde_on_grid(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Evaluate a Gaussian KDE on a common grid; NaN row if too few samples."""
    if samples.size < 10:
        return np.full_like(grid, np.nan, dtype=np.float32)
    try:
        kde = gaussian_kde(samples)
        return kde(grid).astype(np.float32)
    except Exception:
        return np.full_like(grid, np.nan, dtype=np.float32)


def compute_word_arrays(
    per_year: dict[int, np.ndarray],
    all_years: list[int],
    min_usages: int,
    max_per_year: int,
    grid: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    """Compute all four arrays for one word on the full all_years grid."""
    Y = len(all_years)
    G = grid.size

    # Filter to years meeting the usage floor.
    strong = {y: per_year[y] for y in per_year if per_year[y].shape[0] >= min_usages}

    mean_full = np.full((Y, Y), np.nan, dtype=np.float32)
    var_full = np.full((Y, Y), np.nan, dtype=np.float32)

    if len(strong) >= 2:
        strong_years, mean_mat, var_mat = cross_period_mean_var_matrix(
            strong, max_per_year=max_per_year, seed=seed
        )
        idx_map = {y: i for i, y in enumerate(all_years)}
        for i, ya in enumerate(strong_years):
            ia = idx_map[ya]
            for j, yb in enumerate(strong_years):
                jb = idx_map[yb]
                mean_full[ia, jb] = mean_mat[i, j]
                var_full[ia, jb] = var_mat[i, j]

    # Within-year KDE per year.
    within = np.full((Y, G), np.nan, dtype=np.float32)
    rng = np.random.default_rng(seed)
    for i, y in enumerate(all_years):
        if y not in strong:
            continue
        embs = strong[y]
        if embs.shape[0] > max_per_year:
            idx = rng.choice(embs.shape[0], size=max_per_year, replace=False)
            embs = embs[idx]
        dists = pairwise_cosine_distances(embs)
        within[i] = kde_on_grid(dists, grid)

    # Consecutive-year KDE: pairs (y, y+1) where both meet the floor.
    consec = np.full((Y - 1, G), np.nan, dtype=np.float32)
    for i in range(Y - 1):
        y1, y2 = all_years[i], all_years[i + 1]
        if y1 not in strong or y2 not in strong:
            continue
        a, b = strong[y1], strong[y2]
        if a.shape[0] > max_per_year:
            a = a[rng.choice(a.shape[0], size=max_per_year, replace=False)]
        if b.shape[0] > max_per_year:
            b = b[rng.choice(b.shape[0], size=max_per_year, replace=False)]
        a32 = a.astype(np.float32)
        b32 = b.astype(np.float32)
        a32 = a32 / (np.linalg.norm(a32, axis=1, keepdims=True) + 1e-10)
        b32 = b32 / (np.linalg.norm(b32, axis=1, keepdims=True) + 1e-10)
        dists = (1.0 - a32 @ b32.T).ravel().astype(np.float32)
        consec[i] = kde_on_grid(dists, grid)

    return {
        "mean": mean_full,
        "var": var_full,
        "within_kde": within,
        "consec_kde": consec,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=1990)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--n-words", type=int, default=100)
    parser.add_argument("--n-strata", type=int, default=10)
    parser.add_argument("--min-usages", type=int, default=150)
    parser.add_argument("--max-per-year", type=int, default=300)
    parser.add_argument("--kde-gridsize", type=int, default=200)
    parser.add_argument("--kde-range", type=str, default="0.0,1.5")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging("24_compute_monosemous_drift")
    logger = logging.getLogger(__name__)

    years = list(range(args.start, args.end + 1))

    # Load the monosemous cohort.
    mono_path = os.path.join(METRICS_DIR, "monosemous_words.json")
    with open(mono_path) as f:
        mono_words: list[str] = json.load(f)
    logger.info(f"Loaded {len(mono_words)} monosemous words from script 23")
    if not mono_words:
        logger.error("Empty monosemous cohort; aborting.")
        return

    # Frequency tally across the cohort, then stratified sample.
    freq = corpus_frequency(BERT_USAGE_INDEX_DIR, range(args.start, args.end + 1),
                            set(mono_words))
    picks = stratified_sample(freq, args.n_words, args.n_strata, args.seed)
    picked_words = [p["word"] for p in picks]

    selection_path = os.path.join(METRICS_DIR, "monosemous_selection.json")
    with open(selection_path, "w") as f:
        json.dump(picks, f, indent=2)
    logger.info(
        f"Wrote selection ({len(picks)} words) to {selection_path}"
    )

    # Load embeddings for just the picked words.
    per_word = load_per_year_embeddings(
        BERT_EMBEDDINGS_DIR, years, words=picked_words, min_usages=1
    )
    logger.info(f"Loaded per-year embeddings for {len(per_word)} picked words")

    # Common KDE grid.
    lo, hi = [float(x) for x in args.kde_range.split(",")]
    grid = np.linspace(lo, hi, args.kde_gridsize).astype(np.float32)

    arrays: dict[str, np.ndarray] = {"grid::": grid}
    for word in picked_words:
        if word not in per_word:
            logger.warning(f"Missing embeddings for picked word: {word}")
            continue
        out = compute_word_arrays(
            per_word[word], years,
            min_usages=args.min_usages,
            max_per_year=args.max_per_year,
            grid=grid,
            seed=args.seed,
        )
        arrays[f"mean::{word}"] = out["mean"]
        arrays[f"var::{word}"] = out["var"]
        arrays[f"within_kde::{word}"] = out["within_kde"]
        arrays[f"consec_kde::{word}"] = out["consec_kde"]
    arrays["years::"] = np.asarray(years, dtype=np.int32)

    out_path = os.path.join(BERT_DIR, "monosemous_drift.npz")
    np.savez_compressed(out_path, **arrays)
    logger.info(f"Wrote {out_path} ({len(arrays)} arrays)")


if __name__ == "__main__":
    main()
