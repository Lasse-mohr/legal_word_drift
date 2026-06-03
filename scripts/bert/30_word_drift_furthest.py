"""MW centroid-AUC drift sweep for the 100 furthest-traveled v_bert words.

Applies the same centroid-AUC Mann–Whitney z drift measure as
``scripts/eurovoc_drift/10_centroid_auc_drift.py`` to the original (non-EuroVoc)
``v_bert`` vocabulary, whose per-year embeddings are cached under
``BERT_EMBEDDINGS_DIR`` (``{year}.npz``, ``w::``-prefixed, float16).

Words of interest are selected by **block-likeness** of their Y×Y median-z
drift matrix, not by raw distance. We score each word's matrix with a contiguous
two-block contrast (the single-changepoint t-statistic): for every split that
partitions the ordered years into an early and a late block (>= ``--min-block``
years each side), compare the cross-block z entries (between) against the
within-block z entries (within) via a Welch two-sample t-statistic, and take the
max over splits. High ``block_score`` = two stable plateaus separated by a sharp
transition; the ``argmax`` split's first late year is recorded as
``transition_year``. We keep the top ``--top-k`` by ``block_score``, then run the
full upper-triangle year-pair sweep on those words. ``dist_total`` (first->last
median-z) is retained as a reference column only.

Per pair, a single k-fold partition (k=4) is built independently on each class
using all available embeddings (no balancing, no cap). Per fold ``f`` with test
sizes ``n' = |test_a|``, ``m' = |test_b|``::

    auc = score_centroid(X_tr, y_tr, X_te, y_te)
    z   = (auc − 0.5) · sqrt(12 · n' · m' / (n' + m' + 1))

The MW null calibration is validated in
``scripts/eurovoc_drift/09_mannwhitney_null_calibration.py``; no permutation null
is computed.

Outputs (under ``--out-dir``, default ``data/models/bert/word_drift_furthest/``):
  word_drift_furthest.csv  # one row per (word, year_a, year_b, fold); schema-
                           # identical to script 10 (``label`` holds the word)
  ranking.csv              # per-word block_score / transition_year used to select
  meta.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.metrics.linear_probe import (
    build_kfold_indices,
    l2_normalize_per_year,
    score_centroid,
)
from src.metrics.temporal_drift import _load_year_npz, load_per_year_embeddings
from src.utils.config import BERT_EMBEDDINGS_DIR, setup_logging


def _pair_z_folds(
    X_a: np.ndarray,
    folds_a: list[tuple[np.ndarray, np.ndarray]],
    X_b: np.ndarray,
    folds_b: list[tuple[np.ndarray, np.ndarray]],
    k_folds: int,
) -> list[float]:
    """Per-fold centroid-AUC z for a single year-pair (no aggregation).

    Both arrays must already be L2-normalised float32 with ``>= k_folds`` rows.
    """
    zs: list[float] = []
    for fold in range(k_folds):
        tr_a, te_a = folds_a[fold]
        tr_b, te_b = folds_b[fold]
        X_tr = np.vstack([X_a[tr_a], X_b[tr_b]])
        y_tr = np.concatenate([
            np.zeros(len(tr_a), dtype=np.int8),
            np.ones(len(tr_b), dtype=np.int8),
        ])
        X_te = np.vstack([X_a[te_a], X_b[te_b]])
        y_te = np.concatenate([
            np.zeros(len(te_a), dtype=np.int8),
            np.ones(len(te_b), dtype=np.int8),
        ])
        auc, _acc, _w = score_centroid(X_tr, y_tr, X_te, y_te)
        n_p, m_p = int(len(te_a)), int(len(te_b))
        zs.append((auc - 0.5) * math.sqrt(12.0 * n_p * m_p / (n_p + m_p + 1)))
    return zs


def _median_z_matrix(
    per_year: dict[int, np.ndarray],
    k_folds: int,
    rng: np.random.Generator,
) -> tuple[list[int], np.ndarray]:
    """Y×Y symmetric median-over-folds centroid-AUC z matrix for one word.

    Returns ``(usable_years, Z)`` where ``Z[i, j]`` is the median z between
    ``usable_years[i]`` and ``usable_years[j]`` (diagonal left NaN). Only years
    with ``>= k_folds`` usages contribute.
    """
    folds_by_year: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for y in sorted(per_year.keys()):
        if per_year[y].shape[0] < k_folds:
            continue
        folds_by_year[y] = build_kfold_indices(per_year[y].shape[0], k_folds, rng)

    usable = sorted(folds_by_year.keys())
    n = len(usable)
    Z = np.full((n, n), np.nan, dtype=np.float64)
    for i, ya in enumerate(usable):
        for j in range(i + 1, n):
            yb = usable[j]
            zs = _pair_z_folds(
                per_year[ya], folds_by_year[ya],
                per_year[yb], folds_by_year[yb], k_folds,
            )
            m = float(np.median(zs))
            Z[i, j] = m
            Z[j, i] = m
    return usable, Z


def _block_score(Z: np.ndarray, min_block: int):
    """Contiguous two-block contrast (single-changepoint Welch t-statistic).

    For each split ``s`` (early = years[:s], late = years[s:]) with at least
    ``min_block`` years on each side, compare cross-block z entries (between)
    against within-block z entries (within) and take the split that maximises
    the Welch two-sample t-statistic. Returns ``(t, s, mean_within, mean_between)``
    for the best split, or ``None`` if no split is valid.
    """
    n = Z.shape[0]
    iu = np.triu_indices(n, k=1)
    ri, ci = iu
    vals = Z[iu]
    best = None
    for s in range(min_block, n - min_block + 1):
        within_mask = ((ri < s) & (ci < s)) | ((ri >= s) & (ci >= s))
        between_mask = (ri < s) & (ci >= s)
        within = vals[within_mask]
        between = vals[between_mask]
        within = within[~np.isnan(within)]
        between = between[~np.isnan(between)]
        if within.size < 2 or between.size < 2:
            continue
        mw, mb = float(within.mean()), float(between.mean())
        vw = float(within.var(ddof=1))
        vb = float(between.var(ddof=1))
        denom = math.sqrt(vb / between.size + vw / within.size)
        if denom == 0.0:
            continue
        t = (mb - mw) / denom
        if best is None or t > best[0]:
            best = (t, s, mw, mb)
    return best


def _rng_for(seed: int, word: str) -> np.random.Generator:
    """Deterministic per-word RNG (same idiom as eurovoc_drift script 10)."""
    ss = np.random.SeedSequence([seed, abs(hash(word)) % (2**32)])
    return np.random.default_rng(ss)


def rank_words(
    embeddings_dir: str,
    years: list[int],
    k_folds: int,
    min_years: int,
    min_block: int,
    seed: int,
    batch_size: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Pass 1: rank words by block-likeness of their Y×Y median-z matrix.

    Two sub-passes keep memory bounded:
      1a. Stream per-year NPZs once, counting eligible years (``>= k_folds``
          usages) per word; retain no embeddings.
      1b. For each batch of eligible words, load their full trajectories, build
          the median-z matrix and score it with the contiguous two-block
          t-statistic (``_block_score``).
    """
    # ── Pass 1a: count eligible years per word (no embeddings retained) ──
    counts: dict[str, int] = {}
    for year in years:
        year_data = _load_year_npz(embeddings_dir, year)
        if year_data is None:
            logger.warning(f"Missing embeddings for {year}")
            continue
        for word, embs in year_data.items():
            if embs.shape[0] < k_folds:
                continue
            counts[word] = counts.get(word, 0) + 1

    eligible = sorted(w for w, c in counts.items() if c >= min_years)
    logger.info(
        f"{len(eligible)} words pass min_years={min_years} "
        f"(of {len(counts)} with >=1 eligible year)"
    )

    # ── Pass 1b: batched matrix build + block scoring ──
    rows: list[dict] = []
    pbar = tqdm(total=len(eligible), desc="block-scoring", unit="word")
    for start in range(0, len(eligible), batch_size):
        batch = eligible[start:start + batch_size]
        raw = load_per_year_embeddings(
            embeddings_dir, years, words=set(batch), min_usages=1,
        )
        per_word = l2_normalize_per_year(raw)
        for word in batch:
            pbar.update(1)
            if word not in per_word:
                continue
            usable, Z = _median_z_matrix(
                per_word[word], k_folds, _rng_for(seed, word),
            )
            if len(usable) < 2 * min_block:
                continue
            best = _block_score(Z, min_block)
            if best is None:
                continue
            t, s, mw, mb = best
            dist_total = float(Z[0, -1]) if not np.isnan(Z[0, -1]) else math.nan
            rows.append({
                "word": word,
                "first_year": int(usable[0]),
                "last_year": int(usable[-1]),
                "span": int(usable[-1] - usable[0]),
                "n_years_ok": int(len(usable)),
                "block_score": float(t),
                "transition_year": int(usable[s]),
                "mean_within": float(mw),
                "mean_between": float(mb),
                "dist_total": dist_total,
            })
    pbar.close()

    ranking = (
        pd.DataFrame(rows)
        .sort_values("block_score", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    logger.info(f"{len(ranking)} words scored for block-likeness")
    return ranking


def sweep_year_pairs(
    per_word: dict[str, dict[int, np.ndarray]],
    words: list[str],
    k_folds: int,
    seed: int,
) -> tuple[list[dict], int, int]:
    """Pass 2: full upper-triangle year-pair sweep (mirrors script 10's loop)."""
    rows: list[dict] = []
    n_pairs = 0
    n_skipped_year = 0
    for word in tqdm(words, desc="sweep", unit="word"):
        per_year = per_word[word]
        rng = _rng_for(seed, word)

        folds_by_year: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        for y in sorted(per_year.keys()):
            n_y = per_year[y].shape[0]
            if n_y < k_folds:
                n_skipped_year += 1
                continue
            folds_by_year[y] = build_kfold_indices(n_y, k_folds, rng)

        usable_years = sorted(folds_by_year.keys())
        for i, ya in enumerate(usable_years):
            X_a = per_year[ya]
            folds_a = folds_by_year[ya]
            for yb in usable_years[i + 1:]:
                X_b = per_year[yb]
                folds_b = folds_by_year[yb]
                n_pairs += 1
                for fold in range(k_folds):
                    tr_a, te_a = folds_a[fold]
                    tr_b, te_b = folds_b[fold]
                    X_tr = np.vstack([X_a[tr_a], X_b[tr_b]])
                    y_tr = np.concatenate([
                        np.zeros(len(tr_a), dtype=np.int8),
                        np.ones(len(tr_b), dtype=np.int8),
                    ])
                    X_te = np.vstack([X_a[te_a], X_b[te_b]])
                    y_te = np.concatenate([
                        np.zeros(len(te_a), dtype=np.int8),
                        np.ones(len(te_b), dtype=np.int8),
                    ])
                    auc, _acc, w_norm = score_centroid(X_tr, y_tr, X_te, y_te)
                    n_p, m_p = int(len(te_a)), int(len(te_b))
                    z = (auc - 0.5) * math.sqrt(
                        12.0 * n_p * m_p / (n_p + m_p + 1)
                    )
                    rows.append({
                        "label": word,
                        "year_a": int(ya),
                        "year_b": int(yb),
                        "fold": fold,
                        "n_train_a": int(len(tr_a)),
                        "n_train_b": int(len(tr_b)),
                        "n_test_a": n_p,
                        "n_test_b": m_p,
                        "auc": float(auc),
                        "z": float(z),
                        "w_norm": float(w_norm),
                    })
    return rows, n_pairs, n_skipped_year


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=1990)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--min-years", type=int, default=10,
                        help="Keep words with at least this many years that "
                             "have >= k_folds embeddings.")
    parser.add_argument("--k-folds", type=int, default=4)
    parser.add_argument("--min-block", type=int, default=3,
                        help="Minimum years on each side of a candidate split "
                             "when scoring block-likeness.")
    parser.add_argument("--top-k", type=int, default=100,
                        help="Number of top block-likeness words to sweep.")
    parser.add_argument("--batch-size", type=int, default=400,
                        help="Words per batch when building median-z matrices "
                             "in Pass 1b (bounds peak memory).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str,
                        default=os.path.join(
                            os.path.dirname(BERT_EMBEDDINGS_DIR),
                            "word_drift_furthest"))
    args = parser.parse_args()

    setup_logging("bert_30_word_drift_furthest")
    logger = logging.getLogger(__name__)

    years = list(range(args.start, args.end + 1))
    embeddings_dir = str(BERT_EMBEDDINGS_DIR)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # ── Pass 1: rank by total distance traversed ──────────────────────────
    logger.info(f"Pass 1: ranking words from {embeddings_dir}, "
                f"years {args.start}..{args.end}")
    ranking = rank_words(
        embeddings_dir, years, args.k_folds, args.min_years, args.min_block,
        args.seed, args.batch_size, logger,
    )
    ranking.to_csv(os.path.join(out_dir, "ranking.csv"), index=False)

    top_words = ranking.head(args.top_k)["word"].tolist()
    logger.info(f"Top {len(top_words)} words: {top_words[:10]} ...")

    # ── Pass 2: full year-pair sweep on the top-k words ───────────────────
    logger.info(f"Pass 2: loading {len(top_words)} top words for sweep")
    raw = load_per_year_embeddings(
        embeddings_dir, years, words=set(top_words), min_usages=1,
    )
    per_word = l2_normalize_per_year(raw)
    rows, n_pairs, n_skipped_year = sweep_year_pairs(
        per_word, top_words, args.k_folds, args.seed,
    )

    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "word_drift_furthest.csv")
    df.to_csv(out_csv, index=False)
    logger.info(f"Wrote {out_csv} ({len(df)} rows)")

    meta = {
        "start": args.start,
        "end": args.end,
        "min_years": args.min_years,
        "k_folds": args.k_folds,
        "min_block": args.min_block,
        "selection": "block_score",
        "top_k": args.top_k,
        "seed": args.seed,
        "n_words_total_ranked": int(len(ranking)),
        "n_words_swept": len(top_words),
        "n_pairs": n_pairs,
        "n_rows": len(df),
        "n_year_skips_low_count": n_skipped_year,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"meta.json: {meta}")


if __name__ == "__main__":
    main()
