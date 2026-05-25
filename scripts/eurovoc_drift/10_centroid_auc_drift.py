"""Centroid-AUC drift sweep for all EuroVoc labels, all year-pairs.

For each EuroVoc label with at least ``--min-years`` years of coverage,
computes the held-out centroid-AUC and analytical Mann–Whitney z-score for
every upper-triangle year-pair ``(ya, yb)`` with ``ya < yb``. Per pair, a
single k-fold partition (k=4) is built independently on each class using all
available embeddings (no balancing, no cap).

Per fold ``f`` with test sizes ``n' = |test_a|``, ``m' = |test_b|``::

    auc = score_centroid(X_tr, y_tr, X_te, y_te)
    z   = (auc − 0.5) · sqrt(12 · n' · m' / (n' + m' + 1))

The MW null calibration is validated in 09_mannwhitney_null_calibration.py
(see ``.research-notes.md`` 2026-05-13); no permutation null is computed.

Outputs (under ``PATHS.eurovoc_drift_centroid_auc_results``):
  centroid_auc_drift.csv  # one row per (label, year_a, year_b, fold)
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
from src.metrics.temporal_drift import load_per_year_embeddings
from src.paths import PATHS
from src.utils.config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=1990)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--min-years", type=int, default=10,
                        help="Keep labels with at least this many years that "
                             "have ≥1 embedding.")
    parser.add_argument("--k-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging("eurovoc_drift_10_centroid_auc_drift")
    logger = logging.getLogger(__name__)

    years = list(range(args.start, args.end + 1))
    embeddings_dir = str(PATHS.eurovoc_drift_embeddings)
    logger.info(f"Loading embeddings from {embeddings_dir}, "
                f"years {args.start}..{args.end}")
    raw = load_per_year_embeddings(
        embeddings_dir, years, words=None, min_usages=1,
    )
    per_word = l2_normalize_per_year(raw)
    logger.info(f"Loaded {len(per_word)} labels")

    labels = sorted(
        w for w, py in per_word.items() if len(py) >= args.min_years
    )
    logger.info(f"{len(labels)} labels pass min_years={args.min_years}")

    out_dir = PATHS.eurovoc_drift_centroid_auc_results
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    n_pairs = 0
    n_skipped_year = 0

    pbar = tqdm(labels, desc="labels", unit="lbl")
    for label in pbar:
        per_year = per_word[label]
        avail_years = sorted(per_year.keys())

        # Deterministic per-label RNG so reruns reproduce identically.
        ss = np.random.SeedSequence([args.seed, abs(hash(label)) % (2**32)])
        rng = np.random.default_rng(ss)

        # Precompute fold indices per year (need k_folds ≤ n).
        folds_by_year: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        for y in avail_years:
            n_y = per_year[y].shape[0]
            if n_y < args.k_folds:
                n_skipped_year += 1
                continue
            folds_by_year[y] = build_kfold_indices(n_y, args.k_folds, rng)

        usable_years = sorted(folds_by_year.keys())
        for i, ya in enumerate(usable_years):
            X_a = per_year[ya]
            folds_a = folds_by_year[ya]
            for yb in usable_years[i + 1:]:
                X_b = per_year[yb]
                folds_b = folds_by_year[yb]
                n_pairs += 1
                for fold in range(args.k_folds):
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
                    n_p = int(len(te_a))
                    m_p = int(len(te_b))
                    z = (auc - 0.5) * math.sqrt(
                        12.0 * n_p * m_p / (n_p + m_p + 1)
                    )
                    rows.append({
                        "label": label,
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

    df = pd.DataFrame(rows)
    out_csv = out_dir / "centroid_auc_drift.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Wrote {out_csv} ({len(df)} rows)")

    meta = {
        "start": args.start,
        "end": args.end,
        "min_years": args.min_years,
        "k_folds": args.k_folds,
        "seed": args.seed,
        "n_labels": len(labels),
        "n_pairs": n_pairs,
        "n_rows": len(df),
        "n_year_skips_low_count": n_skipped_year,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"meta.json: {meta}")


if __name__ == "__main__":
    main()
