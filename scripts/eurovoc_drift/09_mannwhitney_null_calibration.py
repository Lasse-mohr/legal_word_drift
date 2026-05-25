"""Mann–Whitney analytical null calibration for the centroid AUROC.

For a single high-count EuroVoc label across two consecutive years, this
script empirically calibrates the analytical Mann–Whitney null for the
centroid-classifier held-out AUC, by comparing it against a per-fold
permutation null.

Conditional on a fixed training fold T, the test-set AUROC under H₀ is a
standard Mann–Whitney U statistic with mean 1/2 and variance
``(n'+m'+1)/(12 n' m')``. We test this prediction across:

  * two regimes — ``within`` (random half-split of one year's pool, true H₀)
    and ``across`` (year-A vs year-B, expected signal),
  * sample sizes ``n ∈ {50, 100, 200, 500}`` per class,
  * ``r=5`` repetitions × ``k=4`` folds,
  * 200 within-fold label permutations.

Optional flag ``--include-logreg`` re-runs the same k-fold + permutation
protocol with the L2-LogReg probe so we can compare its real AUC and
permutation null to the centroid's.

Outputs (under ``data/results/metrics/eurovoc_drift_mw_null/<label>_<ya>_<yb>/``):
  auc_runs.csv     # one row per (classifier, regime, n, rep, fold)
  auc_perms.parquet  # long form per (classifier, regime, n, rep, fold, perm_id)
  meta.json

Figures (under ``data/results/figures/eurovoc_drift/mw_null/<label>_<ya>_<yb>/``):
  z_qq_within_centroid.png, z_qq_across_centroid.png
  perm_var_vs_analytical_centroid.png
  auc_real_vs_perm_per_n_{centroid,logreg}.png
  auc_centroid_vs_logreg_real.png       # only if --include-logreg
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.metrics.linear_probe import (
    build_kfold_indices,
    score_centroid,
    score_logreg,
)
from src.metrics.temporal_drift import load_per_year_embeddings
from src.paths import PATHS
from src.utils.config import setup_logging
from src.visualization.linear_probe_plots import (
    plot_auc_real_vs_perm_per_n,
    plot_classifier_real_auc_pair,
    plot_zscore_qq_grid,
)


ScoreFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float | None]


def _centroid_scorer(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray
) -> float | None:
    auc, _acc, _w = score_centroid(X_tr, y_tr, X_te, y_te)
    return auc


def _make_logreg_scorer(seed: int) -> ScoreFn:
    def _f(X_tr, y_tr, X_te, y_te):
        return score_logreg(X_tr, y_tr, X_te, y_te, seed=seed)
    return _f


def load_pool_for_label(
    label: str, years: list[int]
) -> dict[int, np.ndarray]:
    """Load the extended pool produced by ``09a_extract_pool_for_label.py``.

    Files are at ``PATHS.eurovoc_drift_mw_null_pool / "{label}_{year}.npz"``
    with key ``w::{label}``. Returns ``{year: ndarray}`` for years that exist.
    """
    out: dict[int, np.ndarray] = {}
    for year in years:
        p = PATHS.eurovoc_drift_mw_null_pool / f"{label}_{year}.npz"
        if not p.exists():
            continue
        with np.load(p) as data:
            key = f"w::{label}"
            if key not in data:
                continue
            out[year] = data[key]
    return out


def auto_pick_year_pair(
    embeddings_dir: str,
    pool_size: int,
    logger: logging.Logger,
) -> tuple[str, int, int]:
    """Scan embeddings, return (label, year_a, year_b) maximising min count.

    Picks the (label, consecutive-year-pair) with the largest min(count_a,
    count_b) such that both years have ≥ ``pool_size`` instances.
    """
    # Discover year files.
    files = sorted(Path(embeddings_dir).glob("*.npz"))
    years = sorted(int(f.stem) for f in files if f.stem.isdigit())
    logger.info(f"Found embeddings for {len(years)} years: {years[0]}..{years[-1]}")

    per_word = load_per_year_embeddings(
        embeddings_dir, years, words=None, min_usages=pool_size,
    )

    best = (None, None, None, -1)
    for word, per_year in per_word.items():
        ys_avail = sorted(per_year.keys())
        for ya, yb in zip(ys_avail[:-1], ys_avail[1:]):
            if yb - ya != 1:
                continue
            m = min(per_year[ya].shape[0], per_year[yb].shape[0])
            if m > best[3]:
                best = (word, ya, yb, m)
    if best[0] is None:
        raise RuntimeError(
            f"No (label, year-pair) found with ≥{pool_size} instances in both years."
        )
    logger.info(
        f"Auto-picked label={best[0]!r}, years=({best[1]},{best[2]}), "
        f"min count = {best[3]}"
    )
    return best[0], best[1], best[2]  # type: ignore[return-value]


def run_one_run(
    X_a: np.ndarray,
    X_b: np.ndarray,
    n_per_class: int,
    k: int,
    n_perms: int,
    scorers: dict[str, ScoreFn],
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict]]:
    """One (regime, n, rep) run across all classifiers.

    ``X_a`` and ``X_b`` are already the n-subsamples for this rep. Performs
    a single k-fold partition (per class) and, for every fold and every
    classifier in ``scorers``, computes real AUC plus ``n_perms`` permuted
    AUCs.

    Returns ``(runs_rows, perms_rows)``.
    """
    assert X_a.shape[0] == n_per_class == X_b.shape[0]
    folds_a = build_kfold_indices(n_per_class, k, rng)
    folds_b = build_kfold_indices(n_per_class, k, rng)

    runs_rows: list[dict] = []
    perms_rows: list[dict] = []

    for fold in range(k):
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
        n_prime = int(len(te_a))
        m_prime = int(len(te_b))
        var_analytical = (n_prime + m_prime + 1) / (12.0 * n_prime * m_prime)

        X_full = np.vstack([X_tr, X_te])
        y_full = np.concatenate([y_tr, y_te])
        n_train = X_tr.shape[0]

        for clf_name, scorer in scorers.items():
            auc_real = scorer(X_tr, y_tr, X_te, y_te)
            if auc_real is None:
                continue

            perm_aucs: list[float] = []
            for perm_id in range(n_perms):
                order = rng.permutation(X_full.shape[0])
                y_perm = y_full[order]
                # Refit on the same row partition but with permuted labels.
                y_tr_p = y_perm[:n_train]
                y_te_p = y_perm[n_train:]
                # Skip degenerate folds (all-one-class).
                if (y_tr_p == 0).sum() < 1 or (y_tr_p == 1).sum() < 1:
                    continue
                if (y_te_p == 0).sum() < 1 or (y_te_p == 1).sum() < 1:
                    continue
                auc_p = scorer(X_tr, y_tr_p, X_te, y_te_p)
                if auc_p is None:
                    continue
                perm_aucs.append(auc_p)
                perms_rows.append({
                    "classifier": clf_name,
                    "fold": fold,
                    "perm_id": perm_id,
                    "auc": auc_p,
                })

            if perm_aucs:
                perm_arr = np.asarray(perm_aucs)
                perm_var_emp = float(perm_arr.var(ddof=1))
                # Two-sided p-value.
                p_perm = float(
                    (np.abs(perm_arr - 0.5) >= abs(auc_real - 0.5)).mean()
                )
            else:
                perm_var_emp = float("nan")
                p_perm = float("nan")

            if clf_name == "centroid":
                z_real = (auc_real - 0.5) * math.sqrt(
                    12.0 * n_prime * m_prime / (n_prime + m_prime + 1)
                )
            else:
                z_real = float("nan")

            runs_rows.append({
                "classifier": clf_name,
                "fold": fold,
                "n_prime": n_prime,
                "m_prime": m_prime,
                "auc_real": float(auc_real),
                "z_real": float(z_real),
                "perm_var_emp": perm_var_emp,
                "perm_var_analytical": var_analytical,
                "p_perm": p_perm,
            })

    return runs_rows, perms_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--year-a", type=int, default=None)
    parser.add_argument("--year-b", type=int, default=None)
    parser.add_argument("--pool-size", type=int, default=1000)
    parser.add_argument("--n-values", type=str, default="50,100,200,500")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--k-folds", type=int, default=4)
    parser.add_argument("--n-perms", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-logreg", action="store_true",
                        help="Also run the LR probe with the same protocol.")
    args = parser.parse_args()

    setup_logging("eurovoc_drift_09_mannwhitney_null_calibration")
    logger = logging.getLogger(__name__)

    embeddings_dir = str(PATHS.eurovoc_drift_embeddings)
    pool_dir = PATHS.eurovoc_drift_mw_null_pool

    if args.label is None or args.year_a is None or args.year_b is None:
        # Prefer auto-pick from the extended pool dir if anything is there.
        pool_files = list(pool_dir.glob("*.npz")) if pool_dir.exists() else []
        if pool_files:
            # Group files by label, then pick a label with two consecutive years.
            from collections import defaultdict as _dd
            by_label: dict[str, list[int]] = _dd(list)
            for f in pool_files:
                # Filename: {label}_{year}.npz; label may contain underscores.
                stem = f.stem
                rsplit = stem.rsplit("_", 1)
                if len(rsplit) != 2:
                    continue
                lbl, ystr = rsplit
                try:
                    by_label[lbl].append(int(ystr))
                except ValueError:
                    continue
            best = None
            for lbl, ys in by_label.items():
                ys_sorted = sorted(ys)
                for a, b in zip(ys_sorted[:-1], ys_sorted[1:]):
                    if b - a == 1:
                        best = (lbl, a, b)
                        break
                if best:
                    break
            if not best:
                raise RuntimeError(
                    f"No consecutive year-pair found in {pool_dir}. "
                    f"Run 09a_extract_pool_for_label.py first."
                )
            label, year_a, year_b = best
            logger.info(f"Auto-picked from mw_null_pool: label={label!r}, "
                        f"years=({year_a},{year_b})")
        else:
            label, year_a, year_b = auto_pick_year_pair(
                embeddings_dir, args.pool_size, logger
            )
    else:
        label, year_a, year_b = args.label, args.year_a, args.year_b
        logger.info(f"Using user-specified label={label!r}, "
                    f"years=({year_a},{year_b})")

    # Load: prefer extended pool dir, fall back to the 100-cap dataset.
    raw_yearmap = load_pool_for_label(label, [year_a, year_b])
    if len(raw_yearmap) == 2:
        logger.info(f"Loaded extended pool from {pool_dir}")
        raw = {label: raw_yearmap}
    else:
        logger.info(f"Extended pool incomplete for {label}; falling back to "
                    f"{embeddings_dir}")
        raw = load_per_year_embeddings(
            embeddings_dir, [year_a, year_b], words={label},
            min_usages=args.pool_size,
        )
    if label not in raw or any(
        raw[label].get(y) is None or raw[label][y].shape[0] < args.pool_size
        for y in (year_a, year_b)
    ):
        avail = {y: (raw.get(label, {}).get(y).shape[0]
                     if raw.get(label, {}).get(y) is not None else 0)
                 for y in (year_a, year_b)}
        raise RuntimeError(
            f"Label {label!r} doesn't have ≥{args.pool_size} embeddings in "
            f"both years. Available: {avail}. Run 09a first or lower --pool-size."
        )
    # Trim the raw (unnormalised) pools once, deterministically. Both
    # normalisation modes downstream use the same trimmed instances so the
    # only difference between modes is whether each row is L2-normalised.
    trim_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 0]))
    raw_a = raw[label][year_a]
    raw_b = raw[label][year_b]
    trim_idx_a = trim_rng.choice(raw_a.shape[0], size=args.pool_size, replace=False)
    trim_idx_b = trim_rng.choice(raw_b.shape[0], size=args.pool_size, replace=False)
    pool_a_raw = raw_a[trim_idx_a].astype(np.float32)
    pool_b_raw = raw_b[trim_idx_b].astype(np.float32)
    logger.info(f"Pool size: {pool_a_raw.shape[0]} per year; dim={pool_a_raw.shape[1]}")

    n_values = [int(x) for x in args.n_values.split(",") if x.strip()]
    for n in n_values:
        if n * 2 > args.pool_size:
            raise ValueError(
                f"n={n} > pool_size/2={args.pool_size // 2}; within-year "
                f"regime needs two disjoint groups of size n."
            )
        if n % args.k_folds != 0:
            logger.warning(
                f"n={n} not divisible by k={args.k_folds}; folds will have "
                f"different sizes (analytical variance computed per fold)."
            )

    scorers: dict[str, ScoreFn] = {"centroid": _centroid_scorer}
    if args.include_logreg:
        # Per-call seeding done within run_one_run via a wrapper? We keep
        # determinism by passing a fixed seed; LogisticRegressionCV's
        # randomness is split selection — same for real and perms keeps
        # comparison fair.
        scorers["logreg"] = _make_logreg_scorer(seed=args.seed)
    logger.info(f"Active classifiers: {list(scorers)}")

    # Two passes: same subsamples, with vs without L2 normalisation. The
    # (regime, n, rep) RNG is seeded purely from (args.seed, regime_id, n,
    # rep) so the two modes draw identical instance indices and identical
    # permutations; only the X matrix differs.
    def _l2_normalise(X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
        return X / norms

    modes: list[tuple[str, np.ndarray, np.ndarray, str]] = [
        ("normalised", _l2_normalise(pool_a_raw),
         _l2_normalise(pool_b_raw), ""),
        ("unnormalised", pool_a_raw, pool_b_raw, "  (unnormalised)"),
    ]

    tag = f"{label}_{year_a}_{year_b}"

    for mode_name, pool_a_full, pool_b_full, title_suffix in modes:
        logger.info(f"=== Mode: {mode_name} ===")
        runs_all: list[dict] = []
        perms_all: list[dict] = []

        total_runs = 2 * len(n_values) * args.reps
        pbar = tqdm(total=total_runs, desc=f"runs[{mode_name}]", unit="run")
        for regime in ("within", "across"):
            for n in n_values:
                for rep in range(args.reps):
                    regime_id = 0 if regime == "within" else 1
                    rng = np.random.default_rng(
                        np.random.SeedSequence(
                            [args.seed, regime_id, n, rep]  # type: ignore[list-item]
                        )
                    )

                    if regime == "within":
                        perm = rng.permutation(args.pool_size)
                        half = args.pool_size // 2
                        idx_a = perm[:half][:n]
                        idx_b = perm[half:half * 2][:n]
                        X_a = pool_a_full[idx_a]
                        X_b = pool_a_full[idx_b]
                    else:
                        idx_a = rng.choice(args.pool_size, size=n, replace=False)
                        idx_b = rng.choice(args.pool_size, size=n, replace=False)
                        X_a = pool_a_full[idx_a]
                        X_b = pool_b_full[idx_b]

                    runs_rows, perms_rows = run_one_run(
                        X_a, X_b, n_per_class=n, k=args.k_folds,
                        n_perms=args.n_perms, scorers=scorers, rng=rng,
                    )
                    for r in runs_rows:
                        r.update({
                            "regime": regime, "n_per_class": n, "rep": rep,
                        })
                        runs_all.append(r)
                    for r in perms_rows:
                        r.update({
                            "regime": regime, "n_per_class": n, "rep": rep,
                        })
                        perms_all.append(r)
                    pbar.update(1)
        pbar.close()

        df_runs = pd.DataFrame(runs_all)
        df_perms = pd.DataFrame(perms_all)
        logger.info(
            f"[{mode_name}] runs rows: {len(df_runs)}; "
            f"perms rows: {len(df_perms)}"
        )

        # ── Save ──────────────────────────────────────────────────────
        csv_dir = PATHS.eurovoc_drift_mw_null_results / tag / mode_name
        fig_dir = PATHS.eurovoc_drift_mw_null_figures / tag / mode_name
        csv_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)

        df_runs.to_csv(csv_dir / "auc_runs.csv", index=False)
        df_perms.to_parquet(csv_dir / "auc_perms.parquet", index=False)
        meta = {
            "label": label,
            "year_a": year_a,
            "year_b": year_b,
            "pool_size": args.pool_size,
            "n_values": n_values,
            "reps": args.reps,
            "k_folds": args.k_folds,
            "n_perms": args.n_perms,
            "seed": args.seed,
            "include_logreg": bool(args.include_logreg),
            "classifiers": list(scorers),
            "normalisation": mode_name,
        }
        (csv_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        logger.info(
            f"[{mode_name}] Wrote auc_runs.csv ({len(df_runs)} rows), "
            f"auc_perms.parquet ({len(df_perms)} rows), meta.json"
        )

        # ── Figures ───────────────────────────────────────────────────
        plot_zscore_qq_grid(
            df_runs, regime="within",
            out_path=str(fig_dir / "z_qq_within_centroid.png"),
            classifier="centroid", title_suffix=title_suffix,
        )
        plot_zscore_qq_grid(
            df_runs, regime="across",
            out_path=str(fig_dir / "z_qq_across_centroid.png"),
            classifier="centroid", title_suffix=title_suffix,
        )
        for clf in scorers:
            plot_auc_real_vs_perm_per_n(
                df_runs, df_perms,
                out_path=str(fig_dir / f"auc_real_vs_perm_per_n_{clf}.png"),
                classifier=clf, title_suffix=title_suffix,
            )
        if "logreg" in scorers:
            plot_classifier_real_auc_pair(
                df_runs,
                out_path=str(fig_dir / "auc_centroid_vs_logreg_real.png"),
                clf_a="centroid", clf_b="logreg",
                title_suffix=title_suffix,
            )

        logger.info(f"[{mode_name}] Figures written to {fig_dir}")


if __name__ == "__main__":
    main()
