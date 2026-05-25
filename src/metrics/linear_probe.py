"""Per-word linear-probe drift statistics.

For a single target word and each consecutive year-pair, fit a binary
classifier on L2-normalised BERT embeddings labelled by year and report
held-out accuracy + ROC-AUC. Two classifiers are provided:

  * ``fit_probe`` — L2-regularised logistic regression (the original probe).
  * ``fit_centroid_probe`` — perpendicular bisector of the train-set class
    means (nearest-centroid / LDA-with-isotropic-covariance). Also reports
    ``w_norm = ||μ_b − μ_a||`` so the magnitude of the centroid shift is
    available alongside accuracy.

Both have signature ``(X_a, X_b, seed) -> dict | None`` and are
interchangeable via ``sweep_consecutive_pairs(..., fit_fn=...)``.

Calibration & design choices shared between classifiers (kept centralised):

  * Class balance by subsampling to ``min(n_a, n_b)``.
  * Stratified 70/30 train/test split, single split (no repeats).
  * Pair-specific seed = ``base_seed + year_a`` for reproducible subsampling.
  * For ``fit_probe``: ``LogisticRegressionCV(Cs=10)``, inner CV scored on
    ROC-AUC, folds capped at ``min(5, smallest_class_in_train)`` (≥2).
  * For ``fit_centroid_probe``: means computed on the train split only so the
    held-out test set never enters the classifier definition; the test score
    is the signed projection ``(x − midpoint) · (μ_b − μ_a)``.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


FitFn = Callable[[np.ndarray, np.ndarray, int], "dict[str, float] | None"]


def build_kfold_indices(
    n: int,
    k: int,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-class k-fold indices: shuffle 0..n-1, split into k contiguous chunks.

    Returns ``[(train_idx, test_idx)]_k``. Train is the complement of test
    within ``0..n-1``. Fold sizes differ by at most one when ``n`` is not
    divisible by ``k``.
    """
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for i, te in enumerate(folds):
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        out.append((tr, te))
    return out


def l2_normalize_per_year(
    per_word: dict[str, dict[int, np.ndarray]],
) -> dict[str, dict[int, np.ndarray]]:
    """L2-normalise every per-(word, year) array as float32."""
    out: dict[str, dict[int, np.ndarray]] = {}
    for word, per_year in per_word.items():
        out[word] = {}
        for year, embs in per_year.items():
            X = embs.astype(np.float32)
            norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
            out[word][year] = X / norms
    return out


def fit_probe(
    X_a: np.ndarray, X_b: np.ndarray, seed: int
) -> dict[str, float] | None:
    """Balance, 70/30 stratified split, L2-LogReg with CV'd C, held-out metrics.

    Returns None if either class has < 4 instances (can't stratify both
    splits and run an internal CV) or if the inner-CV fold floor is unmet.
    """
    split = _balance_and_split(X_a, X_b, seed)
    if split is None:
        return None
    X_tr, X_te, y_tr, y_te, n = split

    n_per_class_train = int(min((y_tr == 0).sum(), (y_tr == 1).sum()))
    cv = min(5, n_per_class_train)
    if cv < 2:
        return None

    clf = LogisticRegressionCV(
        Cs=10, l1_ratios=(0,), cv=cv, scoring="roc_auc",
        max_iter=1000, n_jobs=1, random_state=seed,
        use_legacy_attributes=False,
    )
    clf.fit(X_tr, y_tr)
    scores = clf.decision_function(X_te)
    preds = clf.predict(X_te)
    auc = float(roc_auc_score(y_te, scores))
    acc = float(accuracy_score(y_te, preds))
    n_test_per_class = int((y_te == 0).sum())
    c_chosen = float(np.atleast_1d(clf.C_)[0])
    return {
        "auc": auc,
        "acc": acc,
        "n_per_class": int(n),
        "n_test_per_class": n_test_per_class,
        "c_chosen": c_chosen,
    }


def _balance_and_split(
    X_a: np.ndarray, X_b: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int] | None:
    """Shared balancing + 70/30 stratified split used by every fit_*_probe.

    Returns ``(X_tr, X_te, y_tr, y_te, n)`` or None if insufficient data.
    """
    n = min(X_a.shape[0], X_b.shape[0])
    if n < 4:
        return None
    rng = np.random.default_rng(seed)
    if X_a.shape[0] > n:
        idx = rng.choice(X_a.shape[0], size=n, replace=False)
        X_a = X_a[idx]
    if X_b.shape[0] > n:
        idx = rng.choice(X_b.shape[0], size=n, replace=False)
        X_b = X_b[idx]

    X = np.vstack([X_a, X_b])
    y = np.concatenate([np.zeros(n, dtype=np.int8), np.ones(n, dtype=np.int8)])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    return X_tr, X_te, y_tr, y_te, n


def score_centroid(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
) -> tuple[float, float, float]:
    """Centroid math, decoupled from balancing/splitting.

    Returns ``(auc, acc, w_norm)``. ``w_norm = ||μ_b − μ_a||`` where the
    means are computed on the train split only; the held-out scores are
    ``(X_te − midpoint) · (μ_b − μ_a)`` and a test point is predicted to
    class b iff its score is positive.
    """
    mu_a = X_tr[y_tr == 0].mean(axis=0)
    mu_b = X_tr[y_tr == 1].mean(axis=0)
    w = mu_b - mu_a
    midpoint = 0.5 * (mu_a + mu_b)
    scores = (X_te - midpoint) @ w
    preds = (scores > 0).astype(np.int8)
    auc = float(roc_auc_score(y_te, scores))
    acc = float(accuracy_score(y_te, preds))
    w_norm = float(np.linalg.norm(w))
    return auc, acc, w_norm


def score_logreg(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    seed: int,
) -> float | None:
    """L2-LogReg with inner-CV-selected C, held-out AUC only.

    Mirrors ``fit_probe`` hyperparameters (Cs=10, scoring=roc_auc) but takes
    an externally supplied train/test split. Returns None if the inner CV
    can't form ≥2 folds (smallest train class < 2).
    """
    n_per_class_train = int(min((y_tr == 0).sum(), (y_tr == 1).sum()))
    cv = min(5, n_per_class_train)
    if cv < 2:
        return None
    clf = LogisticRegressionCV(
        Cs=10, l1_ratios=(0,), cv=cv, scoring="roc_auc",
        max_iter=1000, n_jobs=1, random_state=seed,
        use_legacy_attributes=False,
    )
    clf.fit(X_tr, y_tr)
    scores = clf.decision_function(X_te)
    return float(roc_auc_score(y_te, scores))


def fit_centroid_probe(
    X_a: np.ndarray, X_b: np.ndarray, seed: int
) -> dict[str, float] | None:
    """Perpendicular bisector of the train-set class means.

    Decision rule on a test point ``x``:
        score(x) = (x - midpoint) · (μ_b - μ_a)
        predict class b iff score(x) > 0.

    Means are computed on the train split only; the test set never enters
    the classifier definition. Same balancing and 70/30 split as
    ``fit_probe`` (sharing ``_balance_and_split``), so per-pair the two
    classifiers see identical train/test partitions.

    Reports ``w_norm = ||μ_b - μ_a||`` — the L2 magnitude of the mean shift
    — alongside acc and AUC.
    """
    split = _balance_and_split(X_a, X_b, seed)
    if split is None:
        return None
    X_tr, X_te, y_tr, y_te, n = split

    auc, acc, w_norm = score_centroid(X_tr, y_tr, X_te, y_te)
    n_test_per_class = int((y_te == 0).sum())
    return {
        "auc": auc,
        "acc": acc,
        "w_norm": w_norm,
        "n_per_class": int(n),
        "n_test_per_class": n_test_per_class,
    }


def sweep_consecutive_pairs(
    word: str,
    per_year: dict[int, np.ndarray],
    years: list[int],
    seed: int,
    fit_fn: FitFn | None = None,
) -> pd.DataFrame:
    """Fit a probe on each consecutive year-pair where both years have data.

    ``fit_fn`` defaults to ``fit_probe`` (L2-LogReg). Pass
    ``fit_centroid_probe`` for the centroid classifier. The returned
    DataFrame carries every key in the fit function's return dict so
    classifier-specific fields (``c_chosen``, ``w_norm``) flow through
    automatically.
    """
    if fit_fn is None:
        fit_fn = fit_probe
    rows: list[dict] = []
    for ya, yb in zip(years[:-1], years[1:]):
        if ya not in per_year or yb not in per_year:
            continue
        X_a, X_b = per_year[ya], per_year[yb]
        pair_seed = seed + ya
        result = fit_fn(X_a, X_b, pair_seed)
        if result is None:
            continue
        row = {
            "word": word,
            "year_a": ya,
            "year_b": yb,
            "n_a_raw": int(X_a.shape[0]),
            "n_b_raw": int(X_b.shape[0]),
        }
        row.update(result)
        rows.append(row)
    return pd.DataFrame(rows)


def permutation_null(
    word: str,
    per_year: dict[int, np.ndarray],
    years: list[int],
    n_perms: int,
    seed: int,
    fit_fn: FitFn | None = None,
) -> pd.DataFrame:
    """Shuffle year labels across all of the word's embeddings, re-sweep.

    Preserves per-year N while destroying year-meaning — a global null
    testing "is there any year structure at all?".
    """
    available_years = [y for y in years if y in per_year]
    sizes = [per_year[y].shape[0] for y in available_years]
    X_all = np.vstack([per_year[y] for y in available_years])
    rng = np.random.default_rng(seed)

    frames: list[pd.DataFrame] = []
    pbar = tqdm(range(n_perms), desc=f"perm {word}", unit="perm")
    running_acc, running_auc = 0.0, 0.0
    for perm_id in pbar:
        order = rng.permutation(X_all.shape[0])
        X_shuffled = X_all[order]
        shuffled_per_year: dict[int, np.ndarray] = {}
        offset = 0
        for y, n in zip(available_years, sizes):
            shuffled_per_year[y] = X_shuffled[offset:offset + n]
            offset += n

        df = sweep_consecutive_pairs(
            word, shuffled_per_year, years,
            seed=seed + 1000 * (perm_id + 1),
            fit_fn=fit_fn,
        )
        df["perm_id"] = perm_id
        frames.append(df)
        running_acc += (df["acc"].mean() - running_acc) / (perm_id + 1)
        running_auc += (df["auc"].mean() - running_auc) / (perm_id + 1)
        pbar.set_postfix(acc=f"{running_acc:.3f}", auc=f"{running_auc:.3f}")

    return pd.concat(frames, ignore_index=True)
