"""Temporal drift metrics from per-year contextualised embeddings.

Two complementary diachronic analyses on top of the per-year BERT
embeddings produced by ``scripts/bert/13_extract_embeddings.py``:

A. **Centroid trajectory drift** — for each word, reduce each year to a
   single 768-d centroid and track how that centroid moves over time.
B. **Cross-period APD matrix** — for each word, compute pairwise cosine
   distance between embeddings of every pair of years, producing a
   (Y x Y) matrix whose diagonal is the within-year APD and whose
   off-diagonal entries reveal regime shifts.

The two functions ``compute_centroid_drift_table`` and
``compute_cross_period_table`` are the entry points used by scripts 15
and 17. Both walk the per-year NPZ files exactly once.
"""
from __future__ import annotations

import logging
import os
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Loading ───────────────────────────────────────────────────────────────


def _load_year_npz(embeddings_dir: str, year: int) -> dict[str, np.ndarray] | None:
    """Load one year's NPZ as a dict {word: (N, 768) float16}.

    Returns None if the file does not exist. Strips the ``w::`` key
    prefix written by script 13.
    """
    path = os.path.join(embeddings_dir, f"{year}.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=False)
    out: dict[str, np.ndarray] = {}
    for key in data.files:
        word = key[3:] if key.startswith("w::") else key
        out[word] = data[key]
    data.close()
    return out


def load_per_year_embeddings(
    embeddings_dir: str,
    years: Sequence[int],
    words: Iterable[str] | None = None,
    min_usages: int = 10,
) -> dict[str, dict[int, np.ndarray]]:
    """Walk the per-year NPZ files once, returning {word: {year: (N, 768)}}.

    Args:
        embeddings_dir: Directory containing ``{year}.npz`` files.
        years: Years to load.
        words: Optional whitelist of target words. If None, every word
            present in any year's NPZ is loaded.
        min_usages: Skip word-years with fewer usages than this.

    Returns:
        Nested dict ``{word: {year: ndarray}}``. Words with no surviving
        year are dropped.
    """
    word_filter = set(words) if words is not None else None
    out: dict[str, dict[int, np.ndarray]] = {}

    for year in years:
        year_data = _load_year_npz(embeddings_dir, year)
        if year_data is None:
            logger.warning(f"Missing embeddings for {year}")
            continue
        for word, embs in year_data.items():
            if word_filter is not None and word not in word_filter:
                continue
            if embs.shape[0] < min_usages:
                continue
            out.setdefault(word, {})[year] = embs

    return out


# ── Centroid trajectory ──────────────────────────────────────────────────


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two vectors."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    na = np.linalg.norm(a) + 1e-10
    nb = np.linalg.norm(b) + 1e-10
    return float(1.0 - np.dot(a, b) / (na * nb))


def compute_centroids(
    per_year_embs: dict[int, np.ndarray],
) -> tuple[list[int], np.ndarray]:
    """Compute one mean vector per year for a single word.

    Returns:
        (years, centroids) where ``years`` is the sorted list of years
        present and ``centroids`` is shaped ``(n_years, hidden_dim)``,
        float32.
    """
    years = sorted(per_year_embs.keys())
    if not years:
        return [], np.empty((0, 0), dtype=np.float32)
    centroids = np.stack(
        [per_year_embs[y].astype(np.float32).mean(axis=0) for y in years]
    )
    return years, centroids


def centroid_drift_series(
    word: str,
    years: list[int],
    centroids: np.ndarray,
    n_usages_per_year: dict[int, int],
    anchor_window: tuple[int, int],
) -> pd.DataFrame:
    """Per-year drift signal for one word.

    Returns a DataFrame with one row per year and columns:
        word, year, step_distance, cumulative_drift, anchor_distance,
        n_usages.

    ``step_distance`` is the cosine distance to the previous valid year
    (NaN for the first year). ``anchor_distance`` is the cosine distance
    to the mean centroid over years falling inside ``anchor_window``
    (inclusive).
    """
    if len(years) == 0:
        return pd.DataFrame({
            "word": [], "year": [], "step_distance": [],
            "cumulative_drift": [], "anchor_distance": [], "n_usages": [],
        })

    # Anchor centroid: mean over years inside the anchor window
    a_lo, a_hi = anchor_window
    anchor_idx = [i for i, y in enumerate(years) if a_lo <= y <= a_hi]
    if anchor_idx:
        anchor_vec = centroids[anchor_idx].mean(axis=0)
    else:
        # Fall back to the first available year if the anchor window is
        # empty for this word
        anchor_vec = centroids[0]

    rows = []
    cumulative = 0.0
    for i, y in enumerate(years):
        if i == 0:
            step = float("nan")
        else:
            step = _cosine_distance(centroids[i - 1], centroids[i])
            cumulative += step
        anchor_d = _cosine_distance(anchor_vec, centroids[i])
        rows.append({
            "word": word,
            "year": y,
            "step_distance": step,
            "cumulative_drift": cumulative,
            "anchor_distance": anchor_d,
            "n_usages": n_usages_per_year.get(y, 0),
        })
    return pd.DataFrame(rows)


def compute_centroid_drift_table(
    embeddings_dir: str,
    years: Sequence[int],
    words: Iterable[str] | None = None,
    min_usages: int = 10,
    min_years: int = 20,
    anchor_window: tuple[int, int] = (1990, 1994),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, list[int]]]:
    """Compute centroid drift for every eligible word.

    Returns:
        drift_df: long-form (word, year, step_distance, cumulative_drift,
            anchor_distance, n_usages).
        ranking_df: one row per word with summary scores.
        centroids: ``{word: (n_years, hidden_dim) float32}``.
        word_years: ``{word: [year, ...]}`` parallel to centroids rows.
    """
    per_word = load_per_year_embeddings(
        embeddings_dir, years, words=words, min_usages=min_usages
    )
    logger.info(
        f"Loaded embeddings for {len(per_word)} words "
        f"(min_usages={min_usages})"
    )

    drift_frames: list[pd.DataFrame] = []
    ranking_rows: list[dict] = []
    centroids_out: dict[str, np.ndarray] = {}
    years_out: dict[str, list[int]] = {}

    for word, per_year in per_word.items():
        word_years, centroids = compute_centroids(per_year)
        if len(word_years) < min_years:
            continue

        n_usages = {y: int(per_year[y].shape[0]) for y in word_years}
        df = centroid_drift_series(word, word_years, centroids, n_usages, anchor_window)
        drift_frames.append(df)
        centroids_out[word] = centroids
        years_out[word] = word_years

        steps = df["step_distance"].dropna().to_numpy()
        if steps.size == 0:
            continue
        max_idx = int(np.argmax(steps))
        # +1 because step at position i corresponds to years[i]
        max_step_year = word_years[max_idx + 1]
        end_vs_start = _cosine_distance(centroids[0], centroids[-1])
        anchor_to_final = float(df["anchor_distance"].iloc[-1])
        ranking_rows.append({
            "word": word,
            "n_years": len(word_years),
            "total_drift": float(steps.sum()),
            "mean_step": float(steps.mean()),
            "max_step": float(steps.max()),
            "max_step_year": int(max_step_year),
            "end_vs_start": end_vs_start,
            "anchor_to_final": anchor_to_final,
        })

    if not drift_frames:
        logger.warning("No words passed the centroid drift filters")
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            centroids_out,
            years_out,
        )

    drift_df = pd.concat(drift_frames, ignore_index=True)
    ranking_df = (
        pd.DataFrame(ranking_rows)
        .sort_values("total_drift", ascending=False)
        .reset_index(drop=True)
    )
    logger.info(f"Centroid drift: {len(ranking_df)} words ranked")
    return drift_df, ranking_df, centroids_out, years_out


# ── Cross-period APD ──────────────────────────────────────────────────────


def cross_period_apd_matrix(
    per_year_embs: dict[int, np.ndarray],
    max_per_year: int = 50,
    seed: int = 42,
) -> tuple[list[int], np.ndarray]:
    """Cross-period average pairwise cosine distance for one word.

    Subsamples up to ``max_per_year`` embeddings per year, normalises,
    then computes a single (Y, Y) matrix whose entry (a, b) is the mean
    cosine distance between embeddings sampled in year a and embeddings
    sampled in year b. Diagonal entries are the within-year APD computed
    over the same subsample.

    Returns ``(years, matrix)``.
    """
    years = sorted(per_year_embs.keys())
    if not years:
        return [], np.empty((0, 0), dtype=np.float32)

    rng = np.random.default_rng(seed)

    # Subsample and normalise per year
    blocks: list[np.ndarray] = []
    sizes: list[int] = []
    for y in years:
        embs = per_year_embs[y].astype(np.float32)
        n = embs.shape[0]
        if n > max_per_year:
            idx = rng.choice(n, size=max_per_year, replace=False)
            embs = embs[idx]
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
        blocks.append(embs / norms)
        sizes.append(embs.shape[0])

    stacked = np.concatenate(blocks, axis=0)  # (sum_M, D)
    sims = stacked @ stacked.T  # (sum_M, sum_M)
    distances = 1.0 - sims

    Y = len(years)
    matrix = np.zeros((Y, Y), dtype=np.float32)
    starts = np.cumsum([0] + sizes[:-1])
    for a in range(Y):
        sa, ea = int(starts[a]), int(starts[a]) + sizes[a]
        for b in range(a, Y):
            sb, eb = int(starts[b]), int(starts[b]) + sizes[b]
            block = distances[sa:ea, sb:eb]
            if a == b:
                # Within-year APD: mean over off-diagonal pairs only
                if block.shape[0] < 2:
                    val = float("nan")
                else:
                    n = block.shape[0]
                    val = float((block.sum() - np.trace(block)) / (n * (n - 1)))
            else:
                val = float(block.mean())
            matrix[a, b] = val
            matrix[b, a] = val

    return years, matrix


def cross_period_drift_score(
    years: list[int], matrix: np.ndarray
) -> dict:
    """Summary statistics for one word's cross-period matrix."""
    Y = len(years)
    if Y < 2:
        return {
            "n_years": Y,
            "mean_diag": float("nan"),
            "mean_off_diag": float("nan"),
            "drift_excess": float("nan"),
            "drift_ratio": float("nan"),
            "max_off_diag": float("nan"),
            "peak_year": int(years[0]) if years else -1,
        }

    diag = np.diagonal(matrix)
    mean_diag = float(np.nanmean(diag))

    mask = ~np.eye(Y, dtype=bool)
    off = matrix[mask]
    mean_off_diag = float(np.nanmean(off))
    max_off_diag = float(np.nanmax(off))

    drift_excess = mean_off_diag - mean_diag
    drift_ratio = mean_off_diag / mean_diag if mean_diag > 1e-10 else float("nan")

    # Row mean excluding the diagonal: which year is most distant from
    # all others on average?
    off_matrix = matrix.copy()
    np.fill_diagonal(off_matrix, np.nan)
    row_means = np.nanmean(off_matrix, axis=1)
    peak_idx = int(np.argmax(row_means))
    peak_year = int(years[peak_idx])

    return {
        "n_years": Y,
        "mean_diag": mean_diag,
        "mean_off_diag": mean_off_diag,
        "drift_excess": float(drift_excess),
        "drift_ratio": float(drift_ratio),
        "max_off_diag": max_off_diag,
        "peak_year": peak_year,
    }


def compute_cross_period_table(
    embeddings_dir: str,
    years: Sequence[int],
    words: Iterable[str] | None = None,
    min_usages: int = 10,
    min_years: int = 20,
    max_per_year: int = 50,
    seed: int = 42,
) -> tuple[dict[str, dict], pd.DataFrame]:
    """Compute cross-period APD matrices for every eligible word.

    Returns:
        matrices: ``{word: {"years": [...], "matrix": (Y, Y) ndarray}}``.
        ranking_df: ``word, n_years, mean_diag, mean_off_diag,
            drift_excess, drift_ratio, max_off_diag, peak_year`` sorted
            by ``drift_excess`` descending.
    """
    per_word = load_per_year_embeddings(
        embeddings_dir, years, words=words, min_usages=min_usages
    )
    logger.info(
        f"Loaded embeddings for {len(per_word)} words "
        f"(min_usages={min_usages})"
    )

    matrices: dict[str, dict] = {}
    ranking_rows: list[dict] = []

    for word, per_year in per_word.items():
        if len(per_year) < min_years:
            continue
        word_years, matrix = cross_period_apd_matrix(
            per_year, max_per_year=max_per_year, seed=seed
        )
        if len(word_years) < min_years:
            continue
        matrices[word] = {"years": word_years, "matrix": matrix}
        scores = cross_period_drift_score(word_years, matrix)
        scores["word"] = word
        ranking_rows.append(scores)

    if not ranking_rows:
        logger.warning("No words passed the cross-period filters")
        return matrices, pd.DataFrame()

    ranking_df = (
        pd.DataFrame(ranking_rows)
        .loc[:, [
            "word", "n_years", "mean_diag", "mean_off_diag",
            "drift_excess", "drift_ratio", "max_off_diag", "peak_year",
        ]]
        .sort_values("drift_excess", ascending=False)
        .reset_index(drop=True)
    )
    logger.info(f"Cross-period APD: {len(ranking_df)} words ranked")
    return matrices, ranking_df
