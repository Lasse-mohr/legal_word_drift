"""Polysemy metrics from contextualized embeddings.

Measures how much a word's meaning varies across contexts using the
Average Pairwise Distance (APD) and self-similarity of its
contextualized BERT embeddings.

Higher APD = more polysemous (embeddings spread across contexts).
Lower self-similarity = more context-dependent.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def pairwise_cosine_distances(embeddings: np.ndarray) -> np.ndarray:
    """Upper-triangle vector of pairwise cosine distances.

    Args:
        embeddings: (N, D) array.

    Returns:
        1-D array of length N*(N-1)/2, float32. Empty if N < 2.
    """
    n = embeddings.shape[0]
    if n < 2:
        return np.empty(0, dtype=np.float32)
    vecs = embeddings.astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    normed = vecs / norms
    sims = normed @ normed.T
    iu = np.triu_indices(n, k=1)
    return (1.0 - sims[iu]).astype(np.float32)


def cosine_distance_to_centroid(embeddings: np.ndarray) -> np.ndarray:
    """Cosine distance from each embedding to the renormalised mean direction.

    Preferable to pairwise distances for unimodality testing: under a unimodal
    Gaussian null its distribution is well-behaved, whereas pairwise distances
    of high-D Gaussian samples can spuriously appear bimodal due to
    concentration of measure.
    """
    n = embeddings.shape[0]
    if n < 2:
        return np.empty(0, dtype=np.float32)
    vecs = embeddings.astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    normed = vecs / norms
    mean_dir = normed.mean(axis=0)
    mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-10)
    return (1.0 - normed @ mean_dir).astype(np.float32)


def within_group_distance_stats(
    embeddings: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Pairwise + centroid distance distributions and summary stats.

    Returns ``(pairwise_dists, centroid_dists, stats)``. ``stats`` always
    contains the keys consumed by ``src.visualization.polysemy_plots`` —
    ``n_usages``, ``mean_dist``/``median_dist``/``std_dist``,
    ``mean_centroid_dist``/``median_centroid_dist``/``std_centroid_dist``,
    ``silhouette_k2``, ``dip``/``dip_pvalue``,
    ``centroid_dip``/``centroid_dip_pvalue``.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import diptest

    dists = pairwise_cosine_distances(embeddings)
    centroid_dists = cosine_distance_to_centroid(embeddings)

    stats: dict = {
        "n_usages": int(embeddings.shape[0]),
        "mean_dist": float(dists.mean()) if dists.size else float("nan"),
        "median_dist": float(np.median(dists)) if dists.size else float("nan"),
        "std_dist": float(dists.std()) if dists.size else float("nan"),
        "mean_centroid_dist": (
            float(centroid_dists.mean()) if centroid_dists.size else float("nan")
        ),
        "median_centroid_dist": (
            float(np.median(centroid_dists)) if centroid_dists.size else float("nan")
        ),
        "std_centroid_dist": (
            float(centroid_dists.std()) if centroid_dists.size else float("nan")
        ),
    }

    if embeddings.shape[0] >= 10:
        km = KMeans(n_clusters=2, n_init="auto", random_state=seed)
        labels = km.fit_predict(embeddings)
        if len(set(labels)) == 2:
            stats["silhouette_k2"] = float(
                silhouette_score(embeddings, labels, metric="cosine")
            )
        else:
            stats["silhouette_k2"] = float("nan")
    else:
        stats["silhouette_k2"] = float("nan")

    if dists.size >= 4:
        dip, pval = diptest.diptest(dists)
        stats["dip"] = float(dip)
        stats["dip_pvalue"] = float(pval)
    else:
        stats["dip"] = float("nan")
        stats["dip_pvalue"] = float("nan")

    if centroid_dists.size >= 4:
        dip_c, pval_c = diptest.diptest(centroid_dists)
        stats["centroid_dip"] = float(dip_c)
        stats["centroid_dip_pvalue"] = float(pval_c)
    else:
        stats["centroid_dip"] = float("nan")
        stats["centroid_dip_pvalue"] = float("nan")

    return dists, centroid_dists, stats


def compute_per_year_pairwise_distances(
    embeddings_dir: str,
    words: Sequence[str],
    years: Sequence[int],
    *,
    min_usages: int,
    max_usages: int | None = None,
    seed: int = 42,
) -> dict[str, dict[int, tuple[np.ndarray, int]]]:
    """For each (word, year), upper-triangle pairwise cosine distances and n.

    Returns ``{word: {year: (dists, n_kept)}}``. Subsamples uniformly to
    ``max_usages`` per (word, year) when set; (word, year) cells below
    ``min_usages`` are dropped. Walks per-year ``{year}.npz`` files via
    :func:`src.metrics.temporal_drift.load_per_year_embeddings`.
    """
    from src.metrics.temporal_drift import load_per_year_embeddings

    rng = np.random.default_rng(seed)
    per_word = load_per_year_embeddings(
        embeddings_dir, years, words=set(words), min_usages=min_usages
    )
    out: dict[str, dict[int, tuple[np.ndarray, int]]] = {}
    for word in words:
        per_year = per_word.get(word, {})
        out[word] = {}
        for year in years:
            embs = per_year.get(year)
            if embs is None or embs.shape[0] < min_usages:
                continue
            if max_usages is not None and embs.shape[0] > max_usages:
                idx = rng.choice(embs.shape[0], size=max_usages, replace=False)
                embs = embs[idx]
            dists = pairwise_cosine_distances(embs)
            if dists.size:
                out[word][year] = (dists, int(embs.shape[0]))
        logger.info(
            f"{word}: {len(out[word])} year(s) with >= {min_usages} usages"
        )
    return out


def average_pairwise_distance(embeddings: np.ndarray) -> float:
    """Mean cosine distance between all pairs of contextualized embeddings.

    Args:
        embeddings: (N, D) array of contextualized word embeddings.

    Returns:
        Mean cosine distance in [0, 2]. Higher = more varied usage.
        Returns NaN if fewer than 2 embeddings.
    """
    n = embeddings.shape[0]
    if n < 2:
        return float("nan")

    # Upcast to float32 for numerical stability
    vecs = embeddings.astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    vecs_normed = vecs / norms

    # Pairwise cosine similarities (upper triangle only)
    sim_matrix = vecs_normed @ vecs_normed.T
    upper_tri = sim_matrix[np.triu_indices(n, k=1)]
    return float(1.0 - np.mean(upper_tri))


def self_similarity(embeddings: np.ndarray) -> float:
    """Mean cosine similarity of each embedding to the centroid.

    Following Ethayarajh (2019): lower self-similarity indicates
    the word is used in more varied contexts (more polysemous).

    Args:
        embeddings: (N, D) array of contextualized word embeddings.

    Returns:
        Mean cosine similarity to centroid in [-1, 1]. Lower = more polysemous.
        Returns NaN if fewer than 2 embeddings.
    """
    n = embeddings.shape[0]
    if n < 2:
        return float("nan")

    vecs = embeddings.astype(np.float32)
    centroid = vecs.mean(axis=0, keepdims=True)

    # Cosine similarity of each embedding to centroid
    norms_v = np.linalg.norm(vecs, axis=1) + 1e-10
    norms_c = np.linalg.norm(centroid) + 1e-10
    sims = (vecs @ centroid.T).squeeze() / (norms_v * norms_c)
    return float(np.mean(sims))


def compute_polysemy_metrics(
    embeddings_dir: str,
    years: Sequence[int],
    min_usages: int = 10,
) -> pd.DataFrame:
    """Compute APD and self-similarity for each word-year from stored embeddings.

    Args:
        embeddings_dir: Directory containing {year}.npz files.
        years: Years to process.
        min_usages: Minimum number of usages required for inclusion.

    Returns:
        DataFrame with columns: word, year, apd, self_similarity, n_usages.
    """
    import os
    rows = []

    for year in years:
        path = os.path.join(embeddings_dir, f"{year}.npz")
        if not os.path.exists(path):
            logger.warning(f"Missing embeddings file: {path}")
            continue

        data = np.load(path, allow_pickle=False)
        for key in data.files:
            embs = data[key]
            if embs.shape[0] < min_usages:
                continue
            word = key[3:] if key.startswith("w::") else key
            rows.append({
                "word": word,
                "year": year,
                "apd": average_pairwise_distance(embs),
                "self_similarity": self_similarity(embs),
                "n_usages": embs.shape[0],
            })
        data.close()

    df = pd.DataFrame(rows)
    logger.info(f"Computed polysemy metrics: {len(df)} word-year observations")
    return df


def compute_polysemy_ranking(
    metrics_df: pd.DataFrame,
    min_years: int = 20,
) -> pd.DataFrame:
    """Rank words by mean APD across years.

    Args:
        metrics_df: Output of compute_polysemy_metrics.
        min_years: Minimum number of years a word must appear in.

    Returns:
        DataFrame with columns: word, mean_apd, mean_self_similarity,
        n_years, sorted by mean_apd descending.
    """
    grouped = metrics_df.groupby("word").agg(
        mean_apd=("apd", "mean"),
        mean_self_similarity=("self_similarity", "mean"),
        n_years=("year", "nunique"),
        mean_n_usages=("n_usages", "mean"),
    ).reset_index()

    ranked = grouped[grouped["n_years"] >= min_years].sort_values(
        "mean_apd", ascending=False
    ).reset_index(drop=True)

    logger.info(f"Polysemy ranking: {len(ranked)} words (min {min_years} years)")
    return ranked
