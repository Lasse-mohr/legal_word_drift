"""Semantic narrowing/broadening metrics: k-NN dispersion, density, neighborhood size.

Measures how concentrated or diffuse a word's semantic neighborhood is.
Decreasing dispersion over time = semantic narrowing (tighter meaning).
Increasing dispersion = semantic broadening.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from gensim.models import KeyedVectors


def knn_dispersion(
    kv: KeyedVectors,
    word: str,
    k: int = 25,
) -> float:
    """Mean cosine distance from a word to its k nearest neighbors.

    dispersion(w) = (1/k) * sum_i (1 - cos(v_w, v_nn_i))
    Lower = tighter neighborhood. Higher = more diffuse.
    """
    if word not in kv:
        return float("nan")

    neighbors = kv.most_similar(word, topn=k)
    distances = [1.0 - sim for _, sim in neighbors]
    return float(np.mean(distances))


def neighborhood_density(
    kv: KeyedVectors,
    word: str,
    k: int = 25,
) -> float:
    """Mean pairwise cosine similarity among a word's k nearest neighbors.

    density(w) = mean_{i,j in kNN, i<j} cos(v_i, v_j)
    Higher = neighbors cluster tightly. Lower = neighbors are dispersed.
    """
    if word not in kv:
        return float("nan")

    neighbor_words = [w for w, _ in kv.most_similar(word, topn=k)]
    if len(neighbor_words) < 2:
        return float("nan")

    vecs = np.array([kv[w] for w in neighbor_words])
    # Normalize for cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    vecs_normed = vecs / norms

    # Pairwise cosine similarities (upper triangle)
    sim_matrix = vecs_normed @ vecs_normed.T
    n = len(neighbor_words)
    upper_tri = sim_matrix[np.triu_indices(n, k=1)]
    return float(np.mean(upper_tri))


def effective_neighborhood_size(
    kv: KeyedVectors,
    word: str,
    threshold: float = 0.5,
) -> int:
    """Count words within cosine similarity threshold of a word.

    A growing neighborhood = broadening. A shrinking one = narrowing.
    """
    if word not in kv:
        return 0

    vec = kv[word]
    sims = kv.cosine_similarities(vec, kv.vectors)
    # Exclude the word itself (similarity ~1.0)
    count = int(np.sum(sims > threshold)) - 1
    return max(count, 0)


def compute_all_dispersion(
    kv: KeyedVectors,
    words: Sequence[str],
    k: int = 25,
    threshold: float = 0.5,
) -> dict[str, dict]:
    """Compute all dispersion metrics for a list of words.

    Returns dict mapping word -> {dispersion, density, neighborhood_size}.
    """
    results = {}
    for word in words:
        results[word] = {
            "dispersion": knn_dispersion(kv, word, k=k),
            "density": neighborhood_density(kv, word, k=k),
            "neighborhood_size": effective_neighborhood_size(kv, word, threshold=threshold),
        }
    return results
