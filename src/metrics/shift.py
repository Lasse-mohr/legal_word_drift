"""Semantic shift metrics: cosine shift, Jaccard neighborhood shift, directional shift.

Measures how far a word has moved in embedding space between time slices.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from gensim.models import KeyedVectors


def cosine_shift(
    kv_t1: KeyedVectors,
    kv_t2: KeyedVectors,
    word: str,
) -> float:
    """Cosine distance of a word's vector between two time slices.

    Returns 1 - cos(v_t1, v_t2).  Range [0, 2]; 0 = identical, 1 = orthogonal.
    """
    if word not in kv_t1 or word not in kv_t2:
        return float("nan")
    v1 = kv_t1[word]
    v2 = kv_t2[word]
    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    return float(1.0 - cos_sim)


def jaccard_shift(
    kv_t1: KeyedVectors,
    kv_t2: KeyedVectors,
    word: str,
    k: int = 25,
) -> float:
    """Jaccard distance of k-nearest-neighbor sets between two time slices.

    Returns 1 - |NN_t1 ∩ NN_t2| / |NN_t1 ∪ NN_t2|.
    Range [0, 1]; 0 = identical neighbors, 1 = completely different.
    """
    if word not in kv_t1 or word not in kv_t2:
        return float("nan")

    nn_t1 = set(w for w, _ in kv_t1.most_similar(word, topn=k))
    nn_t2 = set(w for w, _ in kv_t2.most_similar(word, topn=k))

    intersection = len(nn_t1 & nn_t2)
    union = len(nn_t1 | nn_t2)
    if union == 0:
        return float("nan")
    return 1.0 - intersection / union


def directional_shift(
    kv_t1: KeyedVectors,
    kv_t2: KeyedVectors,
    word: str,
    k: int = 10,
) -> dict:
    """Analyze the direction of semantic shift.

    Computes the shift direction as the change in neighborhood centroid,
    then finds words closest to that direction vector.

    Returns:
        Dict with 'direction_norm', 'nearest_to_direction' (words nearest
        to the shift direction), and 'gained_neighbors' / 'lost_neighbors'.
    """
    if word not in kv_t1 or word not in kv_t2:
        return {"direction_norm": float("nan")}

    nn_t1 = [w for w, _ in kv_t1.most_similar(word, topn=k)]
    nn_t2 = [w for w, _ in kv_t2.most_similar(word, topn=k)]

    # Centroid of neighbors in each period (in aligned space, so comparable)
    def _centroid(kv: KeyedVectors, words: list[str]) -> np.ndarray:
        vecs = [kv[w] for w in words if w in kv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(kv.vector_size)

    c1 = _centroid(kv_t2, nn_t1)  # where old neighbors are now
    c2 = _centroid(kv_t2, nn_t2)  # where new neighbors are

    direction = c2 - c1
    direction_norm = float(np.linalg.norm(direction))

    # Find words in t2 closest to the shift direction
    nearest_to_dir = []
    if direction_norm > 1e-6:
        direction_unit = direction / direction_norm
        sims = kv_t2.cosine_similarities(direction_unit, kv_t2.vectors)
        top_idx = np.argsort(sims)[-5:][::-1]
        nearest_to_dir = [kv_t2.index_to_key[i] for i in top_idx]

    gained = set(nn_t2) - set(nn_t1)
    lost = set(nn_t1) - set(nn_t2)

    return {
        "direction_norm": direction_norm,
        "nearest_to_direction": nearest_to_dir,
        "gained_neighbors": sorted(gained),
        "lost_neighbors": sorted(lost),
    }


def compute_all_shifts(
    kv_t1: KeyedVectors,
    kv_t2: KeyedVectors,
    words: Sequence[str],
    k: int = 25,
) -> dict[str, dict]:
    """Compute cosine shift and Jaccard shift for a list of words.

    Returns dict mapping word -> {cosine_shift, jaccard_shift, combined_shift}.
    """
    results = {}
    for word in words:
        cs = cosine_shift(kv_t1, kv_t2, word)
        js = jaccard_shift(kv_t1, kv_t2, word, k=k)

        if np.isnan(cs) or np.isnan(js):
            combined = float("nan")
        else:
            combined = 0.5 * cs + 0.5 * js

        results[word] = {
            "cosine_shift": cs,
            "jaccard_shift": js,
            "combined_shift": combined,
        }
    return results
