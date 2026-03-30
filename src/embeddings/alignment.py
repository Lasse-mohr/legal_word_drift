"""Orthogonal Procrustes alignment for diachronic word embeddings.

Aligns word2vec models trained on different time slices into a shared
vector space using Orthogonal Procrustes (Hamilton et al., 2016).

The alignment finds a rotation matrix W* = argmin ||WA - B||_F subject
to W^T W = I, mapping model A's space into model B's space while
preserving internal distances.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

import numpy as np
from gensim.models import KeyedVectors
from scipy.linalg import orthogonal_procrustes

logger = logging.getLogger(__name__)


def _get_shared_vocab(
    kv_a: KeyedVectors,
    kv_b: KeyedVectors,
    anchor_words: Optional[Sequence[str]] = None,
) -> list[str]:
    """Get the intersection vocabulary between two KeyedVectors.

    If anchor_words is provided, restrict to those words that exist in both.
    """
    vocab_a = set(kv_a.key_to_index.keys())
    vocab_b = set(kv_b.key_to_index.keys())

    if anchor_words is not None:
        return [w for w in anchor_words if w in vocab_a and w in vocab_b]
    return sorted(vocab_a & vocab_b)


def _build_matrix(kv: KeyedVectors, words: list[str]) -> np.ndarray:
    """Build a matrix of word vectors for the given words."""
    return np.array([kv[w] for w in words])


def align_pair(
    source_kv: KeyedVectors,
    target_kv: KeyedVectors,
    anchor_words: Optional[Sequence[str]] = None,
) -> tuple[np.ndarray, list[str]]:
    """Compute the Procrustes rotation from source space to target space.

    Args:
        source_kv: KeyedVectors to align (will be rotated).
        target_kv: KeyedVectors to align to (reference frame).
        anchor_words: Words to use for alignment. If None, uses full
                     intersection vocabulary.

    Returns:
        (rotation_matrix, shared_words): The orthogonal matrix W such that
        source_vectors @ W ≈ target_vectors, and the words used for alignment.
    """
    shared = _get_shared_vocab(source_kv, target_kv, anchor_words)
    if len(shared) < 10:
        raise ValueError(
            f"Only {len(shared)} shared words for alignment. "
            f"Need at least 10 for stable Procrustes."
        )

    A = _build_matrix(source_kv, shared)
    B = _build_matrix(target_kv, shared)

    # scipy.linalg.orthogonal_procrustes finds R minimizing ||A @ R - B||_F
    R, _scale = orthogonal_procrustes(A, B)

    return R, shared


def apply_rotation(kv: KeyedVectors, rotation: np.ndarray) -> KeyedVectors:
    """Apply a rotation matrix to all vectors in a KeyedVectors.

    Returns a new KeyedVectors with rotated vectors.
    """
    words = list(kv.key_to_index.keys())
    vectors = kv.vectors @ rotation

    aligned_kv = KeyedVectors(vector_size=kv.vector_size)
    aligned_kv.add_vectors(words, vectors)
    return aligned_kv


def align_to_reference(
    model_kvs: dict[str, KeyedVectors],
    reference_label: str,
    anchor_words: Optional[Sequence[str]] = None,
) -> dict[str, KeyedVectors]:
    """Align all models to a single reference frame.

    All models are aligned to the reference model's vector space.
    Uses chain alignment: sorts labels, aligns each to its nearest
    already-aligned neighbor to minimize Procrustes error propagation.

    Args:
        model_kvs: Dict mapping label -> KeyedVectors.
        reference_label: Label of the reference model (stays unrotated).
        anchor_words: Words to use for Procrustes alignment.

    Returns:
        Dict mapping label -> aligned KeyedVectors.
    """
    if reference_label not in model_kvs:
        raise ValueError(f"Reference '{reference_label}' not in model_kvs")

    aligned = {reference_label: model_kvs[reference_label]}
    labels = sorted(model_kvs.keys())

    # Align models in order of proximity to reference
    # Strategy: work outward from the reference in both directions
    ref_idx = labels.index(reference_label)

    # Forward: reference+1 aligned to reference, reference+2 to reference+1, etc.
    for i in range(ref_idx + 1, len(labels)):
        label = labels[i]
        prev_label = labels[i - 1]
        logger.info(f"Aligning {label} -> {prev_label}")
        R, shared = align_pair(model_kvs[label], aligned[prev_label], anchor_words)
        aligned[label] = apply_rotation(model_kvs[label], R)
        logger.info(f"  Used {len(shared)} anchor words")

    # Backward: reference-1 aligned to reference, reference-2 to reference-1, etc.
    for i in range(ref_idx - 1, -1, -1):
        label = labels[i]
        next_label = labels[i + 1]
        logger.info(f"Aligning {label} -> {next_label}")
        R, shared = align_pair(model_kvs[label], aligned[next_label], anchor_words)
        aligned[label] = apply_rotation(model_kvs[label], R)
        logger.info(f"  Used {len(shared)} anchor words")

    return aligned


def save_aligned(
    aligned_kvs: dict[str, KeyedVectors],
    output_dir: str,
) -> dict[str, str]:
    """Save aligned KeyedVectors to disk.

    Returns dict mapping label -> file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    for label, kv in aligned_kvs.items():
        path = os.path.join(output_dir, f"{label}.kv")
        kv.save(path)
        paths[label] = path
    return paths


def load_aligned(
    output_dir: str,
    labels: Optional[Sequence[str]] = None,
) -> dict[str, KeyedVectors]:
    """Load aligned KeyedVectors from disk.

    If labels is None, loads all .kv files in the directory.
    """
    kvs = {}
    if labels is None:
        for fname in sorted(os.listdir(output_dir)):
            if fname.endswith(".kv"):
                label = fname[:-3]
                kvs[label] = KeyedVectors.load(os.path.join(output_dir, fname))
    else:
        for label in labels:
            path = os.path.join(output_dir, f"{label}.kv")
            kvs[label] = KeyedVectors.load(path)
    return kvs
