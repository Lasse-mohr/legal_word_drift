"""Bootstrap confidence intervals for drift metrics.

Trains M independent word2vec models per time slice (different random
seeds), computes metrics on each replicate, and reports mean + 95% CI.
"""
from __future__ import annotations

import logging
from typing import Callable, Sequence

import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import LineSentence

from src.embeddings.trainer import W2VConfig
from src.embeddings.alignment import align_pair, apply_rotation

logger = logging.getLogger(__name__)


def train_bootstrap_replicate(
    sentence_files: list[str],
    config: W2VConfig,
    seed: int,
) -> Word2Vec:
    """Train a single Word2Vec replicate with a specific seed."""

    class MultiFileSentences:
        def __init__(self, paths: list[str]):
            self.paths = paths
        def __iter__(self):
            for path in self.paths:
                yield from LineSentence(path)

    return Word2Vec(
        sentences=MultiFileSentences(sentence_files),
        vector_size=config.vector_size,
        window=config.window,
        min_count=config.min_count,
        sg=config.sg,
        negative=config.negative,
        sample=config.sample,
        epochs=config.epochs,
        seed=seed,
        workers=1,  # deterministic
    )


def bootstrap_metric(
    sentence_files_t1: list[str],
    sentence_files_t2: list[str],
    metric_fn: Callable[[KeyedVectors, KeyedVectors, str], float],
    word: str,
    config: W2VConfig | None = None,
    anchor_words: Sequence[str] | None = None,
    n_replicates: int = 10,
    base_seed: int = 42,
) -> dict:
    """Compute a metric with bootstrap confidence intervals.

    Args:
        sentence_files_t1: Sentence files for time slice 1.
        sentence_files_t2: Sentence files for time slice 2.
        metric_fn: Function(kv_t1, kv_t2, word) -> float.
        word: The word to compute the metric for.
        config: Word2Vec config. Defaults to W2VConfig().
        anchor_words: Anchor words for Procrustes alignment.
        n_replicates: Number of bootstrap replicates.
        base_seed: Starting seed (incremented per replicate).

    Returns:
        Dict with 'mean', 'std', 'ci_lower', 'ci_upper', 'values'.
    """
    config = config or W2VConfig()
    values = []

    for i in range(n_replicates):
        seed = base_seed + i
        model_t1 = train_bootstrap_replicate(sentence_files_t1, config, seed)
        model_t2 = train_bootstrap_replicate(sentence_files_t2, config, seed + 1000)

        # Align t1 to t2
        try:
            R, _ = align_pair(model_t1.wv, model_t2.wv, anchor_words)
            aligned_t1 = apply_rotation(model_t1.wv, R)
        except ValueError:
            logger.warning(f"Replicate {i}: alignment failed, skipping")
            continue

        val = metric_fn(aligned_t1, model_t2.wv, word)
        if not np.isnan(val):
            values.append(val)

    if not values:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "n_valid": 0,
            "values": [],
        }

    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "ci_lower": float(np.percentile(arr, 2.5)),
        "ci_upper": float(np.percentile(arr, 97.5)),
        "n_valid": len(values),
        "values": values,
    }
