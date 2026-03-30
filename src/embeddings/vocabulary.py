"""Shared vocabulary management for diachronic analysis.

Builds three vocabulary tiers:
- V_global:   words appearing in 3+ time slices with min_count >= threshold
- V_analysis: content words in V_global appearing in 10+ slices
- V_anchor:   top-N most stable words for Procrustes alignment
"""
from __future__ import annotations

import json
import logging
import os
from collections import Counter
from typing import Optional

from gensim.models import Word2Vec
from scipy.stats import hmean

logger = logging.getLogger(__name__)


def get_word_frequencies(model: Word2Vec) -> dict[str, int]:
    """Extract word frequency counts from a trained Word2Vec model."""
    return {
        word: model.wv.get_vecattr(word, "count")
        for word in model.wv.key_to_index
    }


def build_v_global(
    models: dict[str, Word2Vec],
    min_slices: int = 3,
) -> set[str]:
    """Build V_global: words present in at least min_slices models.

    Each model's vocabulary already respects its min_count threshold
    from training.
    """
    word_slice_counts: Counter[str] = Counter()
    for label, model in models.items():
        for word in model.wv.key_to_index:
            word_slice_counts[word] += 1

    v_global = {w for w, c in word_slice_counts.items() if c >= min_slices}
    logger.info(f"V_global: {len(v_global)} words (in {min_slices}+ slices)")
    return v_global


def build_v_analysis(
    models: dict[str, Word2Vec],
    v_global: set[str],
    min_slices: int = 10,
    exclude_tokens: Optional[set[str]] = None,
) -> set[str]:
    """Build V_analysis: content words in V_global present in 10+ slices.

    Excludes placeholder tokens (__CASEREF__, etc.) by default.
    """
    default_exclude = {"__caseref__", "__ecrref__", "__ojref__", "__pararef__", "__num__"}
    exclude = (exclude_tokens or set()) | default_exclude

    word_slice_counts: Counter[str] = Counter()
    for model in models.values():
        for word in model.wv.key_to_index:
            if word in v_global:
                word_slice_counts[word] += 1

    v_analysis = {
        w for w, c in word_slice_counts.items()
        if c >= min_slices and w not in exclude
    }
    logger.info(f"V_analysis: {len(v_analysis)} words (in {min_slices}+ slices)")
    return v_analysis


def build_v_anchor(
    models: dict[str, Word2Vec],
    v_analysis: set[str],
    top_n: int = 500,
) -> list[str]:
    """Build V_anchor: most stable high-frequency words for Procrustes alignment.

    Selects the top-N words by harmonic mean frequency across all time
    slices in which they appear. Harmonic mean favors words that are
    consistently common (not just common in one period).
    """
    word_freqs: dict[str, list[int]] = {}
    for model in models.values():
        for word in model.wv.key_to_index:
            if word in v_analysis:
                count = model.wv.get_vecattr(word, "count")
                word_freqs.setdefault(word, []).append(count)

    # Compute harmonic mean frequency for each word
    word_scores: dict[str, float] = {}
    for word, freqs in word_freqs.items():
        if len(freqs) >= 3:  # Need presence in multiple slices
            word_scores[word] = float(hmean(freqs))

    # Sort by harmonic mean frequency, take top N
    ranked = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
    v_anchor = [w for w, _ in ranked[:top_n]]

    logger.info(
        f"V_anchor: {len(v_anchor)} words "
        f"(hmean freq range: {ranked[0][1]:.0f} - {ranked[min(top_n, len(ranked))-1][1]:.0f})"
    )
    return v_anchor


def save_vocab(words: set[str] | list[str], path: str) -> None:
    """Save a vocabulary set/list to a JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(words), f, indent=2)


def load_vocab(path: str) -> list[str]:
    """Load a vocabulary list from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
