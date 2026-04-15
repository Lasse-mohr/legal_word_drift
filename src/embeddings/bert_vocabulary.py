"""BERT vocabulary bridge: filter V_analysis for contextualized embedding analysis.

Builds V_bert — a subset of V_analysis containing only single words that
can be reliably located in BERT-tokenized text.
"""
from __future__ import annotations

import json
import logging
import os

from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


def build_v_bert(
    v_analysis: list[str],
    tokenizer: PreTrainedTokenizerFast,
    min_length: int = 4,
    max_subwords: int = 5,
) -> dict[str, list[int]]:
    """Build V_bert from V_analysis by filtering to BERT-compatible single words.

    Returns a dict mapping each word to its subword token ID sequence.

    Filtering criteria:
    - Single words only (no underscores — excludes bigram phrases)
    - Minimum character length (default 4, excludes abbreviations)
    - Maximum subword token count (default 5, excludes heavily fragmented words)
    """
    v_bert: dict[str, list[int]] = {}
    skipped_phrase = 0
    skipped_short = 0
    skipped_subwords = 0

    for word in v_analysis:
        if "_" in word:
            skipped_phrase += 1
            continue
        if len(word) < min_length:
            skipped_short += 1
            continue

        # Tokenize the word in isolation (no special tokens)
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(token_ids) > max_subwords:
            skipped_subwords += 1
            continue

        v_bert[word] = token_ids

    logger.info(
        f"V_bert: {len(v_bert)} words "
        f"(skipped {skipped_phrase} phrases, {skipped_short} short, "
        f"{skipped_subwords} over {max_subwords} subwords)"
    )
    return v_bert


def save_v_bert(v_bert: dict[str, list[int]], path: str) -> None:
    """Save V_bert to a JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(v_bert, f, indent=2)
    logger.info(f"Saved V_bert ({len(v_bert)} words) to {path}")


def load_v_bert(path: str) -> dict[str, list[int]]:
    """Load V_bert from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
