"""Collect and sample word usages from paragraph JSONL files.

Scans raw paragraph text for target word occurrences, builds a usage
index, and samples a fixed number of usages per word per year.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from random import Random

logger = logging.getLogger(__name__)

MIN_PARAGRAPH_LENGTH = 50


@dataclass
class Usage:
    """A single occurrence of a target word in a paragraph."""

    word: str
    celex: str
    para_idx: int
    char_start: int
    char_end: int


def _build_word_pattern(words: list[str]) -> re.Pattern:
    """Build a compiled regex that matches any of the target words.

    Uses word boundaries and case-insensitive matching.
    Words are sorted longest-first to ensure correct greedy matching.
    """
    escaped = [re.escape(w) for w in sorted(words, key=len, reverse=True)]
    pattern = r"\b(" + "|".join(escaped) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


def build_usage_index(
    paragraphs_path: str,
    target_words: list[str],
) -> dict[str, list[Usage]]:
    """Scan a year's paragraph JSONL file and find all target word occurrences.

    Args:
        paragraphs_path: Path to a {year}.jsonl file.
        target_words: List of target words to search for.

    Returns:
        Dict mapping each word to its list of Usage objects.
    """
    pattern = _build_word_pattern(target_words)
    target_set = {w.lower() for w in target_words}
    index: dict[str, list[Usage]] = {w: [] for w in target_words}

    with open(paragraphs_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            celex = doc["celex"]
            paragraphs = doc.get("paragraphs", [])

            for para_idx, para_text in enumerate(paragraphs):
                if len(para_text) < MIN_PARAGRAPH_LENGTH:
                    continue

                for match in pattern.finditer(para_text):
                    word = match.group(1).lower()
                    if word in target_set:
                        index[word].append(Usage(
                            word=word,
                            celex=celex,
                            para_idx=para_idx,
                            char_start=match.start(),
                            char_end=match.end(),
                        ))

    n_words_found = sum(1 for usages in index.values() if usages)
    n_total_usages = sum(len(usages) for usages in index.values())
    logger.info(f"Found {n_total_usages} usages of {n_words_found}/{len(target_words)} words")
    return index


def sample_usages(
    index: dict[str, list[Usage]],
    n_per_word: int = 100,
    seed: int = 42,
) -> dict[str, list[Usage]]:
    """Sample up to n_per_word usages per word from the index.

    Returns a new dict with sampled usages. Words with fewer than
    n_per_word usages keep all their usages.
    """
    rng = Random(seed)
    sampled: dict[str, list[Usage]] = {}

    for word, usages in index.items():
        if not usages:
            continue
        if len(usages) <= n_per_word:
            sampled[word] = list(usages)
        else:
            sampled[word] = rng.sample(usages, n_per_word)

    return sampled


def get_paragraphs_to_encode(
    sampled: dict[str, list[Usage]],
    paragraphs_path: str,
) -> dict[tuple[str, int], str]:
    """Load the unique paragraphs needed for encoding.

    Returns dict mapping (celex, para_idx) -> paragraph text.
    """
    # Collect unique (celex, para_idx) pairs
    needed: set[tuple[str, int]] = set()
    for usages in sampled.values():
        for u in usages:
            needed.add((u.celex, u.para_idx))

    # Load from JSONL
    paragraphs: dict[tuple[str, int], str] = {}
    with open(paragraphs_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            celex = doc["celex"]
            for para_idx, para_text in enumerate(doc.get("paragraphs", [])):
                key = (celex, para_idx)
                if key in needed:
                    paragraphs[key] = para_text

    logger.info(f"Loaded {len(paragraphs)} unique paragraphs for encoding")
    return paragraphs


def save_usage_index(index: dict[str, list[Usage]], path: str) -> None:
    """Save a usage index to a JSONL file (one line per usage)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for usages in index.values():
            for u in usages:
                f.write(json.dumps(asdict(u)) + "\n")


def load_usage_index(path: str) -> dict[str, list[Usage]]:
    """Load a usage index from a JSONL file."""
    index: dict[str, list[Usage]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            u = Usage(**data)
            index.setdefault(u.word, []).append(u)
    return index
