"""Assemble year-by-year sentence files from paragraph records.

Reads parsed paragraph JSONL files, applies preprocessing, and writes
tokenized sentence files in gensim LineSentence format (one sentence
per line, tokens space-separated).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from src.preprocessing.legal_tokenizer import preprocess_paragraph
from src.preprocessing.phrase_detector import PhraseDetector
from src.utils.io import read_jsonl

logger = logging.getLogger(__name__)


def build_sentences_for_year(
    paragraphs_path: str,
    output_path: str,
    section_filter: str = "grounds",
    phrase_detector: Optional[PhraseDetector] = None,
) -> int:
    """Build a sentence file for a single year.

    Args:
        paragraphs_path: Path to the year's JSONL file (from 02_fetch_texts).
        output_path: Path to write the sentence file.
        section_filter: Only include paragraphs from this section
                       ("grounds", "operative", or None for all).
        phrase_detector: Optional trained PhraseDetector for multi-word terms.

    Returns:
        Number of sentences written.
    """
    records = read_jsonl(paragraphs_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sentence_count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            if record.get("status") != "ok":
                continue

            paragraphs = record.get("paragraphs", [])
            for para_text in paragraphs:
                # We don't have section info in the JSONL from text_fetcher
                # (it stores raw paragraph text). For section filtering,
                # we'd need to re-parse or store section info. For now,
                # process all paragraphs.
                tokens = preprocess_paragraph(para_text)
                if not tokens:
                    continue

                if phrase_detector:
                    tokens = phrase_detector.apply(tokens)

                f.write(" ".join(tokens) + "\n")
                sentence_count += 1

    logger.info(f"  {output_path}: {sentence_count} sentences")
    return sentence_count
