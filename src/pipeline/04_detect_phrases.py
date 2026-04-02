"""Train phrase (collocation) models and re-tokenize with detected phrases.

Two-step process:
1. Train bigram+trigram models on the initial sentence files (from 03_preprocess)
2. Re-tokenize all sentence files with phrase detection applied

Usage:
    python -m src.pipeline.04_detect_phrases [--start 1990] [--end 2025]
"""
from __future__ import annotations

import argparse
import logging
import os
from src.preprocessing.phrase_detector import PhraseDetector
from src.preprocessing.corpus_builder import build_sentences_for_year
from src.utils.config import SENTENCES_DIR, PARAGRAPHS_DIR, VOCAB_DIR, setup_logging

setup_logging("04_detect_phrases")
logger = logging.getLogger(__name__)

PHRASE_MODEL_DIR = os.path.join(VOCAB_DIR, "phrases")


def iter_sentences_from_files(start: int, end: int):
    """Yield token lists from existing sentence files."""
    for year in range(start, end + 1):
        path = os.path.join(SENTENCES_DIR, f"{year}.txt")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if tokens:
                    yield tokens


def main():
    parser = argparse.ArgumentParser(description="Train phrase models and re-tokenize")
    parser.add_argument("--start", type=int, default=1990, help="Start year")
    parser.add_argument("--end", type=int, default=2025, help="End year")
    parser.add_argument("--min-count", type=int, default=30, help="Phrase min count")
    parser.add_argument("--threshold", type=float, default=10.0, help="Phrase threshold")
    parser.add_argument("--max-phrase-len", type=int, default=3, help="Max words per phrase (2=bigrams only, 3=up to trigrams)")
    args = parser.parse_args()

    # Step 1: Train phrase model on existing sentence files
    logger.info("Step 1: Training phrase model...")
    sentences = list(iter_sentences_from_files(args.start, args.end))
    logger.info(f"  Loaded {len(sentences)} sentences for phrase training")

    detector = PhraseDetector.train(
        sentences,
        min_count=args.min_count,
        threshold=args.threshold,
        max_phrase_len=args.max_phrase_len,
    )
    detector.save(PHRASE_MODEL_DIR)
    logger.info(f"  Phrase model saved to {PHRASE_MODEL_DIR}")

    # Step 2: Re-tokenize with phrases applied
    logger.info("Step 2: Re-tokenizing with phrase detection...")
    total = 0
    for year in range(args.start, args.end + 1):
        para_path = os.path.join(PARAGRAPHS_DIR, f"{year}.jsonl")
        if not os.path.exists(para_path):
            continue
        out_path = os.path.join(SENTENCES_DIR, f"{year}.txt")
        n = build_sentences_for_year(
            para_path, out_path, phrase_detector=detector,
        )
        total += n

    logger.info(f"Done. Total sentences after phrase detection: {total}")


if __name__ == "__main__":
    main()
