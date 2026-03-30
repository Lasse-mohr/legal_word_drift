"""Tokenize paragraph records into sentence files for word2vec training.

Reads JSONL from data/processed/paragraphs/{year}.jsonl,
applies the legal tokenizer, and writes sentence files to
data/processed/sentences/{year}.txt (gensim LineSentence format).

Usage:
    python -m src.pipeline.03_preprocess [--start 1990] [--end 2025]
"""
from __future__ import annotations

import argparse
import glob
import logging
import os

from src.preprocessing.corpus_builder import build_sentences_for_year
from src.preprocessing.phrase_detector import PhraseDetector
from src.utils.config import PARAGRAPHS_DIR, SENTENCES_DIR, VOCAB_DIR, setup_logging

setup_logging("03_preprocess")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Tokenize judgment texts")
    parser.add_argument("--start", type=int, default=1990, help="Start year")
    parser.add_argument("--end", type=int, default=2025, help="End year")
    parser.add_argument(
        "--phrases-dir", type=str, default=None,
        help="Path to trained phrase model directory (from 04_detect_phrases). "
             "If not provided, skips phrase detection.",
    )
    args = parser.parse_args()

    # Load phrase detector if available
    phrase_detector = None
    if args.phrases_dir and os.path.exists(args.phrases_dir):
        logger.info(f"Loading phrase model from {args.phrases_dir}")
        phrase_detector = PhraseDetector.load(args.phrases_dir)

    os.makedirs(SENTENCES_DIR, exist_ok=True)
    total_sentences = 0

    for year in range(args.start, args.end + 1):
        para_path = os.path.join(PARAGRAPHS_DIR, f"{year}.jsonl")
        if not os.path.exists(para_path):
            logger.warning(f"Year {year}: no paragraph file found, skipping")
            continue

        out_path = os.path.join(SENTENCES_DIR, f"{year}.txt")
        n = build_sentences_for_year(
            para_path, out_path, phrase_detector=phrase_detector,
        )
        total_sentences += n

    logger.info(f"Done. Total sentences: {total_sentences}")


if __name__ == "__main__":
    main()
