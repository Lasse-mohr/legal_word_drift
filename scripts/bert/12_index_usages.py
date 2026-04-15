"""Index target word occurrences in paragraph JSONL files.

For each year, scans all paragraphs and records every occurrence of
every V_bert target word with character offsets.

Output: data/models/bert/usage_index/{year}.jsonl
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.embeddings.bert_vocabulary import load_v_bert
from src.embeddings.usage_collector import build_usage_index, save_usage_index
from src.utils.config import BERT_DIR, BERT_USAGE_INDEX_DIR, PARAGRAPHS_DIR, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Index word usages in paragraph files")
    parser.add_argument("--start", type=int, default=1990)
    parser.add_argument("--end", type=int, default=2025)
    args = parser.parse_args()

    setup_logging("12_index_usages")
    logger = logging.getLogger(__name__)

    # Load V_bert
    v_bert_path = os.path.join(BERT_DIR, "v_bert.json")
    v_bert = load_v_bert(v_bert_path)
    target_words = list(v_bert.keys())
    logger.info(f"Loaded V_bert: {len(target_words)} target words")

    os.makedirs(BERT_USAGE_INDEX_DIR, exist_ok=True)

    for year in range(args.start, args.end + 1):
        paragraphs_path = os.path.join(PARAGRAPHS_DIR, f"{year}.jsonl")
        if not os.path.exists(paragraphs_path):
            logger.warning(f"Missing paragraphs file: {paragraphs_path}")
            continue

        output_path = os.path.join(BERT_USAGE_INDEX_DIR, f"{year}.jsonl")
        if os.path.exists(output_path):
            logger.info(f"Skipping {year} (already indexed)")
            continue

        logger.info(f"Indexing {year}...")
        index = build_usage_index(paragraphs_path, target_words)

        # Log statistics
        usage_counts = [len(usages) for usages in index.values() if usages]
        if usage_counts:
            logger.info(
                f"  {year}: {len(usage_counts)} words found, "
                f"usages per word: mean={np.mean(usage_counts):.0f}, "
                f"median={np.median(usage_counts):.0f}, "
                f"max={max(usage_counts)}"
            )

        save_usage_index(index, output_path)


if __name__ == "__main__":
    main()
