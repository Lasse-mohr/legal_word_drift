"""Extract contextualized embeddings for sampled word usages.

For each year: loads the usage index, samples N usages per word,
deduplicates paragraphs, encodes them through BERT, and extracts
target word embeddings. Checkpointed per year.

Output:
  data/models/bert/sampled_usages/{year}.jsonl
  data/models/bert/embeddings/{year}.npz
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.embeddings.bert_encoder import extract_embedding, encode_paragraphs, load_model
from src.embeddings.usage_collector import (
    get_paragraphs_to_encode,
    load_usage_index,
    sample_usages,
    save_usage_index,
)
from src.utils.config import (
    BERT_EMBEDDINGS_DIR,
    BERT_SAMPLED_DIR,
    BERT_USAGE_INDEX_DIR,
    PARAGRAPHS_DIR,
    setup_logging,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract BERT embeddings for target words")
    parser.add_argument("--start", type=int, default=1990)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--n-usages", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging("13_extract_embeddings")
    logger = logging.getLogger(__name__)

    # Load model
    model, tokenizer, device = load_model(device=args.device)

    os.makedirs(BERT_SAMPLED_DIR, exist_ok=True)
    os.makedirs(BERT_EMBEDDINGS_DIR, exist_ok=True)

    for year in range(args.start, args.end + 1):
        embeddings_path = os.path.join(BERT_EMBEDDINGS_DIR, f"{year}.npz")
        if os.path.exists(embeddings_path):
            logger.info(f"Skipping {year} (embeddings already exist)")
            continue

        index_path = os.path.join(BERT_USAGE_INDEX_DIR, f"{year}.jsonl")
        if not os.path.exists(index_path):
            logger.warning(f"Missing usage index: {index_path}")
            continue

        paragraphs_path = os.path.join(PARAGRAPHS_DIR, f"{year}.jsonl")
        if not os.path.exists(paragraphs_path):
            logger.warning(f"Missing paragraphs file: {paragraphs_path}")
            continue

        logger.info(f"Processing {year}...")

        # Load and sample usages
        full_index = load_usage_index(index_path)
        sampled = sample_usages(full_index, n_per_word=args.n_usages, seed=args.seed)
        logger.info(f"  Sampled usages for {len(sampled)} words")

        # Save sampled usages
        sampled_path = os.path.join(BERT_SAMPLED_DIR, f"{year}.jsonl")
        save_usage_index(sampled, sampled_path)

        # Load paragraphs needed for encoding
        paragraphs = get_paragraphs_to_encode(sampled, paragraphs_path)

        # Build paragraph key -> text mapping for batch encoding
        para_keys = list(paragraphs.keys())
        para_texts = [paragraphs[k] for k in para_keys]

        if not para_texts:
            logger.warning(f"  No paragraphs to encode for {year}")
            continue

        # Encode all unique paragraphs
        logger.info(f"  Encoding {len(para_texts)} paragraphs...")
        encoded_list = encode_paragraphs(
            para_texts, model, tokenizer, device, batch_size=args.batch_size
        )

        # Map paragraph keys to encoded results
        encoded_map = dict(zip(para_keys, encoded_list))

        # Extract embeddings per word
        word_embeddings: dict[str, list[np.ndarray]] = defaultdict(list)
        n_failures = 0

        for word, usages in sampled.items():
            for usage in usages:
                key = (usage.celex, usage.para_idx)
                encoded = encoded_map.get(key)
                if encoded is None:
                    n_failures += 1
                    continue

                emb = extract_embedding(encoded, usage.char_start, usage.char_end)
                if emb is None:
                    n_failures += 1
                    continue

                word_embeddings[word].append(emb)

        if n_failures:
            logger.warning(f"  {n_failures} extraction failures")

        # Save embeddings as NPZ (word -> (N, 768) float16 array).
        # Prefix keys with "w::" to avoid collisions with savez_compressed's
        # reserved kwargs (e.g. "file") when a target word matches one.
        arrays = {}
        for word, embs in word_embeddings.items():
            if embs:
                arrays[f"w::{word}"] = np.stack(embs).astype(np.float16)

        np.savez_compressed(embeddings_path, **arrays)
        logger.info(
            f"  Saved embeddings for {len(arrays)} words to {embeddings_path}"
        )


if __name__ == "__main__":
    main()
