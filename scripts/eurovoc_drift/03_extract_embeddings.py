"""Extract contextualised embeddings for sampled EuroVoc-label usages.

For each year: load the usage index built by ``02_index_label_usages.py``,
sample N usages per label, encode the unique paragraphs through EURLEX-BERT,
and mean-pool subwords inside the matched character span.

Output:
  data/models/eurovoc_drift/sampled_usages/{year}.jsonl
  data/models/eurovoc_drift/embeddings/{year}.npz
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.embeddings.bert_encoder import encode_paragraphs, extract_embedding, load_model
from src.embeddings.usage_collector import (
    get_paragraphs_to_encode,
    load_usage_index,
    sample_usages,
    save_usage_index,
)
from src.paths import PATHS
from src.utils.config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract BERT embeddings for EuroVoc labels")
    parser.add_argument("--start", type=int, default=1990)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--n-usages", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    setup_logging("eurovoc_drift_03_extract_embeddings")
    logger = logging.getLogger(__name__)

    model, tokenizer, device = load_model(device=args.device)

    PATHS.eurovoc_drift_sampled.mkdir(parents=True, exist_ok=True)
    PATHS.eurovoc_drift_embeddings.mkdir(parents=True, exist_ok=True)

    for year in range(args.start, args.end + 1):
        emb_path = PATHS.eurovoc_drift_embeddings_year(year)
        if emb_path.exists() and not args.overwrite:
            logger.info(f"Skipping {year} (embeddings exist)")
            continue

        index_path = PATHS.eurovoc_drift_usage_index_year(year)
        if not index_path.exists():
            logger.warning(f"Missing usage index for {year}")
            continue

        para_path = PATHS.paragraphs_year(year)
        if not para_path.exists():
            logger.warning(f"Missing paragraphs for {year}")
            continue

        logger.info(f"Processing {year}...")

        full_index = load_usage_index(str(index_path))
        sampled = sample_usages(full_index, n_per_word=args.n_usages, seed=args.seed)
        logger.info(f"  Sampled usages for {len(sampled)} labels")

        save_usage_index(sampled, str(PATHS.eurovoc_drift_sampled_year(year)))

        paragraphs = get_paragraphs_to_encode(sampled, str(para_path))
        para_keys = list(paragraphs.keys())
        para_texts = [paragraphs[k] for k in para_keys]

        if not para_texts:
            logger.warning(f"  No paragraphs to encode for {year}")
            continue

        logger.info(f"  Encoding {len(para_texts)} paragraphs...")
        encoded_list = encode_paragraphs(
            para_texts, model, tokenizer, device, batch_size=args.batch_size
        )
        encoded_map = dict(zip(para_keys, encoded_list))

        label_embeddings: dict[str, list[np.ndarray]] = defaultdict(list)
        n_failures = 0
        for label, usages in sampled.items():
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
                label_embeddings[label].append(emb)

        if n_failures:
            logger.warning(f"  {n_failures} extraction failures")

        arrays = {
            f"w::{label}": np.stack(embs).astype(np.float16)
            for label, embs in label_embeddings.items()
            if embs
        }
        np.savez_compressed(emb_path, **arrays)
        logger.info(f"  Saved embeddings for {len(arrays)} labels → {emb_path}")


if __name__ == "__main__":
    main()
