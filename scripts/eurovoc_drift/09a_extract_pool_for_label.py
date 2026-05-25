"""Extract a large embedding pool for a single EuroVoc label across two years.

The standard EuroVoc-drift extractor (``03_extract_embeddings.py``) caps
sampling at ~100 usages per (label, year). For the Mann–Whitney null
calibration in ``09_mannwhitney_null_calibration.py`` we want a much larger
pool (e.g. 1000) for a single label across two years.

This standalone script reads the *full* per-year usage index already built
by ``02_index_label_usages.py``, filters to the requested label, samples
up to ``--pool-size`` usages, encodes them through EURLEX-BERT and writes
to a dedicated directory so the main embeddings pool is untouched.

Output:
  data/models/eurovoc_drift/mw_null_pool/{label}_{year}.npz   # key "w::{label}"
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.embeddings.bert_encoder import encode_paragraphs, extract_embedding, load_model
from src.embeddings.usage_collector import (
    get_paragraphs_to_encode,
    load_usage_index,
    sample_usages,
)
from src.paths import PATHS
from src.utils.config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--years", type=str, required=True,
                        help="Comma-separated year list, e.g. 2021,2022")
    parser.add_argument("--pool-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    setup_logging("eurovoc_drift_09a_extract_pool_for_label")
    logger = logging.getLogger(__name__)

    years = [int(y) for y in args.years.split(",") if y.strip()]
    label = args.label

    out_dir = PATHS.eurovoc_drift_mw_null_pool
    out_dir.mkdir(parents=True, exist_ok=True)

    model = tokenizer = device = None

    for year in years:
        out_path = out_dir / f"{label}_{year}.npz"
        if out_path.exists() and not args.overwrite:
            logger.info(f"Skipping {label}/{year} (exists at {out_path})")
            continue

        index_path = PATHS.eurovoc_drift_usage_index_year(year)
        if not index_path.exists():
            logger.error(f"Missing usage index for {year} at {index_path}")
            continue

        logger.info(f"Loading usage index for {year}...")
        full_index = load_usage_index(str(index_path))
        if label not in full_index or not full_index[label]:
            logger.error(f"Label {label!r} has no usages in {year}")
            continue
        n_avail = len(full_index[label])
        logger.info(f"  {label}/{year}: {n_avail} total usages")

        # Restrict the index to just this label so sample_usages only does work for us.
        single_label_index = {label: full_index[label]}
        sampled = sample_usages(
            single_label_index, n_per_word=args.pool_size, seed=args.seed
        )
        usages = sampled.get(label, [])
        logger.info(f"  sampled {len(usages)} usages")

        para_path = PATHS.paragraphs_year(year)
        if not para_path.exists():
            logger.error(f"Missing paragraphs for {year} at {para_path}")
            continue
        paragraphs = get_paragraphs_to_encode(sampled, str(para_path))
        if not paragraphs:
            logger.error(f"No paragraphs to encode for {label}/{year}")
            continue

        if model is None:
            logger.info("Loading EURLEX-BERT model...")
            model, tokenizer, device = load_model(device=args.device)

        para_keys = list(paragraphs.keys())
        para_texts = [paragraphs[k] for k in para_keys]
        logger.info(f"  encoding {len(para_texts)} paragraphs...")
        encoded_list = encode_paragraphs(
            para_texts, model, tokenizer, device, batch_size=args.batch_size
        )
        encoded_map = dict(zip(para_keys, encoded_list))

        embs: list[np.ndarray] = []
        n_fail = 0
        for u in usages:
            enc = encoded_map.get((u.celex, u.para_idx))
            if enc is None:
                n_fail += 1
                continue
            e = extract_embedding(enc, u.char_start, u.char_end)
            if e is None:
                n_fail += 1
                continue
            embs.append(e)
        if n_fail:
            logger.warning(f"  {n_fail} extraction failures")
        if not embs:
            logger.error(f"  no embeddings extracted for {label}/{year}")
            continue

        arr = np.stack(embs).astype(np.float16)
        np.savez_compressed(out_path, **{f"w::{label}": arr})
        logger.info(f"  Saved {arr.shape} → {out_path}")


if __name__ == "__main__":
    main()
