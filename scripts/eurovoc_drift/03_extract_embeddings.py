"""Extract contextualised embeddings for sampled EuroVoc-label usages.

For each year: load the usage index built by ``02_index_label_usages.py``,
sample N usages per label, encode the unique paragraphs through EURLEX-BERT,
and mean-pool subwords inside the matched character span.

Two granularities via ``--unit``:

  - ``year`` (default): one bag of usages per (label, year); sampling caps
    at ``--n-usages`` total per label per year.
  - ``judgment``: usages are stratified by ``celex`` within each
    (label, year); each judgment contributes at most ``--max-per-judgment``
    usages. Per-judgment downstream code (script 13) needs this so a
    high-frequency year doesn't squeeze individual judgments to 1–2 hits.

Outputs (under ``data/models/eurovoc_drift/``):
  sampled_usages/{unit}/{year}.jsonl
  embeddings/{unit}/{year}.npz
  embeddings_index/{unit}/{year}.jsonl
        Row-aligned with the NPZ. One JSON record per surviving embedding
        (failed extractions are dropped from BOTH files, in lockstep).
        Fields: ``label, row_idx, celex, para_idx, char_start, char_end``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from random import Random

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.embeddings.bert_encoder import (
    encode_paragraphs,
    extract_embedding,
    load_model,
    resolve_model,
)
from src.embeddings.usage_collector import (
    Usage,
    get_paragraphs_to_encode,
    load_usage_index,
    sample_usages,
    save_usage_index,
)
from src.paths import PATHS
from src.utils.config import setup_logging


def sample_usages_per_judgment(
    index: dict[str, list[Usage]],
    max_per_judgment: int,
    seed: int,
) -> dict[str, list[Usage]]:
    """Cap each (label, celex) bucket at ``max_per_judgment`` usages.

    Mirrors ``sample_usages`` but with celex-stratified subsampling: every
    judgment that mentions a label contributes all its hits up to the cap,
    so judgments with few mentions are never crowded out by a busy one.
    Labels with no usages are dropped.
    """
    rng = Random(seed)
    sampled: dict[str, list[Usage]] = {}
    for label, usages in index.items():
        if not usages:
            continue
        by_celex: dict[str, list[Usage]] = defaultdict(list)
        for u in usages:
            by_celex[u.celex].append(u)
        kept: list[Usage] = []
        for celex in sorted(by_celex.keys()):
            bucket = by_celex[celex]
            if len(bucket) <= max_per_judgment:
                kept.extend(bucket)
            else:
                kept.extend(rng.sample(bucket, max_per_judgment))
        sampled[label] = kept
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract BERT embeddings for EuroVoc labels")
    parser.add_argument("--unit", choices=["year", "judgment"], default="year",
                        help="Granularity of the upstream usage index and the "
                             "output directories.")
    parser.add_argument("--model", type=str, default="eurlex",
                        help="Encoder (friendly name). 'eurlex' writes to the "
                             "legacy paths; any control model is namespaced "
                             "under models/<name>/.")
    parser.add_argument("--start", type=int, default=1990)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--n-usages", type=int, default=100,
                        help="(year unit) Per-(label, year) usage cap.")
    parser.add_argument("--max-per-judgment", type=int, default=50,
                        help="(judgment unit) Per-(label, celex) usage cap.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    setup_logging(f"eurovoc_drift_03_extract_embeddings_{args.unit}")
    logger = logging.getLogger(__name__)
    hf_id, slug = resolve_model(args.model)
    logger.info(
        f"unit={args.unit}  model={args.model} ({hf_id}) slug={slug}  "
        f"{'n_usages=' + str(args.n_usages) if args.unit == 'year' else 'max_per_judgment=' + str(args.max_per_judgment)}"
    )

    model, tokenizer, device = load_model(model_name=hf_id, device=args.device)

    sampled_dir = PATHS.eurovoc_drift_sampled_for(args.unit, slug)
    embeddings_dir = PATHS.eurovoc_drift_embeddings_for(args.unit, slug)
    embeddings_index_dir = PATHS.eurovoc_drift_embeddings_index_for(args.unit, slug)
    sampled_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    embeddings_index_dir.mkdir(parents=True, exist_ok=True)

    for year in range(args.start, args.end + 1):
        emb_path = PATHS.eurovoc_drift_embeddings_year_for(args.unit, year, slug)
        if emb_path.exists() and not args.overwrite:
            logger.info(f"Skipping {year} (embeddings exist)")
            continue

        index_path = PATHS.eurovoc_drift_usage_index_year_for(args.unit, year)
        if not index_path.exists():
            logger.warning(f"Missing usage index for {year}: {index_path}")
            continue

        para_path = PATHS.paragraphs_year(year)
        if not para_path.exists():
            logger.warning(f"Missing paragraphs for {year}")
            continue

        logger.info(f"Processing {year}...")

        full_index = load_usage_index(str(index_path))
        if args.unit == "year":
            sampled = sample_usages(full_index, n_per_word=args.n_usages, seed=args.seed)
        else:
            sampled = sample_usages_per_judgment(
                full_index, max_per_judgment=args.max_per_judgment, seed=args.seed,
            )
        logger.info(f"  Sampled usages for {len(sampled)} labels")

        save_usage_index(sampled, str(PATHS.eurovoc_drift_sampled_year_for(args.unit, year, slug)))

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
        # Row-aligned index: parallel to label_embeddings[label] order.
        label_index: dict[str, list[dict]] = defaultdict(list)
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
                row_idx = len(label_embeddings[label])
                label_embeddings[label].append(emb)
                label_index[label].append({
                    "label": label,
                    "row_idx": row_idx,
                    "celex": usage.celex,
                    "para_idx": usage.para_idx,
                    "char_start": usage.char_start,
                    "char_end": usage.char_end,
                })

        if n_failures:
            logger.warning(f"  {n_failures} extraction failures")

        arrays = {
            f"w::{label}": np.stack(embs).astype(np.float16)
            for label, embs in label_embeddings.items()
            if embs
        }
        np.savez_compressed(emb_path, **arrays)
        logger.info(f"  Saved embeddings for {len(arrays)} labels → {emb_path}")

        index_path_out = PATHS.eurovoc_drift_embeddings_index_year_for(args.unit, year, slug)
        with open(index_path_out, "w", encoding="utf-8") as f:
            for label in label_index:
                if not label_embeddings.get(label):
                    continue
                for rec in label_index[label]:
                    f.write(json.dumps(rec) + "\n")
        logger.info(f"  Saved row-aligned index → {index_path_out}")


if __name__ == "__main__":
    main()
