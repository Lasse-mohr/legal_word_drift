"""Index EuroVoc-label usages in per-year paragraph JSONLs.

For each label in ``selected_labels.parquet``, scans the paragraph JSONLs
and records every occurrence with character offsets. Multi-word labels are
matched with whitespace-tolerant regex (``\\s+`` between tokens).

Output: ``data/models/eurovoc_drift/usage_index/{year}.jsonl``
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from dataclasses import asdict

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.embeddings.usage_collector import MIN_PARAGRAPH_LENGTH, Usage
from src.paths import PATHS
from src.utils.config import setup_logging

# Single regex with thousands of alternations — chunk to keep individual
# patterns under Python's group-count limit and to bound compile time.
CHUNK_SIZE = 500


def _build_chunked_patterns(labels: list[str]) -> list[re.Pattern]:
    """Compile chunked, whitespace-tolerant, case-insensitive label regexes."""
    sorted_labels = sorted(labels, key=len, reverse=True)
    patterns: list[re.Pattern] = []
    for i in range(0, len(sorted_labels), CHUNK_SIZE):
        chunk = sorted_labels[i : i + CHUNK_SIZE]
        # Replace literal whitespace inside each label with \s+ so multi-space /
        # newline-wrapped occurrences still match.
        alts = [re.sub(r"\s+", r"\\s+", re.escape(w)) for w in chunk]
        patterns.append(re.compile(r"\b(" + "|".join(alts) + r")\b", re.IGNORECASE))
    return patterns


def _normalise_match(text: str) -> str:
    """Lowercase and collapse internal whitespace to single spaces."""
    return re.sub(r"\s+", " ", text).strip().lower()


def index_year(
    paragraphs_path: str,
    labels: list[str],
    patterns: list[re.Pattern],
) -> dict[str, list[Usage]]:
    target_set = {l.lower() for l in labels}
    index: dict[str, list[Usage]] = defaultdict(list)

    with open(paragraphs_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            celex = doc["celex"]
            for para_idx, para_text in enumerate(doc.get("paragraphs", [])):
                if len(para_text) < MIN_PARAGRAPH_LENGTH:
                    continue
                for pattern in patterns:
                    for match in pattern.finditer(para_text):
                        norm = _normalise_match(match.group(1))
                        if norm in target_set:
                            index[norm].append(
                                Usage(
                                    word=norm,
                                    celex=celex,
                                    para_idx=para_idx,
                                    char_start=match.start(),
                                    char_end=match.end(),
                                )
                            )
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Index EuroVoc-label usages")
    parser.add_argument(
        "--unit", choices=["year", "judgment"], default="year",
        help="Which selected_labels_{unit}.parquet to use; also routes the "
             "output index into usage_index/{unit}/.",
    )
    parser.add_argument("--start", type=int, default=1990)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    setup_logging(f"eurovoc_drift_02_index_label_usages_{args.unit}")
    logger = logging.getLogger(__name__)

    selected_path = PATHS.eurovoc_drift_selected_labels_for(args.unit)
    selected = pd.read_parquet(selected_path)
    labels = sorted({str(l).lower() for l in selected["label"].tolist()})
    logger.info(f"Loaded {len(labels):,} unique labels from {selected_path.name}")

    patterns = _build_chunked_patterns(labels)
    logger.info(f"Compiled {len(patterns)} regex chunk(s) (chunk size {CHUNK_SIZE})")

    out_dir = PATHS.eurovoc_drift_usage_index_for(args.unit)
    out_dir.mkdir(parents=True, exist_ok=True)

    for year in range(args.start, args.end + 1):
        para_path = PATHS.paragraphs_year(year)
        if not para_path.exists():
            logger.warning(f"Missing paragraphs for {year}: {para_path}")
            continue

        out_path = PATHS.eurovoc_drift_usage_index_year_for(args.unit, year)
        if out_path.exists() and not args.overwrite:
            logger.info(f"Skipping {year} (already indexed)")
            continue

        logger.info(f"Indexing {year}...")
        index = index_year(str(para_path), labels, patterns)
        n_usages = sum(len(v) for v in index.values())
        n_labels = sum(1 for v in index.values() if v)
        logger.info(f"  {year}: {n_usages:,} usages of {n_labels:,}/{len(labels):,} labels")

        with open(out_path, "w", encoding="utf-8") as f:
            for usages in index.values():
                for u in usages:
                    f.write(json.dumps(asdict(u)) + "\n")


if __name__ == "__main__":
    main()
