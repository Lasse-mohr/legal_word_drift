"""Fetch full judgment XHTML texts from CELLAR REST API.

Reads the metadata index from 01_fetch_metadata, then fetches and parses
XHTML for each judgment. Outputs per-year JSONL files to
data/processed/paragraphs/{year}.jsonl with checkpoint-based resume.

Usage:
    python -m src.pipeline.02_fetch_texts [--concurrency 10] [--max-items 0]
    python -m src.pipeline.02_fetch_texts --year 2020   # single year
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os

import pandas as pd

from src.text_fetcher import fetch_texts_async
from src.utils.config import RAW_METADATA_DIR, PARAGRAPHS_DIR, setup_logging

setup_logging("02_fetch_texts")
logger = logging.getLogger(__name__)


def load_index() -> pd.DataFrame:
    """Load the consolidated metadata index."""
    index_path = os.path.join(RAW_METADATA_DIR, "index.parquet")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Metadata index not found at {index_path}. "
            "Run 01_fetch_metadata.py first."
        )
    return pd.read_parquet(index_path)


async def fetch_year(
    year: int,
    celex_list: list[str],
    concurrency: int = 10,
    max_items: int | None = None,
    languages: tuple[str, ...] = ("eng",),
):
    """Fetch texts for a single year."""
    os.makedirs(PARAGRAPHS_DIR, exist_ok=True)
    output_path = os.path.join(PARAGRAPHS_DIR, f"{year}.jsonl")

    logger.info(f"Year {year}: {len(celex_list)} judgments to process")
    await fetch_texts_async(
        celex_list=celex_list,
        output_path=output_path,
        concurrency=concurrency,
        max_items=max_items,
        languages=languages,
    )


def main():
    parser = argparse.ArgumentParser(description="Fetch CJEU judgment texts")
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent requests")
    parser.add_argument("--max-items", type=int, default=0, help="Limit per year (0=all)")
    parser.add_argument("--year", type=int, default=None, help="Fetch a single year only")
    parser.add_argument(
        "--lang", type=str, default="eng",
        help="Language codes, comma-separated with fallback priority (default: eng)",
    )
    args = parser.parse_args()

    languages = tuple(args.lang.split(","))
    max_items = args.max_items or None

    index_df = load_index()

    # Extract year from date column
    index_df["year"] = pd.to_datetime(index_df["date"]).dt.year

    if args.year:
        years = [args.year]
    else:
        years = sorted(index_df["year"].unique())

    for year in years:
        year_celex = index_df[index_df["year"] == year]["celex"].unique().tolist()
        if not year_celex:
            logger.info(f"Year {year}: no judgments, skipping")
            continue
        asyncio.run(fetch_year(year, year_celex, args.concurrency, max_items, languages))

    logger.info("Done.")


if __name__ == "__main__":
    main()
