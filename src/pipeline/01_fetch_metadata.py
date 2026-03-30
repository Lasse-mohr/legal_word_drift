"""Fetch CJEU judgment metadata year by year via SPARQL.

Saves per-year parquet files to data/raw/metadata/ and a consolidated
index to data/raw/metadata/index.parquet.

Also fetches subject-matter codes for domain-level analysis.

Usage:
    python -m src.pipeline.01_fetch_metadata [--start 1990] [--end 2025]
"""
from __future__ import annotations

import argparse
import logging
import os
import pandas as pd

from src.cjeu_cellar_client import CjeuCellarClient, DOC_TYPE_JUDGMENTS
from src.utils.config import RAW_METADATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def fetch_metadata(start_year: int = 1990, end_year: int = 2025) -> pd.DataFrame:
    """Fetch judgment metadata for each year and save to parquet."""
    os.makedirs(RAW_METADATA_DIR, exist_ok=True)
    client = CjeuCellarClient()

    all_frames: list[pd.DataFrame] = []

    for year in range(start_year, end_year + 1):
        out_path = os.path.join(RAW_METADATA_DIR, f"{year}.parquet")

        if os.path.exists(out_path):
            logger.info(f"{year}: already fetched, loading from {out_path}")
            df = pd.read_parquet(out_path)
        else:
            logger.info(f"{year}: fetching decisions...")
            df = client.fetch_decisions(
                date_from=f"{year}-01-01",
                date_to=f"{year}-12-31",
                doc_types=DOC_TYPE_JUDGMENTS,
            )
            if df.empty:
                logger.warning(f"{year}: no decisions found")
            else:
                df.to_parquet(out_path, index=False)
                logger.info(f"{year}: {len(df)} decisions saved to {out_path}")

        all_frames.append(df)

    # Consolidated index
    index_df = pd.concat(all_frames, ignore_index=True)
    index_path = os.path.join(RAW_METADATA_DIR, "index.parquet")
    index_df.to_parquet(index_path, index=False)
    logger.info(
        f"Consolidated index: {len(index_df)} decisions, "
        f"{index_df['celex'].nunique()} unique CELEX, "
        f"saved to {index_path}"
    )

    return index_df


def fetch_subjects(index_df: pd.DataFrame) -> pd.DataFrame:
    """Fetch subject-matter codes for all judgments in the index."""
    out_path = os.path.join(RAW_METADATA_DIR, "subjects.parquet")

    if os.path.exists(out_path):
        logger.info(f"Subject matter: already fetched, loading from {out_path}")
        return pd.read_parquet(out_path)

    client = CjeuCellarClient()
    celex_list = index_df["celex"].unique().tolist()

    logger.info(f"Fetching subject matter for {len(celex_list)} judgments...")
    subjects_df = client.fetch_subject_matter(celex_list=celex_list)

    subjects_df.to_parquet(out_path, index=False)
    logger.info(f"Subject matter: {len(subjects_df)} rows saved to {out_path}")
    return subjects_df


def main():
    parser = argparse.ArgumentParser(description="Fetch CJEU judgment metadata")
    parser.add_argument("--start", type=int, default=1990, help="Start year (default: 1990)")
    parser.add_argument("--end", type=int, default=2025, help="End year (default: 2025)")
    parser.add_argument("--skip-subjects", action="store_true", help="Skip subject matter fetch")
    args = parser.parse_args()

    index_df = fetch_metadata(start_year=args.start, end_year=args.end)

    if not args.skip_subjects and not index_df.empty:
        fetch_subjects(index_df)

    logger.info("Done.")


if __name__ == "__main__":
    main()
