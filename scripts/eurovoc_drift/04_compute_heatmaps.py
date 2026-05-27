"""Compute cross-period APD matrices for EuroVoc labels.

Reads per-year embeddings written by ``03_extract_embeddings.py`` and uses
``temporal_drift.compute_cross_period_table`` to produce one (Y x Y)
mean-pairwise-cosine-distance matrix per label.

Outputs:
  data/models/eurovoc_drift/cross_period_apd.npz
  data/results/metrics/eurovoc_drift_ranking.parquet
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.embeddings.bert_encoder import resolve_model
from src.metrics.temporal_drift import compute_cross_period_table
from src.paths import PATHS
from src.utils.config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute cross-period APD for EuroVoc labels")
    parser.add_argument("--unit", choices=["year", "judgment"], default="year",
                        help="Routes input embeddings and output paths through "
                             "the {unit}/ subdir; logic is identical for both.")
    parser.add_argument("--model", type=str, default="eurlex",
                        help="Encoder (friendly name). 'eurlex' reads/writes the "
                             "legacy paths; a control model uses models/<name>/.")
    parser.add_argument("--start", type=int, default=1990)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--min-usages", type=int, default=10)
    parser.add_argument("--min-years", type=int, default=10)
    parser.add_argument("--max-per-year", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging(f"eurovoc_drift_04_compute_heatmaps_{args.unit}")
    logger = logging.getLogger(__name__)
    _, slug = resolve_model(args.model)
    logger.info(f"model={args.model} slug={slug}")

    years = range(args.start, args.end + 1)

    matrices, ranking_df = compute_cross_period_table(
        str(PATHS.eurovoc_drift_embeddings_for(args.unit, slug)),
        years,
        words=None,
        min_usages=args.min_usages,
        min_years=args.min_years,
        max_per_year=args.max_per_year,
        seed=args.seed,
    )

    if ranking_df.empty:
        logger.error("No matrices computed — check that embeddings exist")
        return

    # Join with selected_labels metadata to attach domain / microthesaurus.
    selected = pd.read_parquet(PATHS.eurovoc_drift_selected_labels_for(args.unit))
    # Labels were lowercased during indexing; left-join is on the lowercased key.
    selected = selected.assign(label=selected["label"].str.lower())
    # A label string may map to multiple concepts (rare alt-label collisions);
    # keep the first by n_present_ge_floor to avoid row blowup in the ranking.
    selected_unique = (
        selected.sort_values("n_present_ge_floor", ascending=False)
        .drop_duplicates(subset=["label"], keep="first")
    )
    ranked = ranking_df.merge(
        selected_unique[
            ["label", "concept_id", "label_type", "domain_name",
             "microthesaurus_name", "pref_label_en"]
        ],
        left_on="word", right_on="label", how="left"
    ).drop(columns=["label"])

    # Year mode keeps the legacy unsuffixed paths so script 05 still works
    # off them; judgment mode and control models get dedicated locations.
    ranking_path = PATHS.eurovoc_drift_ranking_for(args.unit, slug)
    apd_path = PATHS.eurovoc_drift_apd_npz_for(args.unit, slug)
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    apd_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_parquet(ranking_path, index=False)
    logger.info(f"Saved ranking ({len(ranked)} labels) → {ranking_path}")

    arrays: dict[str, np.ndarray] = {}
    for label, info in matrices.items():
        arrays[f"w::{label}"] = info["matrix"].astype(np.float32)
        arrays[f"y::{label}"] = np.asarray(info["years"], dtype=np.int32)
    np.savez_compressed(apd_path, **arrays)
    logger.info(f"Saved {len(matrices)} matrices → {apd_path}")

    logger.info("\nTop 20 by drift_excess:")
    for _, row in ranked.head(20).iterrows():
        logger.info(
            f"  {row['word'][:30]:30s} [{(row['domain_name'] or '?')[:18]:18s}] "
            f"excess={row['drift_excess']:.4f} ratio={row['drift_ratio']:.3f} "
            f"diag={row['mean_diag']:.4f} peak={int(row['peak_year'])}"
        )


if __name__ == "__main__":
    main()
