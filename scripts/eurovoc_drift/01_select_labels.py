"""Select EuroVoc labels for the cross-period drift sweep.

Two granularities are supported via ``--unit``:

  - ``year`` (default): reads ``label_year_counts.parquet`` and keeps
    labels with ``--count-floor`` hits in ``--min-present`` distinct years.
    Mirrors the original behaviour (defaults 10 / 5).

  - ``judgment``: reads ``label_celex_counts.parquet`` and keeps labels
    with ``--count-floor`` hits in ``--min-present`` distinct judgments
    (CJEU celex IDs).

Either way, domain-root concepts (microthesaurus_id null) are dropped.

Output: ``data/processed/eurovoc_drift/selected_labels_{unit}.parquet``
with columns
``label, concept_id, label_type, domain_name, microthesaurus_name,
pref_label_en, n_present_ge_floor, total_count, unit, count_floor,
min_present``.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.paths import PATHS
from src.utils.config import setup_logging

DEFAULTS = {
    "year": {"count_floor": 10, "min_present": 5},
    "judgment": {"count_floor": 2, "min_present": 30},
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select EuroVoc labels for the cross-period drift sweep."
    )
    parser.add_argument(
        "--unit", choices=["year", "judgment"], default="year",
        help="Granularity of the count and presence filter.",
    )
    parser.add_argument(
        "--count-floor", type=int, default=None,
        help="Min per-unit count to count that unit as 'present'.",
    )
    parser.add_argument(
        "--min-present", type=int, default=None,
        help="Min number of units in which the label is present.",
    )
    args = parser.parse_args()

    unit = args.unit
    count_floor = args.count_floor if args.count_floor is not None else DEFAULTS[unit]["count_floor"]
    min_present = args.min_present if args.min_present is not None else DEFAULTS[unit]["min_present"]

    setup_logging(f"eurovoc_drift_01_select_labels_{unit}")
    logger = logging.getLogger(__name__)
    logger.info(
        f"unit={unit}  count_floor={count_floor}  min_present={min_present}"
    )

    if unit == "year":
        counts_path = PATHS.eurovoc_coverage_label_year_counts
        unit_col = "year"
    else:
        counts_path = PATHS.eurovoc_coverage_label_celex_counts
        unit_col = "celex"

    concepts_path = PATHS.eurovoc / "concepts_enriched.csv"

    counts = pd.read_parquet(counts_path)
    concepts = pd.read_csv(concepts_path, dtype={"concept_id": str})
    logger.info(
        f"Loaded {len(counts):,} label-{unit_col} rows from {counts_path.name}, "
        f"{len(concepts):,} concepts"
    )

    # For judgment unit: each row is (concept_id, label, label_type, celex, year, count)
    # — count is already per-celex, so "presence" = count >= floor.
    # For year unit: each row is (concept_id, label, label_type, year, count) — same logic.
    eligible = counts[counts["count"] >= count_floor]
    per_label = (
        eligible.groupby(["concept_id", "label", "label_type"])
        .agg(
            n_present_ge_floor=(unit_col, "nunique"),
            total_count=("count", "sum"),
        )
        .reset_index()
    )

    n_pre_filter = len(per_label)
    per_label = per_label[per_label["n_present_ge_floor"] >= min_present]
    logger.info(
        f"{unit_col}-floor filter (≥{count_floor} hits/{unit_col}, "
        f"≥{min_present} such {unit_col}s): {n_pre_filter:,} → {len(per_label):,} labels"
    )

    # Drop domain roots: concepts without a microthesaurus_id
    concept_meta = concepts[
        ["concept_id", "domain_name", "microthesaurus_id", "microthesaurus_name", "pref_label_en"]
    ]
    merged = per_label.merge(concept_meta, on="concept_id", how="left")

    n_pre_root = len(merged)
    merged = merged[merged["microthesaurus_id"].notna()]
    logger.info(
        f"Drop domain-root concepts (microthesaurus_id null): "
        f"{n_pre_root:,} → {len(merged):,} labels"
    )

    merged = merged.assign(
        unit=unit,
        count_floor=count_floor,
        min_present=min_present,
    )

    out = (
        merged[
            [
                "label",
                "concept_id",
                "label_type",
                "domain_name",
                "microthesaurus_name",
                "pref_label_en",
                "n_present_ge_floor",
                "total_count",
                "unit",
                "count_floor",
                "min_present",
            ]
        ]
        .sort_values(
            ["domain_name", "n_present_ge_floor", "total_count"],
            ascending=[True, False, False],
        )
        .reset_index(drop=True)
    )

    out_path = PATHS.eurovoc_drift_selected_labels_for(unit)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    logger.info(f"Wrote {len(out):,} labels → {out_path}")

    # Summaries
    by_dom = (
        out.groupby("domain_name", dropna=False)
        .agg(
            n_labels=("label", "count"),
            n_concepts=("concept_id", "nunique"),
            median_present=("n_present_ge_floor", "median"),
            median_total=("total_count", "median"),
        )
        .reset_index()
        .sort_values("n_labels", ascending=False)
    )
    logger.info("Per-domain selected labels:\n%s", by_dom.to_string(index=False))

    by_type = out["label_type"].value_counts()
    logger.info("Label type breakdown:\n%s", by_type.to_string())

    n_tokens = out["label"].str.split().str.len()
    logger.info(
        "Label token-length distribution: 1=%d, 2=%d, 3=%d, ≥4=%d (max=%d)",
        int((n_tokens == 1).sum()),
        int((n_tokens == 2).sum()),
        int((n_tokens == 3).sum()),
        int((n_tokens >= 4).sum()),
        int(n_tokens.max()),
    )


if __name__ == "__main__":
    main()
