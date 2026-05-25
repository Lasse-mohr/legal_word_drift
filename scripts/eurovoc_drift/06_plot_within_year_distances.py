"""Plot pooled-across-years pairwise + centroid distance distributions for
the EuroVoc codes flagged in `.research-notes.md` (2026-05-12 entry).

For each highlighted label, concatenates per-year embeddings from
``PATHS.eurovoc_drift_embeddings_year(year)`` over the requested year range
and computes the within-corpus distance distributions. Writes paginated 5×5
grids grouped by EuroVoc domain via
``src.visualization.polysemy_plots.plot_distance_grids``.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.metrics.polysemy import within_group_distance_stats
from src.paths import PATHS
from src.utils.config import setup_logging
from src.visualization.polysemy_plots import plot_distance_grids

# Highlighted EuroVoc labels from the 2026-05-12 research note, grouped by
# domain. Labels match the lowercase keys (without the "w::" prefix) used in
# eurovoc_drift embedding files.
HIGHLIGHTED: dict[str, list[str]] = {
    "business & competition": ["leadership", "organization", "foundation", "branch", "composition"],
    "production tech & research": ["wire", "tube"],
    "social questions": ["orphan"],
    "industry": ["carbon", "engine"],
    "agriculture forestry & fisheries": [
        "cow", "harvest", "orange", "butter", "beef",
        "tobacco", "wheat", "distillation", "cereals",
    ],
    "politics": ["opposition", "citizen"],
    "law": ["succession", "accused", "inheritance", "notary", "punishment"],
    "international relations": ["resolution", "reunification"],
    "geography": ["scotland", "ukraine", "africa"],
    "finance": ["borrowing"],
    "environment": ["plain", "ice", "atlantic", "bird"],
    "employment": ["stress"],
    "education & communications": [
        "web", "forum", "mail", "media", "communications", "translations",
    ],
    "economics": ["agriculture"],
    "other": ["iata", "innovation"],
}


def collect_pooled_embeddings(
    labels: set[str],
    start: int,
    end: int,
) -> dict[str, np.ndarray]:
    """Concatenate per-year embeddings for each label across [start, end]."""
    logger = logging.getLogger(__name__)
    chunks: dict[str, list[np.ndarray]] = {label: [] for label in labels}
    for year in range(start, end + 1):
        path = PATHS.eurovoc_drift_embeddings_year(year)
        if not path.exists():
            logger.warning(f"Missing embeddings for {year}: {path}")
            continue
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                label = key[3:] if key.startswith("w::") else key
                if label in labels:
                    chunks[label].append(data[key].astype(np.float32))
    return {
        label: np.concatenate(arrs, axis=0)
        for label, arrs in chunks.items()
        if arrs
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pooled-across-years distance grids for highlighted EuroVoc labels"
    )
    parser.add_argument("--start", type=int, default=1990)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir", type=str,
        default=str(PATHS.eurovoc_drift_figures / "within_year_distances"),
    )
    parser.add_argument(
        "--metrics-path", type=str, default=None,
        help="Optional CSV path for per-label summary stats. "
             "Defaults to <out-dir>/pooled_<start>-<end>.csv.",
    )
    args = parser.parse_args()

    setup_logging("eurovoc_drift_06_plot_within_year")
    logger = logging.getLogger(__name__)

    wanted = {label for labels in HIGHLIGHTED.values() for label in labels}
    pooled = collect_pooled_embeddings(wanted, args.start, args.end)
    logger.info(
        f"Pooled embeddings for {len(pooled)}/{len(wanted)} labels "
        f"across {args.start}-{args.end}"
    )

    pairwise: dict[str, np.ndarray] = {}
    centroid: dict[str, np.ndarray] = {}
    stats_map: dict[str, dict] = {}
    rows: list[dict] = []
    skipped: list[tuple[str, str]] = []

    for group, labels in HIGHLIGHTED.items():
        for label in labels:
            embs = pooled.get(label)
            # Need ≥2 to have any pairwise distance at all.
            if embs is None or embs.shape[0] < 2:
                skipped.append((group, f"{label} (n={0 if embs is None else embs.shape[0]})"))
                continue
            d, c, s = within_group_distance_stats(embs, seed=args.seed)
            pairwise[label] = d
            centroid[label] = c
            stats_map[label] = s
            rows.append({"label": label, "group": group, **s})

    if skipped:
        logger.warning(f"Skipped {len(skipped)} labels with n<2:")
        for g, item in skipped:
            logger.warning(f"  [{g}] {item}")

    if not pairwise:
        raise RuntimeError("No labels with ≥2 pooled embeddings")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"pooled_{args.start}-{args.end}"
    metrics_path = (
        args.metrics_path or os.path.join(args.out_dir, f"{tag}.csv")
    )
    pd.DataFrame(rows).sort_values(
        ["group", "mean_dist"], ascending=[True, False]
    ).to_csv(metrics_path, index=False)
    logger.info(f"Wrote summary to {metrics_path}")

    for p in plot_distance_grids(
        pairwise, stats_map, HIGHLIGHTED, args.out_dir, tag,
        title_prefix=f"EuroVoc pooled pairwise cosine distances ({args.start}-{args.end})",
        kind="pairwise", pval_key="dip_pvalue", xlabel="pairwise cos dist",
        palette="blues", filename_prefix="eurovoc_within_year",
    ):
        logger.info(f"Wrote figure {p}")
    for p in plot_distance_grids(
        centroid, stats_map, HIGHLIGHTED, args.out_dir, tag,
        title_prefix=f"EuroVoc pooled cos distance to centroid ({args.start}-{args.end})",
        kind="centroid", pval_key="centroid_dip_pvalue",
        xlabel="cos dist to centroid", palette="greens",
        filename_prefix="eurovoc_within_year",
    ):
        logger.info(f"Wrote figure {p}")


if __name__ == "__main__":
    main()
