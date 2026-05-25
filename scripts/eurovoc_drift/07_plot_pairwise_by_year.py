"""Per-year pairwise cosine distance distributions for highlighted EuroVoc labels.

For every label listed in ``configs/eurovoc_block_structure.yaml`` and every
year in [start, end] with at least ``--min-usages`` BERT embeddings under
``PATHS.eurovoc_drift_embeddings_year(year)``, compute the upper-triangle
pairwise cosine distance distribution and render a histogram + gaussian-KDE
line in a grid where rows are years and columns are labels.

Labels are ordered by EuroVoc domain (alphabetical), then by label name
within domain, so adjacent grid columns cluster by domain. Pages split
across labels via ``--words-per-page``.

Outputs:
  data/results/figures/eurovoc_drift/pairwise_by_year/eurovoc_pairwise_by_year_p{NN}.png
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.metrics.polysemy import compute_per_year_pairwise_distances
from src.paths import PATHS
from src.utils.config import PROJECT_ROOT, setup_logging
from src.visualization.polysemy_plots import plot_pairwise_by_year_grid


def load_highlighted_labels(config_path: str) -> list[str]:
    """Read every (label, domain) entry from the clarity tiers and return
    a list of lowercase labels ordered by (domain, label).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    entries: list[tuple[str, str]] = []  # (domain, label_lower)
    seen: set[str] = set()
    for tier, items in cfg.items():
        if tier == "subsets" or not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict) or "label" not in item:
                continue
            label = str(item["label"]).lower()
            domain = str(item.get("domain", "(unspecified)"))
            if label in seen:
                continue
            seen.add(label)
            entries.append((domain, label))

    entries.sort(key=lambda dl: (dl[0], dl[1]))
    return [label for _domain, label in entries]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(PROJECT_ROOT, "configs", "eurovoc_block_structure.yaml"),
    )
    parser.add_argument("--start", type=int, default=1990)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument(
        "--min-usages", type=int, default=50,
        help="Drop (label, year) cells with fewer usages. EuroVoc per-year "
             "counts are often low; consider lowering to ~20 if many cells "
             "are blank.",
    )
    parser.add_argument(
        "--max-usages", type=int, default=1000,
        help="Subsample to this many per (label, year); set 0 to disable.",
    )
    parser.add_argument("--words-per-page", type=int, default=8)
    parser.add_argument("--nbins", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir", type=str,
        default=str(PATHS.eurovoc_drift_figures / "pairwise_by_year"),
    )
    args = parser.parse_args()

    setup_logging("eurovoc_drift_07_plot_pairwise_by_year")
    logger = logging.getLogger(__name__)

    labels = load_highlighted_labels(args.config)
    logger.info(f"Loaded {len(labels)} labels from {args.config}")

    years = list(range(args.start, args.end + 1))
    max_usages = args.max_usages if args.max_usages > 0 else None

    distances = compute_per_year_pairwise_distances(
        str(PATHS.eurovoc_drift_embeddings), labels, years,
        min_usages=args.min_usages,
        max_usages=max_usages,
        seed=args.seed,
    )

    kept = [w for w in labels if distances.get(w)]
    dropped = [w for w in labels if not distances.get(w)]
    if dropped:
        logger.warning(f"No data for: {dropped}")
    if not kept:
        logger.error("No label had any year with enough usages; aborting.")
        return

    written = plot_pairwise_by_year_grid(
        distances, kept, years, args.out_dir,
        words_per_page=args.words_per_page,
        nbins=args.nbins,
        filename_prefix="eurovoc_pairwise_by_year",
        title="EuroVoc per-year pairwise cosine distance distributions",
    )
    for p in written:
        logger.info(f"Wrote {p}")


if __name__ == "__main__":
    main()
