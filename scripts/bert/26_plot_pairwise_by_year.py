"""Per-year pairwise cosine distance distributions for pilot words.

For every word listed in ``configs/polysemy_pilot.yaml`` and every year in
[start, end] with at least ``--min-usages`` BERT embeddings, compute the
upper-triangle pairwise cosine distance distribution and render a histogram
+ gaussian-KDE line in a grid where rows are years and columns are words.

If there are more words than ``--words-per-page``, the grid is split across
multiple figures. Each subplot is kept short so all years fit in one figure.

Outputs:
  data/results/figures/polysemy_pilot/pairwise_by_year_p{NN}.png
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.metrics.polysemy import compute_per_year_pairwise_distances
from src.utils.config import (
    BERT_EMBEDDINGS_DIR,
    FIGURES_DIR,
    PROJECT_ROOT,
    setup_logging,
)
from src.visualization.polysemy_plots import plot_pairwise_by_year_grid


CATEGORIES = ("polysemous", "monosemantic", "extras", "frequency_sampled")


def load_word_list(config_path: str) -> list[str]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    words: list[str] = []
    seen: set[str] = set()
    for cat in CATEGORIES:
        for w in cfg.get(cat) or []:
            wl = w.lower()
            if wl not in seen:
                seen.add(wl)
                words.append(wl)
    return words


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str,
                        default=os.path.join(PROJECT_ROOT, "configs",
                                             "polysemy_pilot.yaml"))
    parser.add_argument("--start", type=int, default=1990)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--min-usages", type=int, default=50)
    parser.add_argument("--max-usages", type=int, default=1000,
                        help="Subsample to this many per (word, year); "
                             "set 0 to disable.")
    parser.add_argument("--words-per-page", type=int, default=8)
    parser.add_argument("--nbins", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str,
                        default=os.path.join(FIGURES_DIR, "polysemy_pilot"))
    args = parser.parse_args()

    setup_logging("26_plot_pairwise_by_year")
    logger = logging.getLogger(__name__)

    words = load_word_list(args.config)
    logger.info(f"Loaded {len(words)} words from {args.config}")

    years = list(range(args.start, args.end + 1))
    max_usages = args.max_usages if args.max_usages > 0 else None

    distances = compute_per_year_pairwise_distances(
        BERT_EMBEDDINGS_DIR, words, years,
        min_usages=args.min_usages,
        max_usages=max_usages,
        seed=args.seed,
    )

    kept = [w for w in words if distances.get(w)]
    dropped = [w for w in words if not distances.get(w)]
    if dropped:
        logger.warning(f"No data for: {dropped}")
    if not kept:
        logger.error("No word had any year with enough usages; aborting.")
        return

    written = plot_pairwise_by_year_grid(
        distances, kept, years, args.out_dir,
        words_per_page=args.words_per_page, nbins=args.nbins,
    )
    for p in written:
        logger.info(f"Wrote {p}")


if __name__ == "__main__":
    main()
