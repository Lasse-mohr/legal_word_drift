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

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.metrics.polysemy import pairwise_cosine_distances
from src.metrics.temporal_drift import load_per_year_embeddings
from src.utils.config import (
    BERT_EMBEDDINGS_DIR,
    FIGURES_DIR,
    PROJECT_ROOT,
    setup_logging,
)
from src.visualization.plot_config import (
    SEQUENTIAL_PALETTES,
    apply_plot_style,
    remove_extra_spines,
)


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


def compute_distances(
    embeddings_dir: str,
    words: list[str],
    years: list[int],
    min_usages: int,
    max_usages: int | None,
    seed: int,
) -> dict[str, dict[int, np.ndarray]]:
    """Returns {word: {year: 1-D pairwise-distance array}}."""
    logger = logging.getLogger(__name__)
    rng = np.random.default_rng(seed)

    per_word = load_per_year_embeddings(
        embeddings_dir, years, words=set(words), min_usages=min_usages
    )
    out: dict[str, dict[int, np.ndarray]] = {}
    for word in words:
        per_year = per_word.get(word, {})
        out[word] = {}
        for year in years:
            embs = per_year.get(year)
            if embs is None or embs.shape[0] < min_usages:
                continue
            if max_usages is not None and embs.shape[0] > max_usages:
                idx = rng.choice(embs.shape[0], size=max_usages, replace=False)
                embs = embs[idx]
            dists = pairwise_cosine_distances(embs)
            if dists.size:
                out[word][year] = (dists, int(embs.shape[0]))
        logger.info(
            f"{word}: {len(out[word])} year(s) with >= {min_usages} usages"
        )
    return out


def plot_grid(
    distances: dict[str, dict[int, tuple[np.ndarray, int]]],
    words: list[str],
    years: list[int],
    out_dir: str,
    words_per_page: int,
    nbins: int,
) -> list[str]:
    apply_plot_style()
    pal = SEQUENTIAL_PALETTES["blues"]
    hist_color, kde_color = pal[2], pal[5]

    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []

    # Per-word shared xlim across years for easier vertical comparison.
    xlims: dict[str, tuple[float, float]] = {}
    for word in words:
        all_d = [d for (d, _n) in distances.get(word, {}).values()]
        if all_d:
            flat = np.concatenate(all_d)
            lo = float(np.quantile(flat, 0.001))
            hi = float(np.quantile(flat, 0.999))
            # Pad a touch.
            span = hi - lo if hi > lo else 1.0
            xlims[word] = (lo - 0.02 * span, hi + 0.02 * span)

    n_pages = (len(words) + words_per_page - 1) // words_per_page
    n_rows = len(years)

    for page in range(n_pages):
        chunk = words[page * words_per_page : (page + 1) * words_per_page]
        n_cols = len(chunk)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(1.8 * n_cols, 0.65 * n_rows),
            squeeze=False,
            sharex="col",
        )

        for ci, word in enumerate(chunk):
            word_d = distances.get(word, {})
            # Column header
            axes[0, ci].set_title(word, fontsize=10, pad=6)
            for ri, year in enumerate(years):
                ax = axes[ri, ci]
                entry = word_d.get(year)
                # Year labels on the left column only.
                if ci == 0:
                    ax.set_ylabel(
                        str(year), rotation=0, fontsize=8,
                        ha="right", va="center", labelpad=12,
                    )
                # Strip y-ticks; keep only x-ticks on the bottom row.
                ax.set_yticks([])
                if ri != n_rows - 1:
                    ax.tick_params(axis="x", labelbottom=False)
                else:
                    ax.tick_params(axis="x", labelsize=7)

                if entry is None:
                    ax.set_facecolor("#f7f7f7")
                    ax.set_xticks([])
                    remove_extra_spines(ax)
                    continue

                dists, n = entry
                ax.hist(
                    dists, bins=nbins, density=True,
                    color=hist_color, edgecolor="white", linewidth=0.2,
                    alpha=0.85,
                )
                try:
                    from scipy.stats import gaussian_kde
                    xs = np.linspace(dists.min(), dists.max(), 200)
                    ax.plot(
                        xs, gaussian_kde(dists)(xs),
                        color=kde_color, linewidth=1.0,
                    )
                except Exception:
                    pass
                ax.text(
                    0.97, 0.88, f"n={n}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=6, color="#555",
                )
                if word in xlims:
                    ax.set_xlim(*xlims[word])
                remove_extra_spines(ax)

            axes[-1, ci].set_xlabel("cos dist", fontsize=8)

        fig.suptitle(
            f"Per-year pairwise cosine distance distributions "
            f"(page {page + 1}/{n_pages})",
            fontsize=12, y=1.00,
        )
        fig.tight_layout()
        out_path = os.path.join(
            out_dir, f"pairwise_by_year_p{page + 1:02d}.png"
        )
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)

    return written


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

    distances = compute_distances(
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

    written = plot_grid(
        distances, kept, years, args.out_dir,
        words_per_page=args.words_per_page, nbins=args.nbins,
    )
    for p in written:
        logger.info(f"Wrote {p}")


if __name__ == "__main__":
    main()
