"""Render per-domain heatmap-grid figures from the EuroVoc drift matrices.

For each EuroVoc domain present in the ranking, picks the top-K labels by
``drift_excess`` and writes a heatmap-grid + marginals-grid + drift-excess
histogram into ``data/results/figures/eurovoc_drift/{domain_slug}/``.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.paths import PATHS
from src.utils.config import setup_logging
from src.visualization.temporal_drift_plots import (
    plot_cross_period_grid,
    plot_cross_period_marginals_grid,
    plot_drift_excess_distribution,
)


def load_matrices(path: str) -> dict[str, dict]:
    data = np.load(path, allow_pickle=False)
    out: dict[str, dict] = {}
    for key in data.files:
        if key.startswith("w::"):
            out.setdefault(key[3:], {})["matrix"] = data[key]
        elif key.startswith("y::"):
            out.setdefault(key[3:], {})["years"] = data[key].tolist()
    data.close()
    return {w: v for w, v in out.items() if "matrix" in v and "years" in v}


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", name.lower()).strip("_")
    return s or "unassigned"


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-domain EuroVoc drift heatmaps")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--ncols", type=int, default=5)
    parser.add_argument(
        "--rank-by", type=str, default="drift_excess",
        choices=["drift_excess", "drift_ratio", "max_off_diag"],
    )
    args = parser.parse_args()

    setup_logging("eurovoc_drift_05_plot_per_domain")
    logger = logging.getLogger(__name__)

    ranking_df = pd.read_parquet(PATHS.eurovoc_drift_ranking)
    matrices = load_matrices(str(PATHS.eurovoc_drift_apd_npz))
    logger.info(f"Loaded {len(ranking_df)} ranked labels, {len(matrices)} matrices")

    out_root = PATHS.eurovoc_drift_figures
    out_root.mkdir(parents=True, exist_ok=True)

    domains = sorted(ranking_df["domain_name"].dropna().unique().tolist())
    if ranking_df["domain_name"].isna().any():
        domains.append(None)

    for domain in domains:
        sub = (
            ranking_df[ranking_df["domain_name"].isna()]
            if domain is None
            else ranking_df[ranking_df["domain_name"] == domain]
        )
        if sub.empty:
            continue

        slug = _slugify(domain or "unassigned")
        dom_dir = out_root / slug
        dom_dir.mkdir(parents=True, exist_ok=True)

        top = (
            sub.sort_values(args.rank_by, ascending=False)
            .head(args.top_k)["word"]
            .tolist()
        )
        top = [w for w in top if w in matrices]
        logger.info(f"  {domain or '(unassigned)'}: {len(sub)} labels, plotting top {len(top)}")
        if not top:
            continue

        plot_cross_period_grid(
            matrices, top,
            save_path=str(dom_dir / f"heatmap_grid_top{args.top_k}.png"),
            ncols=args.ncols,
        )
        plot_cross_period_marginals_grid(
            matrices, top,
            save_path=str(dom_dir / f"marginals_grid_top{args.top_k}.png"),
            ncols=args.ncols,
        )
        plot_drift_excess_distribution(
            sub,
            save_path=str(dom_dir / "drift_excess_hist.png"),
        )

    # Also a global top-K for cross-domain context.
    global_top = (
        ranking_df.sort_values(args.rank_by, ascending=False)
        .head(args.top_k)["word"]
        .tolist()
    )
    global_top = [w for w in global_top if w in matrices]
    if global_top:
        plot_cross_period_grid(
            matrices, global_top,
            save_path=str(out_root / f"heatmap_grid_global_top{args.top_k}.png"),
            ncols=args.ncols,
        )
        plot_cross_period_marginals_grid(
            matrices, global_top,
            save_path=str(out_root / f"marginals_grid_global_top{args.top_k}.png"),
            ncols=args.ncols,
        )
        plot_drift_excess_distribution(
            ranking_df,
            save_path=str(out_root / "drift_excess_hist_global.png"),
        )


if __name__ == "__main__":
    main()
