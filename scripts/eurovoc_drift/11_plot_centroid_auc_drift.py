"""Visualisations of the centroid-AUC drift sweep (script 10's CSV).

Reads ``PATHS.eurovoc_drift_centroid_auc_results / "centroid_auc_drift.csv"``
and writes the following into
``PATHS.eurovoc_drift_centroid_auc_figures``:

  z_heatmap_grid.png            # per-label Y×Y median-z heatmaps (upper triangle)
  consecutive_z_lines.html      # plotly: one thin line per label, hover for name
  z_histogram.png               # pooled (label, pair, fold) z + N(0,1)
  drift_ranking_median.png      # per-label bar chart of median z (sorted)
  drift_ranking_mean_abs.png    # per-label bar chart of mean |z|
  z_vs_year_gap.png             # violins of z by (yb − ya)
  z_vs_w_norm.png               # scatter z vs ||μ_b − μ_a||

Existing figures in other directories are not overwritten.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.paths import PATHS
from src.utils.config import setup_logging
from src.visualization.linear_probe_plots import (
    plot_centroid_z_heatmap_grid,
    plot_consecutive_z_lines_sleeping_beauty,
    plot_consecutive_z_lines_top_bottom,
    plot_consecutive_z_plotly,
    plot_drift_rank_histograms_by_group,
    plot_drift_ranking_top_bottom,
    plot_global_z_histogram,
    plot_per_label_drift_ranking,
    plot_score_median_vs_span,
    plot_z_vs_w_norm,
    plot_z_vs_year_gap,
)


def load_eurovoc_domain_map(parquet_path: str) -> dict[str, str]:
    """Return label → EuroVoc top-level domain (all 21 domains)."""
    sel = pd.read_parquet(parquet_path)
    out: dict[str, str] = {}
    for _, row in sel.iterrows():
        lab = str(row["label"]).strip()
        dom = row.get("domain_name")
        if lab and isinstance(dom, str) and dom:
            out[lab] = dom
    return out


def build_z_matrices(df: pd.DataFrame) -> dict[str, dict]:
    """Build per-label upper-triangle Y×Y median-z matrices.

    For each label, the index is ``sorted(unique years observed in either
    year_a or year_b)``. Cell (i, j) with ``i < j`` holds the median z over
    folds for (years[i], years[j]); diagonal and lower triangle are NaN.
    Pairs that aren't in the CSV (e.g. dropped because a year had < k_folds
    embeddings) are left NaN as well.
    """
    out: dict[str, dict] = {}
    agg = (
        df.groupby(["label", "year_a", "year_b"])["z"].median().reset_index()
    )
    for label, sub in agg.groupby("label"):
        years = sorted(set(sub["year_a"].tolist()) | set(sub["year_b"].tolist()))
        idx = {y: i for i, y in enumerate(years)}
        M = np.full((len(years), len(years)), np.nan, dtype=np.float32)
        for _, r in sub.iterrows():
            i = idx[int(r["year_a"])]
            j = idx[int(r["year_b"])]
            M[i, j] = float(r["z"])
        out[str(label)] = {"years": years, "matrix": M}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=str, default=None,
                        help="Override CSV path (default: script-10 output).")
    parser.add_argument("--heatmap-ncols", type=int, default=10,
                        help="Columns per heatmap page (rows are filled to "
                             "the same count, so each page holds ncols² panels).")
    parser.add_argument("--ranking-top-k", type=int, default=None,
                        help="If set, show only the top-K labels in the "
                             "ranking bar chart.")
    args = parser.parse_args()

    setup_logging("eurovoc_drift_11_plot_centroid_auc_drift")
    logger = logging.getLogger(__name__)

    csv_path = (
        args.csv if args.csv is not None
        else str(PATHS.eurovoc_drift_centroid_auc_results / "centroid_auc_drift.csv")
    )
    logger.info(f"Loading {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(
        f"Rows: {len(df):,}; labels: {df['label'].nunique()}; "
        f"pairs: {df.groupby(['label','year_a','year_b']).ngroups:,}"
    )

    out_dir = PATHS.eurovoc_drift_centroid_auc_figures
    out_dir.mkdir(parents=True, exist_ok=True)

    matrices = build_z_matrices(df)
    label_order = sorted(matrices.keys())
    logger.info(f"Built z-matrices for {len(matrices)} labels")

    ncols = args.heatmap_ncols
    per_page = ncols * ncols
    n_pages = (len(label_order) + per_page - 1) // per_page
    width = max(2, len(str(n_pages)))
    for page in range(n_pages):
        chunk = label_order[page * per_page:(page + 1) * per_page]
        p_heat = out_dir / f"z_heatmap_grid_{page + 1:0{width}d}.png"
        plot_centroid_z_heatmap_grid(
            matrices, chunk, str(p_heat), ncols=ncols,
        )
        logger.info(f"Wrote {p_heat}  ({len(chunk)} labels)")

    p_plotly = out_dir / "consecutive_z_lines.html"
    plot_consecutive_z_plotly(df, str(p_plotly))
    logger.info(f"Wrote {p_plotly}")

    p_hist = out_dir / "z_histogram.png"
    plot_global_z_histogram(df, str(p_hist))
    logger.info(f"Wrote {p_hist}")

    p_rank_med = out_dir / "drift_ranking_median.png"
    plot_per_label_drift_ranking(
        df, str(p_rank_med), top_k=args.ranking_top_k, agg="median",
    )
    logger.info(f"Wrote {p_rank_med}")

    p_rank_abs = out_dir / "drift_ranking_mean_abs.png"
    plot_per_label_drift_ranking(
        df, str(p_rank_abs), top_k=args.ranking_top_k, agg="mean_abs",
    )
    logger.info(f"Wrote {p_rank_abs}")

    p_gap = out_dir / "z_vs_year_gap.png"
    plot_z_vs_year_gap(df, str(p_gap))
    logger.info(f"Wrote {p_gap}")

    p_wnorm = out_dir / "z_vs_w_norm.png"
    plot_z_vs_w_norm(df, str(p_wnorm))
    logger.info(f"Wrote {p_wnorm}")

    p_span_z = out_dir / "fold_dispersion_z.png"
    plot_score_median_vs_span(df, str(p_span_z), metric="z", metric_label="z")
    logger.info(f"Wrote {p_span_z}")

    p_span_auc = out_dir / "fold_dispersion_auc.png"
    plot_score_median_vs_span(df, str(p_span_auc), metric="auc",
                              metric_label="AUC")
    logger.info(f"Wrote {p_span_auc}")

    p_rank_tb = out_dir / "drift_ranking_top_bottom50.png"
    plot_drift_ranking_top_bottom(df, str(p_rank_tb), n_each=50, agg="median")
    logger.info(f"Wrote {p_rank_tb}")

    sel_path = str(PATHS.eurovoc_drift_selected_labels)
    if os.path.exists(sel_path):
        domain_map = load_eurovoc_domain_map(sel_path)
        n_total = df["label"].nunique()
        n_mapped = sum(1 for lab in df["label"].unique()
                       if str(lab) in domain_map)
        logger.info(
            f"Domain map: {n_mapped}/{n_total} labels mapped to EuroVoc domains"
        )
        p_hist_dom = out_dir / "drift_rank_hist_by_domain.png"
        plot_drift_rank_histograms_by_group(
            df, domain_map, str(p_hist_dom), agg="median",
        )
        logger.info(f"Wrote {p_hist_dom}")
    else:
        logger.warning(f"selected_labels parquet not found: {sel_path}")

    p_top_bot_lines = out_dir / "consecutive_z_lines_top_bottom50.png"
    plot_consecutive_z_lines_top_bottom(df, str(p_top_bot_lines), n_each=50)
    logger.info(f"Wrote {p_top_bot_lines}")

    p_sb_lines = out_dir / "consecutive_z_lines_sleeping_beauty50.png"
    plot_consecutive_z_lines_sleeping_beauty(
        df, str(p_sb_lines), n_each=50,
    )
    logger.info(f"Wrote {p_sb_lines}")

    logger.info(f"All figures written to {out_dir}")


if __name__ == "__main__":
    main()
