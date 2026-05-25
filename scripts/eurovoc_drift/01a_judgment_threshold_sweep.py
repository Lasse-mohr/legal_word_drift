"""Exploratory sweep for picking judgment-unit thresholds.

For each EuroVoc domain, sweep (count_floor x min_judgments) and report:
  - median judgments-per-label among eligible labels
  - % of labels retained vs the no-filter baseline

Baseline = labels in the domain with any corpus hit at all (any celex with
count >= 1). Domain-root concepts (microthesaurus_id null) are dropped to
match the policy in 01_select_labels.py.

Inputs:
    data/processed/eurovoc_coverage/label_celex_counts.parquet
    data/processed/eurovoc/concepts_enriched.csv

Outputs (data/results/figures/eurovoc_drift/judgment_threshold_sweep/):
    median_judgments_per_label.pdf
    pct_labels_retained.pdf
    grid.csv     long-form (domain, floor, min_judgments, n_eligible,
                 baseline_n, median_judgments, pct_retained)
"""
from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.paths import PATHS
from src.utils.config import setup_logging
from src.visualization.plot_config import apply_plot_style, get_heatmap_cmap

FLOORS = [1, 2, 3, 5, 10]
MIN_JUDGMENTS = [5, 10, 20, 30, 50, 100]

OUT_DIR = PATHS.eurovoc_drift_figures / "judgment_threshold_sweep"


def compute_grid(label_celex: pd.DataFrame) -> pd.DataFrame:
    """Compute (domain, floor, min_judgments) -> (n_eligible, baseline, median_j, pct).

    `label_celex` must already be filtered to non-root concepts and joined
    with `domain_name`. One row per (concept_id, label, celex, year).
    A label is identified by (concept_id, label) — pref + alt count
    independently, mirroring 01_select_labels.py.
    """
    # Baseline per domain: count of distinct (concept_id, label) with any hit.
    baseline = (
        label_celex.groupby("domain_name")[["concept_id", "label"]]
        .apply(lambda df: df.drop_duplicates().shape[0])
        .rename("baseline_n")
    )

    records: list[dict] = []
    # For each floor: filter to (concept_id, label, celex) rows where count >= floor,
    # then per (domain, concept_id, label) count distinct celexes — that's the
    # "judgments_present_ge_floor" statistic. Apply min_judgments filter on it.
    for floor in FLOORS:
        keep = label_celex[label_celex["count"] >= floor]
        # judgments per (domain, label) under this floor
        j_per_label = (
            keep.groupby(["domain_name", "concept_id", "label"])["celex"]
            .nunique()
            .rename("n_judgments")
            .reset_index()
        )
        for min_j in MIN_JUDGMENTS:
            eligible = j_per_label[j_per_label["n_judgments"] >= min_j]
            # per-domain stats
            agg = (
                eligible.groupby("domain_name")["n_judgments"]
                .agg(["count", "median"])
                .rename(columns={"count": "n_eligible", "median": "median_judgments"})
            )
            agg = agg.reindex(baseline.index, fill_value=0)
            for dom in baseline.index:
                n_elig = int(agg.loc[dom, "n_eligible"])
                med = agg.loc[dom, "median_judgments"]
                base = int(baseline.loc[dom])
                records.append(
                    {
                        "domain_name": dom,
                        "floor": floor,
                        "min_judgments": min_j,
                        "n_eligible": n_elig,
                        "baseline_n": base,
                        "median_judgments": float(med) if n_elig > 0 else float("nan"),
                        "pct_retained": (100.0 * n_elig / base) if base > 0 else float("nan"),
                    }
                )
    return pd.DataFrame.from_records(records)


def _grid_layout(n: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def plot_heatmaps(
    grid: pd.DataFrame,
    value_col: str,
    title: str,
    cbar_label: str,
    out_path: Path,
    fmt: str = "{:.1f}",
    use_log: bool = True,
) -> None:
    """Small-multiples heatmap: one panel per domain.

    Floor on x-axis, min_judgments on y-axis. Value annotated per cell.
    Blue cmap with LogNorm (clipped to >0 for log safety).
    """
    apply_plot_style()
    cmap = get_heatmap_cmap()

    domains = sorted(grid["domain_name"].dropna().unique().tolist())
    n_dom = len(domains)
    rows, cols = _grid_layout(n_dom)

    # Global vmin/vmax across non-null values for shared color scale
    vals_all = grid[value_col].dropna()
    vals_pos = vals_all[vals_all > 0]
    if use_log and len(vals_pos):
        vmin = max(float(vals_pos.min()), 1e-3)
        vmax = float(vals_pos.max())
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        vmin = float(vals_all.min()) if len(vals_all) else 0.0
        vmax = float(vals_all.max()) if len(vals_all) else 1.0
        norm = None

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 2.8, rows * 2.4), squeeze=False, constrained_layout=True
    )

    for k, dom in enumerate(domains):
        ax = axes[k // cols][k % cols]
        sub = grid[grid["domain_name"] == dom]
        mat = (
            sub.pivot(index="min_judgments", columns="floor", values=value_col)
            .reindex(index=MIN_JUDGMENTS, columns=FLOORS)
        )
        data = mat.to_numpy(dtype=float)
        # For LogNorm safety: mask non-positive
        plot_data = data.copy()
        if use_log:
            plot_data[~(plot_data > 0)] = np.nan
        im = ax.imshow(
            plot_data,
            aspect="auto",
            cmap=cmap,
            norm=norm if use_log else None,
            vmin=None if use_log else vmin,
            vmax=None if use_log else vmax,
            origin="upper",
        )
        # Annotate every cell with the underlying value (incl. zeros / nans)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if np.isnan(v):
                    txt = "—"
                else:
                    txt = fmt.format(v)
                # contrast: white text on dark cells
                if not use_log:
                    rel = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.0
                else:
                    rel = (
                        (math.log10(max(v, vmin)) - math.log10(vmin))
                        / (math.log10(vmax) - math.log10(vmin))
                        if (use_log and v > 0 and vmax > vmin)
                        else 0.0
                    )
                color = "white" if rel > 0.55 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)
        ax.set_xticks(range(len(FLOORS)))
        ax.set_xticklabels(FLOORS, fontsize=8)
        ax.set_yticks(range(len(MIN_JUDGMENTS)))
        ax.set_yticklabels(MIN_JUDGMENTS, fontsize=8)
        ax.set_title(str(dom), fontsize=9)
        if k // cols == rows - 1:
            ax.set_xlabel("count floor", fontsize=8)
        if k % cols == 0:
            ax.set_ylabel("min judgments", fontsize=8)

    # Hide unused axes
    for k in range(n_dom, rows * cols):
        axes[k // cols][k % cols].axis("off")

    cbar = fig.colorbar(im, ax=axes, shrink=0.6, location="right", pad=0.02)
    cbar.set_label(cbar_label, fontsize=9)
    fig.suptitle(title, fontsize=12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_logging("eurovoc_drift_01a_judgment_threshold_sweep")
    logger = logging.getLogger(__name__)

    counts_path = PATHS.eurovoc_coverage / "label_celex_counts.parquet"
    concepts_path = PATHS.eurovoc / "concepts_enriched.csv"
    if not counts_path.exists():
        raise FileNotFoundError(
            f"{counts_path} not found. Run 02_corpus_coverage.py first."
        )

    counts = pd.read_parquet(counts_path)
    concepts = pd.read_csv(concepts_path, dtype={"concept_id": str})
    logger.info(
        f"Loaded {len(counts):,} label-celex rows, {len(concepts):,} concepts"
    )

    # Drop domain-root concepts (microthesaurus_id null) per 01_select_labels policy
    non_root = concepts[concepts["microthesaurus_id"].notna()][
        ["concept_id", "domain_name"]
    ]
    df = counts.merge(non_root, on="concept_id", how="inner")
    logger.info(
        f"After dropping domain-root concepts: {len(df):,} rows over "
        f"{df['concept_id'].nunique():,} concepts in "
        f"{df['domain_name'].nunique()} domains"
    )

    logger.info("Computing sweep grid...")
    grid = compute_grid(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUT_DIR / "grid.csv", index=False)
    logger.info(f"Wrote {OUT_DIR / 'grid.csv'}  ({len(grid):,} rows)")

    logger.info("Plotting median judgments-per-label heatmaps...")
    plot_heatmaps(
        grid,
        value_col="median_judgments",
        title="Median judgments per eligible label, by domain",
        cbar_label="median judgments / label (log)",
        out_path=OUT_DIR / "median_judgments_per_label.pdf",
        fmt="{:.0f}",
        use_log=True,
    )

    logger.info("Plotting % labels retained heatmaps...")
    plot_heatmaps(
        grid,
        value_col="pct_retained",
        title="% of labels retained vs no-filter baseline, by domain",
        cbar_label="% labels retained (log)",
        out_path=OUT_DIR / "pct_labels_retained.pdf",
        fmt="{:.0f}",
        use_log=True,
    )

    logger.info(f"Wrote heatmaps to {OUT_DIR}")


if __name__ == "__main__":
    main()
