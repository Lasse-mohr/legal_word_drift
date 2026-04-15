"""Compute per-word quantiles of year-to-year centroid shift distances.

Loads ``centroid_drift.parquet`` (output of script 15), groups step_distance
by word, and computes quantiles across the word's yearly shifts.  Words with
few non-NaN steps are excluded.

The quantile table is the raw material for identifying words that are:
  - mostly stable but occasionally spike  (low q0.2, high q0.8)
  - uniformly drifting                    (low q0.1, high q0.9, similar ratio)
  - monotonically drifting vs noisy       (compare spread ratios)

Outputs
-------
  data/results/metrics/shift_quantiles.parquet   — one row per word
  data/results/figures/shift_quantiles_scatter.png
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import FIGURES_DIR, METRICS_DIR, setup_logging

QUANTILES = [0.0, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 1.0]
Q_COLS = [f"q{int(q*100):02d}" for q in QUANTILES]  # q00, q10, ..., q100


def compute_shift_quantiles(
    drift_df: pd.DataFrame,
    min_steps: int = 10,
) -> pd.DataFrame:
    """Return one row per word with quantile columns plus q100/q90 ratio.

    Args:
        drift_df: Long-form DataFrame with columns ``word``, ``step_distance``.
        min_steps: Minimum number of non-NaN step values required.

    Returns:
        DataFrame with columns: word, q00, q10, …, q100, n_steps,
        spread_ratio (q100/q90, NaN when q90 == 0).
    """
    rows = []
    for word, grp in drift_df.groupby("word"):
        steps = grp["step_distance"].dropna().to_numpy(dtype=np.float64)
        if steps.size < min_steps:
            continue
        quantile_vals = np.quantile(steps, QUANTILES)
        row = {"word": word, "n_steps": int(steps.size)}
        for col, val in zip(Q_COLS, quantile_vals):
            row[col] = float(val)
        q_low = row["q90"]
        q_high = row["q100"]
        row["spread_ratio"] = float(q_high / q_low) if q_low > 1e-10 else float("nan")
        rows.append(row)

    col_order = ["word", "n_steps"] + Q_COLS + ["spread_ratio"]
    return pd.DataFrame(rows)[col_order].sort_values("spread_ratio", ascending=False)


def plot_low_vs_high(
    quantile_df: pd.DataFrame,
    output_path: str,
    n_label: int = 30,
) -> None:
    """Scatter of q_low (default q90) vs q_high (default q100), coloured by spread_ratio.

    Words in the top-right are broadly drifty.
    Words near the bottom-left are mostly stable.
    Words with low x but high y are the interesting ones: stable most years,
    but with a few large shifts.

    The n_label words with the highest spread_ratio are annotated.
    """
    df = quantile_df.dropna(subset=["spread_ratio"]).copy()

    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        df["q90"],
        df["q100"],
        c=df["spread_ratio"],
        cmap="plasma",
        alpha=0.6,
        s=18,
        linewidths=0,
    )
    ax.set_yscale('log')
    ax.set_xscale('log')
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("spread ratio  (q100/ q90)", fontsize=9)

    # Annotate top-n by spread_ratio
    top = df.nlargest(n_label, "spread_ratio")
    path_effects = [pe.withStroke(linewidth=2, foreground="white")]
    for _, row in top.iterrows():
        ax.annotate(
            row["word"],
            xy=(row["q90"], row["q100"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6.5,
            path_effects=path_effects,
        )

    # Reference line: equal percentiles (y = x * ratio at median ratio)
    xlim = ax.get_xlim()
    median_ratio = float(df["spread_ratio"].median())
    xs = np.array(xlim)
    ax.plot(xs, xs * median_ratio, ls="--", lw=0.8, color="gray",
            label=f"median ratio  ({median_ratio:.1f}×)")
    ax.set_xlim(xlim)

    ax.set_xlabel("q90 of yearly shift  (typical low-drift year)", fontsize=10)
    ax.set_ylabel("q100 of yearly shift  (typical high-drift year)", fontsize=10)
    ax.set_title(
        "Per-word shift distribution: q90 vs q100\n"
        "Low q90 + high q100 → stable word with occasional large shifts",
        fontsize=11,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute shift-distribution quantiles and plot q90 vs q100"
    )
    parser.add_argument(
        "--min-steps", type=int, default=10,
        help="Minimum non-NaN step_distance values required per word (default: 10)",
    )
    parser.add_argument(
        "--n-label", type=int, default=30,
        help="Number of words to annotate on the scatter plot (default: 30)",
    )
    args = parser.parse_args()

    setup_logging("19_compute_shift_quantiles")
    logger = logging.getLogger(__name__)

    drift_path = os.path.join(METRICS_DIR, "centroid_drift.parquet")
    if not os.path.exists(drift_path):
        logger.error(
            f"Missing {drift_path} — run script 15 (compute_centroid_drift) first"
        )
        sys.exit(1)

    logger.info(f"Loading {drift_path} …")
    drift_df = pd.read_parquet(drift_path)
    logger.info(f"  {len(drift_df):,} word-year rows, {drift_df['word'].nunique()} words")

    quantile_df = compute_shift_quantiles(drift_df, min_steps=args.min_steps)
    logger.info(
        f"Computed quantiles for {len(quantile_df)} words "
        f"(min_steps={args.min_steps})"
    )

    os.makedirs(METRICS_DIR, exist_ok=True)
    out_parquet = os.path.join(METRICS_DIR, "shift_quantiles.parquet")
    quantile_df.to_parquet(out_parquet, index=False)
    logger.info(f"Saved quantile table to {out_parquet}")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_fig = os.path.join(FIGURES_DIR, "shift_quantiles_scatter.png")
    plot_low_vs_high(quantile_df, out_fig, n_label=args.n_label)
    logger.info(f"Saved scatter plot to {out_fig}")

    logger.info("\nTop 20 words by spread_ratio (q100 / q90):")
    pd.set_option("display.float_format", "{:.5f}".format)
    logger.info(
        quantile_df.head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()
