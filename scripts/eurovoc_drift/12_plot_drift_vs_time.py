"""Plot per-step centroid drift vs time, with a year-shuffled null.

For every selected EuroVoc label, the per-year BERT centroids give a
sequence of step-distances (cosine and Euclidean) between consecutive
year means. We pool those across labels and look at how the distribution
behaves over the years 1990–2025 — to eyeball whether drift might be
accelerating.

The null is a *per-word, stratified* shuffle of year labels: within each
word, every embedding is reassigned to a uniformly-random year drawn
from the same word's year coverage, while preserving per-year sample
sizes exactly. Centroids are then recomputed under that shuffle.

Note: case counts under this null are identical to the real counts by
construction (the shuffle preserves per-year N). The count-vs-count
plots therefore exist only in their real form; the figure caption
states this explicitly.

Outputs in ``data/results/figures/eurovoc_drift/drift_vs_time/``:

    fig_drift_vs_time_all.png
    fig_drift_vs_time_by_domain_cosine.png
    fig_drift_vs_time_by_domain_euclidean.png
    fig_count_vs_count_all.png
    fig_count_vs_count_by_domain.png

Cached intermediate: ``data/processed/eurovoc_drift/usage_counts_by_year.parquet``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.embeddings.bert_encoder import resolve_model
from src.metrics.temporal_drift import (
    _cosine_distance,
    compute_centroids,
    load_per_year_embeddings,
)
from src.paths import PATHS
from src.utils.config import setup_logging
from src.visualization.plot_config import (
    apply_plot_style,
    get_heatmap_cmap,
    remove_extra_spines,
)

DIP_YEARS = (2003, 2004)
LOGGER = logging.getLogger(__name__)


# ── data assembly ─────────────────────────────────────────────────────────


def step_drifts_for_word(
    per_year: dict[int, np.ndarray],
) -> list[tuple[int, int, float, float]]:
    """(year_t, year_t1, cos, eucl) for every truly consecutive year pair."""
    years, centroids = compute_centroids(per_year)
    rows: list[tuple[int, int, float, float]] = []
    for i in range(1, len(years)):
        y0, y1 = years[i - 1], years[i]
        if y1 - y0 != 1:
            continue
        cos = _cosine_distance(centroids[i - 1], centroids[i])
        eucl = float(np.linalg.norm(centroids[i].astype(np.float32) - centroids[i - 1].astype(np.float32)))
        rows.append((y0, y1, cos, eucl))
    return rows


def stratified_year_shuffle(
    per_year: dict[int, np.ndarray],
    rng: np.random.Generator,
) -> dict[int, np.ndarray]:
    """Permute year labels for one word, preserving per-year sample sizes."""
    years = sorted(per_year.keys())
    sizes = [per_year[y].shape[0] for y in years]
    stacked = np.concatenate([per_year[y] for y in years], axis=0)
    perm = rng.permutation(stacked.shape[0])
    shuffled = stacked[perm]
    out: dict[int, np.ndarray] = {}
    start = 0
    for y, n in zip(years, sizes):
        out[y] = shuffled[start:start + n]
        start += n
    return out


def build_drift_table(
    embeddings: dict[str, dict[int, np.ndarray]],
    domain_by_word: dict[str, str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """One row per (word, year_t) with real and null drift columns."""
    rows: list[dict] = []
    for word, per_year in embeddings.items():
        domain = domain_by_word.get(word, "(unknown)")
        real = step_drifts_for_word(per_year)
        null_pyr = stratified_year_shuffle(per_year, rng)
        null = step_drifts_for_word(null_pyr)
        null_map = {(t0, t1): (c, e) for t0, t1, c, e in null}
        for t0, t1, cos_r, eucl_r in real:
            cos_n, eucl_n = null_map.get((t0, t1), (np.nan, np.nan))
            rows.append({
                "word": word,
                "domain": domain,
                "year_t": t0,
                "year_t1": t1,
                "cos_real": cos_r,
                "eucl_real": eucl_r,
                "cos_null": cos_n,
                "eucl_null": eucl_n,
            })
    return pd.DataFrame(rows)


def load_usage_counts(years: range, unit: str) -> pd.DataFrame:
    """(word, year, count) from the usage index, cached to parquet."""
    base_cache = PATHS.eurovoc_drift_usage_counts
    cache = (
        base_cache if unit == "year"
        else base_cache.with_name(
            f"{base_cache.stem}_{unit}{base_cache.suffix}"
        )
    )
    if cache.exists():
        LOGGER.info(f"Loading cached usage counts from {cache}")
        return pd.read_parquet(cache)

    LOGGER.info("Counting usage-index lines per (word, year)…")
    cache.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for year in years:
        path = PATHS.eurovoc_drift_usage_index_year_for(unit, year)
        if not path.exists():
            LOGGER.warning(f"Missing usage index for {year}")
            continue
        counter: Counter = Counter()
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                counter[rec["word"]] += 1
        for word, count in counter.items():
            rows.append({"word": word, "year": year, "count": count})
    df = pd.DataFrame(rows)
    df.to_parquet(cache, index=False)
    LOGGER.info(f"Cached usage counts to {cache} ({len(df)} rows)")
    return df


def build_count_pairs(counts: pd.DataFrame, domain_by_word: dict[str, str]) -> pd.DataFrame:
    """Per (word, year_t) wide pairs of count_t and count_{t+1}."""
    wide = counts.pivot(index="word", columns="year", values="count").fillna(0).astype(int)
    years = sorted(wide.columns)
    rows: list[dict] = []
    for word, row in wide.iterrows():
        domain = domain_by_word.get(word, "(unknown)")
        for t in years[:-1]:
            t1 = t + 1
            if t1 not in wide.columns:
                continue
            c0, c1 = int(row[t]), int(row[t1])
            if c0 == 0 and c1 == 0:
                continue
            rows.append({
                "word": word, "domain": domain,
                "year_t": t, "count_t": c0, "count_t1": c1,
            })
    return pd.DataFrame(rows)


# ── plotting helpers ──────────────────────────────────────────────────────


def shade_dip(ax: plt.Axes) -> None:
    ax.axvspan(DIP_YEARS[0] - 0.5, DIP_YEARS[1] + 0.5,
               color="#dddddd", alpha=0.6, zorder=0,
               label="2003–04 corpus dip")


def drift_hist2d_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    y_edges: np.ndarray | None = None,
) -> None:
    """2D histogram with one bin per year on x; log-transformed drift on y."""
    if df.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        return
    x = df["year_t"].to_numpy()
    y = df[metric_col].to_numpy()
    mask = np.isfinite(y) & (y > 0)
    x, y = x[mask], y[mask]
    if x.size == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "no positive data", transform=ax.transAxes, ha="center", va="center")
        return
    logy = np.log(y)

    y_lo = int(np.floor(x.min()))
    y_hi = int(np.ceil(x.max()))
    x_edges = np.arange(y_lo - 0.5, y_hi + 1.5, 1.0)
    if y_edges is None:
        y_edges = np.linspace(logy.min(), logy.max(), 26)

    ax.hist2d(x, logy, bins=[x_edges, y_edges], cmap=get_heatmap_cmap(), cmin=1)

    med = (
        pd.DataFrame({"x": x, "logy": logy})
        .groupby("x")["logy"].median().sort_index()
    )
    ax.plot(med.index, med.values, color="#d94801", lw=1.4, label="median per year")

    shade_dip(ax)
    ax.set_xlabel("year t (of step t → t+1)")
    ax.set_ylabel(f"log {ylabel}")
    ax.set_title(title)
    remove_extra_spines(ax)


def hexbin_count_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
) -> None:
    if df.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        return
    x = df["count_t"].to_numpy().astype(float)
    y = df["count_t1"].to_numpy().astype(float)
    mask = (x > 0) & (y > 0)
    x, y = x[mask], y[mask]
    if x.size == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        return
    ax.hexbin(
        x, y, xscale="log", yscale="log", gridsize=30,
        cmap=get_heatmap_cmap(), mincnt=1,
    )
    lo = max(min(x.min(), y.min()), 1.0)
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], color="#666666", lw=0.8, ls="--", label="y=x")
    ax.set_xlabel("count at t")
    ax.set_ylabel("count at t+1")
    ax.set_title(title)
    remove_extra_spines(ax)


# ── figure builders ──────────────────────────────────────────────────────


def _shared_log_edges(df: pd.DataFrame, real_col: str, null_col: str, n: int = 26) -> np.ndarray:
    vals = np.concatenate([df[real_col].to_numpy(), df[null_col].to_numpy()])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    return np.linspace(np.log(vals.min()), np.log(vals.max()), n)


def plot_all_labels(df: pd.DataFrame, out_dir) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    cos_edges = _shared_log_edges(df, "cos_real", "cos_null")
    eucl_edges = _shared_log_edges(df, "eucl_real", "eucl_null")

    drift_hist2d_panel(axes[0, 0], df, "cos_real",
                       "Real • cosine step-distance", "cos(μ_t, μ_{t+1})", cos_edges)
    drift_hist2d_panel(axes[0, 1], df, "eucl_real",
                       "Real • Euclidean step-distance", "‖μ_{t+1} − μ_t‖", eucl_edges)
    drift_hist2d_panel(axes[1, 0], df, "cos_null",
                       "Year-shuffled null • cosine", "cos(μ_t, μ_{t+1})", cos_edges)
    drift_hist2d_panel(axes[1, 1], df, "eucl_null",
                       "Year-shuffled null • Euclidean", "‖μ_{t+1} − μ_t‖", eucl_edges)

    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle(
        f"Per-step centroid drift vs time — all {df['word'].nunique()} EuroVoc labels"
    )
    out = out_dir / "fig_drift_vs_time_all.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOGGER.info(f"Wrote {out}")


def plot_by_domain_drift(df: pd.DataFrame, metric: str, out_dir) -> None:
    """5x5 grid of domain panels, real on top of null per panel via two subplots.

    We render a single panel per domain showing the real median trend and
    the null median trend overlaid, plus a translucent real hexbin
    background. This keeps the figure readable.
    """
    metric_real, metric_null, ylabel = {
        "cosine": ("cos_real", "cos_null", "cos(μ_t, μ_{t+1})"),
        "euclidean": ("eucl_real", "eucl_null", "‖μ_{t+1} − μ_t‖"),
    }[metric]

    domains = sorted(df["domain"].dropna().unique())
    n = len(domains)
    cols = 5
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 2.6),
                             constrained_layout=True, sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    y_edges = _shared_log_edges(df, metric_real, metric_null)
    year_min = int(df["year_t"].min())
    year_max = int(df["year_t"].max())
    x_edges = np.arange(year_min - 0.5, year_max + 1.5, 1.0)

    for ax, domain in zip(axes.ravel(), domains):
        sub = df[df["domain"] == domain]
        x = sub["year_t"].to_numpy()
        yr = sub[metric_real].to_numpy()
        yn = sub[metric_null].to_numpy()

        m_real = np.isfinite(yr) & (yr > 0)
        if m_real.any():
            ax.hist2d(x[m_real], np.log(yr[m_real]),
                      bins=[x_edges, y_edges],
                      cmap=get_heatmap_cmap(), cmin=1, alpha=0.85)

        df_real = pd.DataFrame({"x": x, "y": yr}).query("y > 0")
        df_null = pd.DataFrame({"x": x, "y": yn}).query("y > 0")
        med_real = np.log(df_real.groupby("x")["y"].median()) if not df_real.empty else pd.Series(dtype=float)
        med_null = np.log(df_null.groupby("x")["y"].median()) if not df_null.empty else pd.Series(dtype=float)
        if not med_real.empty:
            ax.plot(med_real.index, med_real.values, color="#d94801", lw=1.3,
                    label="real median")
        if not med_null.empty:
            ax.plot(med_null.index, med_null.values, color="#2171b5", lw=1.3,
                    ls="--", label="null median")

        shade_dip(ax)
        ax.set_title(f"{domain}  (n={sub['word'].nunique()})", fontsize=9)
        remove_extra_spines(ax)

    # Hide unused axes
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel("year t")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"log {ylabel}")

    # Single shared legend in first visible axis
    axes.ravel()[0].legend(loc="upper left", fontsize=7)

    fig.suptitle(f"Per-step centroid drift vs time, by EuroVoc domain • {metric}")
    out = out_dir / f"fig_drift_vs_time_by_domain_{metric}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOGGER.info(f"Wrote {out}")


def plot_counts_all(df: pd.DataFrame, out_dir) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6), constrained_layout=True)
    hexbin_count_panel(ax, df, f"Case-count at t vs t+1 — {df['word'].nunique()} labels")
    ax.legend(loc="upper left", fontsize=8)
    fig.text(
        0.01, 0.01,
        "Null version omitted: the per-word stratified shuffle preserves "
        "per-year sample sizes by construction, so the null is identical "
        "to the real data here.",
        fontsize=7, color="#555555",
    )
    out = out_dir / "fig_count_vs_count_all.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOGGER.info(f"Wrote {out}")


def plot_counts_by_domain(df: pd.DataFrame, out_dir) -> None:
    domains = sorted(df["domain"].dropna().unique())
    n = len(domains)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.8),
                             constrained_layout=True, sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for ax, domain in zip(axes.ravel(), domains):
        sub = df[df["domain"] == domain]
        hexbin_count_panel(ax, sub, f"{domain}  (n={sub['word'].nunique()})")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle("Case-count(t) vs case-count(t+1), by EuroVoc domain  (log-log)")
    out = out_dir / "fig_count_vs_count_by_domain.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOGGER.info(f"Wrote {out}")


# ── main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot drift vs time with year-shuffle null")
    parser.add_argument("--unit", choices=["year", "judgment"], default="year",
                        help="Routes embeddings/selection/output through {unit}/.")
    parser.add_argument("--model", type=str, default="eurlex",
                        help="Encoder (friendly name). 'eurlex' reads/writes the "
                             "legacy paths; a control model uses models/<name>/.")
    parser.add_argument("--start", type=int, default=1990)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--min-usages", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    setup_logging(f"eurovoc_drift_12_plot_drift_vs_time_{args.unit}")
    apply_plot_style()
    _, slug = resolve_model(args.model)
    LOGGER.info(f"model={args.model} slug={slug}")

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = PATHS.eurovoc_drift_figures_drift_vs_time_for(args.unit, slug)
    out_dir.mkdir(parents=True, exist_ok=True)

    years = range(args.start, args.end + 1)
    rng = np.random.default_rng(args.seed)

    # Label → domain map
    selected = pd.read_parquet(PATHS.eurovoc_drift_selected_labels_for(args.unit))
    selected["label"] = selected["label"].str.lower()
    domain_by_word = dict(zip(selected["label"], selected["domain_name"].fillna("(unassigned)")))

    LOGGER.info("Loading per-year embeddings…")
    embeddings = load_per_year_embeddings(
        str(PATHS.eurovoc_drift_embeddings_for(args.unit, slug)),
        years,
        words=None,
        min_usages=args.min_usages,
    )
    LOGGER.info(f"Loaded embeddings for {len(embeddings)} words")

    LOGGER.info("Computing real + null drift table…")
    drift = build_drift_table(embeddings, domain_by_word, rng)
    LOGGER.info(f"Drift rows: {len(drift)}")

    # Sanity assertion: under stratified shuffle, both real and null are
    # computed from the same per-year sample sizes — there should be
    # exactly one null row for every real row.
    assert drift["cos_null"].notna().sum() == drift["cos_real"].notna().sum(), \
        "null/real row mismatch — stratified shuffle is broken"

    plot_all_labels(drift, out_dir)
    plot_by_domain_drift(drift, "cosine", out_dir)
    plot_by_domain_drift(drift, "euclidean", out_dir)

    LOGGER.info("Building case-count table…")
    counts = load_usage_counts(years, args.unit)
    pairs = build_count_pairs(counts, domain_by_word)
    LOGGER.info(f"Count pairs: {len(pairs)}")

    plot_counts_all(pairs, out_dir)
    plot_counts_by_domain(pairs, out_dir)

    LOGGER.info(f"Done. Figures in {out_dir}")


if __name__ == "__main__":
    main()
