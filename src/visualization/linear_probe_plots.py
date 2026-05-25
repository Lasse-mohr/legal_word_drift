"""Plotting helpers for the linear-probe drift statistic.

Shared across script 27 (calibration) and the EuroVoc per-label sweep.

  * ``plot_metric_distribution_real`` — histogram of held-out metric across
    all (word, year-pair) rows, with a per-word rug.
  * ``plot_metric_over_time_per_word`` — per-word trajectory grid.
  * ``plot_permutation_null`` — real-vs-permuted comparison for one word
    (built for bounded metrics like accuracy/AUC; null line at 0.5).
  * ``plot_w_norm_over_time`` — real trajectory + 5/95-percentile null band
    for an unbounded magnitude statistic (used for ``w_norm``).
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import kstest, norm

from src.visualization.plot_config import (
    apply_plot_style,
    get_categorical_colors,
    remove_extra_spines,
    get_named_cmap
)


def plot_metric_distribution_real(
    df: pd.DataFrame,
    words: list[str],
    out_path: str,
    metric: str,
    metric_label: str,
) -> None:
    apply_plot_style()
    colors = get_categorical_colors(len(words))
    word_color = dict(zip(words, colors))

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    bins = np.linspace(0.3, 1.0, 30)
    ax.hist(df[metric], bins=bins, color="#bbbbbb",
            edgecolor="white", linewidth=0.4)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.0,
               label=f"null {metric_label} = 0.5")

    rng = np.random.default_rng(0)
    for word in words:
        sub = df[df["word"] == word]
        if sub.empty:
            continue
        y = -1.5 - rng.uniform(0, 0.8, size=len(sub))
        ax.scatter(sub[metric], y, color=word_color[word], s=10,
                   alpha=0.85, label=word)

    ax.set_xlabel(f"held-out {metric_label}")
    ax.set_ylabel("count (year-pairs)")
    ax.set_title(
        f"Linear-probe {metric_label} across words & consecutive year-pairs"
    )
    ax.legend(loc="upper right", fontsize=8)
    remove_extra_spines(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_over_time_per_word(
    df: pd.DataFrame,
    words: list[str],
    out_path: str,
    metric: str,
    metric_label: str,
    n_cols: int = 2,
    panel_size: tuple[float, float] = (5.0, 3.0),
    titles: dict[str, str] | None = None,
) -> None:
    """Grid of per-word metric-vs-time panels.

    ``titles`` lets callers override panel titles (e.g. to add clarity tag).
    """
    apply_plot_style()
    colors = get_categorical_colors(max(len(words), 2))

    n_rows = math.ceil(len(words) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(panel_size[0] * n_cols, panel_size[1] * n_rows),
        sharey=True, squeeze=False,
    )
    for ax, word, color in zip(axes.flat, words, colors):
        title = (titles or {}).get(word, word)
        sub = df[df["word"] == word].sort_values("year_a")
        if sub.empty:
            ax.text(0.5, 0.5, f"no data: {word}",
                    transform=ax.transAxes, ha="center", va="center")
            ax.set_title(title)
            continue
        midpoints = sub["year_a"] + 0.5
        ax.plot(midpoints, sub[metric], color=color, linewidth=1.4, alpha=0.9)
        ax.scatter(midpoints, sub[metric], color=color, s=14, zorder=3)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.7)
        ax.set_title(title)
        ax.set_ylim(0.3, 1.02)
        ax.set_xlabel("year-pair midpoint")
        ax.set_ylabel(metric_label)
        for _, row in sub.iterrows():
            ax.text(row["year_a"] + 0.5, row[metric] + 0.015,
                    str(int(row["n_per_class"])),
                    fontsize=6, ha="center", color="#555")
        remove_extra_spines(ax)

    # Hide any unused panels.
    for ax in axes.flat[len(words):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Per-word linear-probe {metric_label} over consecutive year-pairs"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_w_norm_over_time(
    df_real: pd.DataFrame,
    df_perm: pd.DataFrame,
    word: str,
    out_path: str,
    title_suffix: str = "",
) -> None:
    """Real ``w_norm`` trajectory vs. permutation null band, one word.

    Null band: per-year-pair 5th/95th percentile from the permutation
    distribution, with the null median as a dashed line.
    """
    apply_plot_style()
    real = df_real[df_real["word"] == word].sort_values("year_a")

    if df_perm.empty or real.empty:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.text(0.5, 0.5, f"no data: {word}",
                transform=ax.transAxes, ha="center", va="center")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    pcts = (
        df_perm.groupby("year_a")["w_norm"]
        .agg(p05=lambda s: np.percentile(s, 5),
             p50=lambda s: np.percentile(s, 50),
             p95=lambda s: np.percentile(s, 95))
        .reset_index()
        .sort_values("year_a")
    )

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x_null = pcts["year_a"] + 0.5
    ax.fill_between(x_null, pcts["p05"], pcts["p95"],
                    color="#bbbbbb", alpha=0.45,
                    label=f"null 5–95 % (×{df_perm['perm_id'].nunique()})")
    ax.plot(x_null, pcts["p50"], color="#666666", linewidth=0.9,
            linestyle="--", label="null median")
    ax.plot(real["year_a"] + 0.5, real["w_norm"],
            color="#d62728", linewidth=1.6, label="real")
    ax.scatter(real["year_a"] + 0.5, real["w_norm"],
               color="#d62728", s=18, zorder=3)
    ax.set_xlabel("year-pair midpoint")
    ax.set_ylabel(r"$\|\mu_b - \mu_a\|$  (w_norm)")
    ax.set_title(f"{word}{title_suffix}: centroid-shift magnitude vs. null")
    ax.legend(loc="upper right", fontsize=8)
    remove_extra_spines(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_zscore_qq_grid(
    df_runs: pd.DataFrame,
    regime: str,
    out_path: str,
    classifier: str = "centroid",
    title_suffix: str = "",
) -> None:
    """QQ plot of analytical z-scores vs N(0,1), one panel per ``n_per_class``.

    Filters ``df_runs`` to ``classifier`` and ``regime``. KS p-value (against
    standard normal) in each panel subtitle.
    """
    apply_plot_style()
    sub_all = df_runs[
        (df_runs["classifier"] == classifier) & (df_runs["regime"] == regime)
    ].dropna(subset=["z_real"])
    if sub_all.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, f"no {classifier}/{regime} z-scores",
                transform=ax.transAxes, ha="center", va="center")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    n_values = sorted(sub_all["n_per_class"].unique())
    n_cols = min(len(n_values), 4)
    n_rows = math.ceil(len(n_values) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.4 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )
    for ax, n_val in zip(axes.flat, n_values):
        z = sub_all[sub_all["n_per_class"] == n_val]["z_real"].to_numpy()
        z_sorted = np.sort(z)
        probs = (np.arange(1, len(z_sorted) + 1) - 0.5) / len(z_sorted)
        theo = norm.ppf(probs)
        ks_stat, ks_p = kstest(z, "norm")
        ax.scatter(theo, z_sorted, s=14, color="#1f77b4", alpha=0.85)
        lim = max(abs(theo).max(), abs(z_sorted).max(), 3.0)
        ax.plot([-lim, lim], [-lim, lim],
                color="black", linestyle="--", linewidth=0.8)
        ax.axhline(0, color="#999", linewidth=0.5)
        ax.axvline(0, color="#999", linewidth=0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("N(0,1) quantile")
        ax.set_ylabel("observed z")
        ax.set_title(f"n={n_val}  (KS p={ks_p:.3g})")
        remove_extra_spines(ax)

    for ax in axes.flat[len(n_values):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Held-out AUC z-score vs N(0,1)  [{classifier} / {regime}]{title_suffix}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



def plot_auc_real_vs_perm_per_n(
    df_runs: pd.DataFrame,
    df_perms: pd.DataFrame,
    out_path: str,
    classifier: str,
    title_suffix: str = "",
) -> None:
    """For each n, side-by-side within/across panels: real AUC vs pooled perm AUC.

    Both inputs are pre-filtered conceptually but this helper also filters by
    ``classifier``. ``df_perms`` long form needs columns:
    ``classifier, regime, n_per_class, auc``.
    """
    apply_plot_style()
    runs = df_runs[df_runs["classifier"] == classifier]
    perms = df_perms[df_perms["classifier"] == classifier]
    if runs.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, f"no {classifier} runs",
                transform=ax.transAxes, ha="center", va="center")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    n_values = sorted(runs["n_per_class"].unique())
    regimes = ["within", "across"]
    n_rows = len(n_values)
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(10, 2.6 * n_rows), sharex=True, squeeze=False,
    )
    bins = np.linspace(0.0, 1.0, 41)
    for row, n_val in enumerate(n_values):
        for col, regime in enumerate(regimes):
            ax = axes[row, col]
            run_sub = runs[(runs["n_per_class"] == n_val)
                           & (runs["regime"] == regime)]
            perm_sub = perms[(perms["n_per_class"] == n_val)
                             & (perms["regime"] == regime)]
            if not perm_sub.empty:
                ax.hist(perm_sub["auc"], bins=bins, color="#bbbbbb",
                        edgecolor="white", linewidth=0.3, alpha=0.85,
                        label="permuted", density=True)
            if not run_sub.empty:
                ax.hist(run_sub["auc_real"], bins=bins, color="#d62728",
                        edgecolor="white", linewidth=0.3, alpha=0.7,
                        label="real", density=True)
            ax.axvline(0.5, color="black", linestyle="--", linewidth=0.8)
            ax.set_title(f"n={n_val}, {regime}")
            ax.set_xlim(0.0, 1.0)
            if row == n_rows - 1:
                ax.set_xlabel("held-out AUC")
            if col == 0:
                ax.set_ylabel("density")
            if row == 0 and col == 0:
                ax.legend(loc="upper left", fontsize=7)
            remove_extra_spines(ax)
    fig.suptitle(
        f"Real vs permuted AUC by sample size & regime [{classifier}]{title_suffix}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_classifier_real_auc_pair(
    df_runs: pd.DataFrame,
    out_path: str,
    clf_a: str = "centroid",
    clf_b: str = "logreg",
    title_suffix: str = "",
) -> None:
    """Paired-by-(rep, fold) scatter of real AUCs for two classifiers.

    One panel per (n, regime). Identity line for reference.
    """
    apply_plot_style()
    join_cols = ["regime", "n_per_class", "rep", "fold"]
    a = (df_runs[df_runs["classifier"] == clf_a]
         [join_cols + ["auc_real"]]
         .rename(columns={"auc_real": f"auc_{clf_a}"}))
    b = (df_runs[df_runs["classifier"] == clf_b]
         [join_cols + ["auc_real"]]
         .rename(columns={"auc_real": f"auc_{clf_b}"}))
    merged = a.merge(b, on=join_cols, how="inner")
    if merged.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no overlapping rows",
                transform=ax.transAxes, ha="center", va="center")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    n_values = sorted(merged["n_per_class"].unique())
    regimes = ["within", "across"]
    n_rows = len(n_values)
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(9, 3.0 * n_rows),
        sharex=True, sharey=True, squeeze=False,
    )
    for row, n_val in enumerate(n_values):
        for col, regime in enumerate(regimes):
            ax = axes[row, col]
            s = merged[(merged["n_per_class"] == n_val)
                       & (merged["regime"] == regime)]
            ax.scatter(s[f"auc_{clf_a}"], s[f"auc_{clf_b}"],
                       color="#1f77b4", alpha=0.8, s=18, edgecolor="white",
                       linewidth=0.4)
            ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8)
            ax.axhline(0.5, color="#999", linewidth=0.4)
            ax.axvline(0.5, color="#999", linewidth=0.4)
            ax.set_title(f"n={n_val}, {regime}")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            if row == n_rows - 1:
                ax.set_xlabel(f"{clf_a} AUC")
            if col == 0:
                ax.set_ylabel(f"{clf_b} AUC")
            remove_extra_spines(ax)
    fig.suptitle(f"Paired real AUC: {clf_a} vs {clf_b}{title_suffix}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_permutation_null(
    df_real: pd.DataFrame,
    df_perm: pd.DataFrame,
    word: str,
    out_path: str,
    metric: str,
    metric_label: str,
    title_suffix: str = "",
) -> None:
    apply_plot_style()
    real = df_real[df_real["word"] == word].sort_values("year_a")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    bins = np.linspace(0.3, 1.0, 30)
    ax.hist(df_perm[metric], bins=bins, color="#bbbbbb",
            edgecolor="white", linewidth=0.4, alpha=0.9,
            label=f"permuted (×{df_perm['perm_id'].nunique()})", density=True)
    ax.hist(real[metric], bins=bins, color="#d62728",
            edgecolor="white", linewidth=0.4, alpha=0.7,
            label="real", density=True)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel(f"held-out {metric_label}")
    ax.set_ylabel("density")
    ax.set_title(f"{word}{title_suffix}: permuted vs real {metric_label}")
    ax.legend(loc="upper right")
    remove_extra_spines(ax)

    ax = axes[1]
    for perm_id, sub in df_perm.groupby("perm_id"):
        sub = sub.sort_values("year_a")
        ax.plot(sub["year_a"] + 0.5, sub[metric],
                color="#888888", linewidth=0.8, alpha=0.25)
    ax.plot(real["year_a"] + 0.5, real[metric],
            color="#d62728", linewidth=1.8, label="real")
    ax.scatter(real["year_a"] + 0.5, real[metric],
               color="#d62728", s=20, zorder=3)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.7)
    ax.set_xlabel("year-pair midpoint")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{word}{title_suffix}: {metric_label} trajectory")
    ax.legend(loc="upper right")
    ax.set_ylim(0.3, 1.02)
    remove_extra_spines(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Centroid-AUC drift summaries (script-10 outputs) ──────────────────────


def plot_centroid_z_heatmap_grid(
    matrices_dict: dict[str, dict],
    labels: list[str],
    out_path: str,
    ncols: int = 5,
    vlim: float | None = None,
) -> None:
    """Per-label Y×Y heatmap grid of median z (upper triangle only).

    ``matrices_dict[label] = {"years": list[int], "matrix": (Y,Y) ndarray}``
    where ``matrix[i, j]`` is the median fold z for ``(years[i], years[j])``
    with ``i < j``; the lower triangle and diagonal are ``NaN`` (left blank
    in the heatmap). Diverging colormap centred on 0 (the MW null mean).
    """
    apply_plot_style()
    available = [w for w in labels if w in matrices_dict]
    if not available:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "no labels", transform=ax.transAxes,
                ha="center", va="center")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    n = len(available)
    nrows = max(1, (n + ncols - 1) // ncols)

    if vlim is None:
        all_vals = np.concatenate([
            matrices_dict[w]["matrix"][~np.isnan(matrices_dict[w]["matrix"])]
            .flatten()
            for w in available
        ])
        vlim = float(np.nanpercentile(np.abs(all_vals), 98)) if all_vals.size else 1.0
        vlim = max(vlim, 1.0)
    cmap = get_named_cmap("blues")

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 2.6, nrows * 2.6),
    )
    axes = np.array(axes).reshape(nrows, ncols)

    last_im = None
    for i, label in enumerate(available):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        info = matrices_dict[label]
        years = info["years"]
        M = info["matrix"]
        last_im = ax.imshow(
            M, cmap=cmap, origin="lower", aspect="equal",
            vmin=-vlim, vmax=vlim,
        )
        tick_idx = [k for k, y in enumerate(years) if y % 5 == 0]
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([str(years[k]) for k in tick_idx],
                           fontsize=6, rotation=45)
        ax.set_yticks(tick_idx)
        ax.set_yticklabels([str(years[k]) for k in tick_idx], fontsize=6)
        ax.set_title(label, fontsize=9, fontweight="bold")

    for i in range(n, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)

    fig.tight_layout(rect=(0, 0, 0.93, 1))
    if last_im is not None:
        cax = fig.add_axes((0.94, 0.15, 0.012, 0.7))
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.set_label("median z", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_global_z_histogram(
    df: pd.DataFrame,
    out_path: str,
    bins: int = 80,
) -> None:
    """Pooled histogram of every (label, pair, fold) z, with N(0,1) overlay."""
    apply_plot_style()
    z = df["z"].to_numpy()
    z = z[np.isfinite(z)]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    lo, hi = float(np.percentile(z, 0.5)), float(np.percentile(z, 99.5))
    span = max(abs(lo), abs(hi), 4.0)
    edges = np.linspace(-span, span, bins + 1)
    ax.hist(z, bins=edges, color="#888888", edgecolor="white",
            linewidth=0.3, density=True, label=f"observed (n={len(z):,})")
    xs = np.linspace(-span, span, 400)
    ax.plot(xs, norm.pdf(xs), color="#d62728", linewidth=1.4,
            label="N(0, 1)")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.7)
    ax.set_xlabel("centroid AUC z-score")
    ax.set_ylabel("density")
    frac_signif = float(np.mean(np.abs(z) > 1.96))
    ax.set_title(
        f"Global z distribution — fraction |z|>1.96: {frac_signif:.2%}"
    )
    ax.legend(loc="upper right", fontsize=8)
    remove_extra_spines(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_label_drift_ranking(
    df: pd.DataFrame,
    out_path: str,
    top_k: int | None = None,
    agg: str = "median",
) -> None:
    """Horizontal bar chart of per-label aggregate z.

    ``agg`` ∈ {``"median"``, ``"mean"``, ``"mean_abs"``}.
    """
    apply_plot_style()
    if agg == "mean_abs":
        score = df.assign(absz=df["z"].abs()).groupby("label")["absz"].mean()
        xlabel = "mean |z| (over all pairs, folds)"
    elif agg == "mean":
        score = df.groupby("label")["z"].mean()
        xlabel = "mean z (over all pairs, folds)"
    else:
        score = df.groupby("label")["z"].median()
        xlabel = "median z (over all pairs, folds)"

    score = score.sort_values(ascending=True)
    if top_k is not None and top_k < len(score):
        score = score.tail(top_k)

    h = max(2.5, 0.16 * len(score))
    fig, ax = plt.subplots(figsize=(6.5, h))
    colors = ["#d62728" if v >= 0 else "#1f77b4" for v in score.to_numpy()]
    ax.barh(score.index.tolist(), score.to_numpy(), color=colors,
            edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("EuroVoc label")
    ax.tick_params(axis="y", labelsize=6)
    ax.set_title(f"Per-label centroid-AUC drift ({agg})")
    remove_extra_spines(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_z_vs_year_gap(
    df: pd.DataFrame,
    out_path: str,
) -> None:
    """Distribution of z by year-gap (yb − ya), pooled over labels and folds."""
    apply_plot_style()
    d = df.assign(gap=df["year_b"] - df["year_a"])
    gaps = sorted(d["gap"].unique())

    fig, ax = plt.subplots(figsize=(8, 3.8))
    data_per_gap = [d.loc[d["gap"] == g, "z"].to_numpy() for g in gaps]
    parts = ax.violinplot(
        data_per_gap, positions=gaps, widths=0.85,
        showmeans=False, showextrema=False, showmedians=True,
    )
    for b in parts["bodies"]:
        b.set_facecolor("#1f77b4")
        b.set_edgecolor("white")
        b.set_alpha(0.55)
    if "cmedians" in parts:
        parts["cmedians"].set_color("#d62728")
        parts["cmedians"].set_linewidth(1.2)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.7)
    ax.axhline(1.96, color="#888", linestyle=":", linewidth=0.6)
    ax.axhline(-1.96, color="#888", linestyle=":", linewidth=0.6)
    ax.set_xlabel("year gap  (year_b − year_a)")
    ax.set_ylabel("z")
    ax.set_title("Centroid-AUC z vs temporal gap (all labels, all folds)")
    remove_extra_spines(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_z_vs_w_norm(
    df: pd.DataFrame,
    out_path: str,
    sample_n: int | None = 20000,
) -> None:
    """Scatter z vs ||μ_b − μ_a|| across all (label, pair, fold)."""
    apply_plot_style()
    d = df[["w_norm", "z"]].dropna()
    if sample_n is not None and len(d) > sample_n:
        d = d.sample(sample_n, random_state=0)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(d["w_norm"], d["z"], s=4, color="#1f77b4",
               alpha=0.25, edgecolor="none")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel(r"$\|\mu_b - \mu_a\|$  (w_norm)")
    ax.set_ylabel("z")
    if len(d) >= 2 and d["w_norm"].std() > 0:
        rho = float(d["w_norm"].corr(d["z"]))
        ax.set_title(f"Centroid shift vs z   (Pearson r = {rho:.3f})")
    else:
        ax.set_title("Centroid shift vs z")
    remove_extra_spines(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_consecutive_z_plotly(
    df: pd.DataFrame,
    out_html: str,
) -> None:
    """Interactive plotly line plot of median consecutive-year z per label.

    Restricts to (year_b − year_a) == 1, aggregates the 4 folds via median,
    one thin line per label. Hovering reveals the label name.
    """
    import plotly.graph_objects as go

    cons = df[df["year_b"] == df["year_a"] + 1]
    agg = (
        cons.groupby(["label", "year_a"])["z"].median().reset_index()
        .rename(columns={"z": "median_z"})
    )
    agg["midpoint"] = agg["year_a"] + 0.5

    fig = go.Figure()
    for label, sub in agg.groupby("label"):
        sub = sub.sort_values("year_a")
        fig.add_trace(go.Scatter(
            x=sub["midpoint"], y=sub["median_z"],
            mode="lines",
            line=dict(width=0.8, color="rgba(31,119,180,0.35)"),
            name=label,
            hovertemplate=(
                f"<b>{label}</b><br>"
                "year-pair midpoint: %{x:.1f}<br>"
                "median z: %{y:.2f}<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.add_hline(y=0, line=dict(color="black", width=0.6, dash="dash"))
    fig.add_hline(y=1.96, line=dict(color="#888", width=0.5, dash="dot"))
    fig.add_hline(y=-1.96, line=dict(color="#888", width=0.5, dash="dot"))
    fig.update_layout(
        title="Centroid-AUC median z, consecutive year-pairs (hover for label)",
        xaxis_title="year-pair midpoint",
        yaxis_title="median z (over 4 folds)",
        template="plotly_white",
        hovermode="closest",
        width=1000,
        height=600,
    )
    fig.write_html(out_html, include_plotlyjs="cdn")


def plot_score_median_vs_span(
    df: pd.DataFrame,
    out_path: str,
    metric: str = "z",
    metric_label: str | None = None,
) -> None:
    """Per (label, year_a, year_b) point plot: median across k folds vs span (max−min).

    Single-series scatter (one point per word-pair) → black points per style spec.
    """
    apply_plot_style()
    metric_label = metric_label or metric
    agg = (
        df.groupby(["label", "year_a", "year_b"])[metric]
        .agg(med="median", lo="min", hi="max")
        .reset_index()
    )
    agg["span"] = agg["hi"] - agg["lo"]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.scatter(
        agg["med"], agg["span"],
        color="black", s=6, alpha=0.35, edgecolor="none",
    )
    if metric == "z":
        ax.axvline(0, color="#888", linewidth=0.5, linestyle="--")
    ax.set_xlabel(f"median {metric_label} across folds")
    ax.set_ylabel(f"max − min {metric_label} across folds")
    ax.set_title(
        f"Per-(label, year-pair) fold dispersion of {metric_label}  "
        f"(n={len(agg):,})"
    )
    remove_extra_spines(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_drift_ranking_top_bottom(
    df: pd.DataFrame,
    out_path: str,
    n_each: int = 50,
    agg: str = "median",
) -> None:
    """Horizontal bar chart of the bottom-N and top-N labels by aggregate z.

    ``agg`` ∈ {``"median"``, ``"mean"``, ``"mean_abs"``}.
    """
    apply_plot_style()
    if agg == "mean_abs":
        score = df.assign(absz=df["z"].abs()).groupby("label")["absz"].mean()
        xlabel = "mean |z|"
    elif agg == "mean":
        score = df.groupby("label")["z"].mean()
        xlabel = "mean z"
    else:
        score = df.groupby("label")["z"].median()
        xlabel = "median z"

    score = score.sort_values(ascending=True)
    if len(score) > 2 * n_each:
        bottom = score.head(n_each)
        top = score.tail(n_each)
        sel = pd.concat([bottom, top])
    else:
        sel = score

    h = max(3.0, 0.18 * len(sel))
    fig, ax = plt.subplots(figsize=(6.5, h))
    ax.barh(
        sel.index.tolist(), sel.to_numpy(),
        color="black", edgecolor="white", linewidth=0.3,
    )
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("EuroVoc label")
    ax.tick_params(axis="y", labelsize=6)
    ax.set_title(
        f"Per-label centroid-AUC drift ({agg}) — bottom {n_each} & top {n_each}"
    )
    remove_extra_spines(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_drift_rank_histograms_by_group(
    df: pd.DataFrame,
    label_to_group: dict[str, str],
    out_path: str,
    agg: str = "median",
    group_order: list[str] | None = None,
    ncols: int = 3,
) -> None:
    """Grid of histograms of per-label aggregate-z, one panel per group.

    ``label_to_group`` maps each EuroVoc label to its category (e.g. clarity
    tier or EuroVoc domain). Labels not in the mapping are skipped.
    """
    apply_plot_style()
    if agg == "mean_abs":
        score = df.assign(absz=df["z"].abs()).groupby("label")["absz"].mean()
        xlabel = "mean |z|"
    elif agg == "mean":
        score = df.groupby("label")["z"].mean()
        xlabel = "mean z"
    else:
        score = df.groupby("label")["z"].median()
        xlabel = "median z"

    rows = []
    for lab, val in score.items():
        g = label_to_group.get(str(lab))
        if g is None:
            continue
        rows.append({"label": str(lab), "group": g, "score": float(val)})
    sdf = pd.DataFrame(rows)

    if sdf.empty:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "no labels matched any group",
                transform=ax.transAxes, ha="center", va="center")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    groups = group_order or sorted(sdf["group"].unique().tolist())
    n = len(groups)
    nrows = max(1, (n + ncols - 1) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.4 * ncols, 2.6 * nrows), squeeze=False,
    )

    lo = float(sdf["score"].min())
    hi = float(sdf["score"].max())
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    edges = np.linspace(lo, hi, 25)

    for ax, g in zip(axes.flat, groups):
        vals = sdf.loc[sdf["group"] == g, "score"].to_numpy()
        ax.hist(vals, bins=edges.tolist(), color="black",
                edgecolor="white", linewidth=0.4)
        ax.axvline(0, color="#888", linewidth=0.6, linestyle="--")
        ax.set_title(f"{g}  (n={len(vals)})", fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        remove_extra_spines(ax)
    for ax in axes.flat[n:]:
        ax.set_visible(False)

    fig.suptitle(f"Per-group distribution of per-label {xlabel}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _consecutive_z_per_label(df: pd.DataFrame) -> pd.DataFrame:
    """Median z over folds for every consecutive-year pair, per label.

    Returns columns ``label, year_a, midpoint, median_z``.
    """
    cons = df[df["year_b"] == df["year_a"] + 1]
    agg = (
        cons.groupby(["label", "year_a"])["z"].median().reset_index()
        .rename(columns={"z": "median_z"})
    )
    agg["midpoint"] = agg["year_a"] + 0.5
    return agg


def _plot_z_lines_two_panel(
    consec: pd.DataFrame,
    ranking: pd.Series,
    out_path: str,
    n_each: int,
    title: str,
    bottom_label: str = "lowest",
    top_label: str = "highest",
) -> None:
    """Two-panel matplotlib z-line plot: bottom-N (left) and top-N (right).

    ``ranking`` is a per-label score; lines are colored along the `blues`
    palette in ranking order so the gradient encodes the ranking.
    """
    apply_plot_style()
    ranking_sorted = ranking.sort_values(ascending=True)
    bottom = ranking_sorted.head(n_each).index.tolist()
    top = ranking_sorted.tail(n_each).index.tolist()

    cmap = get_named_cmap("blues")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)

    for ax, members, sub_title in (
        (axes[0], bottom, f"{bottom_label} {n_each}"),
        (axes[1], top, f"{top_label} {n_each}"),
    ):
        if not members:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center")
            continue
        for rank, lab in enumerate(members):
            color = cmap(0.15 + 0.8 * rank / max(1, len(members) - 1))
            sub = consec[consec["label"] == lab].sort_values("year_a")
            if sub.empty:
                continue
            ax.plot(
                sub["midpoint"], sub["median_z"],
                color=color, linewidth=0.9, alpha=0.75,
            )
        ax.axhline(0, color="black", linestyle="--", linewidth=0.7)
        ax.axhline(1.96, color="#888", linestyle=":", linewidth=0.5)
        ax.axhline(-1.96, color="#888", linestyle=":", linewidth=0.5)
        ax.set_xlabel("year-pair midpoint")
        ax.set_title(sub_title)
        remove_extra_spines(ax)
    axes[0].set_ylabel("median z (over folds)")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_consecutive_z_lines_top_bottom(
    df: pd.DataFrame,
    out_path: str,
    n_each: int = 50,
) -> None:
    """Top-N / bottom-N drifters by median z over consecutive year-pairs."""
    consec = _consecutive_z_per_label(df)
    ranking = consec.groupby("label")["median_z"].median()
    _plot_z_lines_two_panel(
        consec, ranking, out_path, n_each,
        title=(
            f"Consecutive-year median z — bottom {n_each} vs top {n_each} "
            "by median(median_z)"
        ),
    )


def compute_sleeping_beauty_coefficients(df: pd.DataFrame) -> pd.DataFrame:
    """Ke-et-al.-2015 B coefficient on per-label per-year median |z| series.

    Series s(t) is the median |z| over folds for each consecutive year-pair
    (year_a, year_a+1), indexed by integer ``t = year_a − min(year_a)``.
    With t_m = argmax s(t), s_0 = s(0), s_m = s(t_m):

        B = Σ_{t=0..t_m} [ ((s_m − s_0)/t_m · t + s_0 − s(t)) / max(1, s(t)) ]

    Labels with fewer than 2 consecutive points or with t_m == 0 get NaN.
    Returns columns ``label, B``.
    """
    cons = df[df["year_b"] == df["year_a"] + 1].copy()
    cons["abs_z"] = cons["z"].abs()
    series = (
        cons.groupby(["label", "year_a"])["abs_z"].median().reset_index()
        .sort_values(["label", "year_a"])
    )

    rows = []
    for label, sub in series.groupby("label"):
        ya = sub["year_a"].to_numpy()
        s = sub["abs_z"].to_numpy()
        if len(s) < 2:
            rows.append({"label": str(label), "B": float("nan")})
            continue
        t = (ya - ya.min()).astype(float)
        t_m_idx = int(np.argmax(s))
        if t_m_idx == 0:
            rows.append({"label": str(label), "B": float("nan")})
            continue
        s_0, s_m = float(s[0]), float(s[t_m_idx])
        t_m = float(t[t_m_idx])
        line = (s_m - s_0) / t_m * t[: t_m_idx + 1] + s_0
        denom = np.maximum(1.0, s[: t_m_idx + 1])
        B = float(np.sum((line - s[: t_m_idx + 1]) / denom))
        rows.append({"label": str(label), "B": B})
    return pd.DataFrame(rows)


def plot_consecutive_z_lines_sleeping_beauty(
    df: pd.DataFrame,
    out_path: str,
    n_each: int = 50,
) -> pd.DataFrame:
    """Top/bottom-N labels by sleeping-beauty coefficient B (z-line plot).

    Returns the per-label B table for downstream use.
    """
    sb = compute_sleeping_beauty_coefficients(df)
    sb_clean = sb.dropna(subset=["B"]).set_index("label")["B"]
    consec = _consecutive_z_per_label(df)
    _plot_z_lines_two_panel(
        consec, sb_clean, out_path, n_each,
        title=(
            f"Consecutive-year median z — bottom {n_each} vs top {n_each} "
            "sleeping-beauty (B on |z|)"
        ),
    )
    return sb
