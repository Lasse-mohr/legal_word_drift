"""C2ST null-variance and normality check.

Reads the per-word permutation outputs of script 27
(``metrics_perm_{word}.csv``) and characterises the null distribution of
held-out **accuracy** (= precision under balanced classes).

Theory recap (conditional-on-H derivation):

  Under H0 (no drift), with balanced held-out classes of size ñ per class,
    E[acc | H] = 1/2  for any hyperplane H, and
    Var(acc)   = E[q_H (1 − q_H)] / (2 ñ)  ≤  1/(8 ñ),
  where q_H = Pr(x lands on the t-side of H), x ~ P. The bound is tight
  when q_H ≈ 1/2 (e.g. an HDLSS L2-LogReg fit centred on P).

What this script reports per (word, year-pair):

  * mean_obs   — observed mean(acc) across permutations (target ≈ 0.5)
  * var_obs    — observed Var(acc) across permutations
  * var_bound  — analytical upper bound 1/(8 ñ)
  * shapiro_p  — Shapiro–Wilk p-value for normality of acc

Outputs:
  data/results/bert/linear_probe_calibration/null_variance_check.csv
  data/results/figures/linear_probe_calibration/var_vs_n_loglog.png
  data/results/figures/linear_probe_calibration/mean_vs_n.png
  data/results/figures/linear_probe_calibration/qq_grid.png
  data/results/bert/linear_probe_calibration/null_model_findings.md
"""
from __future__ import annotations

import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import FIGURES_DIR, RESULTS_DIR, setup_logging
from src.visualization.plot_config import (
    apply_plot_style,
    get_categorical_colors,
    remove_extra_spines,
)


TARGET_WORDS = ["organization", "general", "citizen", "referring"]


# ── Per-pair summary ─────────────────────────────────────────────────────


def summarise_pairs(perm_per_word: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per (word, year-pair) with mean, variance, Shapiro p-value."""
    rows: list[dict] = []
    for word, df in perm_per_word.items():
        for (ya, yb), grp in df.groupby(["year_a", "year_b"], sort=True):
            acc = grp["acc"].to_numpy()
            n_test = int(grp["n_test_per_class"].iloc[0])
            assert (grp["n_test_per_class"] == n_test).all(), (
                f"n_test_per_class varies within ({word}, {ya}, {yb})"
            )
            # Shapiro requires >=3 samples and is meaningful for n in [3, 5000].
            if len(acc) >= 3:
                _, sh_p = stats.shapiro(acc)
            else:
                sh_p = np.nan
            rows.append({
                "word": word,
                "year_a": int(ya),
                "year_b": int(yb),
                "n_test_per_class": n_test,
                "n_perms": len(acc),
                "mean_obs": float(acc.mean()),
                "var_obs": float(acc.var(ddof=1)),
                "var_bound": 1.0 / (8.0 * n_test),
                "shapiro_p": float(sh_p),
            })
    return pd.DataFrame(rows)


# ── Plots ────────────────────────────────────────────────────────────────


def plot_var_vs_n_loglog(summary: pd.DataFrame, out_path: str) -> None:
    apply_plot_style()
    words = sorted(summary["word"].unique())
    colors = get_categorical_colors(len(words))
    word_color = dict(zip(words, colors))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    # Combined log-log scatter.
    ax = axes[0]
    n_grid = np.logspace(
        np.log10(max(1, summary["n_test_per_class"].min())),
        np.log10(summary["n_test_per_class"].max()),
        200,
    )
    ax.plot(n_grid, 1.0 / (8.0 * n_grid), color="black", linestyle="--",
            linewidth=1.0, label="bound: 1 / (8 ñ)")
    for word in words:
        sub = summary[summary["word"] == word]
        ax.scatter(sub["n_test_per_class"], sub["var_obs"],
                   color=word_color[word], s=24, alpha=0.85, label=word)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ñ (held-out points per class)")
    ax.set_ylabel("observed Var(acc)")
    ax.set_title("Permutation-null variance vs. analytical bound")
    ax.legend(loc="upper right")
    remove_extra_spines(ax)

    # Per-word ratio panel: var_obs / var_bound.
    ax = axes[1]
    for word in words:
        sub = summary[summary["word"] == word].sort_values("n_test_per_class")
        ratio = sub["var_obs"] / sub["var_bound"]
        ax.scatter(sub["n_test_per_class"], ratio,
                   color=word_color[word], s=24, alpha=0.85, label=word)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0,
               label="ratio = 1")
    ax.set_xscale("log")
    ax.set_xlabel("ñ (held-out points per class)")
    ax.set_ylabel("var_obs / (1/(8ñ))")
    ax.set_title("Tightness of the bound")
    ax.legend(loc="upper right")
    remove_extra_spines(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mean_vs_n(summary: pd.DataFrame, out_path: str) -> None:
    apply_plot_style()
    words = sorted(summary["word"].unique())
    colors = get_categorical_colors(len(words))
    word_color = dict(zip(words, colors))

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for word in words:
        sub = summary[summary["word"] == word].sort_values("n_test_per_class")
        # MC standard error of mean(acc) given the bound: sqrt(1/(8ñ) / n_perms)
        se = np.sqrt(sub["var_bound"] / sub["n_perms"])
        ax.errorbar(sub["n_test_per_class"], sub["mean_obs"], yerr=se,
                    fmt="o", color=word_color[word], markersize=5,
                    capsize=2, alpha=0.85, label=word)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0,
               label="theoretical mean = 0.5")
    ax.set_xscale("log")
    ax.set_xlabel("ñ (held-out points per class)")
    ax.set_ylabel("observed mean(acc)")
    ax.set_title("Permutation-null mean (error bars: MC SE under bound)")
    ax.legend(loc="upper right")
    remove_extra_spines(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _pick_n_buckets(
    sub_summary: pd.DataFrame, k: int = 3
) -> list[pd.Series]:
    """Pick k year-pair rows representative of small / median / large ñ."""
    if sub_summary.empty:
        return []
    s = sub_summary.sort_values("n_test_per_class").reset_index(drop=True)
    if len(s) <= k:
        return [s.iloc[i] for i in range(len(s))]
    idxs = np.linspace(0, len(s) - 1, k).round().astype(int)
    return [s.iloc[int(i)] for i in idxs]


def plot_qq_grid(
    summary: pd.DataFrame,
    perm_per_word: dict[str, pd.DataFrame],
    out_path: str,
    k_buckets: int = 3,
) -> None:
    apply_plot_style()
    words = sorted(summary["word"].unique())

    fig, axes = plt.subplots(
        len(words), k_buckets,
        figsize=(3.6 * k_buckets, 3.0 * len(words)),
        squeeze=False,
    )
    for r, word in enumerate(words):
        picks = _pick_n_buckets(summary[summary["word"] == word], k=k_buckets)
        for c in range(k_buckets):
            ax = axes[r, c]
            if c >= len(picks):
                ax.set_visible(False)
                continue
            row = picks[c]
            ya, yb = int(row["year_a"]), int(row["year_b"])
            acc = (
                perm_per_word[word]
                .query("year_a == @ya and year_b == @yb")["acc"]
                .to_numpy()
            )
            stats.probplot(acc, dist="norm", plot=ax)
            # Override default styling.
            ax.get_lines()[0].set_marker(".")
            ax.get_lines()[0].set_markersize(3)
            ax.get_lines()[0].set_color("#444")
            ax.get_lines()[1].set_color("#d62728")
            ax.set_title(
                f"{word} {ya}-{yb}\nñ={int(row['n_test_per_class'])}, "
                f"Shapiro p={row['shapiro_p']:.2g}",
                fontsize=9,
            )
            remove_extra_spines(ax)

    fig.suptitle("Q–Q (Normal) of permutation accuracy per (word, year-pair)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Findings report ─────────────────────────────────────────────────────


def write_findings(
    summary: pd.DataFrame, csv_dir: str, fig_dir: str, out_path: str
) -> None:
    n_pairs = len(summary)
    mean_ratio = (summary["var_obs"] / summary["var_bound"]).mean()
    median_ratio = (summary["var_obs"] / summary["var_bound"]).median()
    frac_over_bound = (summary["var_obs"] > summary["var_bound"]).mean()

    # Mean check: is empirical mean within MC SE of 0.5?
    se = np.sqrt(summary["var_bound"] / summary["n_perms"])
    z_mean = (summary["mean_obs"] - 0.5) / se
    frac_mean_outside_2se = (z_mean.abs() > 2).mean()

    # Normality.
    shapiro_pass_uncorrected = (summary["shapiro_p"] > 0.05).mean()
    bonf_alpha = 0.05 / max(1, n_pairs)
    shapiro_pass_bonf = (summary["shapiro_p"] > bonf_alpha).mean()

    per_word = summary.groupby("word").agg(
        pairs=("var_obs", "size"),
        mean_var_ratio=(
            "var_obs",
            lambda v: float((v / summary.loc[v.index, "var_bound"]).mean()),
        ),
        median_mean_acc=("mean_obs", "median"),
        median_shapiro_p=("shapiro_p", "median"),
    )

    md = []
    md.append("# C2ST null-model findings (script 28)\n")
    md.append(f"Total (word, year-pair) cells: **{n_pairs}**.\n")
    md.append("## Variance vs. analytical bound 1/(8ñ)\n")
    md.append(
        f"- Mean ratio var_obs / var_bound: **{mean_ratio:.3f}** "
        f"(median **{median_ratio:.3f}**)."
    )
    md.append(
        f"- Fraction of pairs with var_obs > var_bound: **{frac_over_bound:.2%}**."
    )
    md.append(
        "- Reading: ratio ≈ 1 means the bound is tight (q_H ≈ 1/2, "
        "consistent with HDLSS data piling); ratio < 1 means the learned "
        "hyperplane tilts P → variance is *smaller* than 1/(8ñ).\n"
    )
    md.append("## Mean check (target = 0.5)\n")
    md.append(
        f"- Fraction of pairs whose observed mean is > 2 MC-SEs from 0.5: "
        f"**{frac_mean_outside_2se:.2%}** "
        "(under H0 with the bound, expect ≈ 5%).\n"
    )
    md.append("## Normality (Shapiro–Wilk on permutation accuracies)\n")
    md.append(
        f"- Pass at α=0.05 (uncorrected): **{shapiro_pass_uncorrected:.2%}** "
        "of pairs."
    )
    md.append(
        f"- Pass at Bonferroni α/{n_pairs}={bonf_alpha:.2g}: "
        f"**{shapiro_pass_bonf:.2%}** of pairs."
    )
    md.append("- Q-Q grid: see `qq_grid.png`.\n")
    md.append("## Per-word summary\n")
    md.append("```\n" + per_word.to_string() + "\n```\n")
    md.append("## Files\n")
    md.append(
        "- Per-pair table: `null_variance_check.csv`\n"
        "- Variance plot: `figures/.../var_vs_n_loglog.png`\n"
        "- Mean plot: `figures/.../mean_vs_n.png`\n"
        "- Q–Q grid: `figures/.../qq_grid.png`\n"
    )
    md.append(
        "## Decision\n"
        "Open. After reviewing the figures + this report, decide whether to "
        "replace permutations with the closed-form null `Normal(0.5, "
        "1/√(8ñ))` for the full vocabulary sweep, keep permutations, or use "
        "a hybrid.\n"
    )

    with open(out_path, "w") as f:
        f.write("\n".join(md))


# ── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    setup_logging("28_c2st_null_variance_check")
    logger = logging.getLogger(__name__)

    csv_dir = os.path.join(RESULTS_DIR, "bert", "linear_probe_calibration")
    fig_dir = os.path.join(FIGURES_DIR, "linear_probe_calibration")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    perm_per_word: dict[str, pd.DataFrame] = {}
    for word in TARGET_WORDS:
        path = os.path.join(csv_dir, f"metrics_perm_{word}.csv")
        if not os.path.isfile(path):
            logger.warning(f"Missing: {path} — skipping {word!r}")
            continue
        perm_per_word[word] = pd.read_csv(path)
        logger.info(f"Loaded {os.path.basename(path)}: {len(perm_per_word[word])} rows")

    if not perm_per_word:
        logger.error("No permutation CSVs found. Run script 27 first.")
        return

    summary = summarise_pairs(perm_per_word)
    summary_path = os.path.join(csv_dir, "null_variance_check.csv")
    summary.to_csv(summary_path, index=False)
    logger.info(f"Wrote {os.path.basename(summary_path)} ({len(summary)} rows)")

    plot_var_vs_n_loglog(
        summary, os.path.join(fig_dir, "var_vs_n_loglog.png")
    )
    plot_mean_vs_n(
        summary, os.path.join(fig_dir, "mean_vs_n.png")
    )
    plot_qq_grid(
        summary, perm_per_word,
        os.path.join(fig_dir, "qq_grid.png"),
    )

    findings_path = os.path.join(csv_dir, "null_model_findings.md")
    write_findings(summary, csv_dir, fig_dir, findings_path)
    logger.info(f"Wrote {os.path.basename(findings_path)}")
    logger.info(f"Figures written to {fig_dir}")


if __name__ == "__main__":
    main()
