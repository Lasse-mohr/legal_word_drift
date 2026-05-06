"""Linear-probe drift statistic — calibration prototype.

For each target word and each consecutive year-pair, fit an L2-regularised
logistic regression on L2-normalised BERT embeddings labelled by year. We
report two held-out statistics per (word, year-pair):

  * **acc** — held-out accuracy (== precision under balanced classes). This
    is the primary statistic. Under H0 of no drift, with balanced held-out
    classes, E[acc] = 1/2 exactly and Var(acc) ≤ 1/(8 ñ) where ñ is the
    held-out points per class.
  * **auc** — held-out ROC-AUC. Kept as a secondary diagnostic.

Permutation null (year labels shuffled across all of a word's embeddings,
sweep re-run) is run on all target words to characterise the null
distribution of acc and auc.

Outputs:
  data/results/bert/linear_probe_calibration/
    counts_per_year.csv
    metrics_real.csv                    # acc + auc, real labels
    metrics_perm_{word}.csv             # acc + auc, permuted labels (per word)
  data/results/figures/linear_probe_calibration/
    acc_distribution_real.png
    acc_over_time_per_word.png
    permutation_null_acc_{word}.png
    auc_distribution_real.png           # legacy AUC view
    auc_over_time_per_word.png
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from src.metrics.temporal_drift import load_per_year_embeddings
from src.utils.config import (
    BERT_EMBEDDINGS_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
    setup_logging,
)
from src.visualization.plot_config import (
    apply_plot_style,
    get_categorical_colors,
    remove_extra_spines,
)


TARGET_WORDS = ["organization", "general", "citizen", "referring"]


# ── Data loading ─────────────────────────────────────────────────────────


def load_word_embeddings(
    embeddings_dir: str, words: list[str], years: list[int]
) -> dict[str, dict[int, np.ndarray]]:
    """Load and L2-normalise per-(word, year) embeddings as float32."""
    raw = load_per_year_embeddings(
        embeddings_dir, years, words=set(words), min_usages=1
    )
    out: dict[str, dict[int, np.ndarray]] = {}
    for word, per_year in raw.items():
        out[word] = {}
        for year, embs in per_year.items():
            X = embs.astype(np.float32)
            norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
            out[word][year] = X / norms
    return out


# ── Probe ────────────────────────────────────────────────────────────────


def fit_probe(
    X_a: np.ndarray, X_b: np.ndarray, seed: int
) -> dict[str, float] | None:
    """Balance, 70/30 stratified split, L2-LogReg with CV'd C, held-out metrics.

    Returns None if either class has < 4 instances (can't stratify both
    splits and run an internal 5-fold CV).
    """
    n = min(X_a.shape[0], X_b.shape[0])
    if n < 4:
        return None
    rng = np.random.default_rng(seed)
    if X_a.shape[0] > n:
        idx = rng.choice(X_a.shape[0], size=n, replace=False)
        X_a = X_a[idx]
    if X_b.shape[0] > n:
        idx = rng.choice(X_b.shape[0], size=n, replace=False)
        X_b = X_b[idx]

    X = np.vstack([X_a, X_b])
    y = np.concatenate([np.zeros(n, dtype=np.int8), np.ones(n, dtype=np.int8)])

    # 70/30 stratified split.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )

    # Internal CV needs at least 2 samples per class per fold; cap folds at
    # min(5, smallest_class_in_train).
    n_per_class_train = int(min((y_tr == 0).sum(), (y_tr == 1).sum()))
    cv = min(5, n_per_class_train)
    if cv < 2:
        return None

    clf = LogisticRegressionCV(
        Cs=10, l1_ratios=(0,), cv=cv, scoring="roc_auc",
        max_iter=1000, n_jobs=1, random_state=seed,
        use_legacy_attributes=False,
    )
    clf.fit(X_tr, y_tr)
    scores = clf.decision_function(X_te)
    preds = clf.predict(X_te)
    auc = float(roc_auc_score(y_te, scores))
    acc = float(accuracy_score(y_te, preds))
    n_test_per_class = int((y_te == 0).sum())
    c_chosen = float(np.atleast_1d(clf.C_)[0])
    return {
        "auc": auc,
        "acc": acc,
        "n_per_class": int(n),
        "n_test_per_class": n_test_per_class,
        "c_chosen": c_chosen,
    }


# ── Sweeps ───────────────────────────────────────────────────────────────


def sweep_consecutive_pairs(
    word: str,
    per_year: dict[int, np.ndarray],
    years: list[int],
    seed: int,
) -> pd.DataFrame:
    """Fit a probe on each consecutive year-pair where both years have data."""
    rows: list[dict] = []
    for ya, yb in zip(years[:-1], years[1:]):
        if ya not in per_year or yb not in per_year:
            continue
        X_a, X_b = per_year[ya], per_year[yb]
        # Seed varies by pair so subsampling differs across pairs but is
        # reproducible.
        pair_seed = seed + ya
        result = fit_probe(X_a, X_b, seed=pair_seed)
        if result is None:
            continue
        rows.append({
            "word": word,
            "year_a": ya,
            "year_b": yb,
            "n_a_raw": int(X_a.shape[0]),
            "n_b_raw": int(X_b.shape[0]),
            "n_per_class": result["n_per_class"],
            "n_test_per_class": result["n_test_per_class"],
            "acc": result["acc"],
            "auc": result["auc"],
            "c_chosen": result["c_chosen"],
        })
    return pd.DataFrame(rows)


def permutation_null(
    word: str,
    per_year: dict[int, np.ndarray],
    years: list[int],
    n_perms: int,
    seed: int,
) -> pd.DataFrame:
    """Shuffle year labels across all of the word's embeddings, re-sweep."""
    available_years = [y for y in years if y in per_year]
    sizes = [per_year[y].shape[0] for y in available_years]
    X_all = np.vstack([per_year[y] for y in available_years])
    rng = np.random.default_rng(seed)

    frames: list[pd.DataFrame] = []
    pbar = tqdm(range(n_perms), desc=f"perm {word}", unit="perm")
    running_acc, running_auc = 0.0, 0.0
    for perm_id in pbar:
        order = rng.permutation(X_all.shape[0])
        X_shuffled = X_all[order]
        # Re-split into per-year arrays of the original sizes.
        shuffled_per_year: dict[int, np.ndarray] = {}
        offset = 0
        for y, n in zip(available_years, sizes):
            shuffled_per_year[y] = X_shuffled[offset:offset + n]
            offset += n

        df = sweep_consecutive_pairs(
            word, shuffled_per_year, years, seed=seed + 1000 * (perm_id + 1)
        )
        df["perm_id"] = perm_id
        frames.append(df)
        # Running mean across permutations seen so far.
        running_acc += (df["acc"].mean() - running_acc) / (perm_id + 1)
        running_auc += (df["auc"].mean() - running_auc) / (perm_id + 1)
        pbar.set_postfix(acc=f"{running_acc:.3f}", auc=f"{running_auc:.3f}")

    return pd.concat(frames, ignore_index=True)


# ── Plotting ─────────────────────────────────────────────────────────────


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

    # Rug plot, jittered along y, coloured by word.
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
    ax.legend(loc="upper right")
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
) -> None:
    apply_plot_style()
    colors = get_categorical_colors(len(words))

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=True)
    for ax, word, color in zip(axes.flat, words, colors):
        sub = df[df["word"] == word].sort_values("year_a")
        if sub.empty:
            ax.text(0.5, 0.5, f"no data: {word}",
                    transform=ax.transAxes, ha="center", va="center")
            ax.set_title(word)
            continue
        midpoints = sub["year_a"] + 0.5
        ax.plot(midpoints, sub[metric], color=color, linewidth=1.4, alpha=0.9)
        ax.scatter(midpoints, sub[metric], color=color, s=14, zorder=3)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.7)
        ax.set_title(word)
        ax.set_ylim(0.3, 1.02)
        ax.set_xlabel("year-pair midpoint")
        ax.set_ylabel(metric_label)

        # Annotate min(n_a, n_b) for each point in tiny text below the line.
        for _, row in sub.iterrows():
            ax.text(row["year_a"] + 0.5, row[metric] + 0.015,
                    str(int(row["n_per_class"])),
                    fontsize=6, ha="center", color="#555")
        remove_extra_spines(ax)

    fig.suptitle(
        f"Per-word linear-probe {metric_label} over consecutive year-pairs"
    )
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
) -> None:
    apply_plot_style()
    real = df_real[df_real["word"] == word].sort_values("year_a")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: histogram comparison.
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
    ax.set_title(f"{word}: permuted vs real {metric_label}")
    ax.legend(loc="upper right")
    remove_extra_spines(ax)

    # Right: trajectories.
    ax = axes[1]
    for perm_id, sub in df_perm.groupby("perm_id"):
        sub = sub.sort_values("year_a")
        ax.plot(sub["year_a"] + 0.5, sub[metric],
                color="#888888", linewidth=0.8, alpha=0.55)
    ax.plot(real["year_a"] + 0.5, real[metric],
            color="#d62728", linewidth=1.8, label="real")
    ax.scatter(real["year_a"] + 0.5, real[metric],
               color="#d62728", s=20, zorder=3)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.7)
    ax.set_xlabel("year-pair midpoint")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{word}: {metric_label} trajectory (real vs permuted)")
    ax.legend(loc="upper right")
    ax.set_ylim(0.3, 1.02)
    remove_extra_spines(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=1990)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--n-perms", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging("27_linear_probe_drift_calibration")
    logger = logging.getLogger(__name__)

    csv_dir = os.path.join(RESULTS_DIR, "bert", "linear_probe_calibration")
    fig_dir = os.path.join(FIGURES_DIR, "linear_probe_calibration")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    years = list(range(args.start, args.end + 1))

    logger.info(f"Loading embeddings for {TARGET_WORDS}, {len(years)} years")
    per_word = load_word_embeddings(BERT_EMBEDDINGS_DIR, TARGET_WORDS, years)
    for word in TARGET_WORDS:
        if word not in per_word:
            logger.warning(f"No embeddings for {word!r} — will be skipped")

    # Counts table.
    count_rows = []
    for word, per_year in per_word.items():
        for year, X in per_year.items():
            count_rows.append({"word": word, "year": year, "n": int(X.shape[0])})
    counts_df = pd.DataFrame(count_rows).sort_values(["word", "year"])
    counts_df.to_csv(os.path.join(csv_dir, "counts_per_year.csv"), index=False)
    logger.info(f"Wrote counts_per_year.csv ({len(counts_df)} rows)")

    # Real sweep.
    real_frames: list[pd.DataFrame] = []
    for word in TARGET_WORDS:
        if word not in per_word:
            continue
        logger.info(f"Real sweep: {word}")
        df_w = sweep_consecutive_pairs(
            word, per_word[word], years, seed=args.seed
        )
        logger.info(
            f"  {word}: {len(df_w)} year-pairs, "
            f"mean acc = {df_w['acc'].mean():.3f}, "
            f"mean AUC = {df_w['auc'].mean():.3f}"
        )
        real_frames.append(df_w)
    df_real = pd.concat(real_frames, ignore_index=True)
    df_real.to_csv(os.path.join(csv_dir, "metrics_real.csv"), index=False)
    logger.info(f"Wrote metrics_real.csv ({len(df_real)} rows)")

    # Permutation null on every target word.
    perm_per_word: dict[str, pd.DataFrame] = {}
    for word in TARGET_WORDS:
        if word not in per_word:
            continue
        logger.info(f"Permutation null: {word} ({args.n_perms} perms)")
        df_perm = permutation_null(
            word, per_word[word], years,
            n_perms=args.n_perms, seed=args.seed,
        )
        out_csv = os.path.join(csv_dir, f"metrics_perm_{word}.csv")
        df_perm.to_csv(out_csv, index=False)
        logger.info(
            f"Wrote {os.path.basename(out_csv)} ({len(df_perm)} rows); "
            f"mean permuted acc = {df_perm['acc'].mean():.3f}, "
            f"AUC = {df_perm['auc'].mean():.3f}"
        )
        perm_per_word[word] = df_perm

    # Figures — primary (precision/accuracy).
    plot_metric_distribution_real(
        df_real, TARGET_WORDS,
        os.path.join(fig_dir, "acc_distribution_real.png"),
        metric="acc", metric_label="accuracy",
    )
    plot_metric_over_time_per_word(
        df_real, TARGET_WORDS,
        os.path.join(fig_dir, "acc_over_time_per_word.png"),
        metric="acc", metric_label="accuracy",
    )
    for word, df_perm in perm_per_word.items():
        plot_permutation_null(
            df_real, df_perm, word,
            os.path.join(fig_dir, f"permutation_null_acc_{word}.png"),
            metric="acc", metric_label="accuracy",
        )

    # Figures — secondary (AUC, kept for continuity with prior calibration).
    plot_metric_distribution_real(
        df_real, TARGET_WORDS,
        os.path.join(fig_dir, "auc_distribution_real.png"),
        metric="auc", metric_label="AUC",
    )
    plot_metric_over_time_per_word(
        df_real, TARGET_WORDS,
        os.path.join(fig_dir, "auc_over_time_per_word.png"),
        metric="auc", metric_label="AUC",
    )
    for word, df_perm in perm_per_word.items():
        plot_permutation_null(
            df_real, df_perm, word,
            os.path.join(fig_dir, f"permutation_null_auc_{word}.png"),
            metric="auc", metric_label="AUC",
        )

    logger.info(f"Figures written to {fig_dir}")


if __name__ == "__main__":
    main()
