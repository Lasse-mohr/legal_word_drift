"""Linear-probe AUC with top-k PCA components removed (k ∈ {0,1,2,3}).

Companion to script 27. The hypothesis under test: a small number of dominant
PCA directions in BERT embeddings carry corpus-level / frequency-driven
variance ("all-but-the-top", Mu & Viswanath 2018). If those directions are
also what the C2ST probe latches onto, then removing them should drop AUC
toward 0.5; if drift signal lives in the trailing directions, AUC should be
robust or even sharpen.

Pipeline per k:
  - Per word, fit PCA on the L2-normalised embeddings stacked across all
    available years (unsupervised, no label leakage).
  - Project out the top-k principal components, then re-normalise to unit
    length.
  - Run the same real sweep + permutation null on `organization` as in 27.

Outputs (under data/results/bert/linear_probe_pca_removal/ and
data/results/figures/linear_probe_pca_removal/):
  auc_real.csv                 — long-format with `k_removed` column
  auc_perm_organization.csv    — long-format with `k_removed` column
  counts_per_year.csv
  auc_over_time_per_word_k{0..3}.png
  permutation_null_organization_k{0..3}.png
  auc_summary_by_k.png         — per-word mean AUC vs k
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

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
PERM_WORD = "organization"
K_VALUES = [0, 1, 2, 3]


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


# ── PCA removal ──────────────────────────────────────────────────────────


def remove_top_k_components(
    per_year: dict[int, np.ndarray], k: int
) -> dict[int, np.ndarray]:
    """Remove top-k PCs (fit on the word's pooled embeddings) and renormalise.

    PCA is unsupervised — fit on the union across years has no label leakage.
    Returns L2-renormalised embeddings; if k == 0, returns inputs unchanged.
    """
    if k == 0:
        return per_year

    years = sorted(per_year.keys())
    sizes = [per_year[y].shape[0] for y in years]
    X_all = np.vstack([per_year[y] for y in years])

    pca = PCA(n_components=k, svd_solver="auto", random_state=0)
    pca.fit(X_all)
    # Project out top-k: x' = x - sum_i (x · v_i) v_i, equivalently
    # x' = x - (x - mean) P P^T  where P = components_.T.
    # PCA centers internally; for cosine-style embeddings we keep the mean
    # in (don't recentre), only strip directions.
    P = pca.components_  # shape (k, d)
    # Per-direction projection: for each component, subtract proj.
    # X_proj = X - (X @ P.T) @ P
    proj = (X_all @ P.T) @ P
    X_resid = X_all - proj
    norms = np.linalg.norm(X_resid, axis=1, keepdims=True) + 1e-10
    X_resid = X_resid / norms

    out: dict[int, np.ndarray] = {}
    offset = 0
    for y, n in zip(years, sizes):
        out[y] = X_resid[offset:offset + n]
        offset += n
    return out


# ── Probe ────────────────────────────────────────────────────────────────


def fit_probe_auc(
    X_a: np.ndarray, X_b: np.ndarray, seed: int
) -> dict[str, float] | None:
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

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )

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
    auc = float(roc_auc_score(y_te, scores))
    c_chosen = float(np.atleast_1d(clf.C_)[0])
    return {"auc": auc, "n_per_class": int(n), "c_chosen": c_chosen}


# ── Sweeps ───────────────────────────────────────────────────────────────


def sweep_consecutive_pairs(
    word: str,
    per_year: dict[int, np.ndarray],
    years: list[int],
    seed: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for ya, yb in zip(years[:-1], years[1:]):
        if ya not in per_year or yb not in per_year:
            continue
        X_a, X_b = per_year[ya], per_year[yb]
        pair_seed = seed + ya
        result = fit_probe_auc(X_a, X_b, seed=pair_seed)
        if result is None:
            continue
        rows.append({
            "word": word,
            "year_a": ya,
            "year_b": yb,
            "n_a_raw": int(X_a.shape[0]),
            "n_b_raw": int(X_b.shape[0]),
            "n_per_class": result["n_per_class"],
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
    logger = logging.getLogger(__name__)

    available_years = [y for y in years if y in per_year]
    sizes = [per_year[y].shape[0] for y in available_years]
    X_all = np.vstack([per_year[y] for y in available_years])
    rng = np.random.default_rng(seed)

    frames: list[pd.DataFrame] = []
    for perm_id in range(n_perms):
        order = rng.permutation(X_all.shape[0])
        X_shuffled = X_all[order]
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
        logger.info(
            f"    perm {perm_id + 1}/{n_perms}: "
            f"mean AUC = {df['auc'].mean():.3f}"
        )

    return pd.concat(frames, ignore_index=True)


# ── Plotting ─────────────────────────────────────────────────────────────


def plot_auc_over_time_per_word(
    df: pd.DataFrame, words: list[str], k: int, out_path: str
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
        ax.plot(midpoints, sub["auc"], color=color, linewidth=1.4, alpha=0.9)
        ax.scatter(midpoints, sub["auc"], color=color, s=14, zorder=3)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.7)
        ax.set_title(word)
        ax.set_ylim(0.3, 1.02)
        ax.set_xlabel("year-pair midpoint")
        ax.set_ylabel("AUC")
        for _, row in sub.iterrows():
            ax.text(row["year_a"] + 0.5, row["auc"] + 0.015,
                    str(int(row["n_per_class"])),
                    fontsize=6, ha="center", color="#555")
        remove_extra_spines(ax)

    fig.suptitle(f"Per-word linear-probe AUC (top-{k} PCs removed)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_permutation_null(
    df_real: pd.DataFrame,
    df_perm: pd.DataFrame,
    word: str,
    k: int,
    out_path: str,
) -> None:
    apply_plot_style()
    real = df_real[df_real["word"] == word].sort_values("year_a")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    bins = np.linspace(0.3, 1.0, 30)
    ax.hist(df_perm["auc"], bins=bins, color="#bbbbbb",
            edgecolor="white", linewidth=0.4, alpha=0.9,
            label=f"permuted (×{df_perm['perm_id'].nunique()})", density=True)
    ax.hist(real["auc"], bins=bins, color="#d62728",
            edgecolor="white", linewidth=0.4, alpha=0.7,
            label="real", density=True)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("held-out AUC")
    ax.set_ylabel("density")
    ax.set_title(f"{word}: permuted vs real (top-{k} PCs removed)")
    ax.legend(loc="upper right")
    remove_extra_spines(ax)

    ax = axes[1]
    for perm_id, sub in df_perm.groupby("perm_id"):
        sub = sub.sort_values("year_a")
        ax.plot(sub["year_a"] + 0.5, sub["auc"],
                color="#888888", linewidth=0.8, alpha=0.55)
    ax.plot(real["year_a"] + 0.5, real["auc"],
            color="#d62728", linewidth=1.8, label="real")
    ax.scatter(real["year_a"] + 0.5, real["auc"],
               color="#d62728", s=20, zorder=3)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.7)
    ax.set_xlabel("year-pair midpoint")
    ax.set_ylabel("AUC")
    ax.set_title(f"{word}: AUC trajectory (top-{k} PCs removed)")
    ax.legend(loc="upper right")
    ax.set_ylim(0.3, 1.02)
    remove_extra_spines(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary_by_k(
    df_real: pd.DataFrame,
    df_perm: pd.DataFrame,
    words: list[str],
    perm_word: str,
    out_path: str,
) -> None:
    apply_plot_style()
    colors = get_categorical_colors(len(words))
    word_color = dict(zip(words, colors))

    fig, ax = plt.subplots(figsize=(7, 4))
    for word in words:
        sub = df_real[df_real["word"] == word]
        if sub.empty:
            continue
        agg = sub.groupby("k_removed")["auc"].agg(["mean", "std", "count"])
        agg = agg.reindex(K_VALUES)
        ax.errorbar(
            agg.index, agg["mean"],
            yerr=agg["std"] / np.sqrt(agg["count"].clip(lower=1)),
            marker="o", color=word_color[word], label=word, linewidth=1.4,
            capsize=3,
        )

    if not df_perm.empty:
        agg_p = df_perm.groupby("k_removed")["auc"].agg(["mean", "std"])
        agg_p = agg_p.reindex(K_VALUES)
        ax.plot(agg_p.index, agg_p["mean"],
                marker="s", color="#888888", linestyle="--",
                label=f"perm null ({perm_word})")
        ax.fill_between(
            agg_p.index,
            agg_p["mean"] - agg_p["std"],
            agg_p["mean"] + agg_p["std"],
            color="#888888", alpha=0.15,
        )

    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.7)
    ax.set_xticks(K_VALUES)
    ax.set_xlabel("# top PCA components removed")
    ax.set_ylabel("mean AUC across year-pairs")
    ax.set_title("Linear-probe AUC vs PCA-component removal")
    ax.legend(loc="best", fontsize=8)
    remove_extra_spines(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=1990)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--n-perms", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging("29_linear_probe_pca_removal")
    logger = logging.getLogger(__name__)

    csv_dir = os.path.join(RESULTS_DIR, "bert", "linear_probe_pca_removal")
    fig_dir = os.path.join(FIGURES_DIR, "linear_probe_pca_removal")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    years = list(range(args.start, args.end + 1))

    logger.info(f"Loading embeddings for {TARGET_WORDS}, {len(years)} years")
    per_word = load_word_embeddings(BERT_EMBEDDINGS_DIR, TARGET_WORDS, years)
    for word in TARGET_WORDS:
        if word not in per_word:
            logger.warning(f"No embeddings for {word!r} — will be skipped")

    count_rows = []
    for word, per_year in per_word.items():
        for year, X in per_year.items():
            count_rows.append({"word": word, "year": year, "n": int(X.shape[0])})
    counts_df = pd.DataFrame(count_rows).sort_values(["word", "year"])
    counts_df.to_csv(os.path.join(csv_dir, "counts_per_year.csv"), index=False)
    logger.info(f"Wrote counts_per_year.csv ({len(counts_df)} rows)")

    real_all: list[pd.DataFrame] = []
    perm_all: list[pd.DataFrame] = []

    for k in K_VALUES:
        logger.info(f"=== k = {k} top PC(s) removed ===")
        per_word_k = {
            word: remove_top_k_components(per_year, k)
            for word, per_year in per_word.items()
        }

        # Real sweep.
        real_frames: list[pd.DataFrame] = []
        for word in TARGET_WORDS:
            if word not in per_word_k:
                continue
            logger.info(f"  Real sweep: {word}")
            df_w = sweep_consecutive_pairs(
                word, per_word_k[word], years, seed=args.seed
            )
            if not df_w.empty:
                logger.info(
                    f"    {word}: {len(df_w)} year-pairs, "
                    f"mean AUC = {df_w['auc'].mean():.3f}"
                )
            real_frames.append(df_w)
        df_real_k = pd.concat(real_frames, ignore_index=True)
        df_real_k["k_removed"] = k
        real_all.append(df_real_k)

        # Permutation null on PERM_WORD.
        if PERM_WORD in per_word_k:
            logger.info(f"  Permutation null: {PERM_WORD} ({args.n_perms} perms)")
            df_perm_k = permutation_null(
                PERM_WORD, per_word_k[PERM_WORD], years,
                n_perms=args.n_perms, seed=args.seed,
            )
            df_perm_k["k_removed"] = k
            perm_all.append(df_perm_k)
            logger.info(
                f"    mean permuted AUC (k={k}) = {df_perm_k['auc'].mean():.3f}"
            )

        # Per-k figures.
        plot_auc_over_time_per_word(
            df_real_k, TARGET_WORDS, k,
            os.path.join(fig_dir, f"auc_over_time_per_word_k{k}.png"),
        )
        if perm_all and perm_all[-1]["k_removed"].iloc[0] == k:
            plot_permutation_null(
                df_real_k, perm_all[-1], PERM_WORD, k,
                os.path.join(fig_dir, f"permutation_null_organization_k{k}.png"),
            )

    df_real = pd.concat(real_all, ignore_index=True)
    df_real.to_csv(os.path.join(csv_dir, "auc_real.csv"), index=False)
    logger.info(f"Wrote auc_real.csv ({len(df_real)} rows)")

    df_perm = (
        pd.concat(perm_all, ignore_index=True) if perm_all else pd.DataFrame()
    )
    if not df_perm.empty:
        df_perm.to_csv(
            os.path.join(csv_dir, "auc_perm_organization.csv"), index=False
        )
        logger.info(f"Wrote auc_perm_organization.csv ({len(df_perm)} rows)")

    plot_summary_by_k(
        df_real, df_perm, TARGET_WORDS, PERM_WORD,
        os.path.join(fig_dir, "auc_summary_by_k.png"),
    )

    # Per-(word, k) summary table.
    summary = (
        df_real.groupby(["word", "k_removed"])["auc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.to_csv(os.path.join(csv_dir, "auc_summary_by_k.csv"), index=False)
    logger.info("\n" + summary.to_string(index=False))
    logger.info(f"Figures written to {fig_dir}")


if __name__ == "__main__":
    main()
