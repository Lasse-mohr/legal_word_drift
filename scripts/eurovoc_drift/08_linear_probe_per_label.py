"""Linear-probe drift sweep on EuroVoc labels with APD block structure.

Loads the shortlist from ``configs/eurovoc_block_structure.yaml``, selects a
named subset (default: ``clarity_spread``), and runs **two** classifiers on
the EuroVoc-drift embeddings for direct comparison:

  * ``logreg`` — L2-regularised logistic regression (the script-27 probe).
  * ``centroid`` — perpendicular bisector of the train-set class means
    (nearest-centroid / isotropic-LDA). Also reports
    ``w_norm = ||μ_b − μ_a||`` per pair so the centroid-shift magnitude is
    available alongside accuracy/AUC, both for the real labels and the
    permutation null.

Both classifiers see the same balanced 70/30 partitions on the same per-pair
seed, so paired comparisons are meaningful.

Outputs (under ``data/results/metrics/eurovoc_drift_linear_probe/<subset>/``):
  counts_per_year.csv
  subset_meta.csv                                       # label / clarity / domain
  metrics_real.csv                                      # both classifiers, classifier column
  metrics_perm_{label}_{classifier}.csv                 # one per (label, classifier)

Figures (under ``data/results/figures/eurovoc_drift/linear_probe/<subset>/``):
  acc_distribution_real_{classifier}.png
  acc_over_time_per_word_{classifier}.png
  permutation_null_acc_{label}_{classifier}.png
  auc_distribution_real_{classifier}.png
  auc_over_time_per_word_{classifier}.png
  permutation_null_auc_{label}_{classifier}.png
  w_norm_real_vs_null_{label}.png                       # centroid only
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.metrics.linear_probe import (
    FitFn,
    fit_centroid_probe,
    fit_probe,
    l2_normalize_per_year,
    permutation_null,
    sweep_consecutive_pairs,
)
from src.metrics.temporal_drift import load_per_year_embeddings
from src.paths import PATHS
from src.utils.config import setup_logging
from src.visualization.linear_probe_plots import (
    plot_metric_distribution_real,
    plot_metric_over_time_per_word,
    plot_permutation_null,
    plot_w_norm_over_time,
)


CONFIG_PATH = PATHS.project_root / "configs" / "eurovoc_block_structure.yaml"

CLASSIFIERS: dict[str, FitFn] = {
    "logreg": fit_probe,
    "centroid": fit_centroid_probe,
}


def load_subset(
    subset_name: str, labels_override: list[str] | None = None
) -> tuple[list[str], pd.DataFrame]:
    """Resolve a subset name (or explicit label list) to (labels, metadata).

    Metadata DataFrame has columns ``label, clarity, domain, notes``.
    """
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    clarity_tiers = [k for k in cfg.keys() if k != "subsets"]
    meta_rows: list[dict] = []
    for tier in clarity_tiers:
        for entry in cfg[tier] or []:
            meta_rows.append({
                "label": entry["label"],
                "clarity": tier,
                "domain": entry.get("domain", ""),
                "notes": entry.get("notes", "") or "",
            })
    meta_all = pd.DataFrame(meta_rows)

    if labels_override is not None:
        labels = list(labels_override)
    else:
        subsets = cfg.get("subsets", {})
        if subset_name not in subsets:
            raise KeyError(
                f"Subset {subset_name!r} not found in {CONFIG_PATH}. "
                f"Available: {list(subsets)}"
            )
        labels = list(subsets[subset_name]["labels"])

    missing = [w for w in labels if w not in set(meta_all["label"])]
    if missing:
        raise ValueError(
            f"Subset labels not present in tier listings: {missing}"
        )

    meta = meta_all.set_index("label").loc[labels].reset_index()
    return labels, meta


def run_sweeps(
    labels: list[str],
    per_word: dict[str, dict[int, "object"]],
    years: list[int],
    seed: int,
    n_perms: int,
    classifier: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run real sweep + permutation null for one classifier."""
    fit_fn = CLASSIFIERS[classifier]

    logger.info(f"=== Classifier: {classifier} ===")

    real_frames: list[pd.DataFrame] = []
    for label in labels:
        if label not in per_word:
            continue
        logger.info(f"  real sweep: {label}")
        df_w = sweep_consecutive_pairs(
            label, per_word[label], years, seed=seed, fit_fn=fit_fn,
        )
        if df_w.empty:
            logger.warning(f"    {label}: no valid year-pairs")
        else:
            msg = (
                f"    {label}: {len(df_w)} pairs, "
                f"mean acc = {df_w['acc'].mean():.3f}, "
                f"mean AUC = {df_w['auc'].mean():.3f}"
            )
            if "w_norm" in df_w.columns:
                msg += f", mean w_norm = {df_w['w_norm'].mean():.3f}"
            logger.info(msg)
        real_frames.append(df_w)
    df_real = pd.concat(real_frames, ignore_index=True)
    df_real["classifier"] = classifier

    perm_per_word: dict[str, pd.DataFrame] = {}
    for label in labels:
        if label not in per_word:
            continue
        logger.info(f"  permutation null ({n_perms} perms): {label}")
        df_perm = permutation_null(
            label, per_word[label], years,
            n_perms=n_perms, seed=seed, fit_fn=fit_fn,
        )
        df_perm["classifier"] = classifier
        perm_per_word[label] = df_perm

    return df_real, perm_per_word


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=str, default="clarity_spread",
                        help="Named subset in configs/eurovoc_block_structure.yaml")
    parser.add_argument("--labels", type=str, default=None,
                        help="Comma-separated label list to override --subset.")
    parser.add_argument("--start", type=int, default=1990)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--n-perms", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--classifiers", type=str, default="logreg,centroid",
        help="Comma-separated list; subset of {logreg, centroid}.",
    )
    args = parser.parse_args()

    setup_logging("eurovoc_drift_08_linear_probe_per_label")
    logger = logging.getLogger(__name__)

    classifiers = [c.strip() for c in args.classifiers.split(",") if c.strip()]
    for c in classifiers:
        if c not in CLASSIFIERS:
            raise ValueError(f"Unknown classifier {c!r}; options: {list(CLASSIFIERS)}")

    labels_override = (
        [x.strip() for x in args.labels.split(",") if x.strip()]
        if args.labels else None
    )
    labels, meta = load_subset(args.subset, labels_override=labels_override)
    subset_tag = args.subset if labels_override is None else "custom"
    logger.info(f"Subset {subset_tag}: {labels}")
    logger.info(f"Classifiers: {classifiers}")

    csv_dir: Path = PATHS.eurovoc_drift_linear_probe_results / subset_tag
    fig_dir: Path = PATHS.eurovoc_drift_linear_probe_figures / subset_tag
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    meta.to_csv(csv_dir / "subset_meta.csv", index=False)
    logger.info(f"Wrote subset_meta.csv ({len(meta)} rows)")

    years = list(range(args.start, args.end + 1))
    embeddings_dir = str(PATHS.eurovoc_drift_embeddings)
    logger.info(f"Loading embeddings from {embeddings_dir}, {len(years)} years")
    raw = load_per_year_embeddings(
        embeddings_dir, years, words=set(labels), min_usages=1
    )
    per_word = l2_normalize_per_year(raw)
    for label in labels:
        if label not in per_word:
            logger.warning(f"No embeddings for {label!r} — will be skipped")

    count_rows = []
    for label, per_year in per_word.items():
        for year, X in per_year.items():
            count_rows.append({"label": label, "year": year, "n": int(X.shape[0])})
    counts_df = pd.DataFrame(count_rows).sort_values(["label", "year"])
    counts_df.to_csv(csv_dir / "counts_per_year.csv", index=False)
    logger.info(f"Wrote counts_per_year.csv ({len(counts_df)} rows)")

    real_all: list[pd.DataFrame] = []
    perm_by_clf: dict[str, dict[str, pd.DataFrame]] = {}
    for clf in classifiers:
        df_real, perm_per_word = run_sweeps(
            labels, per_word, years,
            seed=args.seed, n_perms=args.n_perms,
            classifier=clf, logger=logger,
        )
        real_all.append(df_real)
        perm_by_clf[clf] = perm_per_word

        for label, df_perm in perm_per_word.items():
            out_csv = csv_dir / f"metrics_perm_{label}_{clf}.csv"
            df_perm.to_csv(out_csv, index=False)
            mean_acc = df_perm["acc"].mean()
            mean_auc = df_perm["auc"].mean()
            msg = (
                f"Wrote {out_csv.name} ({len(df_perm)} rows); "
                f"null mean acc = {mean_acc:.3f}, AUC = {mean_auc:.3f}"
            )
            if "w_norm" in df_perm.columns:
                msg += f", w_norm = {df_perm['w_norm'].mean():.3f}"
            logger.info(msg)

    df_real_all = pd.concat(real_all, ignore_index=True)
    df_real_all = df_real_all.merge(
        meta.rename(columns={"label": "word"}), on="word", how="left"
    )
    df_real_all.to_csv(csv_dir / "metrics_real.csv", index=False)
    logger.info(f"Wrote metrics_real.csv ({len(df_real_all)} rows)")

    # ── Plots ────────────────────────────────────────────────────────────
    titles = {
        row["label"]: f"{row['label']} — {row['clarity']} / {row['domain']}"
        for _, row in meta.iterrows()
    }

    for clf in classifiers:
        df_real_clf = df_real_all[df_real_all["classifier"] == clf]
        perm_per_word = perm_by_clf[clf]

        plot_metric_distribution_real(
            df_real_clf, labels,
            str(fig_dir / f"acc_distribution_real_{clf}.png"),
            metric="acc", metric_label="accuracy",
        )
        plot_metric_over_time_per_word(
            df_real_clf, labels,
            str(fig_dir / f"acc_over_time_per_word_{clf}.png"),
            metric="acc", metric_label="accuracy",
            n_cols=2, panel_size=(5.5, 3.0), titles=titles,
        )
        plot_metric_distribution_real(
            df_real_clf, labels,
            str(fig_dir / f"auc_distribution_real_{clf}.png"),
            metric="auc", metric_label="AUC",
        )
        plot_metric_over_time_per_word(
            df_real_clf, labels,
            str(fig_dir / f"auc_over_time_per_word_{clf}.png"),
            metric="auc", metric_label="AUC",
            n_cols=2, panel_size=(5.5, 3.0), titles=titles,
        )
        for label, df_perm in perm_per_word.items():
            clarity = meta.set_index("label").loc[label, "clarity"]
            plot_permutation_null(
                df_real_clf, df_perm, label,
                str(fig_dir / f"permutation_null_acc_{label}_{clf}.png"),
                metric="acc", metric_label="accuracy",
                title_suffix=f" [{clf}, {clarity}]",
            )
            plot_permutation_null(
                df_real_clf, df_perm, label,
                str(fig_dir / f"permutation_null_auc_{label}_{clf}.png"),
                metric="auc", metric_label="AUC",
                title_suffix=f" [{clf}, {clarity}]",
            )

    # w_norm plots — centroid classifier only.
    if "centroid" in classifiers:
        df_real_centroid = df_real_all[df_real_all["classifier"] == "centroid"]
        for label, df_perm in perm_by_clf["centroid"].items():
            clarity = meta.set_index("label").loc[label, "clarity"]
            plot_w_norm_over_time(
                df_real_centroid, df_perm, label,
                str(fig_dir / f"w_norm_real_vs_null_{label}.png"),
                title_suffix=f" ({clarity})",
            )

    logger.info(f"Figures written to {fig_dir}")


if __name__ == "__main__":
    main()
