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

Core probe + plotting logic lives in ``src.metrics.linear_probe`` and
``src.visualization.linear_probe_plots`` so the same machinery can be reused
by the EuroVoc per-label sweep.

Outputs:
  data/results/bert/linear_probe_calibration/
    counts_per_year.csv
    metrics_real.csv                    # acc + auc, real labels
    metrics_perm_{word}.csv             # acc + auc, permuted labels (per word)
  data/results/figures/linear_probe_calibration/
    acc_distribution_real.png
    acc_over_time_per_word.png
    permutation_null_acc_{word}.png
    auc_distribution_real.png
    auc_over_time_per_word.png
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.metrics.linear_probe import (
    l2_normalize_per_year,
    permutation_null,
    sweep_consecutive_pairs,
)
from src.metrics.temporal_drift import load_per_year_embeddings
from src.utils.config import (
    BERT_EMBEDDINGS_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
    setup_logging,
)
from src.visualization.linear_probe_plots import (
    plot_metric_distribution_real,
    plot_metric_over_time_per_word,
    plot_permutation_null,
)


TARGET_WORDS = ["organization", "general", "citizen", "referring"]


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
    raw = load_per_year_embeddings(
        BERT_EMBEDDINGS_DIR, years, words=set(TARGET_WORDS), min_usages=1
    )
    per_word = l2_normalize_per_year(raw)
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

    plot_metric_distribution_real(
        df_real, TARGET_WORDS,
        os.path.join(fig_dir, "acc_distribution_real.png"),
        metric="acc", metric_label="accuracy",
    )
    plot_metric_over_time_per_word(
        df_real, TARGET_WORDS,
        os.path.join(fig_dir, "acc_over_time_per_word.png"),
        metric="acc", metric_label="accuracy",
        n_cols=2, panel_size=(5.0, 3.0),
    )
    for word, df_perm in perm_per_word.items():
        plot_permutation_null(
            df_real, df_perm, word,
            os.path.join(fig_dir, f"permutation_null_acc_{word}.png"),
            metric="acc", metric_label="accuracy",
        )

    plot_metric_distribution_real(
        df_real, TARGET_WORDS,
        os.path.join(fig_dir, "auc_distribution_real.png"),
        metric="auc", metric_label="AUC",
    )
    plot_metric_over_time_per_word(
        df_real, TARGET_WORDS,
        os.path.join(fig_dir, "auc_over_time_per_word.png"),
        metric="auc", metric_label="AUC",
        n_cols=2, panel_size=(5.0, 3.0),
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
