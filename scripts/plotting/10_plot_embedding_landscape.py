"""Plot a 2D embedding landscape of the most frequent words with drift trajectories.

Projects the top-N most frequent words from the last time slice into 2D using
UMAP, then produces one figure per top-drifting word, each showing the same
background scatter with cluster labels and a single word's trajectory overlaid.

Depends on:
    - Aligned embeddings in data/models/aligned/
    - frequency_drift.parquet from 07b_frequency_adjusted_drift
    - word_frequencies.parquet from 07b_frequency_adjusted_drift

Produces:
    - data/results/figures/landscape_trajectories/<word>.png  (one per drifter)

Usage:
    python -m src.pipeline.10_plot_embedding_landscape [--top-n 15000] [--m 20]
"""
from __future__ import annotations

import argparse
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from umap import UMAP

from sklearn.cluster import HDBSCAN

from src.embeddings.alignment import load_aligned
from src.metrics.clustering import LEGAL_DOMAINS
from src.visualization.plot_config import apply_plot_style, get_sequential_cmap
from src.utils.config import ALIGNED_DIR, METRICS_DIR, FIGURES_DIR, setup_logging

setup_logging("10_plot_embedding_landscape")
logger = logging.getLogger(__name__)


def _time_label_to_float(label: str) -> float:
    """Convert a window label like 'w1990_1994' to a midpoint float year."""
    if label.startswith("w"):
        parts = label[1:].split("_")
        return (int(parts[0]) + int(parts[1])) / 2.0
    elif label.startswith("y"):
        return float(label[1:])
    return 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Plot 2D embedding landscape with drift trajectories"
    )
    parser.add_argument("--top-n", type=int, default=15000,
                        help="Number of most frequent words to include in background")
    parser.add_argument("--m", type=int, default=20,
                        help="Number of top drifters whose trajectories to show")
    parser.add_argument("--umap-neighbors", type=int, default=15,
                        help="UMAP n_neighbors parameter")
    parser.add_argument("--umap-min-dist", type=float, default=0.3,
                        help="UMAP min_dist parameter")
    parser.add_argument("--min-cluster", type=int, default=50,
                        help="Minimum cluster size for labeling")
    args = parser.parse_args()

    apply_plot_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    logger.info("Loading aligned embeddings...")
    aligned_kvs = load_aligned(ALIGNED_DIR)
    time_labels = sorted(aligned_kvs.keys())
    last_label = time_labels[-1]
    ref_kv = aligned_kvs[last_label]

    logger.info("Loading frequency and drift data...")
    freq_df = pd.read_parquet(os.path.join(METRICS_DIR, "word_frequencies.parquet"))
    drift_df = pd.read_parquet(os.path.join(METRICS_DIR, "frequency_drift.parquet"))

    # ── Select top-N most frequent words (by total count across all slices) ──
    total_freq = freq_df.groupby("word")["count"].sum().reset_index()
    total_freq = total_freq.sort_values("count", ascending=False)

    # Keep only words present in the reference embedding
    ref_vocab = set(ref_kv.key_to_index.keys())
    total_freq = total_freq[total_freq["word"].isin(ref_vocab)]
    bg_words = total_freq.head(args.top_n)["word"].tolist()
    logger.info(f"Background: {len(bg_words)} most frequent words (of {len(ref_vocab)} in vocab)")

    # ── Get top-m drifters ───────────────────────────────────────────────
    top_drifters = (
        drift_df.sort_values("adjusted_drift", ascending=False)
        .head(args.m)["word"]
        .tolist()
    )
    # Filter to words present in reference
    top_drifters = [w for w in top_drifters if w in ref_vocab]
    logger.info(f"Plotting trajectories for {len(top_drifters)} top drifters")

    # ── Build embedding matrix for UMAP ──────────────────────────────────
    # Include all background words + all drifter words (union, no duplicates)
    all_words = list(dict.fromkeys(bg_words + top_drifters))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    matrix = np.array([ref_kv[w] for w in all_words])

    logger.info(f"Running UMAP on {matrix.shape[0]} words ({matrix.shape[1]}D -> 2D)...")
    reducer = UMAP(
        n_components=2,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric="cosine",
        random_state=42,
    )
    coords_2d = reducer.fit_transform(matrix)

    # ── Project drifter trajectories into the fitted UMAP space ──────────
    # For each time slice, transform the drifter vectors through the fitted UMAP
    logger.info("Projecting drifter trajectories across time slices...")
    # drifter_trajectories[word] = list of (year_float, x, y)
    drifter_trajectories = {w: [] for w in top_drifters}

    for label in time_labels:
        kv = aligned_kvs[label]
        year = _time_label_to_float(label)
        # Collect vectors for drifters present in this time slice
        words_this_slice = [w for w in top_drifters if w in kv]
        if not words_this_slice:
            continue
        vecs = np.array([kv[w] for w in words_this_slice])
        projected = reducer.transform(vecs)
        for w, xy in zip(words_this_slice, projected):
            drifter_trajectories[w].append((year, xy[0], xy[1]))

    # ── Cluster background words for labeling ──────────────────────────
    logger.info("Clustering background words with HDBSCAN...")
    bg_only_words = [w for w in bg_words if w not in set(top_drifters)]
    bg_only_indices = [word_to_idx[w] for w in bg_only_words]
    bg_xy = coords_2d[bg_only_indices]

    # Build word -> total frequency lookup
    total_freq_map = dict(zip(total_freq["word"], total_freq["count"]))

    clusterer = HDBSCAN(min_cluster_size=args.min_cluster)
    cluster_labels = clusterer.fit_predict(bg_xy)
    n_clusters = len(set(cluster_labels) - {-1})
    logger.info(f"Found {n_clusters} clusters (min_cluster_size={args.min_cluster})")

    # Build inverted domain lookup: word -> set of domain names
    domain_lookup: dict[str, set[str]] = {}
    for domain_name, domain_words in LEGAL_DOMAINS.items():
        for w in domain_words:
            domain_lookup.setdefault(w, set()).add(domain_name)

    # For each cluster, determine a label
    cluster_annotations: list[tuple[float, float, str]] = []
    for cid in sorted(set(cluster_labels) - {-1}):
        mask = cluster_labels == cid
        cluster_words = [bg_only_words[i] for i in range(len(bg_only_words)) if mask[i]]
        cluster_xy = bg_xy[mask]

        # Count domain hits: how many words from each domain appear in this cluster
        domain_hits: dict[str, int] = {}
        for w in cluster_words:
            for d in domain_lookup.get(w, []):
                domain_hits[d] = domain_hits.get(d, 0) + 1

        # Use domain name if ≥3 words from that domain land in this cluster
        best_domain = None
        best_count = 0
        for d, count in domain_hits.items():
            if count >= 3 and count > best_count:
                best_domain = d
                best_count = count

        if best_domain is not None:
            label = best_domain.replace("_", " ").title()
        else:
            # Fall back to the most frequent word in the cluster
            most_freq_word = max(cluster_words, key=lambda w: total_freq_map.get(w, 0))
            label = most_freq_word.replace("_", " ")

        centroid = cluster_xy.mean(axis=0)
        cluster_annotations.append((centroid[0], centroid[1], label))

    logger.info(f"Labeling {len(cluster_annotations)} clusters")

    # ── Plot one figure per drifter word ────────────────────────────────
    cmap = get_sequential_cmap()
    year_min = _time_label_to_float(time_labels[0])
    year_max = _time_label_to_float(time_labels[-1])

    out_dir = os.path.join(FIGURES_DIR, "landscape_trajectories")
    os.makedirs(out_dir, exist_ok=True)

    for wi, word in enumerate(top_drifters):
        pts = drifter_trajectories[word]
        if len(pts) < 2:
            logger.info(f"  Skipping {word} (insufficient trajectory points)")
            continue

        logger.info(f"  [{wi+1}/{len(top_drifters)}] Plotting {word}...")
        fig, ax = plt.subplots(figsize=(16, 12))

        # Background scatter
        ax.scatter(bg_xy[:, 0], bg_xy[:, 1], c="#CCCCCC", s=3, alpha=0.3,
                   zorder=1, rasterized=True)

        # Cluster labels
        for cx, cy, label in cluster_annotations:
            ax.annotate(
                label, (cx, cy),
                fontsize=9, fontweight="bold", color="#444444", alpha=0.8,
                ha="center", va="center", zorder=2,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#AAAAAA",
                          alpha=0.75, linewidth=0.5),
            )

        # Single trajectory
        pts.sort(key=lambda p: p[0])
        years = [p[0] for p in pts]
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        colors = [cmap((y - year_min) / max(year_max - year_min, 1)) for y in years]

        ax.plot(xs, ys, color="#555555", alpha=0.4, linewidth=1.2, zorder=2)

        for x, y, c in zip(xs, ys, colors):
            ax.scatter(x, y, c=[c], s=40, zorder=3, edgecolors="white", linewidths=0.5)

        # Start marker
        ax.scatter(xs[0], ys[0], c=[colors[0]], s=80, zorder=4,
                   edgecolors="black", linewidths=0.8, marker="o")

        # Label at final position
        ax.annotate(
            word.replace("_", " "),
            (xs[-1], ys[-1]),
            fontsize=10, fontweight="bold", color="#222222",
            xytext=(6, 6), textcoords="offset points",
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
        )

        # Colorbar
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=year_min, vmax=year_max),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02, aspect=30)
        cbar.set_label("Year (window midpoint)", fontsize=12, fontweight="bold")
        cbar.ax.tick_params(labelsize=10)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_title(
            f"Semantic drift: {word.replace('_', ' ')}",
            fontsize=14, fontweight="bold",
        )

        save_path = os.path.join(out_dir, f"{word}.png")
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    logger.info(f"Saved {len(top_drifters)} plots to {out_dir}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
