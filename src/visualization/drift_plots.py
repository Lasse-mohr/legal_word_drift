"""Hamilton-style drift visualization using PCA or UMAP.

For each target word, computes a 2D embedding where background neighbor
words are fixed at their positions from the most recent time slice, and
the target word's position is computed per time slice as the centroid of
its k'-nearest neighbors' fixed positions.

This follows the procedure from Hamilton et al. (2016), "Diachronic Word
Embeddings Reveal Statistical Laws of Semantic Change", adapted to use
PCA or UMAP instead of t-SNE.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA

from src.visualization.plot_config import get_sequential_cmap, remove_extra_spines

logger = logging.getLogger(__name__)


def _time_label_to_float(label: str) -> float:
    """Convert a window label like 'w1990_1994' or 'y2005' to a float year."""
    if label.startswith("w"):
        parts = label[1:].split("_")
        return (int(parts[0]) + int(parts[1])) / 2.0
    elif label.startswith("y"):
        return float(label[1:])
    return 0.0


def get_neighbor_union(
    word: str,
    aligned_kvs: dict[str, KeyedVectors],
    k: int = 10,
) -> list[str]:
    """Find the union of word's k nearest neighbors across all time slices."""
    neighbors = set()
    for kv in aligned_kvs.values():
        if word in kv:
            for w, _ in kv.most_similar(word, topn=k):
                neighbors.add(w)
    neighbors.discard(word)
    return sorted(neighbors)


def compute_background_positions(
    neighbor_words: list[str],
    reference_kv: KeyedVectors,
    method: str = "pca",
) -> dict[str, np.ndarray]:
    """Compute 2D positions for background words using the most recent time slice.

    Args:
        neighbor_words: Words to embed.
        reference_kv: KeyedVectors from the most recent time slice.
        method: "pca" or "umap".

    Returns:
        Dict mapping word -> 2D position array.
    """
    present = [w for w in neighbor_words if w in reference_kv]
    if len(present) < 3:
        return {}

    matrix = np.array([reference_kv[w] for w in present])

    if method == "umap":
        from umap import UMAP
        n_neighbors = min(15, len(present) - 1)
        reducer = UMAP(
            n_components=2, n_neighbors=n_neighbors,
            min_dist=0.1, metric="cosine", random_state=42,
        )
        coords = reducer.fit_transform(matrix)
    else:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(matrix)

    return {w: coords[i] for i, w in enumerate(present)}


def compute_target_position(
    target_word: str,
    time_slice_kv: KeyedVectors,
    background_positions: dict[str, np.ndarray],
    neighbor_words: list[str],
    k_prime: int = 5,
) -> np.ndarray | None:
    """Compute the target word's 2D position for one time slice.

    Uses the centroid of the k'-nearest neighbors (from the background set)
    in the high-dimensional embedding space of this time slice, mapped to
    their fixed 2D positions.
    """
    if target_word not in time_slice_kv:
        return None

    # Find neighbors present in both the time slice and the background
    candidates = [w for w in neighbor_words if w in time_slice_kv and w in background_positions]
    if not candidates:
        return None

    # Get similarities to target in this time slice
    target_vec = time_slice_kv[target_word]
    sims = []
    for w in candidates:
        w_vec = time_slice_kv[w]
        cos_sim = np.dot(target_vec, w_vec) / (
            np.linalg.norm(target_vec) * np.linalg.norm(w_vec) + 1e-10
        )
        sims.append((w, cos_sim))

    # Take top k' by similarity
    sims.sort(key=lambda x: x[1], reverse=True)
    top_neighbors = sims[:k_prime]

    # Centroid of their fixed 2D positions, weighted by similarity
    positions = np.array([background_positions[w] for w, _ in top_neighbors])
    weights = np.array([s for _, s in top_neighbors])
    weights = np.maximum(weights, 0)  # clip negative similarities
    weight_sum = weights.sum()
    if weight_sum < 1e-10:
        return positions.mean(axis=0)
    return np.average(positions, axis=0, weights=weights)


def compute_word_trajectory(
    target_word: str,
    aligned_kvs: dict[str, KeyedVectors],
    k: int = 10,
    k_prime: int = 5,
    k_label: int = 3,
    method: str = "pca",
) -> dict:
    """Compute full trajectory data for one word.

    Returns dict with:
        - 'target_word': str
        - 'neighbor_words': list[str]
        - 'background_positions': dict[str, ndarray]
        - 'trajectory': dict[str, ndarray] (time_label -> 2D position)
        - 'label_words': set[str] — words that are among the top k_label
          nearest neighbors in at least one time slice (for plot labels)
    """
    neighbor_words = get_neighbor_union(target_word, aligned_kvs, k=k)

    # Use most recent time slice as reference
    time_labels = sorted(aligned_kvs.keys())
    reference_kv = aligned_kvs[time_labels[-1]]

    background_positions = compute_background_positions(
        neighbor_words, reference_kv, method=method,
    )

    trajectory = {}
    label_words = set()
    for label in time_labels:
        kv = aligned_kvs[label]
        pos = compute_target_position(
            target_word, kv,
            background_positions, neighbor_words, k_prime=k_prime,
        )
        if pos is not None:
            trajectory[label] = pos
        # Collect top k_label neighbors in this time slice
        if target_word in kv:
            for w, _ in kv.most_similar(target_word, topn=k_label):
                if w in background_positions:
                    label_words.add(w)

    return {
        "target_word": target_word,
        "neighbor_words": neighbor_words,
        "background_positions": background_positions,
        "trajectory": trajectory,
        "label_words": label_words,
    }


def plot_word_drift_subplot(
    ax: plt.Axes,
    trajectory_data: dict,
    time_labels: list[str],
    cmap,
) -> None:
    """Plot one word's drift trajectory on a subplot."""
    bg = trajectory_data["background_positions"]
    traj = trajectory_data["trajectory"]
    word = trajectory_data["target_word"]

    if not bg or not traj:
        ax.set_title(word, fontsize=9)
        ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="grey")
        return

    # Background words as grey dots
    bg_words = list(bg.keys())
    bg_coords = np.array([bg[w] for w in bg_words])
    ax.scatter(bg_coords[:, 0], bg_coords[:, 1], c="#BBBBBB", s=12, alpha=0.5, zorder=1)

    # Label background words that are top-k nearest to the target in any time slice
    label_words = trajectory_data.get("label_words", set())
    for i, w in enumerate(bg_words):
        if w in label_words:
            ax.annotate(
                w, bg_coords[i],
                fontsize=5, alpha=0.6, color="#666666",
                xytext=(2, 2), textcoords="offset points",
            )

    # Target word trajectory
    sorted_labels = [l for l in time_labels if l in traj]
    if len(sorted_labels) < 2:
        ax.set_title(word, fontsize=9)
        return

    positions = np.array([traj[l] for l in sorted_labels])
    n_points = len(sorted_labels)
    colors = [cmap(i / max(n_points - 1, 1)) for i in range(n_points)]

    # Draw trajectory line
    ax.plot(positions[:, 0], positions[:, 1], color="#333333", alpha=0.3,
            linewidth=1, zorder=2)

    # Draw points colored by time
    for i, (pos, color) in enumerate(zip(positions, colors)):
        ax.scatter(pos[0], pos[1], c=[color], s=40, zorder=3,
                   edgecolors="white", linewidths=0.5)

    # Mark start and end
    ax.annotate(
        sorted_labels[0], positions[0],
        fontsize=5, fontweight="bold", color=colors[0],
        xytext=(-8, -8), textcoords="offset points",
    )
    ax.annotate(
        sorted_labels[-1], positions[-1],
        fontsize=5, fontweight="bold", color=colors[-1],
        xytext=(4, 4), textcoords="offset points",
    )

    ax.set_title(word, fontsize=9, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    remove_extra_spines(ax)


def plot_drift_grid(
    trajectory_data_list: list[dict],
    time_labels: list[str],
    ncols: int = 5,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot a grid of drift subplots for multiple words."""
    n = len(trajectory_data_list)
    nrows = max(1, (n + ncols - 1) // ncols)
    cmap = get_sequential_cmap()

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 3.5, nrows * 3.5),
    )
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for i, traj_data in enumerate(trajectory_data_list):
        row, col = divmod(i, ncols)
        plot_word_drift_subplot(axes[row, col], traj_data, time_labels, cmap)

    # Hide unused subplots
    for i in range(n, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved drift grid to {save_path}")

    return fig
