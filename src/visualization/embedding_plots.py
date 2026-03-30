"""Stacked embedding visualization via PCA and aligned UMAP.

Stacks aligned word vectors from all time slices into one matrix,
reduces to 2D, and plots word trajectories through semantic space
over time.  Each (word, time_slice) pair is a point; same-word points
are connected with lines to show drift trajectories.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def stack_embeddings(
    aligned_kvs: dict[str, KeyedVectors],
    words: Sequence[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Stack aligned word vectors across all time slices.

    For each (word, time_slice) pair where the word exists, extract
    the vector. Returns:
        - matrix: (N, dim) array of stacked vectors
        - word_labels: length-N list of word strings
        - time_labels: length-N list of time-slice labels
    """
    vectors = []
    word_labels = []
    time_labels = []

    for label in sorted(aligned_kvs.keys()):
        kv = aligned_kvs[label]
        for word in words:
            if word in kv:
                vectors.append(kv[word])
                word_labels.append(word)
                time_labels.append(label)

    matrix = np.array(vectors) if vectors else np.empty((0, 0))
    return matrix, word_labels, time_labels


def reduce_pca(matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Reduce embedding matrix to n_components dimensions via PCA."""
    pca = PCA(n_components=n_components)
    return pca.fit_transform(matrix)


def reduce_umap(
    matrix: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """Reduce embedding matrix to n_components dimensions via UMAP."""
    from umap import UMAP
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(matrix)


def _time_label_to_float(label: str) -> float:
    """Convert a window label like 'w1990_1994' or 'y2005' to a float year."""
    if label.startswith("w"):
        parts = label[1:].split("_")
        return (int(parts[0]) + int(parts[1])) / 2.0
    elif label.startswith("y"):
        return float(label[1:])
    else:
        try:
            return float(label)
        except ValueError:
            return 0.0


def plot_trajectories(
    coords_2d: np.ndarray,
    word_labels: list[str],
    time_labels: list[str],
    words_to_plot: Sequence[str] | None = None,
    method: str = "PCA",
    figsize: tuple[float, float] = (14, 10),
    show_labels: bool = True,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot word trajectories through 2D embedding space over time.

    Each word gets a colored line connecting its positions across time
    slices. Points are colored by time (early=light, late=dark).

    Args:
        coords_2d: (N, 2) array from reduce_pca or reduce_umap.
        word_labels: length-N word strings.
        time_labels: length-N time-slice labels.
        words_to_plot: Subset of words to plot. None = plot all.
        method: Label for axes ("PCA" or "UMAP").
        figsize: Figure size.
        show_labels: Whether to annotate start/end points with word names.
        save_path: If provided, save figure to this path.
    """
    unique_words = sorted(set(word_labels))
    if words_to_plot is not None:
        unique_words = [w for w in unique_words if w in set(words_to_plot)]

    # Assign a color to each word
    n_words = len(unique_words)
    word_colors = {w: cm.tab20(i % 20) for i, w in enumerate(unique_words)}

    # Sort time labels for ordering
    all_times = sorted(set(time_labels), key=_time_label_to_float)
    time_to_idx = {t: i for i, t in enumerate(all_times)}
    n_times = len(all_times)

    fig, ax = plt.subplots(figsize=figsize)

    for word in unique_words:
        # Gather this word's points, sorted by time
        indices = [
            i for i, (w, t) in enumerate(zip(word_labels, time_labels))
            if w == word
        ]
        if len(indices) < 2:
            continue

        # Sort by time
        sorted_indices = sorted(indices, key=lambda i: time_to_idx[time_labels[i]])
        xs = coords_2d[sorted_indices, 0]
        ys = coords_2d[sorted_indices, 1]
        times = [time_labels[i] for i in sorted_indices]

        color = word_colors[word]

        # Plot trajectory line
        ax.plot(xs, ys, color=color, alpha=0.6, linewidth=1.5, zorder=1)

        # Plot points with time-based alpha (lighter = earlier)
        for j, (x, y, t) in enumerate(zip(xs, ys, times)):
            alpha = 0.3 + 0.7 * (time_to_idx[t] / max(n_times - 1, 1))
            ax.scatter(x, y, color=color, alpha=alpha, s=30, zorder=2,
                       edgecolors="white", linewidths=0.3)

        # Label start and end points
        if show_labels and len(xs) > 0:
            ax.annotate(
                word, (xs[-1], ys[-1]),
                fontsize=7, alpha=0.8,
                xytext=(4, 4), textcoords="offset points",
            )

    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.set_title(f"Word trajectories in {method} space ({all_times[0]} to {all_times[-1]})")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved to {save_path}")

    return fig


def plot_snapshot(
    kv: KeyedVectors,
    words: Sequence[str],
    method: str = "PCA",
    figsize: tuple[float, float] = (12, 8),
    save_path: str | None = None,
) -> plt.Figure:
    """Plot a single time slice's embedding in 2D.

    Useful for inspecting cluster structure at a single point in time.
    """
    present = [w for w in words if w in kv]
    if len(present) < 3:
        raise ValueError(f"Only {len(present)} words found in KeyedVectors")

    matrix = np.array([kv[w] for w in present])

    if method.upper() == "UMAP":
        coords = reduce_umap(matrix)
    else:
        coords = reduce_pca(matrix)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(coords[:, 0], coords[:, 1], s=20, alpha=0.6)

    for i, word in enumerate(present):
        ax.annotate(word, (coords[i, 0], coords[i, 1]),
                     fontsize=6, alpha=0.7,
                     xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.set_title(f"Embedding snapshot ({method})")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
