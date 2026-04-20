"""Pilot polysemy analysis: within-period distance distributions.

For each word in configs/polysemy_pilot.yaml, collects up to N occurrences from
a single year (or a multi-year window via --window), extracts contextualized
BERT embeddings (pooled over layers 8-10), and computes:
  - pairwise cosine distance distribution
  - cosine distance from each embedding to the (renormalized) mean direction
  - silhouette at k=2 and Hartigan dip test on each distance distribution

The centroid-distance is preferable for unimodality testing: under a unimodal
Gaussian null its distribution is well-behaved, whereas pairwise distances of
high-D Gaussian samples can spuriously appear bimodal due to concentration of
measure.

Outputs:
  data/results/metrics/polysemy_pilot_{tag}.csv                  (per-word stats)
  data/results/metrics/polysemy_pilot_{tag}_dists.npz            (pairwise dists)
  data/results/metrics/polysemy_pilot_{tag}_centroid_dists.npz   (centroid dists)
  data/results/figures/polysemy_pilot_{tag}_pairwise_p{N}.png    (pairwise grid)
  data/results/figures/polysemy_pilot_{tag}_centroid_p{N}.png    (centroid grid)

Rerun with --plot-only to redraw figures from cached distances.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from random import Random

import diptest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import silhouette_score
from sklearn.mixture import BayesianGaussianMixture

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.embeddings.bert_encoder import (
    DEFAULT_LAYERS,
    encode_paragraphs,
    extract_embedding,
    load_model,
)
from src.embeddings.usage_collector import Usage, build_usage_index
from src.utils.config import (
    FIGURES_DIR,
    METRICS_DIR,
    PARAGRAPHS_DIR,
    PROJECT_ROOT,
    setup_logging,
)
from src.visualization.plot_config import (
    SEQUENTIAL_PALETTES,
    apply_plot_style,
    remove_extra_spines,
)

CATEGORIES = ("polysemous", "monosemantic", "extras", "frequency_sampled")
TOKEN_PATTERN = re.compile(r"[A-Za-z]+")
MIN_TOKEN_LEN = 3
GRID_ROWS = 5
GRID_COLS = 5


def load_pilot_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in CATEGORIES:
        cfg.setdefault(key, [])
    return cfg


def count_tokens(years: list[int]) -> Counter:
    """Unigram frequency table across paragraphs in the given year(s)."""
    logger = logging.getLogger(__name__)
    counts: Counter = Counter()
    for year in years:
        p = os.path.join(PARAGRAPHS_DIR, f"{year}.jsonl")
        if not os.path.exists(p):
            logger.warning(f"Missing paragraphs: {p}")
            continue
        logger.info(f"Counting tokens in {year}...")
        with open(p, encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                for para in doc.get("paragraphs", []):
                    if len(para) < 50:
                        continue
                    for tok in TOKEN_PATTERN.findall(para.lower()):
                        counts[tok] += 1
    return counts


def frequency_weighted_sample(
    counts: Counter,
    n: int,
    min_freq: int,
    excluded: set[str],
    seed: int,
) -> list[str]:
    """Sample n distinct words with probability proportional to frequency."""
    logger = logging.getLogger(__name__)
    pool = {
        w: c for w, c in counts.items()
        if c >= min_freq
        and len(w) >= MIN_TOKEN_LEN
        and w not in ENGLISH_STOP_WORDS
        and w not in excluded
    }
    logger.info(
        f"Candidate pool: {len(pool)} words (freq>={min_freq}, not stopword, not already listed)"
    )
    if len(pool) == 0:
        return []
    if len(pool) <= n:
        logger.warning(f"Pool has only {len(pool)} words; returning the whole pool.")
        return sorted(pool)
    words = np.array(list(pool.keys()))
    freqs = np.array([pool[w] for w in words], dtype=np.float64)
    probs = freqs / freqs.sum()
    rng = np.random.default_rng(seed)
    sampled = rng.choice(words, size=n, replace=False, p=probs)
    return sorted(sampled.tolist())


def write_config(cfg: dict, path: str) -> None:
    """Persist the config back to YAML (loses comments, preserves field order)."""
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def collect_usages(
    words: list[str],
    years: list[int],
    max_per_word: int,
    seed: int,
) -> dict[str, list[Usage]]:
    """Scan paragraph files across years and sample up to max_per_word."""
    logger = logging.getLogger(__name__)
    merged: dict[str, list[Usage]] = defaultdict(list)

    for year in years:
        paragraphs_path = os.path.join(PARAGRAPHS_DIR, f"{year}.jsonl")
        if not os.path.exists(paragraphs_path):
            logger.warning(f"Missing paragraphs for {year}: {paragraphs_path}")
            continue
        logger.info(f"Scanning {year}...")
        index = build_usage_index(paragraphs_path, words)
        for w, usages in index.items():
            merged[w].extend(usages)

    rng = Random(seed)
    sampled: dict[str, list[Usage]] = {}
    for w in words:
        usages = merged.get(w, [])
        if len(usages) > max_per_word:
            sampled[w] = rng.sample(usages, max_per_word)
        else:
            sampled[w] = list(usages)
        logger.info(f"  {w}: kept {len(sampled[w])} of {len(usages)} occurrences")
    return sampled


def encode_and_extract(
    sampled: dict[str, list[Usage]],
    years: list[int],
    batch_size: int,
    device: str,
) -> dict[str, np.ndarray]:
    """Encode paragraphs and extract per-word embedding arrays."""
    logger = logging.getLogger(__name__)
    model, tokenizer, dev = load_model(device=device)

    # Load unique paragraphs across all years in the window.
    needed: set[tuple[str, int]] = set()
    for usages in sampled.values():
        for u in usages:
            needed.add((u.celex, u.para_idx))

    paragraphs: dict[tuple[str, int], str] = {}

    for year in years:
        p = os.path.join(PARAGRAPHS_DIR, f"{year}.jsonl")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                celex = doc["celex"]
                for idx, text in enumerate(doc.get("paragraphs", [])):
                    if (celex, idx) in needed:
                        paragraphs[(celex, idx)] = text
    logger.info(f"Loaded {len(paragraphs)} unique paragraphs for encoding")

    keys = list(paragraphs.keys())
    texts = [paragraphs[k] for k in keys]
    encoded_list = encode_paragraphs(
        texts, model, tokenizer, dev, batch_size=batch_size, layers=DEFAULT_LAYERS
    )
    encoded_map = dict(zip(keys, encoded_list))

    word_embeddings: dict[str, list[np.ndarray]] = defaultdict(list)
    n_fail = 0
    for word, usages in sampled.items():
        for u in usages:
            enc = encoded_map.get((u.celex, u.para_idx))
            if enc is None:
                n_fail += 1
                continue
            emb = extract_embedding(enc, u.char_start, u.char_end)
            if emb is None:
                n_fail += 1
                continue
            word_embeddings[word].append(emb)
    if n_fail:
        logger.warning(f"{n_fail} extraction failures (truncation or offset mismatch)")

    return {w: np.stack(e).astype(np.float32) for w, e in word_embeddings.items() if e}


def collect_token_embeddings_for_pca(
    n_paragraphs: int,
    batch_size: int,
    device: str,
    seed: int,
    max_tokens: int = 200_000,
) -> np.ndarray:
    """Sample paragraphs uniformly across years, encode, gather token embeddings.

    Returns (N, hidden_dim) float32 array of non-special-token hidden states.
    """
    logger = logging.getLogger(__name__)
    rng = Random(seed)

    if not os.path.isdir(PARAGRAPHS_DIR):
        raise FileNotFoundError(f"No paragraphs directory: {PARAGRAPHS_DIR}")
    year_files = sorted(
        os.path.join(PARAGRAPHS_DIR, f) for f in os.listdir(PARAGRAPHS_DIR)
        if f.endswith(".jsonl")
    )
    if not year_files:
        raise FileNotFoundError(f"No .jsonl in {PARAGRAPHS_DIR}")

    counts_per_file: list[int] = []
    for fp in year_files:
        with open(fp, encoding="utf-8") as f:
            counts_per_file.append(sum(1 for _ in f))
    total_docs = sum(counts_per_file)
    logger.info(
        f"PCA sampling: {len(year_files)} year files, {total_docs} total docs"
    )

    per_file = [
        max(1, round(n_paragraphs * c / total_docs)) for c in counts_per_file
    ]

    sampled_paragraphs: list[str] = []
    for fp, n_to_sample in zip(year_files, per_file):
        paras_in_file: list[str] = []
        with open(fp, encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                for para in doc.get("paragraphs", []):
                    if len(para) >= 50:
                        paras_in_file.append(para)
        if len(paras_in_file) <= n_to_sample:
            sampled_paragraphs.extend(paras_in_file)
        else:
            sampled_paragraphs.extend(rng.sample(paras_in_file, n_to_sample))

    rng.shuffle(sampled_paragraphs)
    logger.info(f"PCA sampling: {len(sampled_paragraphs)} paragraphs to encode")

    model, tokenizer, dev = load_model(device=device)
    encoded_list = encode_paragraphs(
        sampled_paragraphs, model, tokenizer, dev,
        batch_size=batch_size, layers=DEFAULT_LAYERS,
    )

    chunks: list[np.ndarray] = []
    n_collected = 0
    for enc in encoded_list:
        keep = [i for i, (s, e) in enumerate(enc.offsets) if s != e]
        if not keep:
            continue
        chunks.append(enc.hidden_states[keep])
        n_collected += len(keep)
        if n_collected >= max_tokens:
            break

    embeddings = np.concatenate(chunks, axis=0).astype(np.float32)
    if embeddings.shape[0] > max_tokens:
        # Random subsample down to max_tokens for tractability.
        rng_np = np.random.default_rng(seed)
        idx = rng_np.choice(embeddings.shape[0], size=max_tokens, replace=False)
        embeddings = embeddings[idx]
    logger.info(f"PCA fit will use {embeddings.shape[0]} token embeddings")
    return embeddings


def fit_or_load_pca(
    cache_path: str,
    n_paragraphs: int,
    n_components: int,
    refit: bool,
    batch_size: int,
    device: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load PCA from cache or fit a new one across all year files.

    Returns (mean, components, explained_variance).
    """
    logger = logging.getLogger(__name__)
    if os.path.exists(cache_path) and not refit:
        logger.info(f"Loading cached PCA from {cache_path}")
        with np.load(cache_path) as cache:
            mean = cache["mean"]
            components = cache["components"]
            ev = cache["explained_variance"]
        logger.info(
            f"Loaded PCA: hidden_dim={mean.shape[0]}, "
            f"cached_components={components.shape[0]}"
        )
        return mean, components, ev

    logger.info(f"Fitting across-time PCA with up to {n_components} components")
    embs = collect_token_embeddings_for_pca(
        n_paragraphs=n_paragraphs, batch_size=batch_size,
        device=device, seed=seed,
    )
    n_components = min(n_components, embs.shape[1], embs.shape[0])
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(embs)
    mean = pca.mean_.astype(np.float32)
    components = pca.components_.astype(np.float32)
    ev = pca.explained_variance_.astype(np.float32)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path, mean=mean, components=components, explained_variance=ev,
    )
    ratio = ev / ev.sum()
    logger.info(
        f"Saved PCA cache to {cache_path}; top-10 explained variance ratio: "
        + ", ".join(f"{r:.3f}" for r in ratio[:10])
    )
    return mean, components, ev


def project_embeddings(
    embeddings: np.ndarray,
    pca_mean: np.ndarray,
    pca_components: np.ndarray,
    all_but_top: int,
    reduc_dim: int,
) -> np.ndarray:
    """Project embeddings onto PCs [all_but_top : all_but_top + reduc_dim]."""
    if all_but_top + reduc_dim > pca_components.shape[0]:
        raise ValueError(
            f"all_but_top ({all_but_top}) + reduc_dim ({reduc_dim}) "
            f"exceeds available PCA components ({pca_components.shape[0]}). "
            "Refit with --refit-pca and a larger --pca-n-components."
        )
    centered = embeddings.astype(np.float32) - pca_mean
    selected = pca_components[all_but_top : all_but_top + reduc_dim]
    return centered @ selected.T


def bootstrap_k_estimate(
    projected: np.ndarray,
    n_bootstrap: int,
    k_max: int,
    weight_threshold: float,
    cov_type: str,
    seed: int,
) -> np.ndarray:
    """Resample with replacement, fit BGMM, count effective components.

    Returns int32 array of effective-K values (length n_bootstrap).
    Iterations whose fit raises an exception are recorded as -1.
    """
    n = projected.shape[0]
    rng = np.random.default_rng(seed)
    k_values = np.empty(n_bootstrap, dtype=np.int32)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = projected[idx]
        bgmm = BayesianGaussianMixture(
            n_components=k_max,
            covariance_type=cov_type,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=1.0 / k_max,
            max_iter=300,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        try:
            bgmm.fit(sample)
            k_values[i] = int((bgmm.weights_ > weight_threshold).sum())
        except Exception:
            k_values[i] = -1
    return k_values


def pairwise_cosine_distances(embeddings: np.ndarray) -> np.ndarray:
    """Return upper-triangle vector of pairwise cosine distances."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    normed = embeddings / norms
    sims = normed @ normed.T
    iu = np.triu_indices(embeddings.shape[0], k=1)
    return 1.0 - sims[iu]


def cosine_distance_to_centroid(embeddings: np.ndarray) -> np.ndarray:
    """Cosine distance from each embedding to the renormalized mean direction."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    normed = embeddings / norms
    mean_dir = normed.mean(axis=0)
    mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-10)
    return 1.0 - normed @ mean_dir


def compute_word_stats(
    embeddings: np.ndarray,
    seed: int,
    *,
    projected: np.ndarray | None = None,
    n_bootstrap: int = 0,
    k_max: int = 6,
    weight_threshold: float = 0.05,
    cov_type: str = "diag",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Compute pairwise + centroid distances, BGMM-bootstrap K, and summary stats.

    If ``projected`` is given and ``n_bootstrap > 0``, runs the BGMM bootstrap on
    the projected (PCA-reduced) embeddings and returns the K-distribution.
    """
    dists = pairwise_cosine_distances(embeddings)
    centroid_dists = cosine_distance_to_centroid(embeddings)

    stats = {
        "n_usages": int(embeddings.shape[0]),
        "mean_dist": float(dists.mean()),
        "median_dist": float(np.median(dists)),
        "std_dist": float(dists.std()),
        "mean_centroid_dist": float(centroid_dists.mean()),
        "median_centroid_dist": float(np.median(centroid_dists)),
        "std_centroid_dist": float(centroid_dists.std()),
    }

    # Silhouette at k=2 on the embeddings themselves.
    if embeddings.shape[0] >= 10:
        km = KMeans(n_clusters=2, n_init="auto", random_state=seed)
        labels = km.fit_predict(embeddings)
        if len(set(labels)) == 2:
            stats["silhouette_k2"] = float(
                silhouette_score(embeddings, labels, metric="cosine")
            )
        else:
            stats["silhouette_k2"] = float("nan")
    else:
        stats["silhouette_k2"] = float("nan")

    # Hartigan dip test on each distance distribution.
    # diptest.diptest returns (dip_statistic, p_value).
    if dists.size >= 4:
        dip, pval = diptest.diptest(dists)
        stats["dip"] = float(dip)
        stats["dip_pvalue"] = float(pval)
    else:
        stats["dip"] = float("nan")
        stats["dip_pvalue"] = float("nan")

    if centroid_dists.size >= 4:
        dip_c, pval_c = diptest.diptest(centroid_dists)
        stats["centroid_dip"] = float(dip_c)
        stats["centroid_dip_pvalue"] = float(pval_c)
    else:
        stats["centroid_dip"] = float("nan")
        stats["centroid_dip_pvalue"] = float("nan")

    # Bootstrap-based K estimation on projected embeddings.
    if projected is not None and n_bootstrap > 0 and projected.shape[0] >= 10:
        k_values = bootstrap_k_estimate(
            projected, n_bootstrap=n_bootstrap, k_max=k_max,
            weight_threshold=weight_threshold, cov_type=cov_type, seed=seed,
        )
        valid = k_values[k_values > 0]
        if valid.size > 0:
            counts = np.bincount(valid, minlength=k_max + 1)
            mode = int(counts.argmax())
            stats["k_mode"] = mode
            stats["k_mode_freq"] = float(counts[mode] / valid.size)
            stats["k_n_valid_bootstraps"] = int(valid.size)
        else:
            stats["k_mode"] = -1
            stats["k_mode_freq"] = float("nan")
            stats["k_n_valid_bootstraps"] = 0
    else:
        k_values = np.empty(0, dtype=np.int32)
        stats["k_mode"] = -1
        stats["k_mode_freq"] = float("nan")
        stats["k_n_valid_bootstraps"] = 0

    return dists, centroid_dists, k_values, stats


def _draw_k_inset(
    ax,
    k_values: np.ndarray | None,
    k_max: int,
) -> None:
    """Render a small histogram of bootstrap K values in the upper-right of ax."""
    if k_values is None or len(k_values) == 0:
        return
    valid = k_values[k_values > 0]
    if valid.size == 0:
        return
    inset = inset_axes(ax, width="32%", height="32%", loc="upper right", borderpad=0.4)
    bins = np.arange(0.5, k_max + 1.5, 1.0)
    inset.hist(valid, bins=bins, color="#555", edgecolor="white", linewidth=0.3)
    inset.set_xticks(range(1, k_max + 1))
    inset.set_xticklabels([str(k) for k in range(1, k_max + 1)], fontsize=5)
    inset.set_yticks([])
    inset.tick_params(axis="x", length=2, pad=1)
    inset.set_title("K", fontsize=6, pad=1)
    for spine in ("top", "right", "left"):
        inset.spines[spine].set_visible(False)


def _draw_subplot(
    ax,
    word: str,
    cat: str,
    dists: np.ndarray,
    s: dict,
    hist_color: str,
    kde_color: str,
    *,
    xlabel: str = "cos dist",
    pval_key: str = "dip_pvalue",
    k_values: np.ndarray | None = None,
    k_max: int = 6,
) -> None:
    ax.hist(
        dists, bins=60, density=True,
        color=hist_color, edgecolor="white", linewidth=0.3, alpha=0.85,
    )
    try:
        from scipy.stats import gaussian_kde
        xs = np.linspace(dists.min(), dists.max(), 200)
        ax.plot(xs, gaussian_kde(dists)(xs), color=kde_color, linewidth=1.3)
    except Exception:
        pass

    sil = s.get("silhouette_k2", float("nan"))
    pval = s.get(pval_key, float("nan"))
    k_mode = s.get("k_mode", -1)
    k_mode_freq = s.get("k_mode_freq", float("nan"))
    parts = [f"n={int(s['n_usages'])}", f"sil₂={sil:.2f}"]
    if not np.isnan(pval):
        parts.append(f"dip p={pval:.1e}")
    if isinstance(k_mode, (int, np.integer)) and k_mode > 0:
        if not np.isnan(k_mode_freq):
            parts.append(f"K̂={int(k_mode)} ({k_mode_freq:.0%})")
        else:
            parts.append(f"K̂={int(k_mode)}")
    ax.set_title(f"{word}\n{'  '.join(parts)}", fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("density", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.text(
        0.02, 0.95, cat[:4],
        transform=ax.transAxes, ha="left", va="top",
        fontsize=7, color="#555",
    )
    remove_extra_spines(ax)
    _draw_k_inset(ax, k_values, k_max)


def plot_grids(
    distances: dict[str, np.ndarray],
    stats: dict[str, dict],
    categories: dict[str, list[str]],
    out_dir: str,
    tag: str,
    title_prefix: str,
    *,
    kind: str = "pairwise",
    pval_key: str = "dip_pvalue",
    xlabel: str = "cos dist",
    palette: str = "blues",
    k_distributions: dict[str, np.ndarray] | None = None,
    k_max: int = 6,
) -> list[str]:
    """Render per-page 5x5 grids, one subplot per word. Returns list of file paths.

    Words are ordered by category (polysemous → monosemantic → extras →
    frequency_sampled) and chunked into pages of GRID_ROWS × GRID_COLS.
    """
    apply_plot_style()
    pal = SEQUENTIAL_PALETTES[palette]
    hist_color, kde_color = pal[2], pal[5]

    ordered: list[tuple[str, str]] = []
    for cat in CATEGORIES:
        for w in categories.get(cat, []):
            if w in distances:
                ordered.append((cat, w))
    if not ordered:
        raise RuntimeError("No words with distances to plot")

    per_page = GRID_ROWS * GRID_COLS
    n_pages = (len(ordered) + per_page - 1) // per_page
    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []

    for page in range(n_pages):
        chunk = ordered[page * per_page : (page + 1) * per_page]
        fig, axes = plt.subplots(
            GRID_ROWS, GRID_COLS,
            figsize=(2.9 * GRID_COLS, 2.3 * GRID_ROWS),
            sharex=False,
        )
        axes = np.atleast_2d(axes).reshape(GRID_ROWS, GRID_COLS)

        for i, (cat, word) in enumerate(chunk):
            ax = axes[i // GRID_COLS, i % GRID_COLS]
            kvals = (
                k_distributions.get(word) if k_distributions is not None else None
            )
            _draw_subplot(
                ax, word, cat, distances[word], stats[word], hist_color, kde_color,
                xlabel=xlabel, pval_key=pval_key,
                k_values=kvals, k_max=k_max,
            )
        for j in range(len(chunk), per_page):
            axes[j // GRID_COLS, j % GRID_COLS].axis("off")

        fig.suptitle(
            f"{title_prefix}  (page {page + 1}/{n_pages})",
            fontsize=13, y=1.00,
        )
        fig.tight_layout()
        out_path = os.path.join(
            out_dir, f"polysemy_pilot_{tag}_{kind}_p{page + 1:02d}.png"
        )
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pilot polysemy analysis: within-period pairwise-distance distributions"
    )
    parser.add_argument(
        "--config",
        default=os.path.join(PROJECT_ROOT, "configs", "polysemy_pilot.yaml"),
    )
    parser.add_argument(
        "--window", action="store_true",
        help="Use multi-year window (window_start..window_end) instead of single year",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Skip encoding; redraw figures from cached distances and summary.",
    )
    parser.add_argument(
        "--sample-frequency-words", type=int, default=0, metavar="N",
        help="Scan the chosen year(s), sample N words with probability "
             "proportional to frequency, append them to the config under "
             "'frequency_sampled', and exit without encoding.",
    )
    parser.add_argument(
        "--sample-min-freq", type=int, default=None,
        help="Minimum corpus frequency for the sampling pool. "
             "Defaults to config's max_occurrences_per_word.",
    )
    # PCA + BGMM-bootstrap parameters.
    parser.add_argument(
        "--reduc-dim", type=int, default=25,
        help="PCA dimensions to keep for BGMM (after dropping --all-but-top).",
    )
    parser.add_argument(
        "--all-but-top", type=int, default=0,
        help="Drop the top N PCs before retaining --reduc-dim (anisotropy correction).",
    )
    parser.add_argument(
        "--pca-cache", type=str,
        default=os.path.join(METRICS_DIR, "pca_acrosstime.npz"),
        help="Cache path for the across-time PCA fit.",
    )
    parser.add_argument(
        "--pca-paragraphs", type=int, default=2000,
        help="Total paragraphs to sample (across all years) for PCA fitting.",
    )
    parser.add_argument(
        "--pca-n-components", type=int, default=100,
        help="Number of PCA components to fit and cache.",
    )
    parser.add_argument(
        "--refit-pca", action="store_true",
        help="Refit PCA even if a cached file exists.",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=100,
        help="Bootstrap iterations per word for BGMM K estimation. Set 0 to skip.",
    )
    parser.add_argument(
        "--k-max", type=int, default=6,
        help="Max BGMM components considered.",
    )
    parser.add_argument(
        "--k-weight-threshold", type=float, default=0.05,
        help="Min weight for a BGMM component to count as 'effective'.",
    )
    parser.add_argument(
        "--cov-type", type=str, default="diag",
        choices=("full", "tied", "diag", "spherical"),
        help="BGMM covariance type. Diagonal is the safe default for ~25-dim BGMMs.",
    )
    args = parser.parse_args()

    setup_logging("21_pilot_polysemy")
    logger = logging.getLogger(__name__)

    cfg = load_pilot_config(args.config)

    if args.window:
        years = list(range(cfg["window_start"], cfg["window_end"] + 1))
        tag = f"{cfg['window_start']}-{cfg['window_end']}"
    else:
        years = [cfg["year"]]
        tag = str(cfg["year"])
    logger.info(f"Years: {years} (tag={tag})")

    if args.sample_frequency_words > 0:
        min_freq = (
            args.sample_min_freq
            if args.sample_min_freq is not None
            else cfg["max_occurrences_per_word"]
        )
        excluded = {
            w.lower() for key in CATEGORIES for w in (cfg.get(key) or [])
        }
        counts = count_tokens(years)
        logger.info(f"Total unique tokens: {len(counts)}")
        sampled = frequency_weighted_sample(
            counts, n=args.sample_frequency_words,
            min_freq=min_freq, excluded=excluded, seed=cfg.get("seed", 42),
        )
        cfg["frequency_sampled"] = sampled
        write_config(cfg, args.config)
        logger.info(
            f"Wrote {len(sampled)} words to '{args.config}' under 'frequency_sampled'. "
            "Review the list, then rerun without --sample-frequency-words to encode."
        )
        return

    categories = {c: list(cfg.get(c, [])) for c in CATEGORIES}
    all_words = sorted({w for ws in categories.values() for w in ws})
    logger.info(
        f"Pilot words ({len(all_words)}): "
        + ", ".join(f"{c}={len(categories[c])}" for c in CATEGORIES)
    )

    os.makedirs(METRICS_DIR, exist_ok=True)
    dists_path = os.path.join(METRICS_DIR, f"polysemy_pilot_{tag}_dists.npz")
    centroid_dists_path = os.path.join(
        METRICS_DIR, f"polysemy_pilot_{tag}_centroid_dists.npz"
    )
    k_dists_path = os.path.join(
        METRICS_DIR, f"polysemy_pilot_{tag}_k_dists.npz"
    )
    summary_path = os.path.join(METRICS_DIR, f"polysemy_pilot_{tag}.csv")

    if args.plot_only:
        required = [dists_path, summary_path]
        missing = [p for p in required if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"--plot-only requires cached files; missing: {missing}"
            )
        cached = np.load(dists_path, allow_pickle=False)
        distances = {k: cached[k] for k in cached.files}
        if os.path.exists(centroid_dists_path):
            cached_c = np.load(centroid_dists_path, allow_pickle=False)
            centroid_distances = {k: cached_c[k] for k in cached_c.files}
        else:
            logger.warning(
                f"No cached centroid distances at {centroid_dists_path}; "
                "skipping centroid figure. Rerun without --plot-only to generate."
            )
            centroid_distances = {}
        if os.path.exists(k_dists_path):
            cached_k = np.load(k_dists_path, allow_pickle=False)
            k_distributions = {k: cached_k[k] for k in cached_k.files}
        else:
            k_distributions = {}
        summary = pd.read_csv(summary_path).set_index("word")
        stats_map = summary.to_dict(orient="index")
        for p in plot_grids(
            distances, stats_map, categories, FIGURES_DIR, tag,
            title_prefix=f"Within-period pairwise cosine distances ({tag})",
            kind="pairwise", pval_key="dip_pvalue", xlabel="pairwise cos dist",
            palette="blues",
            k_distributions=k_distributions, k_max=args.k_max,
        ):
            logger.info(f"Wrote figure {p}")
        if centroid_distances:
            for p in plot_grids(
                centroid_distances, stats_map, categories, FIGURES_DIR, tag,
                title_prefix=f"Within-period cos distance to centroid ({tag})",
                kind="centroid", pval_key="centroid_dip_pvalue",
                xlabel="cos dist to centroid", palette="greens",
                k_distributions=k_distributions, k_max=args.k_max,
            ):
                logger.info(f"Wrote figure {p}")
        return

    # Fit (or load) the across-time PCA before any per-word work.
    pca_mean = pca_components = None
    if args.n_bootstrap > 0:
        pca_mean, pca_components, _pca_ev = fit_or_load_pca(
            cache_path=args.pca_cache,
            n_paragraphs=args.pca_paragraphs,
            n_components=args.pca_n_components,
            refit=args.refit_pca,
            batch_size=args.batch_size,
            device=args.device,
            seed=cfg["seed"],
        )

    # Collect usages, encode, extract embeddings.
    sampled = collect_usages(
        all_words,
        years,
        max_per_word=cfg["max_occurrences_per_word"],
        seed=cfg["seed"],
    )
    # Drop words below the minimum before encoding to save compute.
    min_n = cfg["min_occurrences_per_word"]
    kept = {w: us for w, us in sampled.items() if len(us) >= min_n}
    dropped = sorted(set(sampled) - set(kept))
    if dropped:
        logger.warning(f"Dropped (below min {min_n}): {dropped}")

    embeddings = encode_and_extract(
        kept, years, batch_size=args.batch_size, device=args.device
    )

    # Compute distances, projections, BGMM bootstrap, stats.
    distances: dict[str, np.ndarray] = {}
    centroid_distances: dict[str, np.ndarray] = {}
    k_distributions: dict[str, np.ndarray] = {}
    per_word_residual: dict[str, float] = {}
    summary_rows = []
    for word, embs in embeddings.items():
        if embs.shape[0] < min_n:
            continue
        if pca_mean is not None and pca_components is not None:
            projected = project_embeddings(
                embs, pca_mean, pca_components,
                all_but_top=args.all_but_top, reduc_dim=args.reduc_dim,
            )
            # Residual variance left outside the retained subspace, per word.
            centered = embs.astype(np.float32) - pca_mean
            total_var = float(np.var(centered, axis=0).sum())
            kept_var = float(np.var(projected, axis=0).sum())
            per_word_residual[word] = (
                1.0 - kept_var / total_var if total_var > 0 else float("nan")
            )
        else:
            projected = None

        d, c, kvals, s = compute_word_stats(
            embs, seed=cfg["seed"],
            projected=projected, n_bootstrap=args.n_bootstrap,
            k_max=args.k_max, weight_threshold=args.k_weight_threshold,
            cov_type=args.cov_type,
        )
        distances[word] = d.astype(np.float32)
        centroid_distances[word] = c.astype(np.float32)
        if kvals.size > 0:
            k_distributions[word] = kvals
        s["per_word_residual_var"] = per_word_residual.get(word, float("nan"))
        row = {"word": word, **s}
        for cat in CATEGORIES:
            if word in categories[cat]:
                row["category"] = cat
                break
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["category", "mean_dist"], ascending=[True, False]
    )
    summary_df.to_csv(summary_path, index=False)
    np.savez_compressed(dists_path, **distances)
    np.savez_compressed(centroid_dists_path, **centroid_distances)
    if k_distributions:
        np.savez_compressed(k_dists_path, **k_distributions)
        logger.info(
            f"Wrote {summary_path}, {dists_path}, {centroid_dists_path}, "
            f"and {k_dists_path}"
        )
    else:
        logger.info(
            f"Wrote {summary_path}, {dists_path}, and {centroid_dists_path}"
        )

    stats_map = summary_df.set_index("word").to_dict(orient="index")
    for p in plot_grids(
        distances, stats_map, categories, FIGURES_DIR, tag,
        title_prefix=f"Within-period pairwise cosine distances ({tag})",
        kind="pairwise", pval_key="dip_pvalue", xlabel="pairwise cos dist",
        palette="blues",
        k_distributions=k_distributions, k_max=args.k_max,
    ):
        logger.info(f"Wrote figure {p}")
    for p in plot_grids(
        centroid_distances, stats_map, categories, FIGURES_DIR, tag,
        title_prefix=f"Within-period cos distance to centroid ({tag})",
        kind="centroid", pval_key="centroid_dip_pvalue",
        xlabel="cos dist to centroid", palette="greens",
        k_distributions=k_distributions, k_max=args.k_max,
    ):
        logger.info(f"Wrote figure {p}")

    # Console summary.
    logger.info("\nSummary (by category, mean_dist desc):")
    for _, row in summary_df.iterrows():
        k_mode_raw = row["k_mode"] if "k_mode" in row else -1
        k_mode = -1 if bool(pd.isna(k_mode_raw)) else int(k_mode_raw)  # type: ignore[arg-type]
        k_mode_freq_raw = (
            row["k_mode_freq"] if "k_mode_freq" in row else float("nan")
        )
        k_mode_freq = (
            float("nan") if bool(pd.isna(k_mode_freq_raw))
            else float(k_mode_freq_raw)  # type: ignore[arg-type]
        )
        k_str = (
            f"K̂={k_mode}({k_mode_freq:.0%})"
            if k_mode > 0 and not np.isnan(k_mode_freq)
            else "K̂=-"
        )
        logger.info(
            f"  [{row.get('category','?'):<13}] {row['word']:<14} "
            f"n={int(row['n_usages']):<5} "
            f"mean={row['mean_dist']:.3f}  "
            f"sil2={row['silhouette_k2']:.3f}  "
            f"dip_p={row['dip_pvalue']:.2e}  "
            f"cdip_p={row['centroid_dip_pvalue']:.2e}  "
            f"{k_str}"
        )


if __name__ == "__main__":
    main()
