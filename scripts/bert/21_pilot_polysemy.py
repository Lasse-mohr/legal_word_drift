"""Pilot polysemy analysis: within-period pairwise cosine distance distributions.

For each word in configs/polysemy_pilot.yaml, collects up to N occurrences from
a single year (or a multi-year window via --window), extracts contextualized
BERT embeddings (pooled over layers 8-10), and computes the pairwise cosine
distance distribution, silhouette at k=2, and the Hartigan dip test.

Outputs:
  data/results/metrics/polysemy_pilot_{tag}.csv        (per-word summary stats)
  data/results/metrics/polysemy_pilot_{tag}_dists.npz  (raw pairwise distances)
  data/results/figures/polysemy_pilot_{tag}.png        (distribution grid)

Rerun with --plot-only to redraw the figure from cached distances.
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
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import silhouette_score

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


def pairwise_cosine_distances(embeddings: np.ndarray) -> np.ndarray:
    """Return upper-triangle vector of pairwise cosine distances."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    normed = embeddings / norms
    sims = normed @ normed.T
    iu = np.triu_indices(embeddings.shape[0], k=1)
    return 1.0 - sims[iu]


def compute_word_stats(
    embeddings: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Compute pairwise-distance vector and summary statistics for one word."""
    dists = pairwise_cosine_distances(embeddings)

    stats = {
        "n_usages": int(embeddings.shape[0]),
        "mean_dist": float(dists.mean()),
        "median_dist": float(np.median(dists)),
        "std_dist": float(dists.std()),
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

    # Hartigan dip test on the distance distribution.
    if dists.size >= 4:
        # diptest.diptest returns (dip_statistic, p_value).
        dip, pval = diptest.diptest(dists)
        stats["dip"] = float(dip)
        stats["dip_pvalue"] = float(pval)
    else:
        stats["dip"] = float("nan")
        stats["dip_pvalue"] = float("nan")

    return dists, stats


def _draw_subplot(
    ax,
    word: str,
    cat: str,
    dists: np.ndarray,
    s: dict,
    hist_color: str,
    kde_color: str,
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
    pval = s.get("dip_pvalue", float("nan"))
    if not np.isnan(pval):
        subtitle = f"n={int(s['n_usages'])}  sil₂={sil:.2f}  dip p={pval:.1e}"
    else:
        subtitle = f"n={int(s['n_usages'])}  sil₂={sil:.2f}"
    ax.set_title(f"{word}\n{subtitle}", fontsize=9)
    ax.set_xlabel("cos dist", fontsize=8)
    ax.set_ylabel("density", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.text(
        0.98, 0.95, cat[:4],
        transform=ax.transAxes, ha="right", va="top",
        fontsize=7, color="#555",
    )
    remove_extra_spines(ax)


def plot_grids(
    distances: dict[str, np.ndarray],
    stats: dict[str, dict],
    categories: dict[str, list[str]],
    out_dir: str,
    tag: str,
    title_prefix: str,
) -> list[str]:
    """Render per-page 5x5 grids, one subplot per word. Returns list of file paths.

    Words are ordered by category (polysemous → monosemantic → extras →
    frequency_sampled) and chunked into pages of GRID_ROWS × GRID_COLS.
    """
    apply_plot_style()
    blues = SEQUENTIAL_PALETTES["blues"]
    hist_color, kde_color = blues[2], blues[5]

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
            _draw_subplot(
                ax, word, cat, distances[word], stats[word], hist_color, kde_color
            )
        for j in range(len(chunk), per_page):
            axes[j // GRID_COLS, j % GRID_COLS].axis("off")

        fig.suptitle(
            f"{title_prefix}  (page {page + 1}/{n_pages})",
            fontsize=13, y=1.00,
        )
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"polysemy_pilot_{tag}_p{page + 1:02d}.png")
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
    summary_path = os.path.join(METRICS_DIR, f"polysemy_pilot_{tag}.csv")

    if args.plot_only:
        if not (os.path.exists(dists_path) and os.path.exists(summary_path)):
            raise FileNotFoundError(
                f"--plot-only requires cached {dists_path} and {summary_path}"
            )
        cached = np.load(dists_path, allow_pickle=False)
        distances = {k: cached[k] for k in cached.files}
        summary = pd.read_csv(summary_path).set_index("word")
        stats = summary.to_dict(orient="index")
        written = plot_grids(
            distances, stats, categories, FIGURES_DIR, tag,
            title_prefix=f"Within-period pairwise cosine distances ({tag})",
        )
        for p in written:
            logger.info(f"Wrote figure {p}")
        return

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

    # Compute distances + stats.
    distances: dict[str, np.ndarray] = {}
    summary_rows = []
    for word, embs in embeddings.items():
        if embs.shape[0] < min_n:
            continue
        d, s = compute_word_stats(embs, seed=cfg["seed"])
        distances[word] = d.astype(np.float32)
        row = {"word": word, **s}
        # Category label for the summary.
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
    logger.info(f"Wrote {summary_path} and {dists_path}")

    stats_map = summary_df.set_index("word").to_dict(orient="index")
    written = plot_grids(
        distances, stats_map, categories, FIGURES_DIR, tag,
        title_prefix=f"Within-period pairwise cosine distances ({tag})",
    )
    for p in written:
        logger.info(f"Wrote figure {p}")

    # Console summary.
    logger.info("\nSummary (by category, mean_dist desc):")
    for _, row in summary_df.iterrows():
        logger.info(
            f"  [{row.get('category','?'):<13}] {row['word']:<14} "
            f"n={int(row['n_usages']):<5} "
            f"mean={row['mean_dist']:.3f}  "
            f"sil2={row['silhouette_k2']:.3f}  "
            f"dip_p={row['dip_pvalue']:.2e}"
        )


if __name__ == "__main__":
    main()
