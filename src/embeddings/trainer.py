"""Word2Vec training for temporal slices.

Trains Skip-gram word2vec models on sentence files, supporting both
single-year and sliding-window configurations.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class W2VConfig:
    """Word2Vec hyperparameters tuned for legal corpus."""

    vector_size: int = 100
    window: int = 5
    min_count: int = 50
    sg: int = 1             # Skip-gram
    negative: int = 10
    sample: float = 1e-4
    epochs: int = 10
    seed: int = 42
    workers: int = 1        # Deterministic (single-threaded)


def train_on_sentences(
    sentence_files: List[str],
    config: Optional[W2VConfig] = None,
) -> Word2Vec:
    """Train a Word2Vec model on one or more sentence files.

    Args:
        sentence_files: Paths to sentence files (gensim LineSentence format).
        config: Hyperparameters. Uses defaults if None.

    Returns:
        Trained gensim Word2Vec model.
    """
    config = config or W2VConfig()

    # Chain multiple files into one sentence iterator
    class MultiFileSentences:
        def __init__(self, paths: List[str]):
            self.paths = paths

        def __iter__(self):
            for path in self.paths:
                yield from LineSentence(path)

    sentences = MultiFileSentences(sentence_files)

    model = Word2Vec(
        sentences=sentences,
        vector_size=config.vector_size,
        window=config.window,
        min_count=config.min_count,
        sg=config.sg,
        negative=config.negative,
        sample=config.sample,
        epochs=config.epochs,
        seed=config.seed,
        workers=config.workers,
    )
    return model


def train_sliding_windows(
    sentences_dir: str,
    output_dir: str,
    start_year: int = 1990,
    end_year: int = 2025,
    window_size: int = 5,
    step: int = 1,
    config: Optional[W2VConfig] = None,
) -> dict[str, str]:
    """Train Word2Vec models on sliding time windows.

    Args:
        sentences_dir: Directory with {year}.txt sentence files.
        output_dir: Directory to save trained models.
        start_year: First year of the first window.
        end_year: Last year of the last window.
        window_size: Number of years per window.
        step: Step between window starts.
        config: Hyperparameters.

    Returns:
        Dict mapping window label (center year) to model path.
    """
    os.makedirs(output_dir, exist_ok=True)
    models = {}

    # Pre-compute all windows for progress bar
    windows = []
    ws = start_year
    while ws + window_size - 1 <= end_year:
        windows.append((ws, ws + window_size - 1))
        ws += step

    for window_start, window_end in tqdm(windows, desc="Window models", unit="model"):
        label = f"w{window_start}_{window_end}"
        model_path = os.path.join(output_dir, f"{label}.model")

        if os.path.exists(model_path):
            models[label] = model_path
            continue

        # Collect sentence files for this window
        files = []
        for year in range(window_start, window_end + 1):
            path = os.path.join(sentences_dir, f"{year}.txt")
            if os.path.exists(path):
                files.append(path)

        if not files:
            logger.warning(f"Window {label}: no sentence files found, skipping")
            continue

        model = train_on_sentences(files, config)
        model.save(model_path)
        models[label] = model_path
        logger.info(f"  {label}: vocab={len(model.wv)} words")

    return models


def train_single_years(
    sentences_dir: str,
    output_dir: str,
    start_year: int = 1990,
    end_year: int = 2025,
    config: Optional[W2VConfig] = None,
) -> dict[int, str]:
    """Train Word2Vec models for individual years.

    Returns:
        Dict mapping year to model path.
    """
    os.makedirs(output_dir, exist_ok=True)
    models = {}

    years = [y for y in range(start_year, end_year + 1)
             if os.path.exists(os.path.join(sentences_dir, f"{y}.txt"))]

    for year in tqdm(years, desc="Yearly models", unit="model"):
        sent_path = os.path.join(sentences_dir, f"{year}.txt")
        model_path = os.path.join(output_dir, f"y{year}.model")

        if os.path.exists(model_path):
            models[year] = model_path
            continue

        model = train_on_sentences([sent_path], config)
        model.save(model_path)
        models[year] = model_path
        logger.info(f"  y{year}: vocab={len(model.wv)} words")

    return models
