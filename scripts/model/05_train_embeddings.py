"""Train word2vec models for each time slice.

Trains sliding-window models (primary) and optionally single-year
models (secondary).

Usage:
    python -m src.pipeline.05_train_embeddings [--start 1990] [--end 2025]
    python -m src.pipeline.05_train_embeddings --single-year  # also train per-year
"""
from __future__ import annotations

import argparse
import logging
import os

from src.embeddings.trainer import W2VConfig, train_sliding_windows, train_single_years
from src.utils.config import SENTENCES_DIR, W2V_MODELS_DIR, setup_logging

setup_logging("05_train_embeddings")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Word2Vec models")
    parser.add_argument("--start", type=int, default=1990)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--window-size", type=int, default=5, help="Sliding window size in years")
    parser.add_argument("--step", type=int, default=1, help="Sliding window step")
    parser.add_argument("--single-year", action="store_true", help="Also train single-year models")
    parser.add_argument("--vector-size", type=int, default=100)
    parser.add_argument("--min-count", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1, help="1 for reproducibility")
    args = parser.parse_args()

    config = W2VConfig(
        vector_size=args.vector_size,
        min_count=args.min_count,
        epochs=args.epochs,
        workers=args.workers,
    )

    window_dir = os.path.join(W2V_MODELS_DIR, "windows")
    logger.info("Training sliding-window models...")
    window_models = train_sliding_windows(
        SENTENCES_DIR, window_dir,
        start_year=args.start, end_year=args.end,
        window_size=args.window_size, step=args.step,
        config=config,
    )
    logger.info(f"Trained {len(window_models)} window models")

    if args.single_year:
        yearly_dir = os.path.join(W2V_MODELS_DIR, "yearly")
        logger.info("Training single-year models...")
        yearly_models = train_single_years(
            SENTENCES_DIR, yearly_dir,
            start_year=args.start, end_year=args.end,
            config=config,
        )
        logger.info(f"Trained {len(yearly_models)} yearly models")

    logger.info("Done.")


if __name__ == "__main__":
    main()
