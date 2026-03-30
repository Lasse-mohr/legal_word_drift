"""Align trained word2vec models into a shared vector space.

Builds vocabulary tiers (V_global, V_analysis, V_anchor), then performs
Orthogonal Procrustes alignment using the most recent window as reference.

Usage:
    python -m src.pipeline.06_align_embeddings
"""
from __future__ import annotations

import argparse
import glob
import logging
import os

from gensim.models import Word2Vec

from src.embeddings.alignment import align_to_reference, save_aligned
from src.embeddings.vocabulary import (
    build_v_global,
    build_v_analysis,
    build_v_anchor,
    save_vocab,
)
from src.utils.config import W2V_MODELS_DIR, ALIGNED_DIR, VOCAB_DIR, setup_logging

setup_logging("06_align_embeddings")
logger = logging.getLogger(__name__)


def load_window_models(models_dir: str) -> dict[str, Word2Vec]:
    """Load all window models from disk."""
    pattern = os.path.join(models_dir, "*.model")
    models = {}
    for path in sorted(glob.glob(pattern)):
        label = os.path.splitext(os.path.basename(path))[0]
        models[label] = Word2Vec.load(path)
        logger.info(f"  Loaded {label}: {len(models[label].wv)} words")
    return models


def main():
    parser = argparse.ArgumentParser(description="Align word2vec models")
    parser.add_argument(
        "--models-dir", type=str,
        default=os.path.join(W2V_MODELS_DIR, "windows"),
        help="Directory with trained .model files",
    )
    parser.add_argument("--anchor-top-n", type=int, default=500)
    parser.add_argument("--min-slices-global", type=int, default=3)
    parser.add_argument("--min-slices-analysis", type=int, default=10)
    args = parser.parse_args()

    # Load models
    logger.info(f"Loading models from {args.models_dir}...")
    models = load_window_models(args.models_dir)
    if not models:
        logger.error("No models found. Run 05_train_embeddings.py first.")
        return

    # Build vocabulary tiers
    logger.info("Building vocabulary tiers...")
    v_global = build_v_global(models, min_slices=args.min_slices_global)
    v_analysis = build_v_analysis(models, v_global, min_slices=args.min_slices_analysis)
    v_anchor = build_v_anchor(models, v_analysis, top_n=args.anchor_top_n)

    os.makedirs(VOCAB_DIR, exist_ok=True)
    save_vocab(v_global, os.path.join(VOCAB_DIR, "v_global.json"))
    save_vocab(v_analysis, os.path.join(VOCAB_DIR, "v_analysis.json"))
    save_vocab(v_anchor, os.path.join(VOCAB_DIR, "v_anchor.json"))

    # Align: use the most recent window as reference (largest corpus, most stable)
    sorted_labels = sorted(models.keys())
    reference_label = sorted_labels[-1]
    logger.info(f"Reference model: {reference_label}")

    # Extract KeyedVectors from models
    model_kvs = {label: model.wv for label, model in models.items()}

    logger.info("Aligning models...")
    aligned_kvs = align_to_reference(model_kvs, reference_label, anchor_words=v_anchor)

    # Save aligned vectors
    paths = save_aligned(aligned_kvs, ALIGNED_DIR)
    logger.info(f"Saved {len(paths)} aligned models to {ALIGNED_DIR}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
