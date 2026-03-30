"""Path constants and project configuration."""

from __future__ import annotations

import os

# Project root: two levels up from this file (src/utils/config.py -> project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ── Data directories ──────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Raw data (as fetched from CELLAR)
RAW_DIR = os.path.join(DATA_DIR, "raw")
RAW_METADATA_DIR = os.path.join(RAW_DIR, "metadata")
RAW_XHTML_DIR = os.path.join(RAW_DIR, "xhtml")

# Processed data
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
PARAGRAPHS_DIR = os.path.join(PROCESSED_DIR, "paragraphs")
SENTENCES_DIR = os.path.join(PROCESSED_DIR, "sentences")
VOCAB_DIR = os.path.join(PROCESSED_DIR, "vocab")

# Models
MODELS_DIR = os.path.join(DATA_DIR, "models")
W2V_MODELS_DIR = os.path.join(MODELS_DIR, "word2vec")
ALIGNED_DIR = os.path.join(MODELS_DIR, "aligned")

# Results
RESULTS_DIR = os.path.join(DATA_DIR, "results")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Default output for text fetcher
RAW_TEXTS_DIR = os.path.join(RAW_DIR, "texts")
