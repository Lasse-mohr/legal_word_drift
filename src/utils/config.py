"""Path constants and project configuration."""

from __future__ import annotations

import logging
import os
from datetime import datetime

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

# BERT contextualised embeddings
BERT_DIR = os.path.join(MODELS_DIR, "bert")
BERT_USAGE_INDEX_DIR = os.path.join(BERT_DIR, "usage_index")
BERT_SAMPLED_DIR = os.path.join(BERT_DIR, "sampled_usages")
BERT_EMBEDDINGS_DIR = os.path.join(BERT_DIR, "embeddings")
BERT_CENTROIDS_PATH = os.path.join(BERT_DIR, "centroids.npz")
BERT_CROSS_PERIOD_APD_PATH = os.path.join(BERT_DIR, "cross_period_apd.npz")

# Results
RESULTS_DIR = os.path.join(DATA_DIR, "results")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Default output for text fetcher
RAW_TEXTS_DIR = os.path.join(RAW_DIR, "texts")

# Logs
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")


def setup_logging(script_name: str) -> None:
    """Configure logging to both console and a timestamped log file.

    Log files are written to logs/{YYYY-MM-DD}_{script_name}.log
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_path = os.path.join(LOGS_DIR, f"{date_str}_{script_name}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path),
        ],
    )
    logging.getLogger(__name__).info(f"Logging to {log_path}")
