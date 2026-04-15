"""Build V_bert: BERT-compatible target vocabulary from V_analysis.

Filters V_analysis to single words that can be reliably located in
BERT-tokenized text, and pre-computes subword token ID sequences.

Output: data/models/bert/v_bert.json
"""
from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.embeddings.bert_encoder import load_model
from src.embeddings.bert_vocabulary import build_v_bert, save_v_bert
from src.embeddings.vocabulary import load_vocab
from src.utils.config import BERT_DIR, VOCAB_DIR, setup_logging


def main() -> None:
    setup_logging("11_build_bert_vocab")
    logger = logging.getLogger(__name__)

    # Load V_analysis
    v_analysis_path = os.path.join(VOCAB_DIR, "v_analysis.json")
    v_analysis = load_vocab(v_analysis_path)
    logger.info(f"Loaded V_analysis: {len(v_analysis)} words")

    # Load tokenizer (model not needed for this step)
    logger.info("Loading EURLEX-BERT tokenizer...")
    _, tokenizer, _ = load_model(device="cpu")

    # Build V_bert
    v_bert = build_v_bert(v_analysis, tokenizer)

    # Save
    os.makedirs(BERT_DIR, exist_ok=True)
    v_bert_path = os.path.join(BERT_DIR, "v_bert.json")
    save_v_bert(v_bert, v_bert_path)

    # Log sample words with subword decompositions
    sample_words = list(v_bert.keys())[:10]
    logger.info("Sample V_bert entries:")
    for word in sample_words:
        subwords = tokenizer.convert_ids_to_tokens(v_bert[word])
        logger.info(f"  {word} -> {subwords}")


if __name__ == "__main__":
    main()
