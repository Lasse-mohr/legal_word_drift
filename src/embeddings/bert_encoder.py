"""EURLEX-BERT model loading and batch encoding.

Loads the EURLEX-BERT model (or a fallback legal-BERT) and encodes
paragraphs in batches, returning hidden states from specified layers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, BertModel, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

MODEL_NAME = "nlpaueb/bert-base-uncased-eurlex"
FALLBACK_MODEL = "nlpaueb/legal-bert-base-uncased"
DEFAULT_LAYERS = (8, 9, 10)
MAX_LENGTH = 512


@dataclass
class EncodedParagraph:
    """Result of encoding a single paragraph through BERT."""

    input_ids: np.ndarray  # (seq_len,)
    offsets: list[tuple[int, int]]  # per-token (char_start, char_end)
    hidden_states: np.ndarray  # (seq_len, hidden_dim) averaged over selected layers


def detect_device(requested: str = "auto") -> torch.device:
    """Select the best available device."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            # Quick smoke test — MPS can be flaky
            t = torch.zeros(1, device="mps")
            _ = t + 1
            return torch.device("mps")
        except Exception:
            logger.warning("MPS available but failed smoke test, falling back to CPU")
    return torch.device("cpu")


def load_model(
    model_name: str = MODEL_NAME,
    device: str = "auto",
) -> tuple[BertModel, PreTrainedTokenizerFast, torch.device]:
    """Load EURLEX-BERT model and tokenizer.

    Returns (model, tokenizer, device).
    """
    dev = detect_device(device)
    logger.info(f"Using device: {dev}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    except Exception as e:
        logger.warning(f"Failed to load {model_name}: {e}. Trying fallback {FALLBACK_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
        model = AutoModel.from_pretrained(FALLBACK_MODEL, output_hidden_states=True)

    model = model.to(dev)
    model.eval()
    logger.info(f"Loaded {model.config._name_or_path} ({model.config.num_hidden_layers} layers, "
                f"hidden_size={model.config.hidden_size})")
    return model, tokenizer, dev


def encode_paragraphs(
    paragraphs: list[str],
    model: BertModel,
    tokenizer: PreTrainedTokenizerFast,
    device: torch.device,
    batch_size: int = 16,
    layers: tuple[int, ...] = DEFAULT_LAYERS,
) -> list[EncodedParagraph]:
    """Encode paragraphs through BERT and extract hidden states.

    For each paragraph, returns the hidden states averaged over the
    specified layers, plus token-to-character offset mappings.
    Paragraphs exceeding MAX_LENGTH subword tokens are truncated.
    """
    results: list[EncodedParagraph] = []
    n_truncated = 0

    for batch_start in range(0, len(paragraphs), batch_size):
        batch_texts = paragraphs[batch_start : batch_start + batch_size]

        encoding = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        offset_mapping = encoding.pop("offset_mapping")  # (batch, seq_len, 2)
        input_ids = encoding["input_ids"]  # (batch, seq_len)

        # Check for truncation
        for i, text in enumerate(batch_texts):
            full_len = len(tokenizer.encode(text, add_special_tokens=True))
            if full_len > MAX_LENGTH:
                n_truncated += 1

        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding)

        # outputs.hidden_states is a tuple of (batch, seq_len, hidden_dim) per layer
        # Stack selected layers and average
        selected = torch.stack([outputs.hidden_states[l] for l in layers], dim=0)
        averaged = selected.mean(dim=0)  # (batch, seq_len, hidden_dim)

        # Move back to CPU as float16
        averaged_np = averaged.cpu().to(torch.float16).numpy()
        input_ids_np = input_ids.cpu().numpy()

        for i in range(len(batch_texts)):
            # Strip padding
            attn_mask = encoding["attention_mask"][i].cpu().numpy()
            seq_len = int(attn_mask.sum())

            results.append(EncodedParagraph(
                input_ids=input_ids_np[i, :seq_len],
                offsets=[(int(s), int(e)) for s, e in offset_mapping[i, :seq_len].tolist()],
                hidden_states=averaged_np[i, :seq_len, :],
            ))

    if n_truncated:
        logger.warning(f"{n_truncated}/{len(paragraphs)} paragraphs truncated to {MAX_LENGTH} tokens")

    return results


def extract_embedding(
    encoded: EncodedParagraph,
    char_start: int,
    char_end: int,
) -> np.ndarray | None:
    """Extract the contextualized embedding for a word at a given character span.

    Finds all subword tokens whose offsets overlap the character span
    and mean-pools their hidden states.

    Returns (hidden_dim,) float16 array, or None if no matching tokens found.
    """
    indices = [
        i for i, (s, e) in enumerate(encoded.offsets)
        if s >= char_start and e <= char_end and s != e  # exclude special tokens with (0,0)
    ]
    if not indices:
        return None
    return encoded.hidden_states[indices].mean(axis=0)
