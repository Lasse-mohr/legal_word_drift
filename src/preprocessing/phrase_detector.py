"""Multi-word term detection using gensim Phrases.

Trains a bigram (and optionally trigram) phrase model on the corpus to
detect collocations like "internal_market", "preliminary_reference",
"direct_effect", "state_aid", "free_movement", etc.

Usage:
    detector = PhraseDetector.train(sentence_files)
    detector.save(path)
    detector = PhraseDetector.load(path)
    phrased = detector.apply(["free", "movement", "of", "goods"])
"""
from __future__ import annotations

import os
from typing import Iterable, List

from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS


class PhraseDetector:
    """Two-pass phrase detector (bigrams then trigrams)."""

    def __init__(self, bigram_model: Phrases, trigram_model: Phrases | None = None):
        self._bigram = bigram_model.freeze()
        self._trigram = trigram_model.freeze() if trigram_model else None

    @classmethod
    def train(
        cls,
        sentences: Iterable[List[str]],
        min_count: int = 30,
        threshold: float = 10.0,
        connector_words: frozenset[str] = ENGLISH_CONNECTOR_WORDS,
    ) -> PhraseDetector:
        """Train bigram and trigram phrase models.

        Args:
            sentences: Iterable of token lists (each list = one sentence).
            min_count: Minimum co-occurrence count for a phrase.
            threshold: Scoring threshold (higher = fewer phrases).
            connector_words: Words allowed in the middle of phrases (e.g., "of" in
                           "free movement of goods" -> "free_movement_of_goods").
        """
        # First pass: bigrams
        bigram = Phrases(
            sentences,
            min_count=min_count,
            threshold=threshold,
            connector_words=connector_words,
        )

        # Second pass: trigrams (bigrams applied to sentences, then detect new bigrams on top)
        frozen_bigram = bigram.freeze()

        def bigram_sentences():
            for sent in sentences:
                yield frozen_bigram[sent]

        trigram = Phrases(
            bigram_sentences(),
            min_count=min_count,
            threshold=threshold,
            connector_words=connector_words,
        )

        return cls(bigram, trigram)

    def apply(self, tokens: List[str]) -> List[str]:
        """Apply phrase detection to a token list."""
        result = self._bigram[tokens]
        if self._trigram:
            result = self._trigram[result]
        return list(result)

    def save(self, directory: str) -> None:
        """Save phrase models to directory."""
        os.makedirs(directory, exist_ok=True)
        self._bigram.save(os.path.join(directory, "bigram.pkl"))
        if self._trigram:
            self._trigram.save(os.path.join(directory, "trigram.pkl"))

    @classmethod
    def load(cls, directory: str) -> PhraseDetector:
        """Load phrase models from directory."""
        from gensim.models.phrases import FrozenPhrases

        bigram = FrozenPhrases.load(os.path.join(directory, "bigram.pkl"))
        trigram_path = os.path.join(directory, "trigram.pkl")
        trigram = FrozenPhrases.load(trigram_path) if os.path.exists(trigram_path) else None

        instance = cls.__new__(cls)
        instance._bigram = bigram
        instance._trigram = trigram
        return instance
