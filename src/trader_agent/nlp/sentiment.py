"""FinBERT-based financial sentiment scorer.

Loads ProsusAI/finbert and scores text chunks as bullish / bearish / neutral
with a normalized confidence value in [-1.0, 1.0].
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class SentimentScore:
    label: str  # "bullish" | "bearish" | "neutral"
    confidence: float  # [-1.0 (max bearish) .. 1.0 (max bullish)]
    raw_probabilities: dict[str, float]


class FinBERTScorer:
    """Lazy-loaded FinBERT pipeline that scores financial text."""

    def __init__(self, model_name: str = "ProsusAI/finbert", device: str = "cpu") -> None:
        self._model_name = model_name
        self._device = device
        self._pipeline = None

    def _load(self):
        from transformers import pipeline as hf_pipeline

        logger.info("loading_finbert", model=self._model_name, device=self._device)
        self._pipeline = hf_pipeline(
            "sentiment-analysis",
            model=self._model_name,
            device=self._device,
        )

    def score(self, text: str) -> SentimentScore:
        """Return a normalized sentiment score for a single text chunk."""
        if self._pipeline is None:
            self._load()
        raise NotImplementedError

    def score_batch(self, texts: list[str]) -> list[SentimentScore]:
        """Score multiple texts in a single forward pass for efficiency."""
        if self._pipeline is None:
            self._load()
        raise NotImplementedError
