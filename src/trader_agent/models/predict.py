"""Inference module -- loads a trained model and scores current feature vectors."""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


def predict(ticker: str) -> dict:
    """Generate a probability score for a given ticker using the latest model.

    Returns a dict with keys: ticker, probability, confidence, timestamp.
    """
    raise NotImplementedError
