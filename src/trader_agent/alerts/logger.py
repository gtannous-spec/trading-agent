"""Prediction logger -- persists daily prediction outputs for audit and backtesting."""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


def log_prediction(prediction: dict) -> None:
    """Write a prediction record to the predictions table and structured log."""
    raise NotImplementedError
