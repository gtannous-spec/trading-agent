"""Model training pipeline -- trains LightGBM / XGBoost on historical feature matrices."""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


def train_model(
    model_type: str = "lightgbm",
) -> None:
    """Train a classification model to predict probability of outperformance.

    Outputs a serialized model artifact and evaluation metrics.
    """
    raise NotImplementedError
