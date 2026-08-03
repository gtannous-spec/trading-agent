"""Feature engineering -- transforms raw DB records into model-ready feature vectors."""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


def build_feature_matrix(
    ticker: str,
    lookback_days: int = 90,
) -> None:
    """Query DB for a ticker's fundamentals, insider trades, and sentiment scores,
    then assemble a feature matrix suitable for the prediction model.
    """
    raise NotImplementedError
