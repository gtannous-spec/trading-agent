"""APScheduler job definitions for periodic data ingestion."""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


def start_scheduler() -> None:
    """Configure and start the APScheduler with all ingestion jobs.

    Each ingester runs on its own interval to respect upstream API rate limits.
    """
    raise NotImplementedError
