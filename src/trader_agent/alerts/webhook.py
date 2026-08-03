"""Webhook alerting -- sends prediction results to Slack, Discord, or generic HTTP endpoints."""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


async def send_webhook(payload: dict) -> None:
    """POST a JSON payload to the configured webhook URL with retry."""
    raise NotImplementedError
