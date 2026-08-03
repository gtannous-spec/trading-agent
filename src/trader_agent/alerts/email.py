"""Email alerting -- sends prediction summaries via SMTP."""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


async def send_email(subject: str, body: str, recipients: list[str]) -> None:
    """Send an email alert with the given subject and body."""
    raise NotImplementedError
