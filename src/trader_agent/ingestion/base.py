"""Abstract base ingester with built-in retry, rate-limit handling, and structured logging.

All concrete ingesters (fundamentals, insider_trades, news, social) inherit from
this class and implement `fetch()` and `transform()`.
"""

from __future__ import annotations

import abc
from datetime import UTC, datetime
from typing import Any

import httpx
import structlog
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from trader_agent.config import settings

logger = structlog.get_logger(__name__)


def _log_retry(retry_state: RetryCallState) -> None:
    """Tenacity callback -- log each retry attempt with structured context."""
    logger.warning(
        "retrying_api_call",
        attempt=retry_state.attempt_number,
        wait_seconds=getattr(retry_state.next_action, "sleep", None),
        exception=str(retry_state.outcome.exception()) if retry_state.outcome else None,
    )


class BaseIngester(abc.ABC):
    """Abstract base for all data ingestion services.

    Subclasses must define:
        service_name: str          -- identifier used in logs and metrics
        async fetch() -> list      -- pull raw data from the upstream API
        async transform(raw) -> list -- normalize raw data for DB storage

    The `run()` method orchestrates fetch -> transform -> store with full
    retry logic and structured logging around every step.
    """

    service_name: str = "base"

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    async def get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.http_timeout_seconds),
                headers={"User-Agent": f"trader-agent/{self.service_name}"},
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    @abc.abstractmethod
    async def fetch(self) -> list[dict[str, Any]]:
        """Pull raw data from the upstream API. Must be implemented by subclass."""
        ...

    @abc.abstractmethod
    async def transform(self, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize raw API response into records ready for DB insertion."""
        ...

    async def store(self, records: list[dict[str, Any]]) -> int:
        """Persist transformed records to the database. Returns count of rows written."""
        # TODO: implement bulk upsert via SQLAlchemy
        logger.info("storing_records", service=self.service_name, count=len(records))
        raise NotImplementedError("DB storage not yet wired up")

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        before_sleep=_log_retry,
        reraise=True,
    )
    async def _fetch_with_retry(self) -> list[dict[str, Any]]:
        """Wrap fetch() with exponential-backoff retry on transient HTTP errors."""
        return await self.fetch()

    async def run(self) -> None:
        """Full ingestion cycle: fetch -> transform -> store, with logging at each stage."""
        started_at = datetime.now(UTC)
        log = logger.bind(service=self.service_name, started_at=started_at.isoformat())

        log.info("ingestion_started")
        try:
            raw = await self._fetch_with_retry()
            log.info("fetch_complete", raw_record_count=len(raw))

            records = await self.transform(raw)
            log.info("transform_complete", record_count=len(records))

            stored = await self.store(records)
            log.info("store_complete", stored_count=stored)
        except Exception:
            log.exception("ingestion_failed")
            raise
        finally:
            elapsed = (datetime.now(UTC) - started_at).total_seconds()
            log.info("ingestion_finished", elapsed_seconds=round(elapsed, 2))
