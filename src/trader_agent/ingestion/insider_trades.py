"""Insider trading data ingester -- pulls SEC Form 4 filings from Finnhub / SEC EDGAR."""

from __future__ import annotations

from trader_agent.ingestion.base import BaseIngester


class InsiderTradesIngester(BaseIngester):
    """Fetch and store insider transaction filings for tracked tickers."""

    service_name = "insider_trades"

    async def fetch(self) -> list[dict]:
        raise NotImplementedError

    async def transform(self, raw: list[dict]) -> list[dict]:
        raise NotImplementedError
