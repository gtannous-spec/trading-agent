"""Fundamental data ingester -- pulls financials, ratios, and price data from Finnhub / yfinance."""

from __future__ import annotations

from trader_agent.ingestion.base import BaseIngester


class FundamentalsIngester(BaseIngester):
    """Fetch and store fundamental financial data for tracked tickers."""

    service_name = "fundamentals"

    async def fetch(self) -> list[dict]:
        raise NotImplementedError

    async def transform(self, raw: list[dict]) -> list[dict]:
        raise NotImplementedError
