"""Financial news ingester -- pulls articles from NewsAPI / Finnhub news endpoints."""

from __future__ import annotations

from trader_agent.ingestion.base import BaseIngester


class NewsIngester(BaseIngester):
    """Fetch and store financial news articles for tracked tickers."""

    service_name = "news"

    async def fetch(self) -> list[dict]:
        raise NotImplementedError

    async def transform(self, raw: list[dict]) -> list[dict]:
        raise NotImplementedError
