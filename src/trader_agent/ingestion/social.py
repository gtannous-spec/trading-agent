"""Social media sentiment ingester -- pulls posts from X (Twitter) / StockTwits."""

from __future__ import annotations

from trader_agent.ingestion.base import BaseIngester


class SocialIngester(BaseIngester):
    """Fetch and store social media posts for tracked tickers and influencer accounts."""

    service_name = "social"

    async def fetch(self) -> list[dict]:
        raise NotImplementedError

    async def transform(self, raw: list[dict]) -> list[dict]:
        raise NotImplementedError
