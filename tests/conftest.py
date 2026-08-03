"""Shared test fixtures -- database sessions, mock API responses, sample data."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_ticker() -> str:
    return "AAPL"


@pytest.fixture
def sample_news_article() -> dict:
    return {
        "title": "Apple Reports Record Q3 Earnings",
        "source": "Reuters",
        "published_at": "2026-07-28T14:30:00Z",
        "summary": "Apple Inc reported better-than-expected quarterly results.",
        "url": "https://example.com/article/1",
        "ticker": "AAPL",
    }
