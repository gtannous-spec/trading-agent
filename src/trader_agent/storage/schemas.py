"""Pydantic schemas for request/response validation and data transfer objects."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Ticker
# ---------------------------------------------------------------------------


class TickerCreate(BaseModel):
    symbol: str = Field(..., max_length=10)
    name: str | None = None
    sector: str | None = None
    industry: str | None = None


class TickerRead(TickerCreate):
    model_config = ConfigDict(from_attributes=True)
    id: int
    is_active: bool
    created_at: datetime


# ---------------------------------------------------------------------------
# Price History
# ---------------------------------------------------------------------------


class PriceRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    ticker_id: int
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


# ---------------------------------------------------------------------------
# Insider Trade
# ---------------------------------------------------------------------------


class InsiderTradeRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    ticker_id: int
    filed_at: datetime
    insider_name: str
    insider_title: str | None = None
    transaction_type: str
    shares: int
    price_per_share: float | None = None
    total_value: float | None = None


# ---------------------------------------------------------------------------
# News Article
# ---------------------------------------------------------------------------


class NewsArticleRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    source: str
    title: str
    url: str
    published_at: datetime
    summary: str | None = None
    related_tickers: list[str] | None = None


# ---------------------------------------------------------------------------
# Social Post
# ---------------------------------------------------------------------------


class SocialPostRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    platform: str
    external_id: str
    author: str
    content: str
    posted_at: datetime
    related_tickers: list[str] | None = None


# ---------------------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------------------


class SentimentResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    source_type: str
    source_id: int
    ticker_symbol: str
    label: str
    score: float = Field(..., ge=-1.0, le=1.0)
    raw_probabilities: dict[str, float] | None = None


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class PredictionResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    ticker_symbol: str
    model_version: str
    probability: float = Field(..., ge=0.0, le=1.0)
    confidence: float | None = None
    horizon_days: int = 90
    predicted_at: datetime | None = None
