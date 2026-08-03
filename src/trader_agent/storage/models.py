"""SQLAlchemy ORM models for the trading agent data layer.

Tables
------
- tickers:          Tracked stock symbols and metadata.
- price_history:    OHLCV time-series (TimescaleDB hypertable candidate).
- fundamentals:     Quarterly/annual financial metrics per ticker.
- insider_trades:   SEC Form 4 insider transaction records.
- news_articles:    Raw financial news articles with JSONB payload.
- social_posts:     Raw social media posts with JSONB payload.
- sentiment_scores: NLP-derived sentiment linked back to source record.
- predictions:      Model output log -- ticker, probability, timestamp.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared base for all ORM models."""

    type_annotation_map: ClassVar[dict] = {dict[str, Any]: JSONB}


class Ticker(Base):
    __tablename__ = "tickers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(10), unique=True, nullable=False, index=True)
    name: Mapped[str | None] = mapped_column(String(255))
    sector: Mapped[str | None] = mapped_column(String(100))
    industry: Mapped[str | None] = mapped_column(String(100))
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    prices: Mapped[list[PriceHistory]] = relationship(back_populates="ticker")
    fundamentals: Mapped[list[Fundamental]] = relationship(back_populates="ticker")
    insider_trades: Mapped[list[InsiderTrade]] = relationship(back_populates="ticker")


class PriceHistory(Base):
    """OHLCV daily price data -- intended as a TimescaleDB hypertable on (time, ticker_id)."""

    __tablename__ = "price_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker_id: Mapped[int] = mapped_column(ForeignKey("tickers.id"), nullable=False)
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int] = mapped_column(Integer, nullable=False)

    ticker: Mapped[Ticker] = relationship(back_populates="prices")

    __table_args__ = (Index("ix_price_history_ticker_time", "ticker_id", "time"),)


class Fundamental(Base):
    __tablename__ = "fundamentals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker_id: Mapped[int] = mapped_column(ForeignKey("tickers.id"), nullable=False)
    report_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    period: Mapped[str] = mapped_column(String(10), nullable=False)  # "Q1", "Q2", "annual"

    revenue: Mapped[float | None] = mapped_column(Float)
    net_income: Mapped[float | None] = mapped_column(Float)
    eps: Mapped[float | None] = mapped_column(Float)
    pe_ratio: Mapped[float | None] = mapped_column(Float)
    debt_to_equity: Mapped[float | None] = mapped_column(Float)
    free_cash_flow: Mapped[float | None] = mapped_column(Float)
    raw_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    ticker: Mapped[Ticker] = relationship(back_populates="fundamentals")

    __table_args__ = (Index("ix_fundamentals_ticker_date", "ticker_id", "report_date"),)


class InsiderTrade(Base):
    __tablename__ = "insider_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker_id: Mapped[int] = mapped_column(ForeignKey("tickers.id"), nullable=False)
    filed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    insider_name: Mapped[str] = mapped_column(String(255), nullable=False)
    insider_title: Mapped[str | None] = mapped_column(String(255))
    transaction_type: Mapped[str] = mapped_column(String(50), nullable=False)  # "buy" | "sell"
    shares: Mapped[int] = mapped_column(Integer, nullable=False)
    price_per_share: Mapped[float | None] = mapped_column(Float)
    total_value: Mapped[float | None] = mapped_column(Float)
    raw_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    ticker: Mapped[Ticker] = relationship(back_populates="insider_trades")

    __table_args__ = (Index("ix_insider_trades_ticker_date", "ticker_id", "filed_at"),)


class NewsArticle(Base):
    __tablename__ = "news_articles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(100), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    summary: Mapped[str | None] = mapped_column(Text)
    raw_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    related_tickers: Mapped[list[str] | None] = mapped_column(JSONB)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (Index("ix_news_articles_published", "published_at"),)


class SocialPost(Base):
    __tablename__ = "social_posts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    platform: Mapped[str] = mapped_column(String(50), nullable=False)  # "twitter" | "stocktwits"
    external_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    author: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    posted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    related_tickers: Mapped[list[str] | None] = mapped_column(JSONB)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (Index("ix_social_posts_posted", "posted_at"),)


class SentimentScore(Base):
    """NLP-derived sentiment linked to a source news article or social post."""

    __tablename__ = "sentiment_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_type: Mapped[str] = mapped_column(String(20), nullable=False)  # "news" | "social"
    source_id: Mapped[int] = mapped_column(Integer, nullable=False)
    ticker_symbol: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    label: Mapped[str] = mapped_column(String(10), nullable=False)  # "bullish"|"bearish"|"neutral"
    score: Mapped[float] = mapped_column(Float, nullable=False)  # [-1.0 .. 1.0]
    raw_probabilities: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_sentiment_ticker_time", "ticker_symbol", "scored_at"),)


class Prediction(Base):
    """Model output log -- one row per ticker per prediction run."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker_symbol: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False, default=90)
    features_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    predicted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (Index("ix_predictions_ticker_time", "ticker_symbol", "predicted_at"),)
