"""Fetch all data modalities for a ticker from yfinance into a single TickerData bundle."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import structlog
import yfinance as yf

logger = structlog.get_logger(__name__)


@dataclass
class TickerData:
    """All raw data needed for analysis, fetched in one shot."""

    symbol: str
    company_name: str = ""
    sector: str = ""
    industry: str = ""
    current_price: float = 0.0
    currency: str = "USD"
    market_cap: float = 0.0
    institutional_pct: float = 0.0
    revenue_growth: float | None = None

    prices: pd.DataFrame = field(default_factory=pd.DataFrame)
    financials: pd.DataFrame = field(default_factory=pd.DataFrame)
    quarterly_financials: pd.DataFrame = field(default_factory=pd.DataFrame)
    balance_sheet: pd.DataFrame = field(default_factory=pd.DataFrame)
    cashflow: pd.DataFrame = field(default_factory=pd.DataFrame)
    recommendations: pd.DataFrame = field(default_factory=pd.DataFrame)
    news: list[dict] = field(default_factory=list)

    institutional_holders: pd.DataFrame = field(default_factory=pd.DataFrame)
    major_holders: pd.DataFrame = field(default_factory=pd.DataFrame)
    insider_transactions: pd.DataFrame = field(default_factory=pd.DataFrame)

    info: dict = field(default_factory=dict)


def fetch_ticker_data(symbol: str, period: str = "1y") -> TickerData:
    """Pull all data modalities for a single ticker from yfinance.

    Returns a TickerData bundle with prices, financials, recommendations, and news.
    Gracefully handles missing data -- individual fields default to empty.
    """
    symbol = symbol.upper().strip()
    logger.info("fetching_ticker_data", symbol=symbol, period=period)

    ticker = yf.Ticker(symbol)
    data = TickerData(symbol=symbol)

    info = _safe_get(lambda: ticker.info, {}, "info", symbol)
    data.info = info
    data.company_name = info.get("longName") or info.get("shortName") or symbol
    data.sector = info.get("sector", "Unknown")
    data.industry = info.get("industry", "Unknown")
    data.currency = info.get("currency", "USD")
    data.current_price = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0
    data.market_cap = info.get("marketCap") or 0.0
    data.institutional_pct = info.get("heldPercentInstitutions") or 0.0
    data.revenue_growth = info.get("revenueGrowth")

    data.prices = _safe_get(lambda: ticker.history(period=period), pd.DataFrame(), "prices", symbol)

    data.financials = _safe_get(lambda: ticker.financials, pd.DataFrame(), "financials", symbol)
    data.quarterly_financials = _safe_get(
        lambda: ticker.quarterly_financials, pd.DataFrame(), "quarterly_financials", symbol
    )
    data.balance_sheet = _safe_get(
        lambda: ticker.quarterly_balance_sheet, pd.DataFrame(), "balance_sheet", symbol
    )
    data.cashflow = _safe_get(lambda: ticker.quarterly_cashflow, pd.DataFrame(), "cashflow", symbol)

    data.recommendations = _safe_get(
        lambda: ticker.recommendations, pd.DataFrame(), "recommendations", symbol
    )

    data.news = _safe_get(lambda: ticker.news or [], [], "news", symbol)

    data.institutional_holders = _safe_get(
        lambda: ticker.institutional_holders, pd.DataFrame(), "institutional_holders", symbol
    )
    data.major_holders = _safe_get(
        lambda: ticker.major_holders, pd.DataFrame(), "major_holders", symbol
    )
    data.insider_transactions = _safe_get(
        lambda: ticker.insider_transactions, pd.DataFrame(), "insider_transactions", symbol
    )

    logger.info(
        "fetch_complete",
        symbol=symbol,
        price_rows=len(data.prices),
        has_financials=not data.financials.empty,
        news_count=len(data.news),
        rec_count=len(data.recommendations),
    )
    return data


def _safe_get(fn, default, label: str, symbol: str):
    """Call fn() and return its result, or default on any exception."""
    try:
        result = fn()
        if result is None:
            return default
        return result
    except Exception:
        logger.warning("fetch_failed", symbol=symbol, field=label, exc_info=True)
        return default
