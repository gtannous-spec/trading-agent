"""Resolve a fuzzy user input (company name, partial ticker, etc.) to a valid stock symbol.

Handles inputs like "apple", "Marvel", "tesla", "MSFT", "google" and resolves
them to their canonical ticker symbols (AAPL, MRVL, TSLA, MSFT, GOOGL).
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog
import yfinance as yf

logger = structlog.get_logger(__name__)


@dataclass
class ResolvedTicker:
    input_query: str
    symbol: str
    company_name: str
    exchange: str = ""
    matched_via: str = "direct"  # "direct" | "search"


def resolve_symbol(query: str) -> ResolvedTicker | None:
    """Resolve a user query to a valid ticker symbol.

    Strategy:
    1. Try the raw input as a ticker symbol (fast path).
    2. If that fails, use yfinance search to find matches by company name.
    3. Return the best match, or None if nothing found.
    """
    query = query.strip()
    if not query:
        return None

    upper = query.upper()
    logger.info("resolving_symbol", query=query)

    direct = _try_direct_lookup(upper)
    if direct is not None:
        logger.info("resolved_direct", query=query, symbol=direct.symbol, name=direct.company_name)
        return direct

    search_result = _try_search(query)
    if search_result is not None:
        logger.info(
            "resolved_via_search",
            query=query,
            symbol=search_result.symbol,
            name=search_result.company_name,
        )
        return search_result

    logger.warning("symbol_resolution_failed", query=query)
    return None


def _try_direct_lookup(symbol: str) -> ResolvedTicker | None:
    """Try treating the input as a ticker symbol and validate it has data."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        name = info.get("longName") or info.get("shortName")
        quote_type = info.get("quoteType", "")

        if price and price > 0 and quote_type == "EQUITY":
            return ResolvedTicker(
                input_query=symbol,
                symbol=symbol,
                company_name=name or symbol,
                exchange=info.get("exchange", ""),
                matched_via="direct",
            )
    except Exception:
        pass
    return None


def _try_search(query: str) -> ResolvedTicker | None:
    """Search Yahoo Finance for the query and return the best equity match."""
    try:
        results = yf.Search(query)
        quotes = getattr(results, "quotes", []) or []

        for quote in quotes:
            quote_type = quote.get("quoteType", "")
            if quote_type != "EQUITY":
                continue

            symbol = quote.get("symbol", "")
            name = quote.get("longname") or quote.get("shortname") or ""
            exchange = quote.get("exchange", "")

            if symbol:
                return ResolvedTicker(
                    input_query=query,
                    symbol=symbol.upper(),
                    company_name=name,
                    exchange=exchange,
                    matched_via="search",
                )

    except Exception:
        logger.debug("search_failed", query=query, exc_info=True)

    return None
