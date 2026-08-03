"""FastAPI web application for the trading agent dashboard."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from trader_agent.analysis.analyst import compute_analyst_signals
from trader_agent.analysis.fetcher import fetch_ticker_data
from trader_agent.analysis.fundamental import compute_fundamental_signals
from trader_agent.analysis.resolver import resolve_symbol
from trader_agent.analysis.scoring import compute_analysis
from trader_agent.analysis.sentiment import compute_sentiment_signals
from trader_agent.analysis.technical import compute_technical_signals

logger = structlog.get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Trader Agent Dashboard", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text()


@app.get("/api/analyze")
async def analyze(query: str, fast: bool = True, period: str = "1y") -> dict[str, Any]:
    """Run the full analysis pipeline and return structured JSON."""
    query = query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required.")

    resolved = resolve_symbol(query)
    if resolved is None:
        raise HTTPException(
            status_code=404,
            detail=f"Could not find a stock matching '{query}'. Try a ticker symbol or company name.",
        )

    data = fetch_ticker_data(resolved.symbol, period=period)
    if data.current_price <= 0:
        raise HTTPException(
            status_code=404,
            detail=f"Found symbol {resolved.symbol} but no price data available.",
        )

    tech = compute_technical_signals(data.prices)
    fund = compute_fundamental_signals(data)
    sent = compute_sentiment_signals(data.news, fast_mode=fast)
    analyst = compute_analyst_signals(data.recommendations)

    result = compute_analysis(
        symbol=data.symbol,
        company_name=data.company_name,
        sector=data.sector,
        current_price=data.current_price,
        currency=data.currency,
        technical=tech,
        fundamental=fund,
        sentiment=sent,
        analyst=analyst,
    )

    payload = asdict(result)
    payload["resolved_from"] = query if resolved.matched_via == "search" else None
    payload["rating"] = result.rating.value
    payload["risk_level"] = result.risk_level.value

    payload["sentiment"].pop("individual_scores", None)

    return payload
