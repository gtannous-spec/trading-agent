"""Analyst consensus signal from yfinance recommendations data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class AnalystSignals:
    total_recommendations: int = 0
    strong_buy: int = 0
    buy: int = 0
    hold: int = 0
    sell: int = 0
    strong_sell: int = 0
    consensus_label: str = "N/A"
    composite: float = 0.0  # [-1, 1]


def compute_analyst_signals(recommendations: pd.DataFrame) -> AnalystSignals:
    """Parse yfinance recommendations into a consensus signal.

    yfinance returns a DataFrame with columns like 'strongBuy', 'buy', 'hold', 'sell', 'strongSell'
    or alternatively 'To Grade' / 'Action' style columns depending on the ticker.
    """
    signals = AnalystSignals()

    if recommendations is None or recommendations.empty:
        logger.info("no_analyst_recommendations")
        return signals

    cols = {c.lower().replace(" ", ""): c for c in recommendations.columns}

    if "strongbuy" in cols:
        signals = _parse_summary_format(recommendations, cols, signals)
    elif "tograde" in cols or "action" in cols:
        signals = _parse_detailed_format(recommendations, cols, signals)
    else:
        logger.warning("unknown_recommendations_format", columns=list(recommendations.columns))
        return signals

    total = signals.strong_buy + signals.buy + signals.hold + signals.sell + signals.strong_sell
    signals.total_recommendations = total

    if total > 0:
        bullish = signals.strong_buy * 2 + signals.buy
        bearish = signals.strong_sell * 2 + signals.sell
        max_score = total * 2
        signals.composite = float(np.clip((bullish - bearish) / max_score, -1.0, 1.0))
        signals.consensus_label = _label_consensus(signals)

    logger.info(
        "analyst_signals_computed",
        total=signals.total_recommendations,
        consensus=signals.consensus_label,
        composite=round(signals.composite, 3),
    )
    return signals


def _parse_summary_format(
    df: pd.DataFrame, cols: dict[str, str], signals: AnalystSignals
) -> AnalystSignals:
    """Parse the summary format with strongBuy/buy/hold/sell/strongSell columns."""
    recent = df.head(3)

    for key, attr in [
        ("strongbuy", "strong_buy"),
        ("buy", "buy"),
        ("hold", "hold"),
        ("sell", "sell"),
        ("strongsell", "strong_sell"),
    ]:
        if key in cols:
            setattr(signals, attr, int(recent[cols[key]].sum()))
    return signals


def _parse_detailed_format(
    df: pd.DataFrame, cols: dict[str, str], signals: AnalystSignals
) -> AnalystSignals:
    """Parse the detailed format with 'To Grade' / 'Action' columns."""
    grade_col = cols.get("tograde") or cols.get("action")
    if grade_col is None:
        return signals

    recent = df.head(20)
    grades = recent[grade_col].str.lower().str.strip()

    grade_map = {
        "strong buy": "strong_buy",
        "buy": "buy",
        "outperform": "buy",
        "overweight": "buy",
        "hold": "hold",
        "neutral": "hold",
        "equal-weight": "hold",
        "market perform": "hold",
        "sell": "sell",
        "underperform": "sell",
        "underweight": "sell",
        "strong sell": "strong_sell",
    }

    for grade in grades:
        attr = grade_map.get(grade, "hold")
        setattr(signals, attr, getattr(signals, attr) + 1)
    return signals


def _label_consensus(signals: AnalystSignals) -> str:
    if signals.composite >= 0.6:
        return "Strong Buy"
    if signals.composite >= 0.3:
        return "Buy"
    if signals.composite >= 0.1:
        return "Moderate Buy"
    if signals.composite >= -0.1:
        return "Hold"
    if signals.composite >= -0.3:
        return "Moderate Sell"
    return "Sell"
