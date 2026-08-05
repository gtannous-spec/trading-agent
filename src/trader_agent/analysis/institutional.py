"""Institutional holdings and insider transaction signals from yfinance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import structlog

from trader_agent.analysis.fetcher import TickerData

logger = structlog.get_logger(__name__)


@dataclass
class InstitutionalSignals:
    institutional_pct: float = 0.0
    institutional_holders_count: int = 0
    ownership_signal: float = 0.0  # [-1, 1]: high ownership = bullish

    insider_buy_count: int = 0
    insider_sell_count: int = 0
    insider_net_signal: float = 0.0  # [-1, 1]: net buying = bullish

    composite: float = 0.0


def compute_institutional_signals(data: TickerData) -> InstitutionalSignals:
    """Derive institutional confidence from holdings data and insider transactions."""
    signals = InstitutionalSignals()

    signals.institutional_pct = data.institutional_pct

    if not data.institutional_holders.empty:
        signals.institutional_holders_count = len(data.institutional_holders)

    signals.ownership_signal = _compute_ownership_signal(signals.institutional_pct)

    buys, sells = _parse_insider_transactions(data.insider_transactions)
    signals.insider_buy_count = buys
    signals.insider_sell_count = sells
    signals.insider_net_signal = _compute_insider_signal(buys, sells)

    active = []
    weights = []

    if signals.institutional_pct > 0:
        active.append(signals.ownership_signal * 0.60)
        weights.append(0.60)
    if buys + sells > 0:
        active.append(signals.insider_net_signal * 0.40)
        weights.append(0.40)

    if weights:
        signals.composite = float(np.clip(sum(active) / sum(weights), -1.0, 1.0))

    logger.info(
        "institutional_signals_computed",
        inst_pct=round(signals.institutional_pct, 3),
        holders=signals.institutional_holders_count,
        insider_buys=buys,
        insider_sells=sells,
        composite=round(signals.composite, 3),
    )
    return signals


def _compute_ownership_signal(institutional_pct: float) -> float:
    """Map institutional ownership % to a [-1, 1] signal.

    >70% = strong bullish, 40-70% = moderate, <20% = bearish (less pro confidence).
    """
    if institutional_pct <= 0:
        return 0.0
    return float(np.clip((institutional_pct - 0.35) * 2.5, -1.0, 1.0))


def _parse_insider_transactions(transactions: pd.DataFrame) -> tuple[int, int]:
    """Count recent insider buys and sells."""
    if transactions is None or transactions.empty:
        return 0, 0

    buys = 0
    sells = 0

    cols = {c.lower().replace(" ", "_"): c for c in transactions.columns}

    shares_col = cols.get("shares") or cols.get("number_of_shares")
    text_col = cols.get("text") or cols.get("transaction") or cols.get("transaction_type")

    if shares_col and shares_col in transactions.columns:
        for val in transactions[shares_col]:
            try:
                v = float(val)
                if v > 0:
                    buys += 1
                elif v < 0:
                    sells += 1
            except (ValueError, TypeError):
                pass
    elif text_col and text_col in transactions.columns:
        for val in transactions[text_col]:
            t = str(val).lower()
            if "buy" in t or "purchase" in t or "acquisition" in t:
                buys += 1
            elif "sell" in t or "sale" in t or "disposition" in t:
                sells += 1

    return buys, sells


def _compute_insider_signal(buys: int, sells: int) -> float:
    """Map net insider activity to a [-1, 1] signal.

    Asymmetric: buying is a stronger signal than selling because
    insiders sell for many benign reasons (taxes, diversification).
    """
    total = buys + sells
    if total == 0:
        return 0.0

    buy_weight = buys * 1.5
    sell_weight = sells * 0.7

    net = buy_weight - sell_weight
    max_possible = total * 1.5

    return float(np.clip(net / max_possible, -1.0, 1.0))
