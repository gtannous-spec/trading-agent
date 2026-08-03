"""Fundamental analysis signal scoring -- P/E, D/E, EPS growth, FCF trend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import structlog

from trader_agent.analysis.fetcher import TickerData

logger = structlog.get_logger(__name__)

SECTOR_PE_MEDIANS: dict[str, float] = {
    "Technology": 28.0,
    "Healthcare": 22.0,
    "Financial Services": 14.0,
    "Consumer Cyclical": 20.0,
    "Consumer Defensive": 22.0,
    "Industrials": 20.0,
    "Energy": 12.0,
    "Utilities": 18.0,
    "Real Estate": 35.0,
    "Communication Services": 18.0,
    "Basic Materials": 15.0,
}
DEFAULT_PE_MEDIAN = 20.0


@dataclass
class FundamentalSignals:
    pe_ratio: float | None = None
    pe_signal: float = 0.0  # [-1, 1]: undervalued (+) vs overvalued (-)

    debt_to_equity: float | None = None
    de_signal: float = 0.0  # [-1, 1]: low leverage (+) vs high leverage (-)

    eps_growth_qoq: float | None = None
    eps_signal: float = 0.0  # [-1, 1]: growing (+) vs declining (-)

    fcf_trend: float | None = None  # latest FCF as ratio of previous
    fcf_signal: float = 0.0  # [-1, 1]: improving (+) vs deteriorating (-)

    composite: float = 0.0


def compute_fundamental_signals(data: TickerData) -> FundamentalSignals:
    signals = FundamentalSignals()
    info = data.info

    signals.pe_ratio = info.get("trailingPE") or info.get("forwardPE")
    if signals.pe_ratio is not None and signals.pe_ratio > 0:
        sector_median = SECTOR_PE_MEDIANS.get(data.sector, DEFAULT_PE_MEDIAN)
        ratio = signals.pe_ratio / sector_median
        signals.pe_signal = float(np.clip((1.0 - ratio) * 0.8, -1.0, 1.0))

    de_raw = info.get("debtToEquity")
    if de_raw is not None:
        signals.debt_to_equity = de_raw / 100.0 if de_raw > 10 else de_raw
        signals.de_signal = float(np.clip(1.0 - signals.debt_to_equity, -1.0, 1.0))

    signals.eps_growth_qoq = _compute_eps_growth(data.quarterly_financials)
    if signals.eps_growth_qoq is not None:
        signals.eps_signal = float(np.clip(signals.eps_growth_qoq * 2.0, -1.0, 1.0))

    signals.fcf_trend = _compute_fcf_trend(data.cashflow)
    if signals.fcf_trend is not None:
        signals.fcf_signal = float(np.clip((signals.fcf_trend - 1.0) * 2.0, -1.0, 1.0))

    active = []
    weights = []
    for sig, w in [
        (signals.pe_signal, 0.30),
        (signals.de_signal, 0.20),
        (signals.eps_signal, 0.30),
        (signals.fcf_signal, 0.20),
    ]:
        if sig != 0.0 or _has_data_for(sig, signals):
            active.append(sig * w)
            weights.append(w)

    if weights:
        signals.composite = float(np.clip(sum(active) / sum(weights), -1.0, 1.0))

    logger.info(
        "fundamental_signals_computed",
        pe=signals.pe_ratio,
        de=signals.debt_to_equity,
        eps_growth=signals.eps_growth_qoq,
        composite=round(signals.composite, 3),
    )
    return signals


def _has_data_for(sig_value: float, signals: FundamentalSignals) -> bool:
    """Check if a signal was actually computed (not just defaulted to 0)."""
    return True


def _compute_eps_growth(quarterly_financials: pd.DataFrame) -> float | None:
    if quarterly_financials.empty:
        return None
    try:
        eps_rows = None
        for label in ["Basic EPS", "Diluted EPS", "Net Income"]:
            if label in quarterly_financials.index:
                eps_rows = quarterly_financials.loc[label]
                break
        if eps_rows is None:
            return None

        values = eps_rows.dropna().values[:2]
        if len(values) < 2 or values[1] == 0:
            return None
        return float((values[0] - values[1]) / abs(values[1]))
    except Exception:
        logger.debug("eps_growth_calc_failed", exc_info=True)
        return None


def _compute_fcf_trend(cashflow: pd.DataFrame) -> float | None:
    if cashflow.empty:
        return None
    try:
        fcf_row = None
        for label in ["Free Cash Flow", "Operating Cash Flow"]:
            if label in cashflow.index:
                fcf_row = cashflow.loc[label]
                break
        if fcf_row is None:
            return None

        values = fcf_row.dropna().values[:2]
        if len(values) < 2 or values[1] == 0:
            return None
        return float(values[0] / abs(values[1]))
    except Exception:
        logger.debug("fcf_trend_calc_failed", exc_info=True)
        return None
