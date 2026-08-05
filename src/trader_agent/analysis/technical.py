"""Technical analysis signal computation -- RSI, MACD, SMA crossovers, Bollinger Bands, volatility."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TechnicalSignals:
    """Computed technical indicators and a composite signal in [-1, 1]."""

    rsi_14: float = 50.0
    rsi_signal: float = 0.0  # [-1, 1]: <30 bullish, >70 bearish

    macd_value: float = 0.0
    macd_signal_line: float = 0.0
    macd_histogram: float = 0.0
    macd_signal: float = 0.0  # [-1, 1]: histogram direction

    sma_50: float = 0.0
    sma_200: float = 0.0
    sma_crossover_signal: float = 0.0  # +1 golden cross, -1 death cross

    bb_upper: float = 0.0
    bb_lower: float = 0.0
    bb_position: float = 0.5  # 0=at lower band, 1=at upper band
    bb_signal: float = 0.0  # [-1, 1]: oversold/overbought

    volatility_30d: float = 0.0  # annualized 30-day std dev of returns
    volatility_annual: float = 0.0

    composite: float = 0.0  # weighted average of all signals


def compute_technical_signals(prices: pd.DataFrame) -> TechnicalSignals:
    """Compute all technical indicators from a DataFrame with a 'Close' column."""
    signals = TechnicalSignals()

    if prices.empty or len(prices) < 30:
        logger.warning("insufficient_price_data", rows=len(prices))
        return signals

    close = prices["Close"].astype(float).dropna()
    if len(close) < 30:
        logger.warning("insufficient_clean_price_data", rows=len(close))
        return signals

    signals.rsi_14 = _rsi(close, 14)
    signals.rsi_signal = _rsi_to_signal(signals.rsi_14)

    macd_line, signal_line, histogram = _macd(close)
    signals.macd_value = macd_line
    signals.macd_signal_line = signal_line
    signals.macd_histogram = histogram
    signals.macd_signal = np.clip(histogram / max(abs(close.iloc[-1]) * 0.02, 1e-9), -1.0, 1.0)

    signals.sma_50 = close.rolling(window=min(50, len(close))).mean().iloc[-1]
    if len(close) >= 200:
        signals.sma_200 = close.rolling(window=200).mean().iloc[-1]
    else:
        signals.sma_200 = close.rolling(window=len(close)).mean().iloc[-1]

    if signals.sma_200 > 0:
        ratio = signals.sma_50 / signals.sma_200
        signals.sma_crossover_signal = np.clip((ratio - 1.0) * 10, -1.0, 1.0)
    else:
        signals.sma_crossover_signal = 0.0

    _bb_mid, bb_upper, bb_lower = _bollinger_bands(close, 20, 2.0)
    signals.bb_upper = bb_upper
    signals.bb_lower = bb_lower
    bb_range = bb_upper - bb_lower
    if bb_range > 0:
        signals.bb_position = (close.iloc[-1] - bb_lower) / bb_range
    else:
        signals.bb_position = 0.5
    signals.bb_signal = np.clip((0.5 - signals.bb_position) * 2, -1.0, 1.0)

    returns = close.pct_change().dropna()
    if len(returns) >= 30:
        signals.volatility_30d = float(returns.tail(30).std() * np.sqrt(252))
    signals.volatility_annual = float(returns.std() * np.sqrt(252))

    def _safe(val: float) -> float:
        return 0.0 if (np.isnan(val) or np.isinf(val)) else val

    signals.rsi_signal = _safe(signals.rsi_signal)
    signals.macd_signal = _safe(signals.macd_signal)
    signals.sma_crossover_signal = _safe(signals.sma_crossover_signal)
    signals.bb_signal = _safe(signals.bb_signal)

    signals.composite = (
        0.25 * signals.rsi_signal
        + 0.30 * signals.macd_signal
        + 0.25 * signals.sma_crossover_signal
        + 0.20 * signals.bb_signal
    )
    signals.composite = float(np.clip(signals.composite, -1.0, 1.0))

    logger.info(
        "technical_signals_computed",
        rsi=round(signals.rsi_14, 1),
        macd_hist=round(signals.macd_histogram, 4),
        sma_cross=round(signals.sma_crossover_signal, 2),
        composite=round(signals.composite, 3),
    )
    return signals


def _rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period, min_periods=1).mean()

    last_gain = gain.iloc[-1]
    last_loss = loss.iloc[-1]
    if pd.isna(last_gain) or pd.isna(last_loss):
        return 50.0
    if last_loss == 0:
        return 100.0 if last_gain > 0 else 50.0
    rs = last_gain / last_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _rsi_to_signal(rsi: float) -> float:
    """Map RSI to [-1, 1]: <30 is bullish (oversold), >70 is bearish (overbought)."""
    return float(np.clip((50.0 - rsi) / 30.0, -1.0, 1.0))


def _macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[float, float, float]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(histogram.iloc[-1])


def _bollinger_bands(
    series: pd.Series, period: int = 20, std_dev: float = 2.0
) -> tuple[float, float, float]:
    mid = series.rolling(window=period).mean().iloc[-1]
    std = series.rolling(window=period).std().iloc[-1]
    return float(mid), float(mid + std_dev * std), float(mid - std_dev * std)
