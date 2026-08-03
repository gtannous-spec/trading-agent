"""Tests for technical signal computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trader_agent.analysis.technical import compute_technical_signals


@pytest.fixture
def uptrend_prices() -> pd.DataFrame:
    """Simulate 250 trading days of a steady uptrend."""
    np.random.seed(42)
    n = 250
    base = 100.0
    returns = np.random.normal(0.0005, 0.015, n)
    close = base * np.cumprod(1 + returns)
    return pd.DataFrame(
        {
            "Open": close * 0.998,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, n),
        }
    )


@pytest.fixture
def downtrend_prices() -> pd.DataFrame:
    """Simulate 250 trading days of a steady downtrend."""
    np.random.seed(123)
    n = 250
    base = 200.0
    returns = np.random.normal(-0.001, 0.02, n)
    close = base * np.cumprod(1 + returns)
    return pd.DataFrame(
        {
            "Open": close * 1.002,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, n),
        }
    )


def test_uptrend_has_positive_composite(uptrend_prices):
    signals = compute_technical_signals(uptrend_prices)
    assert signals.composite > -1.0
    assert signals.composite <= 1.0
    assert signals.rsi_14 > 0
    assert signals.volatility_annual > 0


def test_downtrend_has_lower_composite_than_uptrend(uptrend_prices, downtrend_prices):
    up_signals = compute_technical_signals(uptrend_prices)
    down_signals = compute_technical_signals(downtrend_prices)
    assert down_signals.composite < up_signals.composite


def test_rsi_in_valid_range(uptrend_prices):
    signals = compute_technical_signals(uptrend_prices)
    assert 0 <= signals.rsi_14 <= 100


def test_empty_prices_returns_defaults():
    signals = compute_technical_signals(pd.DataFrame())
    assert signals.composite == 0.0
    assert signals.rsi_14 == 50.0


def test_short_series_returns_defaults():
    short = pd.DataFrame({"Close": [100, 101, 102]})
    signals = compute_technical_signals(short)
    assert signals.composite == 0.0
