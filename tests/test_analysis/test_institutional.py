"""Tests for institutional holdings and insider transaction signals."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from trader_agent.analysis.institutional import (
    InstitutionalSignals,
    _compute_insider_signal,
    _compute_ownership_signal,
    compute_institutional_signals,
)


def _make_ticker_data(**kwargs):
    """Build a minimal TickerData-like object for testing."""
    mock = MagicMock()
    mock.institutional_pct = kwargs.get("institutional_pct", 0.0)
    mock.institutional_holders = kwargs.get("institutional_holders", pd.DataFrame())
    mock.major_holders = kwargs.get("major_holders", pd.DataFrame())
    mock.insider_transactions = kwargs.get("insider_transactions", pd.DataFrame())
    return mock


def test_compute_institutional_signals_high_ownership():
    data = _make_ticker_data(
        institutional_pct=0.75,
        institutional_holders=pd.DataFrame({"Name": ["Vanguard", "BlackRock"]}),
    )
    signals = compute_institutional_signals(data)
    assert isinstance(signals, InstitutionalSignals)
    assert signals.institutional_pct == 0.75
    assert signals.ownership_signal > 0
    assert signals.institutional_holders_count == 2


def test_compute_institutional_signals_low_ownership():
    data = _make_ticker_data(institutional_pct=0.10)
    signals = compute_institutional_signals(data)
    assert signals.ownership_signal < 0


def test_compute_institutional_signals_with_insider_buys():
    tx = pd.DataFrame({"Shares": [1000, 500, -200]})
    data = _make_ticker_data(institutional_pct=0.50, insider_transactions=tx)
    signals = compute_institutional_signals(data)
    assert signals.insider_buy_count == 2
    assert signals.insider_sell_count == 1
    assert signals.insider_net_signal > 0


def test_compute_institutional_signals_empty():
    data = _make_ticker_data()
    signals = compute_institutional_signals(data)
    assert signals.composite == 0.0
    assert signals.insider_buy_count == 0


def test_ownership_signal_positive_for_high_ownership():
    assert _compute_ownership_signal(0.80) > 0.5


def test_ownership_signal_negative_for_low_ownership():
    assert _compute_ownership_signal(0.10) < 0


def test_ownership_signal_zero_for_no_data():
    assert _compute_ownership_signal(0.0) == 0.0


def test_insider_signal_bullish_net_buying():
    signal = _compute_insider_signal(buys=5, sells=1)
    assert signal > 0


def test_insider_signal_bearish_net_selling():
    signal = _compute_insider_signal(buys=0, sells=5)
    assert signal < 0


def test_insider_signal_zero_no_activity():
    assert _compute_insider_signal(0, 0) == 0.0


def test_insider_signal_asymmetric():
    """Buying should weigh more than selling."""
    buy_signal = _compute_insider_signal(buys=3, sells=0)
    sell_signal = _compute_insider_signal(buys=0, sells=3)
    assert abs(buy_signal) > abs(sell_signal)
