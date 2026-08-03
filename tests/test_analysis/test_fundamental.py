"""Tests for fundamental signal scoring."""

from __future__ import annotations

import pandas as pd
import pytest

from trader_agent.analysis.fetcher import TickerData
from trader_agent.analysis.fundamental import compute_fundamental_signals


@pytest.fixture
def strong_company() -> TickerData:
    """Company with good fundamentals -- low P/E, low debt, growing EPS."""
    data = TickerData(symbol="GOOD", sector="Technology")
    data.info = {
        "trailingPE": 15.0,
        "debtToEquity": 40.0,
    }
    data.quarterly_financials = pd.DataFrame(
        {"2026Q2": [5.0], "2026Q1": [4.0]},
        index=["Basic EPS"],
    )
    data.cashflow = pd.DataFrame(
        {"2026Q2": [1_000_000], "2026Q1": [800_000]},
        index=["Free Cash Flow"],
    )
    return data


@pytest.fixture
def weak_company() -> TickerData:
    """Company with weak fundamentals -- high P/E, high debt, declining EPS."""
    data = TickerData(symbol="WEAK", sector="Technology")
    data.info = {
        "trailingPE": 80.0,
        "debtToEquity": 350.0,
    }
    data.quarterly_financials = pd.DataFrame(
        {"2026Q2": [2.0], "2026Q1": [4.0]},
        index=["Basic EPS"],
    )
    data.cashflow = pd.DataFrame(
        {"2026Q2": [500_000], "2026Q1": [1_000_000]},
        index=["Free Cash Flow"],
    )
    return data


def test_strong_company_positive_composite(strong_company):
    signals = compute_fundamental_signals(strong_company)
    assert signals.composite > 0.0
    assert signals.pe_signal > 0.0
    assert signals.eps_signal > 0.0


def test_weak_company_negative_composite(weak_company):
    signals = compute_fundamental_signals(weak_company)
    assert signals.composite < 0.0
    assert signals.pe_signal < 0.0


def test_strong_beats_weak(strong_company, weak_company):
    strong = compute_fundamental_signals(strong_company)
    weak = compute_fundamental_signals(weak_company)
    assert strong.composite > weak.composite


def test_missing_data_returns_zero():
    empty = TickerData(symbol="EMPTY")
    empty.info = {}
    signals = compute_fundamental_signals(empty)
    assert signals.composite == 0.0
