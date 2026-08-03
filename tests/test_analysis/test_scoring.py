"""Tests for the scoring engine -- aggregation, rating, price targets, risk."""

from __future__ import annotations

from trader_agent.analysis.analyst import AnalystSignals
from trader_agent.analysis.fundamental import FundamentalSignals
from trader_agent.analysis.scoring import (
    Rating,
    RiskLevel,
    _classify_risk,
    _map_rating,
    compute_analysis,
)
from trader_agent.analysis.sentiment import SentimentSignals
from trader_agent.analysis.technical import TechnicalSignals


def test_map_rating_strong_buy():
    assert _map_rating(0.7) == Rating.STRONG_BUY


def test_map_rating_buy():
    assert _map_rating(0.4) == Rating.BUY


def test_map_rating_weak_buy():
    assert _map_rating(0.15) == Rating.WEAK_BUY


def test_map_rating_hold():
    assert _map_rating(0.0) == Rating.HOLD


def test_map_rating_dont_buy():
    assert _map_rating(-0.3) == Rating.DONT_BUY


def test_classify_risk_low():
    assert _classify_risk(0.20, 0.5) == RiskLevel.LOW


def test_classify_risk_medium():
    assert _classify_risk(0.30, 0.8) == RiskLevel.MEDIUM


def test_classify_risk_high_volatility():
    assert _classify_risk(0.45, 0.5) == RiskLevel.HIGH


def test_classify_risk_high_leverage():
    assert _classify_risk(0.20, 2.5) == RiskLevel.HIGH


def test_full_analysis_bullish():
    result = compute_analysis(
        symbol="BULL",
        company_name="Bull Corp",
        sector="Technology",
        current_price=150.0,
        currency="USD",
        technical=TechnicalSignals(composite=0.5),
        fundamental=FundamentalSignals(composite=0.4),
        sentiment=SentimentSignals(composite=0.6),
        analyst=AnalystSignals(composite=0.7),
    )
    assert result.composite_score > 0.3
    assert result.rating in (Rating.STRONG_BUY, Rating.BUY)
    assert len(result.price_targets) == 3
    assert result.price_targets[0].bull_case > result.current_price


def test_full_analysis_bearish():
    result = compute_analysis(
        symbol="BEAR",
        company_name="Bear Corp",
        sector="Energy",
        current_price=50.0,
        currency="USD",
        technical=TechnicalSignals(composite=-0.6, volatility_annual=0.5),
        fundamental=FundamentalSignals(composite=-0.5, debt_to_equity=3.0),
        sentiment=SentimentSignals(composite=-0.4),
        analyst=AnalystSignals(composite=-0.3),
    )
    assert result.composite_score < -0.1
    assert result.rating == Rating.DONT_BUY
    assert result.risk_level == RiskLevel.HIGH


def test_price_targets_have_correct_ordering():
    result = compute_analysis(
        symbol="TEST",
        company_name="Test",
        sector="Technology",
        current_price=100.0,
        currency="USD",
        technical=TechnicalSignals(composite=0.3, volatility_annual=0.25),
        fundamental=FundamentalSignals(composite=0.2),
        sentiment=SentimentSignals(composite=0.1),
        analyst=AnalystSignals(composite=0.2),
    )
    for pt in result.price_targets:
        assert pt.bear_case < pt.base_case < pt.bull_case
