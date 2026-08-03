"""Scoring engine -- weighted aggregation, rating, price targets, risk classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import structlog

from trader_agent.analysis.analyst import AnalystSignals
from trader_agent.analysis.fundamental import FundamentalSignals
from trader_agent.analysis.sentiment import SentimentSignals
from trader_agent.analysis.technical import TechnicalSignals

logger = structlog.get_logger(__name__)


class Rating(StrEnum):
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    WEAK_BUY = "WEAK BUY"
    HOLD = "HOLD"
    DONT_BUY = "DON'T BUY"


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


SIGNAL_WEIGHTS = {
    "technical": 0.30,
    "fundamental": 0.25,
    "sentiment": 0.25,
    "analyst": 0.20,
}


@dataclass
class PriceTarget:
    horizon_label: str  # "1 Month", "2 Months", "3 Months"
    horizon_months: int
    bear_case: float
    base_case: float
    bull_case: float


@dataclass
class AnalysisResult:
    """Complete analysis output for a single ticker."""

    symbol: str
    company_name: str
    sector: str
    current_price: float
    currency: str

    composite_score: float = 0.0
    rating: Rating = Rating.HOLD
    risk_level: RiskLevel = RiskLevel.MEDIUM

    technical: TechnicalSignals = field(default_factory=TechnicalSignals)
    fundamental: FundamentalSignals = field(default_factory=FundamentalSignals)
    sentiment: SentimentSignals = field(default_factory=SentimentSignals)
    analyst: AnalystSignals = field(default_factory=AnalystSignals)

    price_targets: list[PriceTarget] = field(default_factory=list)

    signal_breakdown: dict[str, float] = field(default_factory=dict)


def compute_analysis(
    symbol: str,
    company_name: str,
    sector: str,
    current_price: float,
    currency: str,
    technical: TechnicalSignals,
    fundamental: FundamentalSignals,
    sentiment: SentimentSignals,
    analyst: AnalystSignals,
) -> AnalysisResult:
    """Aggregate all signals into a final recommendation."""
    result = AnalysisResult(
        symbol=symbol,
        company_name=company_name,
        sector=sector,
        current_price=current_price,
        currency=currency,
        technical=technical,
        fundamental=fundamental,
        sentiment=sentiment,
        analyst=analyst,
    )

    result.signal_breakdown = {
        "technical": technical.composite,
        "fundamental": fundamental.composite,
        "sentiment": sentiment.composite,
        "analyst": analyst.composite,
    }

    result.composite_score = sum(
        result.signal_breakdown[k] * SIGNAL_WEIGHTS[k] for k in SIGNAL_WEIGHTS
    )
    result.composite_score = float(np.clip(result.composite_score, -1.0, 1.0))

    result.rating = _map_rating(result.composite_score)

    result.risk_level = _classify_risk(
        volatility_annual=technical.volatility_annual,
        debt_to_equity=fundamental.debt_to_equity,
    )

    result.price_targets = _project_prices(
        current_price=current_price,
        composite_score=result.composite_score,
        volatility_annual=technical.volatility_annual,
    )

    logger.info(
        "analysis_complete",
        symbol=symbol,
        composite=round(result.composite_score, 3),
        rating=result.rating.value,
        risk=result.risk_level.value,
    )
    return result


def _map_rating(score: float) -> Rating:
    if score >= 0.6:
        return Rating.STRONG_BUY
    if score >= 0.3:
        return Rating.BUY
    if score >= 0.1:
        return Rating.WEAK_BUY
    if score >= -0.1:
        return Rating.HOLD
    return Rating.DONT_BUY


def _classify_risk(
    volatility_annual: float,
    debt_to_equity: float | None,
) -> RiskLevel:
    de = debt_to_equity if debt_to_equity is not None else 0.5

    if volatility_annual >= 0.40 or de >= 2.0:
        return RiskLevel.HIGH
    if volatility_annual < 0.25 and de < 1.0:
        return RiskLevel.LOW
    return RiskLevel.MEDIUM


def _project_prices(
    current_price: float,
    composite_score: float,
    volatility_annual: float,
) -> list[PriceTarget]:
    if current_price <= 0:
        return []

    vol = max(volatility_annual, 0.10)
    monthly_drift = composite_score * 0.015

    targets = []
    for months, label in [(1, "1 Month"), (2, "2 Months"), (3, "3 Months")]:
        base = current_price * (1 + monthly_drift * months)
        monthly_vol = vol / np.sqrt(12)
        std_range = current_price * monthly_vol * np.sqrt(months)

        targets.append(
            PriceTarget(
                horizon_label=label,
                horizon_months=months,
                bear_case=round(base - std_range, 2),
                base_case=round(base, 2),
                bull_case=round(base + std_range, 2),
            )
        )
    return targets
