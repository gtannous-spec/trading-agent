"""Scoring engine -- weighted aggregation, rating, price targets, risk classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import structlog

from trader_agent.analysis.analyst import AnalystSignals
from trader_agent.analysis.fundamental import FundamentalSignals
from trader_agent.analysis.institutional import InstitutionalSignals
from trader_agent.analysis.sentiment import SentimentSignals
from trader_agent.analysis.social import SocialSignals
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
    "technical": 0.25,
    "fundamental": 0.20,
    "sentiment": 0.20,
    "analyst": 0.15,
    "social": 0.10,
    "institutional": 0.10,
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
    social: SocialSignals = field(default_factory=SocialSignals)
    institutional: InstitutionalSignals = field(default_factory=InstitutionalSignals)

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
    social: SocialSignals | None = None,
    institutional: InstitutionalSignals | None = None,
) -> AnalysisResult:
    """Aggregate all signals into a final recommendation."""
    social = social or SocialSignals()
    institutional = institutional or InstitutionalSignals()

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
        social=social,
        institutional=institutional,
    )

    def _safe(val: float) -> float:
        return 0.0 if (np.isnan(val) or np.isinf(val)) else float(val)

    result.signal_breakdown = {
        "technical": _safe(technical.composite),
        "fundamental": _safe(fundamental.composite),
        "sentiment": _safe(sentiment.composite),
        "analyst": _safe(analyst.composite),
        "social": _safe(social.composite),
        "institutional": _safe(institutional.composite),
    }

    result.composite_score = sum(
        result.signal_breakdown[k] * SIGNAL_WEIGHTS[k] for k in SIGNAL_WEIGHTS
    )
    result.composite_score = float(np.clip(result.composite_score, -1.0, 1.0))

    result.rating = _map_rating(result.composite_score)

    result.risk_level = _classify_risk(
        volatility_annual=technical.volatility_annual,
        debt_to_equity=fundamental.debt_to_equity,
        sector=sector,
        market_cap=fundamental.market_cap,
        institutional_pct=fundamental.institutional_pct,
        revenue_growth=fundamental.revenue_growth,
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


SECTOR_VOLATILITY_NORMS: dict[str, float] = {
    "Technology": 0.35,
    "Healthcare": 0.30,
    "Financial Services": 0.25,
    "Consumer Cyclical": 0.28,
    "Consumer Defensive": 0.22,
    "Industrials": 0.25,
    "Energy": 0.40,
    "Utilities": 0.18,
    "Real Estate": 0.25,
    "Communication Services": 0.30,
    "Basic Materials": 0.28,
}
DEFAULT_VOLATILITY_NORM = 0.28

SECTOR_DE_NORMS: dict[str, float] = {
    "Technology": 0.8,
    "Healthcare": 0.6,
    "Financial Services": 3.0,
    "Consumer Cyclical": 1.2,
    "Consumer Defensive": 1.0,
    "Industrials": 1.0,
    "Energy": 0.8,
    "Utilities": 1.5,
    "Real Estate": 1.8,
    "Communication Services": 1.0,
    "Basic Materials": 0.7,
}
DEFAULT_DE_NORM = 1.0


def _classify_risk(
    volatility_annual: float,
    debt_to_equity: float | None,
    sector: str = "Unknown",
    market_cap: float = 0.0,
    institutional_pct: float = 0.0,
    revenue_growth: float | None = None,
) -> RiskLevel:
    """Multi-factor risk scoring: 0.0 (safest) to 1.0 (riskiest)."""
    sub_scores: list[tuple[float, float]] = []

    vol_norm = SECTOR_VOLATILITY_NORMS.get(sector, DEFAULT_VOLATILITY_NORM)
    vol_ratio = volatility_annual / vol_norm if vol_norm > 0 else 1.0
    vol_risk = float(np.clip((vol_ratio - 0.5) / 1.5, 0.0, 1.0))
    sub_scores.append((vol_risk, 0.25))

    de = debt_to_equity if debt_to_equity is not None else 0.5
    de_norm = SECTOR_DE_NORMS.get(sector, DEFAULT_DE_NORM)
    de_ratio = de / de_norm if de_norm > 0 else 1.0
    de_risk = float(np.clip((de_ratio - 0.3) / 2.0, 0.0, 1.0))
    sub_scores.append((de_risk, 0.20))

    if market_cap > 200_000_000_000:
        cap_risk = 0.05
    elif market_cap > 50_000_000_000:
        cap_risk = 0.15
    elif market_cap > 10_000_000_000:
        cap_risk = 0.30
    elif market_cap > 2_000_000_000:
        cap_risk = 0.55
    elif market_cap > 300_000_000:
        cap_risk = 0.75
    else:
        cap_risk = 0.95
    sub_scores.append((cap_risk, 0.25))

    inst_risk = float(np.clip(1.0 - institutional_pct * 1.3, 0.0, 1.0))
    sub_scores.append((inst_risk, 0.15))

    rev_risk = float(np.clip(0.5 - revenue_growth, 0.0, 1.0)) if revenue_growth is not None else 0.5
    sub_scores.append((rev_risk, 0.15))

    total_weight = sum(w for _, w in sub_scores)
    risk_score = sum(s * w for s, w in sub_scores) / total_weight if total_weight > 0 else 0.5

    if risk_score >= 0.60:
        return RiskLevel.HIGH
    if risk_score <= 0.35:
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
