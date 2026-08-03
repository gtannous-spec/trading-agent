"""Sentiment scoring for news headlines -- FinBERT (full) or VADER (fast mode)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SentimentSignals:
    headlines_scored: int = 0
    avg_score: float = 0.0  # [-1 (bearish) .. +1 (bullish)]
    bullish_pct: float = 0.0
    bearish_pct: float = 0.0
    neutral_pct: float = 0.0
    individual_scores: list[tuple[str, float]] = field(default_factory=list)
    composite: float = 0.0
    method: str = "none"


def compute_sentiment_signals(
    news: list[dict],
    fast_mode: bool = False,
) -> SentimentSignals:
    """Score news headlines and return aggregated sentiment signals.

    Args:
        news: list of yfinance news dicts (each has 'title' key)
        fast_mode: if True, use VADER (instant); if False, use FinBERT (slower but better)
    """
    signals = SentimentSignals()

    headlines = _extract_headlines(news)
    if not headlines:
        logger.info("no_headlines_to_score")
        return signals

    if fast_mode:
        scores = _score_vader(headlines)
        signals.method = "vader"
    else:
        scores = _score_finbert(headlines)
        signals.method = "finbert"

    signals.headlines_scored = len(scores)
    signals.individual_scores = list(zip(headlines, scores, strict=False))

    if scores:
        arr = np.array(scores)
        signals.avg_score = float(np.mean(arr))
        signals.bullish_pct = float(np.mean(arr > 0.05) * 100)
        signals.bearish_pct = float(np.mean(arr < -0.05) * 100)
        signals.neutral_pct = float(np.mean(np.abs(arr) <= 0.05) * 100)
        signals.composite = float(np.clip(signals.avg_score, -1.0, 1.0))

    logger.info(
        "sentiment_signals_computed",
        method=signals.method,
        headlines=signals.headlines_scored,
        avg_score=round(signals.avg_score, 3),
        composite=round(signals.composite, 3),
    )
    return signals


def _extract_headlines(news: list[dict]) -> list[str]:
    headlines = []
    for item in news:
        title = item.get("title") or ""
        if not title:
            content = item.get("content", {})
            if isinstance(content, dict):
                title = content.get("title", "")
        title = title.strip()
        if title and len(title) > 10:
            headlines.append(title)
    return headlines[:50]


def _score_vader(headlines: list[str]) -> list[float]:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    scores = []
    for h in headlines:
        vs = analyzer.polarity_scores(h)
        scores.append(vs["compound"])
    return scores


def _score_finbert(headlines: list[str]) -> list[float]:
    from transformers import pipeline as hf_pipeline

    logger.info("loading_finbert_for_sentiment")
    pipe = hf_pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        device="cpu",
        top_k=None,
    )

    scores = []
    results = pipe(headlines, batch_size=16, truncation=True, max_length=512)
    for result in results:
        score_map = {r["label"]: r["score"] for r in result}
        pos = score_map.get("positive", 0.0)
        neg = score_map.get("negative", 0.0)
        scores.append(pos - neg)

    return scores
