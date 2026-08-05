"""Social sentiment -- StockTwits public stream + Finviz news headlines.

Both sources are free, require no API keys, and are reliably accessible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from html.parser import HTMLParser

import httpx
import numpy as np
import structlog

logger = structlog.get_logger(__name__)

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

FETCH_TIMEOUT = 12.0


@dataclass
class SocialSignals:
    post_count: int = 0
    avg_score: float = 0.0
    bullish_pct: float = 0.0
    bearish_pct: float = 0.0
    neutral_pct: float = 0.0
    source_breakdown: dict[str, dict] = field(default_factory=dict)
    composite: float = 0.0


def compute_social_signals(
    symbol: str,
    company_name: str = "",
    fast_mode: bool = True,
) -> SocialSignals:
    """Fetch and score social posts from StockTwits and Finviz."""
    signals = SocialSignals()

    all_texts: list[str] = []
    all_weights: list[float] = []
    all_sources: list[str] = []

    st_posts = _fetch_stocktwits(symbol)
    for post in st_posts:
        all_texts.append(post["text"])
        all_weights.append(post["weight"])
        all_sources.append("stocktwits")

    time.sleep(1.0)

    fv_headlines = _fetch_finviz_news(symbol)
    for headline in fv_headlines:
        all_texts.append(headline["text"])
        all_weights.append(headline["weight"])
        all_sources.append("finviz")

    if not all_texts:
        logger.info("no_social_posts_found", symbol=symbol)
        return signals

    scores = _score_texts(all_texts, fast_mode)
    signals.post_count = len(scores)

    per_source: dict[str, list[float]] = {}
    weighted_scores: list[float] = []
    for score, weight, source in zip(scores, all_weights, all_sources, strict=True):
        weighted_scores.append(score * weight)
        per_source.setdefault(source, []).append(score)

    arr = np.array(scores)
    signals.avg_score = float(np.mean(arr))
    signals.bullish_pct = float(np.mean(arr > 0.05) * 100)
    signals.bearish_pct = float(np.mean(arr < -0.05) * 100)
    signals.neutral_pct = float(np.mean(np.abs(arr) <= 0.05) * 100)

    total_weight = sum(all_weights)
    if total_weight > 0:
        signals.composite = float(np.clip(sum(weighted_scores) / total_weight, -1.0, 1.0))
    else:
        signals.composite = float(np.clip(signals.avg_score, -1.0, 1.0))

    for source, source_scores in per_source.items():
        sub_arr = np.array(source_scores)
        signals.source_breakdown[source] = {
            "count": len(source_scores),
            "avg_score": round(float(np.mean(sub_arr)), 3),
        }

    logger.info(
        "social_signals_computed",
        symbol=symbol,
        posts=signals.post_count,
        avg_score=round(signals.avg_score, 3),
        composite=round(signals.composite, 3),
        sources=list(per_source.keys()),
    )
    return signals


# ---------------------------------------------------------------------------
# StockTwits: public symbol stream (no API key needed)
# ---------------------------------------------------------------------------

_STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
_SENTIMENT_MAP = {"Bullish": 0.5, "Bearish": -0.5}


def _fetch_stocktwits(symbol: str) -> list[dict]:
    """Return recent messages from the StockTwits public stream."""
    url = _STOCKTWITS_URL.format(symbol=symbol.upper())
    try:
        resp = httpx.get(
            url,
            headers={"User-Agent": BROWSER_HEADERS["User-Agent"]},
            timeout=FETCH_TIMEOUT,
            follow_redirects=True,
        )
        if resp.status_code in (403, 429, 404):
            logger.info("stocktwits_unavailable", symbol=symbol, status=resp.status_code)
            return []
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.info("stocktwits_fetch_failed", symbol=symbol)
        return []

    results: list[dict] = []
    for msg in data.get("messages", []):
        body = (msg.get("body") or "").strip()
        if not body or len(body) < 10:
            continue
        sentiment = msg.get("entities", {}).get("sentiment", {})
        label = sentiment.get("basic") if sentiment else None
        preset = _SENTIMENT_MAP.get(label, 0.0) if label else 0.0
        likes = msg.get("likes", {}).get("total", 0) if isinstance(msg.get("likes"), dict) else 0
        weight = 1.0 + np.log1p(likes) * 0.3
        if preset != 0.0:
            weight *= 1.2
        results.append({"text": body, "weight": float(weight), "preset_score": preset})

    logger.info("stocktwits_fetched", symbol=symbol, messages=len(results))
    return results


# ---------------------------------------------------------------------------
# Finviz: news headlines from the ticker page
# ---------------------------------------------------------------------------

_FINVIZ_URL = "https://finviz.com/quote.ashx?t={symbol}&p=d"


class _FinvizNewsParser(HTMLParser):
    """Minimal HTML parser that extracts news headline text from the Finviz
    ticker page.  Headlines live inside ``<a>`` tags within the news table
    (id="news-table").
    """

    def __init__(self) -> None:
        super().__init__()
        self.in_news_table = False
        self.in_link = False
        self.headlines: list[str] = []
        self._current: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict = dict(attrs)
        if tag == "table" and attr_dict.get("id") == "news-table":
            self.in_news_table = True
        if self.in_news_table and tag == "a" and attr_dict.get("class") == "tab-link-news":
            self.in_link = True
            self._current = []

    def handle_endtag(self, tag: str) -> None:
        if self.in_link and tag == "a":
            text = "".join(self._current).strip()
            if text and len(text) >= 15:
                self.headlines.append(text)
            self.in_link = False

    def handle_data(self, data: str) -> None:
        if self.in_link:
            self._current.append(data)


def _fetch_finviz_news(symbol: str) -> list[dict]:
    """Scrape news headlines from the Finviz ticker page."""
    url = _FINVIZ_URL.format(symbol=symbol.upper())
    try:
        resp = httpx.get(url, headers=BROWSER_HEADERS, timeout=FETCH_TIMEOUT, follow_redirects=True)
        if resp.status_code in (403, 429, 404):
            logger.info("finviz_unavailable", symbol=symbol, status=resp.status_code)
            return []
        resp.raise_for_status()
    except Exception:
        logger.info("finviz_fetch_failed", symbol=symbol)
        return []

    parser = _FinvizNewsParser()
    try:
        parser.feed(resp.text)
    except Exception:
        logger.info("finviz_parse_failed", symbol=symbol)
        return []

    results: list[dict] = []
    for headline in parser.headlines[:30]:
        results.append({"text": headline, "weight": 1.0})

    logger.info("finviz_fetched", symbol=symbol, headlines=len(results))
    return results


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score_texts(texts: list[str], fast_mode: bool) -> list[float]:
    if fast_mode:
        return _score_vader(texts)
    return _score_finbert(texts)


def _score_vader(texts: list[str]) -> list[float]:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    return [analyzer.polarity_scores(t)["compound"] for t in texts]


def _score_finbert(texts: list[str]) -> list[float]:
    from transformers import pipeline as hf_pipeline

    pipe = hf_pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        device="cpu",
        top_k=None,
    )
    scores = []
    results = pipe(
        [t[:512] for t in texts],
        batch_size=16,
        truncation=True,
        max_length=512,
    )
    for result in results:
        score_map = {r["label"]: r["score"] for r in result}
        scores.append(score_map.get("positive", 0.0) - score_map.get("negative", 0.0))
    return scores
