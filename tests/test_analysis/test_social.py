"""Tests for social sentiment (StockTwits + Finviz) with mocked HTTP responses."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from trader_agent.analysis.social import (
    SocialSignals,
    _fetch_finviz_news,
    _fetch_stocktwits,
    compute_social_signals,
)


def _mock_stocktwits_response(messages: list[dict]) -> dict:
    return {"messages": messages}


SAMPLE_ST_MESSAGES = [
    {
        "body": "AAPL earnings look fantastic, this stock is going to moon soon",
        "entities": {"sentiment": {"basic": "Bullish"}},
        "likes": {"total": 10},
    },
    {
        "body": "Apple is overvalued here, be careful with this one honestly",
        "entities": {"sentiment": {"basic": "Bearish"}},
        "likes": {"total": 3},
    },
    {
        "body": "short msg",
        "entities": {},
        "likes": {"total": 0},
    },
]

SAMPLE_FINVIZ_HTML = """
<html><body>
<table id="news-table">
<tr><td>Jul-30</td><td><a class="tab-link-news" href="https://example.com/1">Apple beats earnings expectations massively this quarter</a></td></tr>
<tr><td>10:30AM</td><td><a class="tab-link-news" href="https://example.com/2">Analysts raise AAPL price target on strong iPhone sales data</a></td></tr>
<tr><td>09:15AM</td><td><a class="tab-link-news" href="https://example.com/3">Short title</a></td></tr>
</table>
</body></html>
"""


@patch("trader_agent.analysis.social.httpx.get")
@patch("trader_agent.analysis.social.time.sleep")
def test_compute_social_signals_basic(mock_sleep, mock_get):
    def side_effect(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        if "stocktwits" in url:
            resp.json.return_value = _mock_stocktwits_response(SAMPLE_ST_MESSAGES)
        elif "finviz" in url:
            resp.text = SAMPLE_FINVIZ_HTML
        return resp

    mock_get.side_effect = side_effect

    signals = compute_social_signals("AAPL", "Apple Inc.", fast_mode=True)

    assert isinstance(signals, SocialSignals)
    assert signals.post_count > 0
    assert -1.0 <= signals.composite <= 1.0
    assert 0.0 <= signals.bullish_pct <= 100.0


@patch("trader_agent.analysis.social.httpx.get")
@patch("trader_agent.analysis.social.time.sleep")
def test_compute_social_signals_empty(mock_sleep, mock_get):
    def side_effect(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        if "stocktwits" in url:
            resp.json.return_value = _mock_stocktwits_response([])
        elif "finviz" in url:
            resp.text = "<html><body></body></html>"
        return resp

    mock_get.side_effect = side_effect

    signals = compute_social_signals("XYZZ", fast_mode=True)
    assert signals.post_count == 0
    assert signals.composite == 0.0


@patch("trader_agent.analysis.social.httpx.get")
@patch("trader_agent.analysis.social.time.sleep")
def test_compute_social_signals_handles_http_error(mock_sleep, mock_get):
    mock_get.side_effect = Exception("Connection failed")
    signals = compute_social_signals("AAPL", fast_mode=True)
    assert signals.post_count == 0
    assert signals.composite == 0.0


@patch("trader_agent.analysis.social.httpx.get")
@patch("trader_agent.analysis.social.time.sleep")
def test_stocktwits_403_finviz_still_works(mock_sleep, mock_get):
    """When StockTwits returns 403, Finviz still provides data."""

    def side_effect(url, **kwargs):
        resp = MagicMock()
        if "stocktwits" in url:
            resp.status_code = 403
        elif "finviz" in url:
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.text = SAMPLE_FINVIZ_HTML
        return resp

    mock_get.side_effect = side_effect

    signals = compute_social_signals("AAPL", "Apple Inc.", fast_mode=True)
    assert signals.post_count > 0
    assert "finviz" in signals.source_breakdown


@patch("trader_agent.analysis.social.httpx.get")
def test_fetch_stocktwits_success(mock_get):
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = _mock_stocktwits_response(SAMPLE_ST_MESSAGES)
    mock_get.return_value = resp

    posts = _fetch_stocktwits("AAPL")
    assert len(posts) == 2
    assert all("text" in p for p in posts)
    assert all("weight" in p for p in posts)


@patch("trader_agent.analysis.social.httpx.get")
def test_fetch_stocktwits_403_returns_empty(mock_get):
    resp = MagicMock()
    resp.status_code = 403
    mock_get.return_value = resp

    assert _fetch_stocktwits("AAPL") == []


@patch("trader_agent.analysis.social.httpx.get")
def test_fetch_finviz_success(mock_get):
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.text = SAMPLE_FINVIZ_HTML
    mock_get.return_value = resp

    headlines = _fetch_finviz_news("AAPL")
    assert len(headlines) == 2
    assert "Apple beats" in headlines[0]["text"]
    assert all(h["weight"] == 1.0 for h in headlines)


@patch("trader_agent.analysis.social.httpx.get")
def test_fetch_finviz_403_returns_empty(mock_get):
    resp = MagicMock()
    resp.status_code = 403
    mock_get.return_value = resp

    assert _fetch_finviz_news("AAPL") == []
