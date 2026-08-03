# Trader Agent

Multi-modal quantitative trading agent that analyzes stocks using technical indicators, fundamentals, news sentiment, and analyst consensus to produce buy/sell recommendations.

## What It Does

Enter a **stock ticker** (`AAPL`, `TSLA`) or a **company name** (`apple`, `marvel`, `tesla`) and the agent:

1. **Resolves** the input to a valid ticker symbol (fuzzy matching via Yahoo Finance search)
2. **Fetches** live market data -- prices, financials, news, analyst recommendations
3. **Computes** four signal categories:
   - **Technical** -- RSI, MACD, SMA crossovers, Bollinger Bands, volatility
   - **Fundamental** -- P/E ratio, debt/equity, EPS growth, free cash flow trend
   - **Sentiment** -- news headlines scored by FinBERT or VADER
   - **Analyst** -- consensus from analyst recommendations
4. **Aggregates** signals into a weighted composite score mapped to a rating (STRONG BUY → DON'T BUY)
5. **Projects** price targets (bear/base/bull) at 1, 2, and 3 month horizons

No API keys required -- all data comes from Yahoo Finance via yfinance.

## Quick Start

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Analyze from the CLI
python -m trader_agent analyze apple --fast
python -m trader_agent analyze TSLA --period 2y

# Launch the web dashboard
python -m trader_agent serve
# Open http://127.0.0.1:8000
```

## Web Dashboard

The built-in dashboard provides a modern dark-themed UI with:

- Search by ticker symbol or company name
- Verdict card with rating, composite score, and risk level
- Signal breakdown gauges (technical, fundamental, sentiment, analyst)
- Price target projections table
- Key metrics grid (RSI, MACD, P/E, D/E, EPS growth, sentiment split, etc.)

```bash
python -m trader_agent serve              # default: localhost:8000
python -m trader_agent serve --port 3000  # custom port
```

## Full Documentation

See [CLAUDE.md](CLAUDE.md) for the full architecture, commands reference, and development rules.

## License

MIT
