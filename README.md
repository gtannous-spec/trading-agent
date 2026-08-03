# Trader Agent

Multi-modal quantitative trading agent that predicts the probability of a stock outperforming the S&P 500 over a configurable time horizon.

## What It Does

Instead of predicting absolute price targets, this agent outputs a **movement confidence score** (0.0 -- 1.0) by combining four data modalities:

- **Fundamentals** -- quarterly financials from Finnhub / yfinance
- **Insider trades** -- SEC Form 4 filings
- **News sentiment** -- financial articles scored by FinBERT
- **Social sentiment** -- influential posts from X / StockTwits scored by FinBERT

A LightGBM model consumes these normalized signals to produce the final probability.

## Quick Start

```bash
# Start the database
docker compose up -d

# Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env with your API keys

# Migrate
alembic upgrade head

# Test
pytest
```

See [CLAUDE.md](CLAUDE.md) for the full architecture, commands reference, and development rules.

## License

MIT
