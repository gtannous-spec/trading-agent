# CLAUDE.md -- Multi-Modal Quantitative Trading Agent

This file provides guidance to AI assistants working with this repository.

## Big Picture

This project is an **algorithmic trading agent** that predicts the probability of a stock's price outperforming the S&P 500 over a configurable time horizon (default: 90 days). It does **not** produce absolute price targets -- it outputs a movement confidence score in the range [0.0, 1.0].

The agent ingests four data modalities:

1. **Fundamental data** -- quarterly financials, ratios, EPS, P/E, FCF (Finnhub, yfinance)
2. **Insider trading filings** -- SEC Form 4 buy/sell transactions (Finnhub, SEC EDGAR)
3. **Financial news sentiment** -- article headlines and summaries (NewsAPI, Finnhub News)
4. **Social media sentiment** -- posts from influential accounts (X/Twitter, StockTwits)

Raw text is scored by a FinBERT NLP pipeline into normalized sentiment (bullish / bearish / neutral), then combined with fundamental and insider signals into a feature vector consumed by a gradient-boosted tree model (LightGBM or XGBoost).

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Data Ingestion Services                       │
│  ┌──────────────┐ ┌───────────────┐ ┌────────┐ ┌──────────────┐ │
│  │ Fundamentals │ │ Insider Trades│ │  News  │ │    Social    │ │
│  └──────┬───────┘ └───────┬───────┘ └───┬────┘ └──────┬───────┘ │
└─────────┼─────────────────┼─────────────┼─────────────┼──────────┘
          │                 │             │             │
          ▼                 ▼             ▼             ▼
┌──────────────────────────────────────────────────────────────────┐
│             TimescaleDB (PostgreSQL 16 + TimescaleDB)            │
│  OHLCV hypertable │ fundamentals │ insider_trades │ JSONB text   │
└────────────────────────────┬─────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
   ┌─────────────────┐          ┌─────────────────────┐
   │   NLP Pipeline   │          │  Feature Engineering │
   │  (FinBERT)       │          │  (Pandas + custom)   │
   └────────┬────────┘          └──────────┬──────────┘
            │                              │
            │   sentiment_scores table     │
            └──────────────┬───────────────┘
                           ▼
                ┌─────────────────────┐
                │  Prediction Engine   │
                │  LightGBM / XGBoost  │
                └──────────┬──────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
   ┌──────────────────┐     ┌──────────────────┐
   │  Webhook / Email  │     │  Paper Trading   │
   │  Alerts           │     │  (future)        │
   └──────────────────┘     └──────────────────┘
```

## Tech Stack

| Layer             | Technology                                          |
|-------------------|-----------------------------------------------------|
| Language          | Python 3.12+                                        |
| Database          | PostgreSQL 16 + TimescaleDB extension               |
| ORM               | SQLAlchemy 2.x + Alembic                            |
| HTTP client       | httpx (async)                                       |
| Retry / backoff   | tenacity                                            |
| NLP               | HuggingFace transformers (ProsusAI/finbert)         |
| ML                | LightGBM (primary), XGBoost (secondary), scikit-learn |
| Data manipulation | Pandas, NumPy                                       |
| Scheduling        | APScheduler                                         |
| Config            | pydantic-settings (env-based)                       |
| Logging           | structlog (structured JSON logs)                    |
| Testing           | pytest + pytest-cov + pytest-asyncio                |
| Linting           | ruff                                                |
| Data source       | yfinance (Yahoo Finance, no API key needed)         |
| CLI display       | rich (formatted terminal tables and panels)          |
| Fast sentiment    | VADER (vaderSentiment, instant fallback for FinBERT) |
| Containers        | Docker Compose (TimescaleDB for local dev)          |

## Project Structure

```
trader-agent/
├── CLAUDE.md                  # This file
├── README.md
├── pyproject.toml             # Dependencies, tool config
├── docker-compose.yml         # TimescaleDB for local dev
├── alembic.ini                # Alembic migration config
├── alembic/
│   ├── env.py
│   └── versions/              # Migration scripts
├── src/
│   └── trader_agent/
│       ├── __init__.py
│       ├── config.py          # Pydantic Settings
│       ├── db.py              # Engine + session factory
│       ├── ingestion/         # Data ingestion services
│       │   ├── base.py        # Abstract base with retry logic
│       │   ├── fundamentals.py
│       │   ├── insider_trades.py
│       │   ├── news.py
│       │   ├── social.py
│       │   └── scheduler.py   # APScheduler jobs
│       ├── nlp/               # Sentiment analysis
│       │   ├── sentiment.py   # FinBERT scorer
│       │   └── preprocessor.py
│       ├── models/            # ML prediction engine
│       │   ├── features.py    # Feature engineering
│       │   ├── train.py       # Training pipeline
│       │   ├── predict.py     # Inference
│       │   └── registry.py    # Model versioning
│       ├── analysis/          # On-demand stock analysis (CLI)
│       │   ├── fetcher.py     # yfinance data fetcher
│       │   ├── technical.py   # RSI, MACD, SMA, Bollinger, volatility
│       │   ├── fundamental.py # P/E, D/E, EPS growth, FCF
│       │   ├── sentiment.py   # FinBERT / VADER sentiment scoring
│       │   ├── analyst.py     # Analyst recommendation consensus
│       │   ├── scoring.py     # Weighted aggregator, rating, risk
│       │   └── report.py      # Rich terminal report renderer
│       ├── cli.py             # CLI entry point (analyze command)
│       ├── __main__.py        # python -m trader_agent support
│       ├── alerts/            # Notifications
│       │   ├── webhook.py
│       │   ├── email.py
│       │   └── logger.py
│       └── storage/           # Data layer
│           ├── models.py      # SQLAlchemy ORM models
│           └── schemas.py     # Pydantic DTOs
├── tests/
│   ├── conftest.py
│   ├── test_analysis/         # Analysis module tests
│   ├── test_ingestion/
│   ├── test_nlp/
│   ├── test_models/
│   ├── test_alerts/
│   └── test_storage/
└── scripts/
    ├── seed_data.py           # Load historical data
    └── run_backtest.py        # Evaluate model on history
```

## Database Schema

All tables live in a single PostgreSQL database with the TimescaleDB extension.

| Table              | Purpose                                        | Key columns                                                 |
|--------------------|------------------------------------------------|-------------------------------------------------------------|
| `tickers`          | Tracked stock symbols                          | symbol, name, sector, industry                              |
| `price_history`    | Daily OHLCV (hypertable candidate)             | ticker_id, time, open, high, low, close, volume             |
| `fundamentals`     | Quarterly/annual financial metrics             | ticker_id, report_date, period, revenue, eps, pe_ratio, ... |
| `insider_trades`   | SEC Form 4 transactions                        | ticker_id, filed_at, insider_name, transaction_type, shares |
| `news_articles`    | Raw news with JSONB payload                    | source, title, url, published_at, related_tickers (JSONB)   |
| `social_posts`     | Raw social posts with JSONB payload            | platform, external_id, author, content, posted_at           |
| `sentiment_scores` | NLP-derived scores linked to source records    | source_type, source_id, ticker_symbol, label, score         |
| `predictions`      | Model output log                               | ticker_symbol, model_version, probability, predicted_at     |

ORM models are in `src/trader_agent/storage/models.py`. Pydantic DTOs are in `schemas.py`.

## Common Commands

### Initial setup

```bash
# Start the database
docker compose up -d

# Create a virtual environment and install deps
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Copy env template and fill in API keys
cp .env.example .env

# Run initial database migration
alembic upgrade head
```

### Analyze a stock

```bash
# Full analysis with FinBERT sentiment (slower, more accurate)
python -m trader_agent analyze AAPL

# Fast mode -- uses VADER sentiment (instant, less accurate)
python -m trader_agent analyze MSFT --fast

# Custom historical period
python -m trader_agent analyze GOOGL --period 2y

# Also available as an installed script
trader-analyze analyze TSLA --fast
```

The `analyze` command fetches live data from Yahoo Finance (via yfinance), computes
technical indicators (RSI, MACD, SMA crossovers, Bollinger Bands), scores fundamentals
(P/E, D/E, EPS growth, FCF trend), runs sentiment analysis on news headlines, and
parses analyst recommendations. It then aggregates all signals into a weighted composite
score that maps to a rating:

| Score Range  | Rating      |
|-------------|-------------|
| >= 0.6      | STRONG BUY  |
| 0.3 to 0.6  | BUY         |
| 0.1 to 0.3  | WEAK BUY    |
| -0.1 to 0.1 | HOLD        |
| <= -0.1     | DON'T BUY   |

The output includes price target projections (bear/base/bull) at 1, 2, and 3 month
horizons, plus a risk classification (LOW / MEDIUM / HIGH) based on volatility and
leverage. No API keys are required -- all data comes from yfinance.

### Day-to-day development

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=trader_agent --cov-report=term-missing

# Run only fast unit tests (skip slow/integration)
pytest -m "not slow and not integration"

# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# Lint and auto-fix
ruff check --fix src/ tests/

# Create a new DB migration after changing ORM models
alembic revision --autogenerate -m "describe the change"

# Apply pending migrations
alembic upgrade head

# Start the ingestion scheduler (runs all ingesters on their cron)
trader-ingest

# Seed historical data for development
python scripts/seed_data.py --tickers AAPL,MSFT,GOOGL --days 365

# Run a backtest
python scripts/run_backtest.py --model-version latest --start 2025-01-01 --end 2026-01-01
```

### Docker

```bash
# Start TimescaleDB
docker compose up -d

# Stop and remove containers
docker compose down

# Stop and remove containers AND data volumes
docker compose down -v

# View DB logs
docker compose logs -f timescaledb
```

## Development Rules

These rules are **mandatory** for all code contributions. AI assistants must follow them.

### 1. Service Decoupling

- Ingestion services (`fundamentals`, `insider_trades`, `news`, `social`) **must not import from each other**. Shared logic belongs in `ingestion/base.py` or a new shared utility.
- The NLP pipeline, prediction engine, and alerting layer communicate only through the database -- never by direct function calls across service boundaries.

### 2. External API Calls

- **Every** HTTP call to an external API must use the `tenacity` retry decorator with exponential backoff. The base pattern is in `ingestion/base.py._fetch_with_retry()`.
- All API clients must respect rate limits. If an API returns HTTP 429, the retry logic must honor the `Retry-After` header.
- Use `httpx.AsyncClient` for all HTTP communication. Do not use `requests`.

### 3. Configuration and Secrets

- **All** configuration comes from environment variables via `trader_agent.config.Settings` (Pydantic Settings). No hardcoded API keys, URLs, or credentials anywhere.
- The `.env` file is gitignored. Only `.env.example` is committed (with empty placeholder values).

### 4. Logging

- Use `structlog` for all logging. Every module should have `logger = structlog.get_logger(__name__)`.
- Log structured key-value pairs, not formatted strings: `logger.info("event_name", key=value)`.
- Every ingestion run must log: start, fetch count, transform count, store count, elapsed time, and any errors.

### 5. Database Discipline

- Schema changes go through Alembic migrations. Never modify the DB schema by hand or via raw DDL in application code.
- ORM models live in `storage/models.py`. Do not define models elsewhere.
- Use `JSONB` columns for semi-structured data (raw API responses, related tickers lists). Do not create separate tables for every nested JSON field.

### 6. Testing

- Every ingester must be independently testable with mocked API responses (use `pytest` fixtures and `httpx` mocking).
- Tests that require a live database must be marked `@pytest.mark.integration`.
- Tests that are slow (>5s) must be marked `@pytest.mark.slow`.
- Target minimum 80% coverage for non-placeholder modules.

### 7. Code Style

- Python 3.12+ -- use modern syntax (`type X = ...` unions, `match` statements where appropriate).
- All code must pass `ruff check` and `ruff format` with the project configuration.
- Use `from __future__ import annotations` in every module for consistent PEP 604 type syntax.
- Type-annotate all public function signatures.

### 8. ML / NLP Conventions

- Model artifacts are saved to `artifacts/models/` (gitignored) with a version tag.
- FinBERT is loaded lazily on first use to avoid slow import times during tests.
- Feature engineering code must be deterministic and reproducible given the same DB state.
