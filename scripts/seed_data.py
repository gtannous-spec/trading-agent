#!/usr/bin/env python3
"""Seed the database with sample historical data for development and backtesting.

Usage:
    python scripts/seed_data.py --tickers AAPL,MSFT,GOOGL --days 365
"""

from __future__ import annotations

import argparse
import structlog

logger = structlog.get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed historical data into the DB")
    parser.add_argument(
        "--tickers",
        default="AAPL,MSFT,GOOGL",
        help="Comma-separated list of tickers to seed",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of historical days to fetch",
    )
    args = parser.parse_args()
    tickers = [t.strip() for t in args.tickers.split(",")]
    logger.info("seeding_data", tickers=tickers, days=args.days)
    raise NotImplementedError("Seed logic not yet implemented")


if __name__ == "__main__":
    main()
