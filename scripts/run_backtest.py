#!/usr/bin/env python3
"""Run the prediction model against historical data and evaluate accuracy.

Usage:
    python scripts/run_backtest.py --model-version latest --start 2025-01-01 --end 2026-01-01
"""

from __future__ import annotations

import argparse
import structlog

logger = structlog.get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the prediction model")
    parser.add_argument("--model-version", default="latest")
    parser.add_argument("--start", required=True, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="Backtest end date (YYYY-MM-DD)")
    args = parser.parse_args()
    logger.info(
        "starting_backtest",
        model_version=args.model_version,
        start=args.start,
        end=args.end,
    )
    raise NotImplementedError("Backtest logic not yet implemented")


if __name__ == "__main__":
    main()
