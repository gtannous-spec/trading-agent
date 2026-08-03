"""CLI entry point for the trading agent -- `analyze` subcommand."""

from __future__ import annotations

import argparse
import sys

import structlog
from rich.console import Console

logger = structlog.get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="trader-agent",
        description="Multi-Modal Quantitative Trading Agent",
    )
    subparsers = parser.add_subparsers(dest="command")

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a stock and produce a buy/sell recommendation",
    )
    analyze_parser.add_argument(
        "ticker",
        type=str,
        help="Stock ticker symbol or company name (e.g. AAPL, 'apple', 'marvel', TSLA)",
    )
    analyze_parser.add_argument(
        "--fast",
        action="store_true",
        default=False,
        help="Use VADER sentiment (instant) instead of FinBERT (slower but more accurate)",
    )
    analyze_parser.add_argument(
        "--period",
        type=str,
        default="1y",
        help="Historical price period to fetch (default: 1y)",
    )

    serve_parser = subparsers.add_parser(
        "serve",
        help="Launch the web dashboard",
    )
    serve_parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Bind host (default: 127.0.0.1)"
    )
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "analyze":
        _run_analyze(args.ticker, fast=args.fast, period=args.period)
    elif args.command == "serve":
        _run_serve(args.host, args.port)


def _run_analyze(ticker: str, fast: bool = False, period: str = "1y") -> None:
    console = Console()

    with console.status(f"[bold blue]Resolving '{ticker}'..."):
        from trader_agent.analysis.resolver import resolve_symbol

        resolved = resolve_symbol(ticker)

    if resolved is None:
        console.print(
            f"[bold red]Could not find a stock matching '{ticker}'.[/]\n"
            "[dim]Try a ticker symbol (AAPL) or company name (Apple).[/dim]"
        )
        sys.exit(1)

    if resolved.matched_via == "search":
        console.print(
            f"[dim]Resolved[/dim] [bold]'{ticker}'[/bold] [dim]-->[/dim] "
            f"[bold bright_cyan]{resolved.symbol}[/bold bright_cyan] "
            f"[dim]({resolved.company_name})[/dim]"
        )

    with console.status(f"[bold blue]Fetching data for {resolved.symbol}..."):
        from trader_agent.analysis.fetcher import fetch_ticker_data

        data = fetch_ticker_data(resolved.symbol, period=period)

    if data.current_price <= 0:
        console.print(f"[bold red]Found symbol {resolved.symbol} but no price data available.[/]")
        sys.exit(1)

    with console.status("[bold blue]Computing technical indicators..."):
        from trader_agent.analysis.technical import compute_technical_signals

        tech = compute_technical_signals(data.prices)

    with console.status("[bold blue]Analyzing fundamentals..."):
        from trader_agent.analysis.fundamental import compute_fundamental_signals

        fund = compute_fundamental_signals(data)

    sentiment_label = "VADER (fast)" if fast else "FinBERT"
    with console.status(f"[bold blue]Scoring sentiment with {sentiment_label}..."):
        from trader_agent.analysis.sentiment import compute_sentiment_signals

        sent = compute_sentiment_signals(data.news, fast_mode=fast)

    with console.status("[bold blue]Parsing analyst recommendations..."):
        from trader_agent.analysis.analyst import compute_analyst_signals

        analyst = compute_analyst_signals(data.recommendations)

    with console.status("[bold blue]Computing final score..."):
        from trader_agent.analysis.scoring import compute_analysis

        result = compute_analysis(
            symbol=data.symbol,
            company_name=data.company_name,
            sector=data.sector,
            current_price=data.current_price,
            currency=data.currency,
            technical=tech,
            fundamental=fund,
            sentiment=sent,
            analyst=analyst,
        )

    from trader_agent.analysis.report import render_report

    render_report(result, console=console)


def _run_serve(host: str, port: int) -> None:
    import uvicorn

    console = Console()
    console.print(
        f"\n  [bold bright_cyan]Trader Agent Dashboard[/bold bright_cyan]"
        f"  →  [link=http://{host}:{port}]http://{host}:{port}[/link]\n"
    )
    uvicorn.run("trader_agent.web.app:app", host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
