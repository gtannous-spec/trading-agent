"""Rich terminal report -- renders the AnalysisResult as a formatted console output."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from trader_agent.analysis.scoring import AnalysisResult, Rating, RiskLevel

RATING_STYLES = {
    Rating.STRONG_BUY: "bold bright_green",
    Rating.BUY: "bold green",
    Rating.WEAK_BUY: "bold yellow",
    Rating.HOLD: "bold white",
    Rating.DONT_BUY: "bold red",
}

RISK_STYLES = {
    RiskLevel.LOW: "green",
    RiskLevel.MEDIUM: "yellow",
    RiskLevel.HIGH: "bold red",
}


def render_report(result: AnalysisResult, console: Console | None = None) -> None:
    """Print the full analysis report to the terminal."""
    if console is None:
        console = Console()

    console.print()
    _render_header(console, result)
    console.print()
    _render_rating_panel(console, result)
    console.print()
    _render_price_targets(console, result)
    console.print()
    _render_signal_breakdown(console, result)
    console.print()
    _render_details(console, result)
    console.print()
    _render_disclaimer(console)
    console.print()


def _render_header(console: Console, r: AnalysisResult) -> None:
    header = Text()
    header.append(f"  {r.symbol}", style="bold bright_white")
    header.append(f"  --  {r.company_name}", style="dim")
    header.append(f"\n  Sector: {r.sector}", style="dim")
    header.append(f"  |  Current Price: {r.currency} {r.current_price:,.2f}", style="bold")

    console.print(Panel(header, title="Stock Analysis", border_style="bright_blue"))


def _render_rating_panel(console: Console, r: AnalysisResult) -> None:
    rating_text = Text()
    rating_text.append("  Recommendation:  ", style="dim")
    rating_text.append(r.rating.value, style=RATING_STYLES[r.rating])
    rating_text.append(f"    (score: {r.composite_score:+.3f})", style="dim")
    rating_text.append("\n  Risk Level:      ", style="dim")
    rating_text.append(r.risk_level.value, style=RISK_STYLES[r.risk_level])

    vol_pct = r.technical.volatility_annual * 100
    rating_text.append(f"    (annual volatility: {vol_pct:.1f}%)", style="dim")

    console.print(Panel(rating_text, title="Verdict", border_style="bright_yellow"))


def _render_price_targets(console: Console, r: AnalysisResult) -> None:
    if not r.price_targets:
        return

    table = Table(title="Price Target Projections", border_style="bright_blue", show_lines=True)
    table.add_column("Horizon", style="bold")
    table.add_column("Bear Case", justify="right", style="red")
    table.add_column("Base Case", justify="right", style="bright_white")
    table.add_column("Bull Case", justify="right", style="green")

    for pt in r.price_targets:
        bear_pct = ((pt.bear_case / r.current_price) - 1) * 100 if r.current_price else 0
        base_pct = ((pt.base_case / r.current_price) - 1) * 100 if r.current_price else 0
        bull_pct = ((pt.bull_case / r.current_price) - 1) * 100 if r.current_price else 0

        table.add_row(
            pt.horizon_label,
            f"{r.currency} {pt.bear_case:,.2f}  ({bear_pct:+.1f}%)",
            f"{r.currency} {pt.base_case:,.2f}  ({base_pct:+.1f}%)",
            f"{r.currency} {pt.bull_case:,.2f}  ({bull_pct:+.1f}%)",
        )
    console.print(table)


def _render_signal_breakdown(console: Console, r: AnalysisResult) -> None:
    table = Table(title="Signal Breakdown", border_style="bright_blue")
    table.add_column("Signal", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Weight", justify="right", style="dim")
    table.add_column("Contribution", justify="right")
    table.add_column("Direction", justify="center")

    weights = {
        "technical": 0.25,
        "fundamental": 0.20,
        "sentiment": 0.20,
        "analyst": 0.15,
        "social": 0.10,
        "institutional": 0.10,
    }

    for name, score in r.signal_breakdown.items():
        w = weights[name]
        contribution = score * w
        arrow = _direction_indicator(score)
        score_style = "green" if score > 0.05 else ("red" if score < -0.05 else "dim")

        table.add_row(
            name.title(),
            Text(f"{score:+.3f}", style=score_style),
            f"{w:.0%}",
            Text(f"{contribution:+.4f}", style=score_style),
            arrow,
        )

    table.add_section()
    total_style = RATING_STYLES[r.rating]
    table.add_row(
        Text("TOTAL", style="bold"),
        Text(f"{r.composite_score:+.3f}", style=total_style),
        "100%",
        Text(f"{r.composite_score:+.4f}", style=total_style),
        _direction_indicator(r.composite_score),
    )
    console.print(table)


def _render_details(console: Console, r: AnalysisResult) -> None:
    table = Table(title="Key Metrics", border_style="bright_blue", show_lines=True)
    table.add_column("Category", style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Technical", "RSI (14)", f"{r.technical.rsi_14:.1f}")
    table.add_row("Technical", "MACD Histogram", f"{r.technical.macd_histogram:+.4f}")
    table.add_row("Technical", "SMA 50/200 Ratio", f"{r.technical.sma_crossover_signal:+.2f}")
    table.add_row("Technical", "30d Volatility (ann.)", f"{r.technical.volatility_30d * 100:.1f}%")

    pe_str = f"{r.fundamental.pe_ratio:.1f}" if r.fundamental.pe_ratio else "N/A"
    de_str = f"{r.fundamental.debt_to_equity:.2f}" if r.fundamental.debt_to_equity else "N/A"
    eps_str = (
        f"{r.fundamental.eps_growth_qoq:+.1%}"
        if r.fundamental.eps_growth_qoq is not None
        else "N/A"
    )
    table.add_row("Fundamental", "P/E Ratio", pe_str)
    table.add_row("Fundamental", "Debt/Equity", de_str)
    table.add_row("Fundamental", "EPS Growth (QoQ)", eps_str)

    table.add_row(
        "Sentiment",
        f"Headlines ({r.sentiment.method})",
        f"{r.sentiment.headlines_scored} scored",
    )
    if r.sentiment.headlines_scored > 0:
        table.add_row("Sentiment", "Avg Score", f"{r.sentiment.avg_score:+.3f}")
        table.add_row(
            "Sentiment",
            "Distribution",
            f"Bullish {r.sentiment.bullish_pct:.0f}% / "
            f"Neutral {r.sentiment.neutral_pct:.0f}% / "
            f"Bearish {r.sentiment.bearish_pct:.0f}%",
        )

    table.add_row("Analyst", "Total Ratings", str(r.analyst.total_recommendations))
    table.add_row("Analyst", "Consensus", r.analyst.consensus_label)
    if r.analyst.total_recommendations > 0:
        table.add_row(
            "Analyst",
            "Breakdown",
            f"Strong Buy: {r.analyst.strong_buy} | Buy: {r.analyst.buy} | "
            f"Hold: {r.analyst.hold} | Sell: {r.analyst.sell} | "
            f"Strong Sell: {r.analyst.strong_sell}",
        )

    table.add_row("Social", "Posts Scored", str(r.social.post_count))
    if r.social.post_count > 0:
        table.add_row("Social", "Avg Score", f"{r.social.avg_score:+.3f}")
        table.add_row(
            "Social",
            "Distribution",
            f"Bullish {r.social.bullish_pct:.0f}% / "
            f"Neutral {r.social.neutral_pct:.0f}% / "
            f"Bearish {r.social.bearish_pct:.0f}%",
        )

    inst_pct_str = f"{r.institutional.institutional_pct:.1%}"
    table.add_row("Institutional", "Ownership", inst_pct_str)
    table.add_row("Institutional", "Holders", str(r.institutional.institutional_holders_count))
    if r.institutional.insider_buy_count + r.institutional.insider_sell_count > 0:
        table.add_row(
            "Institutional",
            "Insider Activity",
            f"Buys: {r.institutional.insider_buy_count} | "
            f"Sells: {r.institutional.insider_sell_count}",
        )

    console.print(table)


def _render_disclaimer(console: Console) -> None:
    console.print(
        Panel(
            "[dim]This analysis is for informational purposes only and does not constitute "
            "financial advice. Past performance is not indicative of future results. "
            "Always do your own research before making investment decisions.[/dim]",
            border_style="dim",
            title="Disclaimer",
        )
    )


def _direction_indicator(score: float) -> Text:
    if score > 0.05:
        return Text("BULLISH", style="bold green")
    if score < -0.05:
        return Text("BEARISH", style="bold red")
    return Text("NEUTRAL", style="dim")
