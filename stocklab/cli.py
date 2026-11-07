"""Command line interface.

Grouped by task: ``analyse``/``value``/``reverse``/``entry`` for a single name,
``screen`` for a universe, ``regime`` for the market, and the ``account``,
``txn``, ``portfolio``, ``rebalance``, ``tax`` and ``risk`` groups for
positions held across accounts.

Every command that produces a judgement also prints its caveats. That is a
deliberate choice: a fair-value estimate without its assumptions, or a screen
without its failure modes, invites more confidence than the number deserves.
"""
from __future__ import annotations

import logging

import click
import pandas as pd

from .company import Company
from .config import configure
from .data.cache import cache
from .reporting import console as ui
from .util.mathx import NAN, is_num

log = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--verbose", "-v", is_flag=True, help="Log data-fetch activity.")
@click.option("--offline", is_flag=True, help="Use only cached data; never fetch.")
@click.option("--data-dir", type=click.Path(), default=None,
              help="Override the cache and portfolio database location.")
@click.version_option("0.1.0", prog_name="stocklab")
def main(verbose: bool, offline: bool, data_dir: str | None) -> None:
    """Equity research, valuation, screening and portfolio management.

    Not investment advice. See docs/METHODOLOGY.md for the assumptions behind
    every model.
    """
    _setup_logging(verbose)
    overrides = {}
    if offline:
        overrides["offline"] = True
    if data_dir:
        from pathlib import Path
        overrides["root"] = Path(data_dir)
    if overrides:
        configure(**overrides)


# ========================================================================== #
# single-name analysis
# ========================================================================== #
@main.command()
@click.argument("symbol")
@click.option("--benchmark", default="SPY", show_default=True)
def quote(symbol: str, benchmark: str) -> None:
    """One-screen snapshot: price, size, quality, value, trend."""
    company = Company.load(symbol, benchmark=benchmark)
    if not company.has_data:
        ui.console.print(f"[red]No usable data for {symbol.upper()}.[/red]")
        raise SystemExit(1)
    m, t = company.metrics, company.technicals

    ui.console.print(ui.header(
        f"{company.symbol}  {company.name}",
        f"{company.sector} / {company.fundamentals.industry}  |  "
        f"{company.quote.get('exchange') or ''}",
    ))
    left = ui.kv_table([
        ("Price", ui.money(company.price, abbreviate=False)),
        ("Market cap", ui.money(m.get("market_cap"))),
        ("Enterprise value", ui.money(m.get("enterprise_value"))),
        ("Net debt", ui.money(m.get("net_debt"))),
        ("Shares out", ui.money(m.get("shares_outstanding"), abbreviate=True).lstrip("$")),
        ("Share count basis", str(m.get("share_count_basis", "-"))),
    ], title="Size")
    mid = ui.kv_table([
        ("P/E (TTM)", ui.ratio(m.get("pe"))),
        ("P/E on 5y avg earnings", ui.ratio(m.get("pe_5y_avg_earnings"))),
        ("EV/EBIT", ui.ratio(m.get("ev_ebit"))),
        ("P/FCF", ui.ratio(m.get("p_fcf"))),
        ("FCF yield", ui.pct(m.get("fcf_yield"))),
        ("Dividend yield", ui.pct(m.get("dividend_yield"))),
        ("Shareholder yield", ui.pct(m.get("shareholder_yield"))),
    ], title="Valuation")
    right = ui.kv_table([
        ("Gross margin", ui.pct(m.get("gross_margin"))),
        ("Operating margin", ui.pct(m.get("operating_margin"))),
        ("ROIC", ui.pct(m.get("roic"))),
        ("ROIC 5y avg", ui.pct(m.get("roic_5y_avg"))),
        ("Revenue CAGR 5y", ui.pct(m.get("revenue_cagr_5y"))),
        ("Net debt / EBITDA", ui.ratio(m.get("net_debt_to_ebitda"))),
        ("Piotroski F", ui.num(m.get("piotroski_f"), 0) + "/9"),
    ], title="Quality")
    from rich.columns import Columns
    ui.console.print(Columns([left, mid, right], padding=(0, 4)))

    if t:
        ui.console.print(ui.kv_table([
            ("vs 200-day", ui.pct(t.get("dist_sma200"), signed=True)),
            ("From 52wk high", ui.pct(t.get("pct_from_52w_high"), signed=True)),
            ("12-1 momentum", ui.pct(t.get("momentum_12_1"), signed=True)),
            ("Excess vs benchmark 12m", ui.pct(t.get("excess_12m"), signed=True)),
            ("RSI(14)", ui.num(t.get("rsi14"), 1)),
            ("Realised vol (3m)", ui.pct(t.get("vol63"))),
            ("Trend template", ui.num(t.get("trend_template_score"), 0) + "/8"),
        ], title="Trend"))

    ui.print_notes(company.notes + list(m.get("share_count_notes") or []), "Data notes")


@main.command("analyse")
@click.argument("symbol")
@click.option("--benchmark", default="SPY", show_default=True)
@click.option("--capital", default=100_000.0, show_default=True,
              help="Capital base used for position sizing.")
@click.option("--risk", "risk_per_trade", default=0.01, show_default=True,
              help="Fraction of capital risked if the stop is hit.")
@click.option("--no-regime", is_flag=True, help="Skip the market regime read.")
def analyse_cmd(symbol: str, benchmark: str, capital: float,
                risk_per_trade: float, no_regime: bool) -> None:
    """Full work-up: fundamentals, valuation range, entry plan."""
    from .analysis.regime import assess
    from .entry.timing import build_plan
    from .valuation import composite

    company = Company.load(symbol, benchmark=benchmark)
    if not company.has_data:
        ui.console.print(f"[red]No usable data for {symbol.upper()}.[/red]")
        raise SystemExit(1)

    ui.console.print(ui.header(
        f"{company.symbol}  {company.name}",
        f"{company.sector} / {company.fundamentals.industry}  |  "
        f"price {ui.money(company.price, abbreviate=False)}  |  "
        f"{int(company.metrics.get('years_of_data', 0))} years of filings",
    ))

    report = composite.value(company.fundamentals, company.metrics, company.price)
    _print_valuation(report)

    regime_score = None
    if not no_regime:
        regime = assess(benchmark)
        regime_score = regime.get("composite_score")
        ui.console.print(ui.kv_table([
            ("Regime", f"{regime['label']} ({ui.num(regime_score, 0)}/100)"),
            ("Exposure multiplier", ui.num(regime.get("exposure_multiplier"), 2) + "x"),
        ], title="Market context"))

    plan = build_plan(
        company.symbol, company.history,
        margin_of_safety=report.margin_of_safety,
        technicals=company.technicals, regime_score=regime_score,
        capital=capital, risk_per_trade=risk_per_trade,
    )
    _print_plan(plan)
    ui.print_notes(company.notes + report.notes[:6], "Caveats")


def _print_valuation(report) -> None:
    frame = report.to_frame()
    ui.console.print(ui.frame_table(
        frame.set_index("model"),
        title="Valuation models",
        formatters={
            "value_per_share": lambda v: ui.money(v, abbreviate=False),
            "upside": lambda v: ui.pct(v, signed=True),
            "weight": lambda v: ui.pct(v, 0),
            "applicable": lambda v: "yes" if v else "no",
        },
        index_label="model",
    ))
    rows = [
        ("Blended fair value", ui.money(report.fair_value, abbreviate=False)),
        ("Range across models",
         f"{ui.money(report.low, abbreviate=False)} - {ui.money(report.high, abbreviate=False)}"),
        ("Margin of safety", ui.signed_text(report.margin_of_safety)),
        ("Model dispersion", ui.pct(report.dispersion, 0)),
        ("Confidence", report.confidence),
    ]
    if is_num(report.implied_return):
        rows += [
            ("Implied return at this price", ui.pct(report.implied_return, 2)),
            ("Risk-free rate", ui.pct(report.risk_free, 2)),
            ("Equity premium offered",
             ui.signed_text(report.implied_equity_premium)),
        ]
    coc = report.cost_of_capital
    rows += [
        ("WACC", ui.pct(coc.wacc, 2)),
        ("Cost of equity", ui.pct(coc.cost_of_equity, 2)),
        ("Beta (adjusted)", ui.num(coc.beta_adjusted, 2)),
        ("Synthetic rating", coc.synthetic_rating),
    ]
    ui.console.print(ui.kv_table(rows, title="Valuation summary", width=32))
    if not report.scenarios.empty:
        ui.console.print(ui.frame_table(
            report.scenarios, title="Scenarios",
            formatters={
                "growth_initial": lambda v: ui.pct(v),
                "margin_target": lambda v: ui.pct(v),
                "wacc": lambda v: ui.pct(v, 2),
                "value_per_share": lambda v: ui.money(v, abbreviate=False),
                "terminal_share": lambda v: ui.pct(v, 0),
            }, index_label="scenario"))


def _print_plan(plan) -> None:
    ui.console.print(ui.kv_table([
        ("Entry score", ui.score_bar(plan.score)),
        ("Verdict", plan.verdict),
        ("Best setup", plan.best_setup or "-"),
        ("Stop", f"{ui.money(plan.stop.stop, abbreviate=False)} ({plan.stop.method})"),
        ("Risk per share",
         f"{ui.money(plan.stop.risk_per_share, abbreviate=False)} "
         f"({ui.pct(plan.stop.risk_pct)})"),
        ("Position size",
         f"{ui.num(plan.size.get('shares'), 0)} sh = "
         f"{ui.money(plan.size.get('value'))} ({ui.pct(plan.size.get('weight'))})"),
        ("Capital at risk",
         f"{ui.money(plan.size.get('risk_dollars'), abbreviate=False)} "
         f"({ui.pct(plan.size.get('risk_pct_of_capital'), 2)})"),
        ("1R / 2R / 3R targets", " / ".join(
            ui.money(v, abbreviate=False) for v in plan.stop.targets.values())),
    ], title="Trade plan", width=28))

    setups = pd.DataFrame([{
        "setup": s.name, "triggered": "yes" if s.triggered else "no",
        "confidence": s.confidence,
        "failed_conditions": "; ".join(s.failed[:3]) or "-",
    } for s in plan.setups.values()]).set_index("setup")
    ui.console.print(ui.frame_table(setups, title="Entry setups",
                                    formatters={"confidence": lambda v: ui.num(v, 0)},
                                    index_label="setup"))
    if not plan.staged_entries.empty:
        ui.console.print(ui.frame_table(
            plan.staged_entries, title="Staged entry ladder",
            formatters={
                "limit_price": lambda v: ui.money(v, abbreviate=False),
                "allocation": lambda v: ui.pct(v, 0),
                "distance_from_spot": lambda v: ui.pct(v, 1, signed=True),
                "r_to_stop": lambda v: ui.num(v, 2),
            }, index_label="tranche"))
    ui.print_notes(plan.invalidation, "Invalidation (exit if any of these occur)", "red")


@main.command()
@click.argument("symbol")
@click.option("--years", default=10, show_default=True, help="Explicit forecast horizon.")
@click.option("--growth", type=float, default=None, help="Override initial revenue growth.")
@click.option("--margin", type=float, default=None, help="Override target operating margin.")
@click.option("--wacc", type=float, default=None, help="Override the discount rate.")
@click.option("--terminal-growth", type=float, default=None)
def value(symbol: str, years: int, growth: float | None, margin: float | None,
          wacc: float | None, terminal_growth: float | None) -> None:
    """DCF detail: projection, sensitivity, scenarios."""
    from dataclasses import replace
    from .valuation.dcf import auto_assumptions, run_dcf, standard_sensitivity
    from .valuation.wacc import estimate_cost_of_capital

    company = Company.load(symbol, with_technicals=False)
    if not company.has_data:
        ui.console.print(f"[red]No usable data for {symbol.upper()}.[/red]")
        raise SystemExit(1)
    if company.fundamentals.is_financial:
        ui.console.print(
            "[yellow]This is a financial company. A FCFF discounted cash flow "
            "is not meaningful where debt is raw material rather than "
            "financing -- use `stocklab analyse` for the residual-income "
            "model instead.[/yellow]")
    coc = estimate_cost_of_capital(company.fundamentals,
                                   company.metrics.get("market_cap", NAN),
                                   company.metrics)
    assumptions = auto_assumptions(company.fundamentals, company.metrics, coc,
                                   years=years, growth_override=growth,
                                   margin_override=margin,
                                   terminal_growth=terminal_growth)
    if wacc is not None:
        assumptions = replace(assumptions, wacc=wacc)
    result = run_dcf(assumptions)

    ui.console.print(ui.header(f"{company.symbol} discounted cash flow",
                               str(coc)))
    ui.console.print(ui.kv_table([
        ("Revenue base", ui.money(assumptions.revenue_base)),
        ("Initial growth", ui.pct(assumptions.revenue_growth_initial)),
        ("Terminal growth", ui.pct(assumptions.terminal_growth)),
        ("Margin now -> target",
         f"{ui.pct(assumptions.operating_margin_base)} -> "
         f"{ui.pct(assumptions.operating_margin_target)}"),
        ("Sales / invested capital", ui.num(assumptions.sales_to_capital, 2)),
        ("Terminal ROIC", ui.pct(assumptions.terminal_roic)),
        ("Tax rate", ui.pct(assumptions.tax_rate)),
    ], title="Assumptions"))

    projection = result.projection
    if not projection.empty:
        shown = projection[["revenue", "revenue_growth", "operating_margin",
                            "ebit", "reinvestment", "fcff", "pv_fcff"]]
        ui.console.print(ui.frame_table(
            shown, title="Projection",
            formatters={
                "revenue": ui.money, "ebit": ui.money, "reinvestment": ui.money,
                "fcff": ui.money, "pv_fcff": ui.money,
                "revenue_growth": lambda v: ui.pct(v),
                "operating_margin": lambda v: ui.pct(v),
            }, index_label="year"))

    ui.console.print(ui.kv_table([
        ("PV of forecast period", ui.money(result.pv_explicit)),
        ("PV of terminal value", ui.money(result.pv_terminal)),
        ("Terminal share of value", ui.pct(result.terminal_share, 0)),
        ("Enterprise value", ui.money(result.enterprise_value)),
        ("Equity value", ui.money(result.equity_value)),
        ("Value per share", ui.money(result.value_per_share, abbreviate=False)),
        ("Upside vs market", ui.signed_text(result.upside_vs(company.price))),
    ], title="Result", width=28))

    grid = standard_sensitivity(assumptions)
    if not grid.empty:
        display = grid.copy()
        display.index = [f"g={v:.1%}" for v in display.index]
        display.columns = [f"wacc={v:.1%}" for v in display.columns]
        ui.console.print(ui.frame_table(
            display, title="Sensitivity: value per share",
            formatters={c: (lambda v: ui.money(v, abbreviate=False))
                        for c in display.columns},
            index_label="terminal growth"))
    ui.print_notes(result.notes + coc.notes, "Caveats")


@main.command()
@click.argument("symbol")
@click.option("--prices", default="", help="Comma-separated prices for the ladder table.")
def reverse(symbol: str, prices: str) -> None:
    """What the current price already assumes (reverse DCF)."""
    from .valuation.dcf import auto_assumptions
    from .valuation.reverse_dcf import analyse, required_growth_table
    from .valuation.wacc import estimate_cost_of_capital

    company = Company.load(symbol, with_technicals=False)
    if not company.has_data:
        ui.console.print(f"[red]No usable data for {symbol.upper()}.[/red]")
        raise SystemExit(1)
    coc = estimate_cost_of_capital(company.fundamentals,
                                   company.metrics.get("market_cap", NAN),
                                   company.metrics)
    assumptions = auto_assumptions(company.fundamentals, company.metrics, coc)
    expectations = analyse(assumptions, company.price, company.metrics)

    ui.console.print(ui.header(
        f"{company.symbol} implied expectations",
        f"price {ui.money(company.price, abbreviate=False)}  |  "
        f"hurdle WACC {ui.pct(coc.wacc, 2)}"))
    ui.console.print(ui.kv_table([
        ("Implied return (IRR)", ui.pct(expectations.implied_return, 2)),
        ("Risk-free rate", ui.pct(coc.risk_free, 2)),
        ("Equity premium offered",
         ui.signed_text(expectations.implied_return - coc.risk_free
                        if is_num(expectations.implied_return) else NAN)),
        ("Implied near-term growth", ui.pct(expectations.implied_growth)),
        ("  vs 5y history", ui.pct(expectations.historical_growth)),
        ("Implied steady growth", ui.pct(expectations.implied_growth_joint)),
        ("Implied terminal growth", ui.pct(expectations.implied_terminal_growth)),
        ("Implied target margin", ui.pct(expectations.implied_margin)),
        ("  vs 5y history", ui.pct(expectations.historical_margin)),
        ("Verdict", expectations.verdict),
    ], title="What the price requires", width=30))

    ladder = [float(p) for p in prices.split(",") if p.strip()] if prices else [
        company.price * f for f in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1)]
    table = required_growth_table(assumptions, ladder)
    ui.console.print(ui.frame_table(
        table, title="Growth required at each price",
        formatters={"implied_initial_growth": lambda v: ui.pct(v),
                    "vs_history": lambda v: ui.pct(v, signed=True)},
        index_label="price"))
    ui.print_notes(expectations.notes, "Caveats")


@main.command()
@click.argument("symbol")
@click.option("--capital", default=100_000.0, show_default=True)
@click.option("--risk", "risk_per_trade", default=0.01, show_default=True)
@click.option("--stop-style", type=click.Choice(["atr", "structure"]), default="atr")
def entry(symbol: str, capital: float, risk_per_trade: float, stop_style: str) -> None:
    """Entry setups, stop placement and position size."""
    from .entry.timing import build_plan
    company = Company.load(symbol)
    if company.history.empty:
        ui.console.print(f"[red]No price history for {symbol.upper()}.[/red]")
        raise SystemExit(1)
    plan = build_plan(company.symbol, company.history,
                      technicals=company.technicals, capital=capital,
                      risk_per_trade=risk_per_trade, stop_style=stop_style)
    ui.console.print(ui.header(f"{company.symbol} entry plan",
                               f"price {ui.money(company.price, abbreviate=False)}"))
    _print_plan(plan)
    ui.print_notes(plan.notes, "Notes")


# ========================================================================== #
# market
# ========================================================================== #
@main.command()
@click.option("--benchmark", default="SPY", show_default=True)
def regime(benchmark: str) -> None:
    """Market regime: trend, breadth, credit, volatility."""
    from .analysis.regime import assess
    state = assess(benchmark)
    ui.console.print(ui.header(
        f"Market regime: {state['label'].upper()}",
        f"composite {ui.num(state['composite_score'], 1)}/100   "
        f"exposure multiplier {ui.num(state['exposure_multiplier'], 2)}x"))

    components = pd.DataFrame({
        "score": state["components"],
        "weight": state["weights"],
    })
    ui.console.print(ui.frame_table(
        components, title="Components",
        formatters={"score": lambda v: ui.num(v, 1),
                    "weight": lambda v: ui.pct(v, 0)},
        index_label="component"))

    trend, breadth, credit = state["trend"], state["breadth"], state["rates"]
    ui.console.print(ui.kv_table([
        ("Index vs 200-day", ui.pct(trend.get("dist_sma200"), signed=True)),
        ("Drawdown from high", ui.pct(trend.get("drawdown_from_high"), signed=True)),
        ("12-month return", ui.pct(trend.get("return_12m"), signed=True)),
        ("Days above 200-day (1y)", ui.pct(trend.get("pct_days_above_sma200_1y"), 0)),
        ("Breadth above 200-day", ui.pct(breadth.get("pct_above_sma200"), 0)),
        ("Equal-weight vs cap-weight (1y)",
         ui.pct(state.get("equal_weight_divergence"), signed=True)),
        ("High-yield spread", ui.num(state["credit"].get("hy_oas"), 2) + "%"),
        ("  change over 3 months",
         ui.num(state["credit"].get("hy_oas_chg_3m"), 2) + "pp"),
        ("VIX", ui.num(credit.get("vix"), 1)),
        ("10y yield", ui.num(credit.get("ust_10y"), 2) + "%"),
        ("10y minus 3m", ui.num(credit.get("curve_10y3m"), 2) + "pp"),
        ("Unemployment", ui.num(credit.get("unemployment"), 1) + "%"),
    ], title="Readings", width=34))
    ui.print_notes(state["caveats"], "How to use this")


@main.command()
@click.argument("preset", required=False)
@click.option("--symbols", default="", help="Comma-separated symbols to screen.")
@click.option("--universe", type=click.Choice(["megacap", "listed"]),
              default="megacap", show_default=True,
              help="`listed` screens every US common stock and takes a long time.")
@click.option("--limit", default=20, show_default=True)
@click.option("--max-symbols", default=250, show_default=True,
              help="Cap on names fetched when using --universe listed.")
@click.option("--list-presets", is_flag=True, help="Show available screens and exit.")
def screen(preset: str | None, symbols: str, universe: str, limit: int,
           max_symbols: int, list_presets: bool) -> None:
    """Run a screening strategy over a universe."""
    from .screening import presets as preset_module
    from .screening.factors import build_matrix

    if list_presets or not preset:
        rows = pd.DataFrame(preset_module.describe_all()).set_index("key")
        ui.console.print(ui.frame_table(rows[["name", "description", "filters"]],
                                        title="Available screens",
                                        index_label="key"))
        return

    try:
        screen_def = preset_module.get(preset)
    except KeyError as exc:
        ui.console.print(f"[red]{exc}[/red]")
        raise SystemExit(1)

    if symbols:
        candidates = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    elif universe == "listed":
        from .data.universe import common_stocks
        frame = common_stocks()
        candidates = list(frame["symbol"].head(max_symbols))
        ui.console.print(
            f"[yellow]Screening {len(candidates)} of {len(frame)} listed names "
            "(rate limits make a full sweep slow; raise --max-symbols to widen "
            "it). Results are cached, so a second run is fast.[/yellow]")
    else:
        from .data.universe import MEGA_CAP_SAMPLE
        candidates = list(MEGA_CAP_SAMPLE)

    with ui.console.status(f"Loading {len(candidates)} companies..."):
        matrix = build_matrix(candidates, with_technicals=True)
    if matrix.matrix.empty:
        ui.console.print("[red]No usable data for any symbol.[/red]")
        raise SystemExit(1)

    result = screen_def.run(matrix.matrix, limit=limit)
    ui.console.print(ui.header(
        screen_def.name,
        f"{len(result)} of {result.universe_size} names passed"
        + (f"  |  {len(matrix.failures)} symbols had no usable data"
           if matrix.failures else "")))
    if screen_def.evidence:
        ui.console.print(ui.notes_panel([screen_def.evidence], "Evidence", "green"))

    if len(result):
        columns = [c for c in ("name", "sector", "market_cap", "score", "pe",
                               "ev_ebit", "fcf_yield", "roic", "revenue_cagr_5y",
                               "dividend_yield", "momentum_12_1")
                   if c in result.passed.columns]
        ui.console.print(ui.frame_table(
            result.passed[columns], title="Results",
            formatters={
                "market_cap": ui.money, "score": lambda v: ui.num(v, 2),
                "pe": lambda v: ui.ratio(v), "ev_ebit": lambda v: ui.ratio(v),
                "fcf_yield": lambda v: ui.pct(v), "roic": lambda v: ui.pct(v),
                "revenue_cagr_5y": lambda v: ui.pct(v),
                "dividend_yield": lambda v: ui.pct(v),
                "momentum_12_1": lambda v: ui.pct(v, signed=True),
            }, index_label="symbol"))

    attrition = result.filter_attrition[result.filter_attrition > 0]
    if not attrition.empty:
        ui.console.print(ui.frame_table(
            attrition.to_frame("rejected"), title="Where names were filtered out",
            index_label="filter"))
    ui.print_notes(screen_def.failure_modes + result.notes,
                   "Failure modes of this strategy")


@main.command()
@click.option("--symbols", required=True,
              help="Comma-separated universe to trade.")
@click.option("--strategy", type=click.Choice(["momentum", "hold"]),
              default="momentum", show_default=True)
@click.option("--top-n", default=3, show_default=True)
@click.option("--lookback", default=252, show_default=True,
              help="Momentum lookback in trading days.")
@click.option("--frequency", type=click.Choice(["D", "W", "M", "Q"]),
              default="M", show_default=True)
@click.option("--period", default="10y", show_default=True)
@click.option("--benchmark", default="SPY", show_default=True)
@click.option("--capital", default=100_000.0, show_default=True)
@click.option("--slippage-bps", default=5.0, show_default=True)
@click.option("--walk-forward", "do_walk_forward", is_flag=True,
              help="Choose parameters in-sample and score the next window only.")
def backtest(symbols: str, strategy: str, top_n: int, lookback: int,
             frequency: str, period: str, benchmark: str, capital: float,
             slippage_bps: float, do_walk_forward: bool) -> None:
    """Backtest a rule over a universe, with costs and bias warnings."""
    from .backtest import engine, metrics
    from .backtest.walkforward import momentum_builder, walk_forward
    from .data.providers import fred
    from .data.providers.yahoo import fetch_closes, fetch_history

    universe = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if len(universe) < 2:
        ui.console.print("[red]Need at least two symbols.[/red]")
        raise SystemExit(1)

    with ui.console.status(f"Loading {len(universe)} price series..."):
        prices = fetch_closes(universe, period=period, field="adj_close")
        prices = prices.dropna(how="all")
        bench = fetch_history(benchmark, period=period)
    if prices.empty or len(prices) < 300:
        ui.console.print("[red]Not enough price history to backtest.[/red]")
        raise SystemExit(1)
    risk_free = fred.risk_free_rate()

    if do_walk_forward:
        builder, grid = momentum_builder(frequency=frequency)
        with ui.console.status("Walking forward..."):
            result = walk_forward(prices, builder, grid, risk_free=risk_free,
                                  initial_capital=capital)
        if not result.folds:
            ui.console.print("[red]" + "; ".join(result.notes) + "[/red]")
            raise SystemExit(1)
        ui.console.print(ui.header(
            "Walk-forward momentum",
            f"{len(result.folds)} fold(s) | parameters chosen in-sample, "
            "scored out-of-sample only"))
        ui.console.print(ui.frame_table(
            result.fold_table(), title="Folds",
            formatters={"in_sample_cagr": lambda v: ui.pct(v, signed=True),
                        "oos_cagr": lambda v: ui.pct(v, signed=True),
                        "oos_sharpe": lambda v: ui.num(v, 2),
                        "oos_max_drawdown": lambda v: ui.pct(v, signed=True)},
            index_label="fold"))
        ui.console.print(ui.kv_table([
            ("Out-of-sample CAGR", ui.pct(result.oos_summary.get("cagr"), signed=True)),
            ("Out-of-sample Sharpe", ui.num(result.oos_summary.get("sharpe"), 2)),
            ("Max drawdown", ui.pct(result.oos_summary.get("max_drawdown"), signed=True)),
            ("In-sample minus out-of-sample", ui.signed_text(result.degradation)),
        ], title="Out-of-sample result", width=32))
        ui.print_notes(result.notes, "Notes")
        return

    if strategy == "hold":
        weights = engine.buy_and_hold_weights(prices)
        label = "equal-weight buy and hold"
    else:
        scores = prices.shift(21) / prices.shift(lookback) - 1.0
        weights = engine.weights_from_ranks(scores, top_n=top_n, frequency=frequency)
        label = f"momentum top {top_n}, {lookback}d lookback, {frequency} rebalance"

    result = engine.run(prices, weights, initial_capital=capital,
                        slippage_bps=slippage_bps,
                        benchmark=bench["adj_close"] if not bench.empty else None,
                        risk_free=risk_free, label=label)
    hold = engine.run(prices, engine.buy_and_hold_weights(prices),
                      initial_capital=capital, slippage_bps=slippage_bps,
                      risk_free=risk_free)

    ui.console.print(ui.header(f"Backtest: {label}",
                               f"{len(universe)} symbols | "
                               f"{prices.index.min().date()} to {prices.index.max().date()}"))
    comparison = {"strategy": result, "equal-weight hold": hold}
    if result.benchmark_summary:
        comparison[benchmark] = type("B", (), {"summary": result.benchmark_summary})()
    frame = metrics.compare(comparison)
    ui.console.print(ui.frame_table(
        frame, title="Comparison (sorted by return per unit of drawdown)",
        formatters={c: (lambda v: ui.pct(v, signed=True))
                    for c in ("cagr", "volatility", "max_drawdown",
                              "time_underwater", "gross_cagr") if c in frame}
        | {c: (lambda v: ui.num(v, 2)) for c in
           ("sharpe", "sortino", "calmar", "ulcer", "annual_turnover") if c in frame},
        index_label="strategy"))

    annual = metrics.annual_table(
        result.returns,
        bench["adj_close"].pct_change() if not bench.empty else None)
    if not annual.empty:
        ui.console.print(ui.frame_table(
            annual, title="Calendar years",
            formatters={c: (lambda v: ui.pct(v, signed=True)) for c in annual.columns},
            index_label="year"))

    sharpe = result.summary.get("sharpe")
    observations = len(result.returns)
    ui.console.print(ui.kv_table([
        ("Sharpe as reported", ui.num(sharpe, 2)),
        ("Deflated for 10 trials",
         ui.num(metrics.deflated_sharpe(sharpe, 10, observations), 2)),
        ("Deflated for 50 trials",
         ui.num(metrics.deflated_sharpe(sharpe, 50, observations), 2)),
    ], title="Selection bias", width=32))
    ui.print_notes(
        result.warnings
        + ["Deflated Sharpe subtracts the expected maximum from N trials. If "
           "you tried several variants and kept this one, the deflated figure "
           "is the honest one."],
        "Why to distrust this")


@main.command()
@click.option("--purge", is_flag=True, help="Delete every cached entry.")
@click.option("--older-than-days", type=float, default=None,
              help="Delete entries older than this many days.")
def cache_cmd(purge: bool, older_than_days: float | None) -> None:
    """Inspect or clear the data cache."""
    store = cache()
    if purge:
        removed = store.purge()
        ui.console.print(f"Removed {removed} cached entries.")
        return
    if older_than_days is not None:
        removed = store.purge(older_than=older_than_days * 86400)
        ui.console.print(f"Removed {removed} entries older than "
                         f"{older_than_days:g} days.")
        return
    stats = store.stats()
    if not stats:
        ui.console.print("Cache is empty.")
        return
    ui.console.print(ui.frame_table(pd.DataFrame(stats).T, title=f"Cache: {store.path}",
                                    index_label="kind"))


main.add_command(cache_cmd, name="cache")


# ========================================================================== #
# portfolio
# ========================================================================== #
def _prices_for(symbols) -> dict[str, float]:
    """Latest price per symbol, skipping anything unavailable."""
    from .data.providers.yahoo import fetch_meta
    out: dict[str, float] = {}
    for symbol in symbols:
        price = (fetch_meta(symbol) or {}).get("price")
        if is_num(price):
            out[symbol] = float(price)
    return out


def _sectors_for(symbols) -> dict[str, str]:
    from .data.providers.sec import load_fundamentals
    out: dict[str, str] = {}
    for symbol in symbols:
        try:
            out[symbol] = load_fundamentals(symbol).sector
        except Exception:  # noqa: BLE001 - a missing sector must not break a report
            out[symbol] = "Unknown"
    return out


def _store():
    from .portfolio.store import store
    return store()


def _ledger(account: str | None = None):
    from .portfolio.ledger import build
    store = _store()
    account_id = None
    if account:
        record = store.get_account(account)
        if record is None:
            ui.console.print(f"[red]No account named {account!r}.[/red]")
            raise SystemExit(1)
        account_id = record.id
    state, accounts = build(store, account_id=account_id)
    return store, state, accounts


@main.group()
def account() -> None:
    """Manage accounts (taxable, IRA, Roth, 401k, HSA...)."""


@account.command("add")
@click.argument("name")
@click.option("--type", "account_type", required=True,
              type=click.Choice([t.value for t in __import__(
                  "stocklab.portfolio.models", fromlist=["AccountType"]).AccountType]))
@click.option("--broker", default="")
@click.option("--owner", default="")
@click.option("--basis-method",
              type=click.Choice(["fifo", "lifo", "hifo", "average", "specific"]),
              default="fifo", show_default=True,
              help="FIFO is the broker default and usually the worst for tax; "
                   "HIFO minimises realised gains.")
def account_add(name: str, account_type: str, broker: str, owner: str,
                basis_method: str) -> None:
    """Create an account."""
    from .portfolio.models import Account, AccountType
    store = _store()
    if store.get_account(name) is not None:
        ui.console.print(f"[red]An account named {name!r} already exists.[/red]")
        raise SystemExit(1)
    record = Account(name=name, type=AccountType(account_type), broker=broker,
                     owner=owner, cost_basis_method=basis_method)
    store.add_account(record)
    ui.console.print(f"Created [bold]{name}[/bold] ({record.type.label}), "
                     f"cost basis {basis_method.upper()}.")
    if record.type.sheltered:
        ui.console.print("[dim]Sheltered account: losses here are not "
                         "deductible, so loss harvesting does not apply.[/dim]")


@account.command("list")
def account_list() -> None:
    """List accounts and their cash balances."""
    store, state, accounts = _ledger()
    records = store.accounts()
    if not records:
        ui.console.print("No accounts yet. Create one with "
                         "[bold]stocklab account add[/bold].")
        return
    rows = []
    for record in records:
        rows.append({
            "name": record.name,
            "type": record.type.label,
            "tax": ("tax-free" if record.type.tax_free else
                    "tax-deferred" if record.type.tax_deferred else "taxable"),
            "basis": record.cost_basis_method.upper(),
            "broker": record.broker or "-",
            "cash": state.total_cash(record.id),
            "positions": len({lot.symbol for lot in state.open_lots(record.id)}),
        })
    ui.console.print(ui.frame_table(
        pd.DataFrame(rows).set_index("name"), title="Accounts",
        formatters={"cash": lambda v: ui.money(v, abbreviate=False)},
        index_label="account"))


@main.group()
def txn() -> None:
    """Record and review transactions."""


@txn.command("add")
@click.option("--account", required=True)
@click.option("--date", "txn_date", required=True, help="YYYY-MM-DD")
@click.option("--type", "txn_type", required=True,
              type=click.Choice([t.value for t in __import__(
                  "stocklab.portfolio.models", fromlist=["TxnType"]).TxnType]))
@click.option("--symbol", default="")
@click.option("--quantity", type=float, default=0.0)
@click.option("--price", type=float, default=0.0,
              help="For a split, pass the ratio here (4.0 for a 4-for-1).")
@click.option("--amount", type=float, default=0.0,
              help="Cash effect; derived for buys and sells when omitted.")
@click.option("--fees", type=float, default=0.0)
@click.option("--notes", default="")
def txn_add(account: str, txn_date: str, txn_type: str, symbol: str,
            quantity: float, price: float, amount: float, fees: float,
            notes: str) -> None:
    """Record one transaction."""
    from .portfolio.models import Transaction, TxnType
    store = _store()
    record = store.get_account(account)
    if record is None:
        ui.console.print(f"[red]No account named {account!r}.[/red]")
        raise SystemExit(1)
    transaction = Transaction(
        account_id=record.id, date=pd.Timestamp(txn_date).date(),
        type=TxnType(txn_type), symbol=symbol, quantity=quantity, price=price,
        amount=amount, fees=fees, notes=notes,
    )
    store.add_transaction(transaction)
    ui.console.print(
        f"Recorded #{transaction.id}: {txn_type} "
        f"{quantity:g} {symbol or ''} @ {ui.money(price, abbreviate=False)} "
        f"in {account}.")


@txn.command("import")
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("--account", required=True)
@click.option("--dry-run", is_flag=True, help="Parse and show without saving.")
def txn_import(csv_path: str, account: str, dry_run: bool) -> None:
    """Import transactions from CSV.

    Expected columns: date, type, symbol, quantity, price, amount, fees, notes.
    Only date and type are required. Column names are matched case-insensitively
    and common broker aliases are accepted.
    """
    from .portfolio.models import Transaction, TxnType
    store = _store()
    record = store.get_account(account)
    if record is None:
        ui.console.print(f"[red]No account named {account!r}.[/red]")
        raise SystemExit(1)

    frame = pd.read_csv(csv_path)
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    aliases = {
        "trade date": "date", "settlement date": "date", "transaction date": "date",
        "action": "type", "transaction type": "type", "activity": "type",
        "ticker": "symbol", "security": "symbol",
        "shares": "quantity", "qty": "quantity",
        "share price": "price", "unit price": "price",
        "commission": "fees", "fee": "fees",
        "net amount": "amount", "value": "amount", "total": "amount",
        "description": "notes",
    }
    frame = frame.rename(columns={k: v for k, v in aliases.items() if k in frame.columns})
    missing = {"date", "type"} - set(frame.columns)
    if missing:
        ui.console.print(f"[red]CSV is missing required column(s): "
                         f"{', '.join(sorted(missing))}.[/red]")
        raise SystemExit(1)

    valid_types = {t.value for t in TxnType}
    parsed, skipped = [], []
    for position, row in frame.iterrows():
        raw_type = str(row.get("type", "")).strip().lower().replace(" ", "_")
        if raw_type not in valid_types:
            skipped.append((position + 2, f"unknown type {row.get('type')!r}"))
            continue
        try:
            parsed.append(Transaction(
                account_id=record.id,
                date=pd.Timestamp(row["date"]).date(),
                type=TxnType(raw_type),
                symbol=str(row.get("symbol") or "").strip(),
                quantity=float(row.get("quantity") or 0.0),
                price=float(row.get("price") or 0.0),
                amount=float(row.get("amount") or 0.0),
                fees=float(row.get("fees") or 0.0),
                notes=str(row.get("notes") or ""),
            ))
        except Exception as exc:  # noqa: BLE001 - report the row, keep going
            skipped.append((position + 2, str(exc)))

    ui.console.print(f"Parsed {len(parsed)} transaction(s); {len(skipped)} skipped.")
    for line, reason in skipped[:10]:
        ui.console.print(f"  [yellow]line {line}: {reason}[/yellow]")
    if dry_run:
        preview = pd.DataFrame([{
            "date": t.date, "type": t.type.value, "symbol": t.symbol,
            "quantity": t.quantity, "price": t.price, "amount": t.amount,
        } for t in parsed[:15]])
        if not preview.empty:
            ui.console.print(ui.frame_table(preview, title="Preview (first 15)"))
        ui.console.print("[dim]Dry run: nothing saved.[/dim]")
        return
    store.add_transactions(parsed)
    ui.console.print(f"Imported {len(parsed)} transaction(s) into {account}.")


@txn.command("list")
@click.option("--account", default=None)
@click.option("--symbol", default=None)
@click.option("--limit", default=40, show_default=True)
def txn_list(account: str | None, symbol: str | None, limit: int) -> None:
    """List recorded transactions."""
    store = _store()
    account_id = None
    if account:
        record = store.get_account(account)
        if record is None:
            ui.console.print(f"[red]No account named {account!r}.[/red]")
            raise SystemExit(1)
        account_id = record.id
    transactions = store.transactions(account_id=account_id, symbol=symbol)
    if not transactions:
        ui.console.print("No transactions recorded.")
        return
    names = {a.id: a.name for a in store.accounts()}
    frame = pd.DataFrame([{
        "id": t.id, "date": t.date, "account": names.get(t.account_id, t.account_id),
        "type": t.type.value, "symbol": t.symbol, "quantity": t.quantity,
        "price": t.price, "amount": t.amount, "fees": t.fees,
    } for t in transactions[-limit:]]).set_index("id")
    ui.console.print(ui.frame_table(
        frame, title=f"Transactions (last {min(limit, len(transactions))})",
        formatters={"price": lambda v: ui.money(v, abbreviate=False),
                    "amount": lambda v: ui.money(v, abbreviate=False),
                    "fees": lambda v: ui.money(v, abbreviate=False),
                    "quantity": lambda v: ui.num(v, 4)},
        index_label="id"))


@main.command()
@click.option("--account", default=None, help="Limit to one account.")
@click.option("--by-account", is_flag=True, help="Split positions by account.")
def portfolio(account: str | None, by_account: bool) -> None:
    """Current positions, cost basis and unrealised gains."""
    store, state, accounts = _ledger(account)
    positions = state.positions(by_account=by_account)
    if not positions:
        ui.console.print("No open positions. Record trades with "
                         "[bold]stocklab txn add[/bold].")
        return
    prices = _prices_for(sorted({p.symbol for p in positions}))
    positions = state.positions(prices=prices, by_account=by_account)

    rows = []
    for position in positions:
        rows.append({
            "symbol": position.symbol,
            "account": (accounts[position.account_id].name
                        if by_account and position.account_id in accounts else "all"),
            "quantity": position.quantity,
            "cost/share": position.cost_per_share,
            "price": position.price,
            "cost basis": position.cost_basis,
            "value": position.market_value,
            "gain": position.unrealized_gain,
            "gain %": position.unrealized_pct,
            "LT shares": position.long_term_quantity,
        })
    frame = pd.DataFrame(rows).set_index("symbol").sort_values("value", ascending=False)
    total_value = float(frame["value"].sum(skipna=True))
    cash = state.total_cash(None if account is None else
                            store.get_account(account).id)
    frame["weight"] = frame["value"] / (total_value + max(cash, 0.0))

    ui.console.print(ui.frame_table(
        frame, title="Positions",
        formatters={
            "quantity": lambda v: ui.num(v, 2),
            "cost/share": lambda v: ui.money(v, abbreviate=False),
            "price": lambda v: ui.money(v, abbreviate=False),
            "cost basis": ui.money, "value": ui.money, "gain": ui.money,
            "gain %": lambda v: ui.pct(v, signed=True),
            "weight": lambda v: ui.pct(v),
            "LT shares": lambda v: ui.num(v, 0),
        }, index_label="symbol"))
    ui.console.print(ui.kv_table([
        ("Market value", ui.money(total_value)),
        ("Cash", ui.money(cash)),
        ("Total", ui.money(total_value + cash)),
        ("Cost basis", ui.money(float(frame["cost basis"].sum(skipna=True)))),
        ("Unrealised gain", ui.signed_text(
            float(frame["gain"].sum(skipna=True)),
            formatter=lambda v: ui.money(v, abbreviate=False))),
    ], title="Totals"))
    ui.print_notes(state.warnings, "Ledger warnings")


@main.command()
@click.option("--account", default=None)
@click.option("--benchmark", default="SPY", show_default=True)
@click.option("--risk-free", type=float, default=None,
              help="Defaults to the current 10-year Treasury yield.")
def performance(account: str | None, benchmark: str, risk_free: float | None) -> None:
    """Time-weighted and money-weighted returns, risk, benchmark comparison."""
    from .data.providers import fred
    from .portfolio.performance import analyse as analyse_performance
    from .portfolio.performance import contribution_by_position

    store, state, accounts = _ledger(account)
    account_id = store.get_account(account).id if account else None
    transactions = store.transactions(account_id=account_id)
    if not transactions:
        ui.console.print("No transactions recorded.")
        return
    rate = fred.risk_free_rate() if risk_free is None else risk_free

    with ui.console.status("Valuing holdings through time..."):
        report = analyse_performance(state, transactions, benchmark=benchmark,
                                     risk_free=rate)
    ui.console.print(ui.header(
        "Performance", f"{report.start} to {report.end}  |  benchmark {benchmark}"))
    ui.console.print(ui.kv_table([
        ("Beginning value", ui.money(report.beginning_value)),
        ("Ending value", ui.money(report.ending_value)),
        ("Net contributions", ui.money(report.net_contributions)),
        ("Gain", ui.signed_text(report.gain,
                                formatter=lambda v: ui.money(v, abbreviate=False))),
        ("", ""),
        ("Time-weighted return (total)", ui.pct(report.twr, signed=True)),
        ("Time-weighted (annualised)", ui.pct(report.twr_annualized, signed=True)),
        ("Money-weighted (IRR)", ui.pct(report.mwr, signed=True)),
        ("Timing effect (MWR - TWR)", ui.signed_text(report.behaviour_gap)),
        ("", ""),
        (f"{benchmark} total return", ui.pct(report.benchmark_return, signed=True)),
    ], title="Returns", width=32))

    if report.risk:
        ui.console.print(ui.kv_table([
            ("Volatility", ui.pct(report.risk.get("volatility"))),
            ("Sharpe", ui.num(report.risk.get("sharpe"), 2)),
            ("Sortino", ui.num(report.risk.get("sortino"), 2)),
            ("Max drawdown", ui.pct(report.risk.get("max_drawdown"), signed=True)),
            ("Calmar", ui.num(report.risk.get("calmar"), 2)),
            ("Beta", ui.num(report.vs_benchmark.get("beta"), 2)),
            ("Alpha (annual)", ui.pct(report.vs_benchmark.get("alpha_annual"), signed=True)),
            ("Correlation", ui.num(report.vs_benchmark.get("correlation"), 2)),
        ], title="Risk", width=32))

    prices = _prices_for(sorted({p.symbol for p in state.positions()}))
    contribution = contribution_by_position(state, prices, transactions)
    if not contribution.empty:
        # Drop the dividend column when no dividends were ever recorded; an
        # all-zero column reads as "this paid nothing", not "nothing logged".
        if contribution["dividends"].abs().sum() == 0:
            contribution = contribution.drop(columns=["dividends"])
        ui.console.print(ui.frame_table(
            contribution.head(15), title="Contribution to gain",
            formatters={c: ui.money for c in
                        ("realized", "unrealized", "total_gain", "cost_basis",
                         "market_value") if c in contribution.columns}
            | {"share_of_gain": lambda v: ui.pct(v),
               "return_on_cost": lambda v: ui.pct(v, signed=True)},
            index_label="symbol"))
    ui.print_notes(report.notes, "Notes")


@main.command()
@click.option("--model", default="default", show_default=True,
              help="Target allocation model name.")
@click.option("--target", "targets", multiple=True,
              help="SYMBOL=WEIGHT, repeatable (e.g. --target AAPL=0.1).")
@click.option("--band", default=0.25, show_default=True,
              help="Relative drift band before trading.")
@click.option("--abs-band", default=0.05, show_default=True,
              help="Absolute drift band; both must be breached.")
@click.option("--tax-budget", type=float, default=None,
              help="Cap on estimated tax this plan may incur.")
@click.option("--fractional", is_flag=True, help="Allow fractional shares.")
@click.option("--save", is_flag=True, help="Store the supplied targets as the model.")
def rebalance(model: str, targets: tuple[str, ...], band: float, abs_band: float,
              tax_budget: float | None, fractional: bool, save: bool) -> None:
    """Generate a tax-aware rebalancing trade list."""
    from .portfolio.rebalance import plan as build_rebalance_plan

    store, state, accounts = _ledger()
    weights = {}
    for item in targets:
        if "=" not in item:
            ui.console.print(f"[red]Bad --target {item!r}; use SYMBOL=WEIGHT.[/red]")
            raise SystemExit(1)
        symbol, weight = item.split("=", 1)
        weights[symbol.strip().upper()] = float(weight)

    if weights and save:
        store.clear_targets(model)
        for symbol, weight in weights.items():
            store.set_target(model, symbol, weight)
        ui.console.print(f"Saved {len(weights)} target(s) to model {model!r}.")
    target_series = pd.Series(weights, dtype=float) if weights else store.targets(model)
    if target_series.empty:
        ui.console.print(
            f"[red]No targets for model {model!r}. Supply --target SYMBOL=WEIGHT "
            "or save a model first.[/red]")
        raise SystemExit(1)

    symbols = sorted(set(target_series.index) | {p.symbol for p in state.positions()})
    symbols = [s for s in symbols if s != "CASH"]
    prices = _prices_for(symbols)
    result = build_rebalance_plan(state, accounts, prices, target_series,
                                  relative_band=band, absolute_band=abs_band,
                                  max_tax_budget=tax_budget,
                                  whole_shares=not fractional)

    ui.console.print(ui.header(
        f"Rebalance to model {model!r}",
        f"portfolio {ui.money(result.total_value)}  |  turnover "
        f"{ui.pct(result.turnover)}  |  estimated tax "
        f"{ui.money(result.estimated_tax, abbreviate=False)}"))
    ui.console.print(ui.frame_table(
        result.drift, title="Drift versus target",
        formatters={"actual": lambda v: ui.pct(v), "target": lambda v: ui.pct(v),
                    "drift": lambda v: ui.pct(v, signed=True),
                    "relative_drift": lambda v: ui.pct(v, 0, signed=True),
                    "out_of_band": lambda v: "yes" if v else "no"},
        index_label="symbol"))
    trades = result.to_frame()
    if trades.empty:
        ui.console.print("[green]No trades required.[/green]")
    else:
        # Reasons and warnings are sentences; as table columns they fold to a
        # few characters per line. They are listed underneath instead.
        compact = trades[["account", "side", "quantity", "price", "value",
                          "est_gain", "est_tax", "term"]].copy()
        compact.index = trades["symbol"]
        compact = compact.rename(columns={"quantity": "qty", "est_gain": "gain",
                                          "est_tax": "tax"})
        ui.console.print(ui.frame_table(
            compact, title="Trades",
            formatters={"price": lambda v: ui.money(v, abbreviate=False),
                        "value": ui.money, "gain": ui.money,
                        "tax": lambda v: ui.money(v, abbreviate=False),
                        "qty": lambda v: ui.num(v, 2)},
            index_label="symbol"))
        reasons = [f"{t.symbol} {t.side} in {t.account_name}: {t.reason}"
                   for t in result.trades]
        ui.print_notes(reasons, "Why each trade", "cyan")
        warnings_out = [f"{t.symbol}: {w}" for t in result.trades for w in t.warnings]
        if warnings_out:
            ui.print_notes(warnings_out, "Trade warnings", "red")
    ui.print_notes(result.notes, "Notes")


@main.command("tax")
@click.option("--year", type=int, default=None, help="Defaults to the current year.")
@click.option("--short-rate", default=0.24, show_default=True,
              help="Marginal ordinary income rate.")
@click.option("--long-rate", default=0.15, show_default=True,
              help="Long-term capital gains rate.")
@click.option("--niit", is_flag=True, help="Add the 3.8% net investment income tax.")
def tax_cmd(year: int | None, short_rate: float, long_rate: float, niit: bool) -> None:
    """Realised gains, wash sales and an estimated federal liability."""
    from .portfolio.tax import annual_report, unrealized_frame
    store, state, accounts = _ledger()
    report = annual_report(state, accounts, year=year, short_term_rate=short_rate,
                           long_term_rate=long_rate, niit=niit)
    ui.console.print(ui.header(f"Tax summary {report.year}",
                               "Federal estimate only; not tax advice"))
    ui.console.print(ui.kv_table([
        ("Short-term gains", ui.signed_text(
            report.short_term_gains, formatter=lambda v: ui.money(v, abbreviate=False))),
        ("Long-term gains", ui.signed_text(
            report.long_term_gains, formatter=lambda v: ui.money(v, abbreviate=False))),
        ("Net", ui.signed_text(report.net_gain,
                               formatter=lambda v: ui.money(v, abbreviate=False))),
        ("Wash sales disallowed", ui.money(report.disallowed_wash_sales,
                                           abbreviate=False)),
        ("Deductible vs income", ui.money(report.deductible_against_income,
                                          abbreviate=False)),
        ("Loss carry-forward", ui.money(report.loss_carryforward, abbreviate=False)),
        ("Estimated federal tax", ui.money(report.estimated_tax, abbreviate=False)),
    ], title="Summary", width=30))
    if not report.realized_detail.empty:
        detail = report.realized_detail.copy()
        # The per-row notes are paragraphs; rendering them as a table column
        # folds them to one character per line. They go below instead.
        row_notes = [
            f"{row['symbol']} sold {row['sell_date']}: {row['notes']}"
            for _, row in detail.iterrows() if str(row.get("notes") or "").strip()
        ]
        compact = detail[["account", "open_date", "sell_date", "quantity",
                          "proceeds", "cost_basis", "gain", "term",
                          "wash_sale_disallowed", "taxable_gain"]].copy()
        compact.index = detail["symbol"]
        compact = compact.rename(columns={
            "open_date": "opened", "sell_date": "sold", "quantity": "qty",
            "cost_basis": "basis", "wash_sale_disallowed": "disallowed",
            "taxable_gain": "taxable",
        })
        ui.console.print(ui.frame_table(
            compact, title="Realised disposals",
            formatters={"proceeds": ui.money, "basis": ui.money,
                        "gain": ui.money, "taxable": ui.money,
                        "disallowed": ui.money,
                        "qty": lambda v: ui.num(v, 2)},
            index_label="symbol", max_rows=25))
        if row_notes:
            ui.print_notes(row_notes, "Wash sales and adjustments")
    prices = _prices_for(sorted({p.symbol for p in state.positions()}))
    unrealized = unrealized_frame(state, accounts, prices)
    if not unrealized.empty:
        soon = unrealized[(unrealized["term"] == "short")
                          & (unrealized["days_to_long_term"] <= 90)
                          & (~unrealized["sheltered"])]
        if not soon.empty:
            ui.console.print(ui.frame_table(
                soon[["account", "open_date", "quantity", "unrealized_gain",
                      "days_to_long_term"]].set_index(soon["symbol"]),
                title="Approaching long-term treatment (taxable accounts)",
                formatters={"unrealized_gain": ui.money,
                            "quantity": lambda v: ui.num(v, 2),
                            "days_to_long_term": lambda v: ui.num(v, 0)},
                index_label="symbol"))
    ui.print_notes(report.notes, "Notes")


@main.command()
@click.option("--min-loss", default=500.0, show_default=True)
@click.option("--short-rate", default=0.24, show_default=True)
@click.option("--long-rate", default=0.15, show_default=True)
def harvest(min_loss: float, short_rate: float, long_rate: float) -> None:
    """Tax-loss harvesting candidates, with wash-sale checks."""
    from .portfolio.tax import harvest_candidates
    store, state, accounts = _ledger()
    prices = _prices_for(sorted({p.symbol for p in state.positions()}))
    candidates = harvest_candidates(state, accounts, prices, min_loss=min_loss,
                                    short_term_rate=short_rate,
                                    long_term_rate=long_rate)
    if not candidates:
        ui.console.print(
            "No harvestable losses above the threshold in taxable accounts.\n"
            "[dim]Losses inside an IRA, Roth or 401(k) are not deductible, so "
            "they are never candidates.[/dim]")
        return
    frame = pd.DataFrame([{
        "symbol": c.symbol, "account": c.account_name, "qty": c.quantity,
        "basis": c.cost_basis, "value": c.market_value, "loss": c.loss,
        "term": c.term, "est saving": c.estimated_tax_saving,
        "wash risk": "YES" if c.wash_sale_risk else "no",
        "clear after": c.blocked_until or "-",
    } for c in candidates]).set_index("symbol")
    ui.console.print(ui.frame_table(
        frame, title="Harvest candidates",
        formatters={"basis": ui.money, "value": ui.money, "loss": ui.money,
                    "est saving": lambda v: ui.money(v, abbreviate=False),
                    "qty": lambda v: ui.num(v, 2)},
        index_label="symbol"))
    total = sum(c.estimated_tax_saving for c in candidates if not c.wash_sale_risk)
    ui.console.print(f"\nEstimated tax saving from unblocked candidates: "
                     f"[bold]{ui.money(total, abbreviate=False)}[/bold]")
    ui.print_notes(
        [n for c in candidates for n in c.notes][:6]
        + ["Harvesting resets your cost basis lower, so it defers tax rather "
           "than eliminating it. The benefit is the time value of the deferral "
           "plus any rate difference, not the full headline saving."],
        "Notes")


@main.command()
@click.option("--benchmark", default="SPY", show_default=True)
def risk(benchmark: str) -> None:
    """Concentration, correlation, exposures and stress scenarios."""
    from .portfolio.risk_report import build as build_risk
    store, state, accounts = _ledger()
    positions = state.positions()
    if not positions:
        ui.console.print("No open positions.")
        return
    symbols = sorted({p.symbol for p in positions})
    with ui.console.status("Loading prices and sectors..."):
        prices = _prices_for(symbols)
        sectors = _sectors_for(symbols)
        report = build_risk(state, accounts, prices, sectors, benchmark=benchmark)

    ui.console.print(ui.header("Portfolio risk",
                               f"total {ui.money(report.total_value)}"))
    concentration = report.concentration
    ui.console.print(ui.kv_table([
        ("Positions", ui.num(concentration.get("positions"), 0)),
        ("Effective holdings", ui.num(concentration.get("effective_holdings"), 1)),
        ("Largest position", ui.pct(concentration.get("largest_weight"))),
        ("Top 3", ui.pct(concentration.get("top_3_weight"))),
        ("Top 5", ui.pct(concentration.get("top_5_weight"))),
        ("Cash", ui.pct(concentration.get("cash_weight"))),
        ("Portfolio beta", ui.num(report.portfolio_beta, 2)),
        ("Portfolio volatility", ui.pct(report.portfolio_vol)),
        ("Average pairwise correlation",
         ui.num(report.risk_stats.get("average_pairwise_correlation"), 2)),
        ("Diversification ratio",
         ui.num(report.risk_stats.get("diversification_ratio"), 2)),
    ], title="Concentration and market risk", width=32))

    if not report.sector_exposure.empty:
        ui.console.print(ui.frame_table(
            report.sector_exposure.to_frame("weight"), title="Sector exposure",
            formatters={"weight": lambda v: ui.pct(v)}, index_label="sector"))
    if not report.stress.empty:
        ui.console.print(ui.frame_table(
            report.stress.set_index("scenario"), title="Stress scenarios",
            formatters={"index_move": lambda v: ui.pct(v, signed=True),
                        "portfolio_move": lambda v: ui.pct(v, signed=True),
                        "value_change": ui.money, "resulting_value": ui.money},
            index_label="scenario"))
    ui.print_notes(report.notes, "Notes")


@main.group()
def watch() -> None:
    """Maintain a watchlist with theses and target buy prices."""


@watch.command("add")
@click.argument("symbol")
@click.option("--list", "list_name", default="default", show_default=True)
@click.option("--thesis", default="")
@click.option("--target-buy", type=float, default=None)
def watch_add(symbol: str, list_name: str, thesis: str,
              target_buy: float | None) -> None:
    """Add a symbol to a watchlist."""
    _store().watch(symbol, list_name, thesis, target_buy)
    ui.console.print(f"Watching {symbol.upper()} in {list_name!r}.")


@watch.command("show")
@click.option("--list", "list_name", default=None)
def watch_show(list_name: str | None) -> None:
    """Show watchlist entries against current prices."""
    frame = _store().watchlist(list_name)
    if frame.empty:
        ui.console.print("Watchlist is empty.")
        return
    prices = _prices_for(list(frame["symbol"]))
    frame["price"] = frame["symbol"].map(prices)
    frame["vs_target"] = (frame["price"] / frame["target_buy"] - 1.0)
    ui.console.print(ui.frame_table(
        frame.set_index("symbol"), title="Watchlist",
        formatters={"price": lambda v: ui.money(v, abbreviate=False),
                    "target_buy": lambda v: ui.money(v, abbreviate=False),
                    "vs_target": lambda v: ui.pct(v, signed=True)},
        index_label="symbol"))


@watch.command("remove")
@click.argument("symbol")
@click.option("--list", "list_name", default="default", show_default=True)
def watch_remove(symbol: str, list_name: str) -> None:
    """Remove a symbol from a watchlist."""
    _store().unwatch(symbol, list_name)
    ui.console.print(f"Removed {symbol.upper()} from {list_name!r}.")


if __name__ == "__main__":  # pragma: no cover
    main()
