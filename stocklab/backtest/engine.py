"""A deliberately pessimistic backtester.

Most backtests are wrong in the same four ways, and each one is addressed here
explicitly rather than assumed away:

1. **Look-ahead.** A signal computed from today's close cannot be traded at
   today's close. Every weight vector is shifted forward one bar before it is
   applied, always, with no option to disable it.
2. **Costs.** Commission and slippage are charged on turnover. A daily-rebalanced
   strategy with 5bp of slippage pays roughly 12% a year in friction, which
   turns most high-turnover "edges" negative.
3. **Survivorship.** A universe drawn from today's listed symbols excludes
   every company that went to zero. This cannot be fixed without a
   point-in-time constituent history, so the result carries a
   ``survivorship_bias`` warning and the backtest refuses to be quiet about it.
4. **Point-in-time fundamentals.** Screening on restated figures means trading
   on numbers nobody had. ``stocklab.data.providers.sec.as_of_filter`` exists
   for this; a fundamental backtest that does not use it is measuring hindsight.

What this is good for: comparing two rules on the same data with the same
costs. What it is not good for: producing a number to show someone.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from ..config import settings
from ..util.mathx import NAN, is_num, safe_div
from .metrics import performance_summary

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    summary: dict
    benchmark: pd.Series | None = None
    benchmark_summary: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    @property
    def total_cost_drag(self) -> float:
        """Annualised performance given up to friction."""
        if self.costs.empty or self.equity.empty:
            return NAN
        years = len(self.returns) / settings().trading_days
        total = float(self.costs.sum())
        start = float(self.equity.iloc[0])
        if years <= 0 or start <= 0:
            return NAN
        return (total / start) / years

    def report(self) -> pd.DataFrame:
        rows = {"strategy": self.summary}
        if self.benchmark_summary:
            rows["benchmark"] = self.benchmark_summary
        return pd.DataFrame(rows)


def run(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    initial_capital: float = 100_000.0,
    commission_per_trade: float | None = None,
    slippage_bps: float | None = None,
    benchmark: pd.Series | None = None,
    risk_free: float = 0.0,
    label: str = "strategy",
) -> BacktestResult:
    """Apply a target-weight schedule to a price panel.

    ``prices`` is dates x symbols of total-return prices (use ``adj_close``);
    ``weights`` is dates x symbols of target weights, which need not sum to 1 --
    the remainder is treated as cash earning the risk-free rate.
    """
    cfg = settings()
    commission = cfg.commission_per_trade if commission_per_trade is None else commission_per_trade
    slippage = (cfg.slippage_bps if slippage_bps is None else slippage_bps) / 10_000.0
    warnings_out: list[str] = [
        "Universe is drawn from currently listed symbols, so companies that "
        "delisted, went bankrupt or were acquired are missing. Survivorship "
        "bias inflates every return shown here, typically by 1-4% a year.",
    ]

    prices = prices.sort_index().ffill()
    common = prices.index.intersection(weights.index)
    if len(common) < 30:
        return BacktestResult(pd.Series(dtype=float), pd.Series(dtype=float),
                              pd.DataFrame(), pd.Series(dtype=float),
                              pd.Series(dtype=float), {},
                              warnings=["Fewer than 30 overlapping dates."])

    prices = prices.loc[common]
    # The shift is the whole point: a weight derived from information available
    # at the close of day t can only be held from day t+1 onward.
    target = weights.loc[common].reindex(columns=prices.columns).fillna(0.0).shift(1).fillna(0.0)

    asset_returns = prices.pct_change().fillna(0.0)
    rf_daily = risk_free / cfg.trading_days

    equity = [float(initial_capital)]
    gross_returns: list[float] = []
    net_returns: list[float] = []
    turnover_series: list[float] = []
    cost_series: list[float] = []
    index = list(prices.index)
    previous = pd.Series(0.0, index=prices.columns)

    for i in range(1, len(index)):
        current_target = target.iloc[i]
        traded = (current_target - previous).abs().sum()
        invested = float(current_target.sum())
        cash_weight = max(0.0, 1.0 - invested)

        period_return = float((current_target * asset_returns.iloc[i]).sum())
        period_return += cash_weight * rf_daily

        # Slippage scales with turnover; commission is per name traded.
        names_traded = int(((current_target - previous).abs() > 1e-6).sum())
        cost_fraction = traded * slippage
        cost_cash = cost_fraction * equity[-1] + commission * names_traded
        cost_fraction_total = safe_div(cost_cash, equity[-1]) or 0.0

        net = period_return - cost_fraction_total
        equity.append(equity[-1] * (1.0 + net))
        gross_returns.append(period_return)
        net_returns.append(net)
        turnover_series.append(traded)
        cost_series.append(cost_cash)
        previous = current_target

    equity_series = pd.Series(equity, index=index[: len(equity)], name="equity")
    returns_series = pd.Series(net_returns, index=index[1: len(equity)], name="return")
    turnover_out = pd.Series(turnover_series, index=index[1: len(equity)], name="turnover")
    costs_out = pd.Series(cost_series, index=index[1: len(equity)], name="cost")

    summary = performance_summary(returns_series, equity_series, risk_free=risk_free)
    summary["annual_turnover"] = float(turnover_out.mean() * cfg.trading_days)
    summary["gross_cagr"] = performance_summary(
        pd.Series(gross_returns, index=returns_series.index),
        None, risk_free=risk_free).get("cagr", NAN)

    bench_summary: dict = {}
    bench_curve = None
    if benchmark is not None and not benchmark.empty:
        bench_aligned = benchmark.reindex(equity_series.index).ffill().dropna()
        if len(bench_aligned) > 30:
            bench_returns = bench_aligned.pct_change().dropna()
            bench_curve = (1.0 + bench_returns).cumprod() * initial_capital
            bench_summary = performance_summary(bench_returns, bench_curve,
                                                risk_free=risk_free)

    if summary.get("annual_turnover", 0) > 4.0:
        warnings_out.append(
            f"Annual turnover of {summary['annual_turnover']:.1f}x. In a taxable "
            "account this converts nearly all gains to short-term, which no "
            "pre-tax backtest shows."
        )
    drag = safe_div(float(costs_out.sum()), float(equity_series.iloc[0]))
    if is_num(drag) and drag > 0.10:
        warnings_out.append(
            f"Trading costs consumed {drag:.1%} of starting capital. "
            f"Gross CAGR {summary.get('gross_cagr', NAN):.2%} versus net "
            f"{summary.get('cagr', NAN):.2%}."
        )
    if len(returns_series) < 252 * 3:
        warnings_out.append(
            f"Only {len(returns_series) / 252:.1f} years of data. Nothing "
            "reliable can be concluded about a strategy from a window this "
            "short -- it covers one market regime at most."
        )
    return BacktestResult(
        equity=equity_series, returns=returns_series, weights=target,
        turnover=turnover_out, costs=costs_out, summary=summary,
        benchmark=bench_curve, benchmark_summary=bench_summary,
        warnings=warnings_out,
        params={"initial_capital": initial_capital, "slippage_bps": slippage * 10_000,
                "commission_per_trade": commission, "label": label},
    )


# --------------------------------------------------------------------------- #
# weight builders
# --------------------------------------------------------------------------- #
def rebalance_schedule(index: pd.DatetimeIndex, frequency: str = "M") -> pd.DatetimeIndex:
    """Dates on which a rebalance happens; weights are held between them."""
    if frequency.upper() in ("D", "DAILY"):
        return index
    alias = {"W": "W-FRI", "M": "ME", "Q": "QE", "A": "YE", "Y": "YE"}.get(
        frequency.upper(), frequency)
    marks = pd.Series(index, index=index).resample(alias).last().dropna()
    return pd.DatetimeIndex(marks.to_numpy())


def weights_from_ranks(
    scores: pd.DataFrame,
    top_n: int = 10,
    frequency: str = "M",
    max_weight: float | None = None,
) -> pd.DataFrame:
    """Equal-weight the top ``top_n`` names by score, rebalanced periodically.

    ``scores`` must already be point-in-time: each row may only contain
    information available on that date. This function cannot check that, and it
    is the easiest place in a backtest to introduce look-ahead.
    """
    if scores.empty:
        return pd.DataFrame()
    marks = rebalance_schedule(pd.DatetimeIndex(scores.index), frequency)
    cap = max_weight or 1.0

    # Build one row per rebalance date, then carry whole rows forward.
    #
    # The obvious shortcut -- writing weights into a zero-filled frame and then
    # `replace(0, nan).ffill()` -- is wrong, and wrong in the most dangerous
    # direction: ffill operates per column, so a name selected once keeps its
    # weight forever and gross exposure ratchets up toward the size of the
    # universe. It silently produced 2.5x leverage here, and with it a 50%
    # annual return out of ten mega-caps.
    rows: dict[pd.Timestamp, pd.Series] = {}
    for stamp in marks:
        if stamp not in scores.index:
            continue
        row = scores.loc[stamp].dropna()
        if row.empty:
            continue
        chosen = row.nlargest(min(top_n, len(row)))
        weight = min(1.0 / len(chosen), cap)
        allocation = pd.Series(0.0, index=scores.columns)
        allocation.loc[chosen.index] = weight
        rows[stamp] = allocation

    if not rows:
        return pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    schedule = pd.DataFrame(rows).T.sort_index()
    return schedule.reindex(scores.index).ffill().fillna(0.0)


def buy_and_hold_weights(prices: pd.DataFrame, symbols: list[str] | None = None
                         ) -> pd.DataFrame:
    """Equal-weight at the start and never trade again.

    The benchmark any active rule must beat *after costs*. It is harder to beat
    than it looks, because it has no turnover, no tax drag and no timing risk.
    """
    symbols = symbols or list(prices.columns)
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    available = [s for s in symbols if s in prices.columns]
    if not available:
        return weights
    weights[available] = 1.0 / len(available)
    return weights
