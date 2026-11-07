"""Portfolio performance: time-weighted and money-weighted returns, attribution.

The two return measures answer different questions and routinely differ by
several points a year:

* **TWR** strips out the timing and size of deposits, so it measures the
  *strategy*. It is what fund managers must report, because they do not control
  when clients add money.
* **MWR (dollar-weighted IRR)** includes them, so it measures the *investor*.
  Someone who added heavily at a bottom earns an MWR well above the TWR, and
  someone who piled in at a top earns far less.

Reporting one alone is how people convince themselves they are doing better
than they are. Both are reported here, with the gap called out, because the gap
is the behavioural cost or benefit of the investor's own timing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from ..analysis.risk import beta_alpha, return_stats, to_returns
from ..data.providers.yahoo import fetch_closes, fetch_history
from ..util.mathx import NAN, is_num, xirr
from .ledger import LedgerState
from .models import TxnType, Transaction


@dataclass
class PerformanceReport:
    start: date
    end: date
    beginning_value: float
    ending_value: float
    net_contributions: float
    gain: float
    twr: float
    twr_annualized: float
    mwr: float
    benchmark_return: float
    benchmark_symbol: str
    equity_curve: pd.Series
    risk: dict = field(default_factory=dict)
    vs_benchmark: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def behaviour_gap(self) -> float:
        """MWR minus TWR: what the investor's timing added or cost."""
        if is_num(self.mwr) and is_num(self.twr_annualized):
            return self.mwr - self.twr_annualized
        return NAN


def daily_holdings(transactions: list[Transaction], start: date, end: date
                   ) -> pd.DataFrame:
    """Share count per symbol for every calendar day in the window.

    Built by cumulating signed share changes and forward-filling, so a
    portfolio can be marked to market on any date without replaying the ledger
    for each one.
    """
    index = pd.date_range(start, end, freq="D")
    deltas: dict[str, pd.Series] = {}
    split_events: list[tuple[pd.Timestamp, str, float]] = []

    for txn in transactions:
        if not txn.symbol or not txn.type.affects_shares:
            continue
        stamp = pd.Timestamp(txn.date)
        if txn.type is TxnType.SPLIT:
            ratio = txn.price if txn.price > 0 else txn.quantity
            if ratio > 0:
                split_events.append((stamp, txn.symbol, ratio))
            continue
        sign = -1.0 if txn.type in (TxnType.SELL, TxnType.TRANSFER_OUT) else 1.0
        series = deltas.setdefault(txn.symbol, pd.Series(0.0, index=index))
        if stamp in series.index:
            series.loc[stamp] += sign * txn.quantity
        elif stamp < index[0]:
            series.iloc[0] += sign * txn.quantity   # opened before the window

    if not deltas:
        return pd.DataFrame(index=index)

    holdings = pd.DataFrame(deltas).cumsum()
    # Splits scale every share held up to that point.
    for stamp, symbol, ratio in sorted(split_events):
        if symbol in holdings.columns and stamp in holdings.index:
            mask = holdings.index >= stamp
            pre = holdings.loc[~mask, symbol]
            holdings.loc[mask, symbol] = holdings.loc[mask, symbol] + pre.iloc[-1] * (ratio - 1.0) if len(pre) else holdings.loc[mask, symbol]
    return holdings.clip(lower=0.0)


def value_series(transactions: list[Transaction], start: date, end: date,
                 cash_series: pd.Series | None = None) -> pd.Series:
    """Daily mark-to-market portfolio value."""
    holdings = daily_holdings(transactions, start, end)
    if holdings.empty:
        return pd.Series(dtype=float)
    symbols = [c for c in holdings.columns if holdings[c].abs().sum() > 0]
    if not symbols:
        return pd.Series(dtype=float)

    span_years = max((end - start).days / 365.25, 1.0)
    period = f"{int(np.ceil(span_years)) + 1}y"
    prices = fetch_closes(symbols, period=period, field="close")
    if prices.empty:
        return pd.Series(dtype=float)

    prices = prices.reindex(holdings.index.union(prices.index)).ffill()
    prices = prices.reindex(holdings.index)
    common = [c for c in symbols if c in prices.columns]
    values = (holdings[common] * prices[common]).sum(axis=1)
    if cash_series is not None:
        values = values.add(cash_series.reindex(values.index).ffill().fillna(0.0),
                            fill_value=0.0)
    return values.dropna()


def time_weighted_return(values: pd.Series,
                         flows: list[tuple[date, float]]) -> tuple[float, pd.Series]:
    """TWR by chaining sub-period returns between external cashflows.

    A deposit raises the portfolio's value without being a gain, so each
    sub-period return subtracts the flow from the ending value before
    comparing. Chaining the sub-periods removes the effect of flow timing
    entirely, which is the point.
    """
    values = values.dropna()
    if len(values) < 2:
        return NAN, pd.Series(dtype=float)

    flow_by_day: dict[pd.Timestamp, float] = {}
    for when, amount in flows:
        stamp = pd.Timestamp(when).normalize()
        flow_by_day[stamp] = flow_by_day.get(stamp, 0.0) + float(amount)

    growth = [1.0]
    index = [values.index[0]]
    for prev, curr in zip(values.index[:-1], values.index[1:]):
        begin = float(values.loc[prev])
        end_value = float(values.loc[curr])
        flow = flow_by_day.get(pd.Timestamp(curr).normalize(), 0.0)
        if begin <= 0:
            # Nothing invested at the start of the sub-period: no return to
            # measure, so the chain simply carries forward.
            growth.append(growth[-1])
            index.append(curr)
            continue
        period_return = (end_value - flow) / begin - 1.0
        growth.append(growth[-1] * (1.0 + period_return))
        index.append(curr)

    curve = pd.Series(growth, index=index)
    return float(curve.iloc[-1] - 1.0), curve


def money_weighted_return(values: pd.Series,
                          flows: list[tuple[date, float]]) -> float:
    """Dollar-weighted IRR: the return the investor actually experienced."""
    values = values.dropna()
    if len(values) < 2:
        return NAN
    cashflows: list[tuple[pd.Timestamp, float]] = [
        (pd.Timestamp(values.index[0]), -float(values.iloc[0]))
    ]
    for when, amount in flows:
        stamp = pd.Timestamp(when)
        if stamp <= values.index[0] or stamp > values.index[-1]:
            continue
        cashflows.append((stamp, -float(amount)))   # a deposit is money in
    cashflows.append((pd.Timestamp(values.index[-1]), float(values.iloc[-1])))
    return xirr(cashflows)


def analyse(state: LedgerState, transactions: list[Transaction],
            *, start: date | None = None, end: date | None = None,
            benchmark: str = "SPY", risk_free: float = 0.0) -> PerformanceReport:
    """Full performance report for a replayed ledger."""
    notes: list[str] = []
    dated = [t for t in transactions if t.date]
    if not dated:
        return PerformanceReport(date.today(), date.today(), 0, 0, 0, 0,
                                 NAN, NAN, NAN, NAN, benchmark,
                                 pd.Series(dtype=float),
                                 notes=["No transactions."])
    start = start or min(t.date for t in dated)
    end = end or date.today()

    values = value_series(dated, start, end)
    if values.empty:
        return PerformanceReport(start, end, 0, 0, 0, 0, NAN, NAN, NAN, NAN,
                                 benchmark, pd.Series(dtype=float),
                                 notes=["Could not value holdings (no price data)."])

    flows = state.external_flows
    net_contributions = float(sum(amount for _, amount in flows))
    twr, curve = time_weighted_return(values, flows)
    years = max((end - start).days / 365.25, 1e-9)
    twr_annual = ((1.0 + twr) ** (1.0 / years) - 1.0) if is_num(twr) and twr > -1 else NAN
    mwr = money_weighted_return(values, flows)

    bench_return = NAN
    vs_bench: dict = {}
    bench = fetch_history(benchmark, period=f"{int(np.ceil(years)) + 1}y")
    if not bench.empty:
        bench_px = bench["adj_close"].reindex(
            bench.index.union(values.index)).ffill().reindex(values.index).dropna()
        if len(bench_px) > 2:
            bench_return = float(bench_px.iloc[-1] / bench_px.iloc[0] - 1.0)
            port_returns = to_returns(curve)
            bench_returns = to_returns(bench_px)
            vs_bench = beta_alpha(port_returns, bench_returns, risk_free)

    risk = return_stats(to_returns(curve), risk_free=risk_free)

    if is_num(twr) and is_num(mwr) and abs(mwr - (twr_annual or 0)) > 0.02:
        notes.append(
            f"Money-weighted return ({mwr:.1%}) differs from time-weighted "
            f"({twr_annual:.1%}) by {abs(mwr - twr_annual):.1%} a year. The gap "
            "is the effect of contribution timing, not of the holdings."
        )
    if len(values) < 90:
        notes.append("Under three months of history: annualised figures and risk "
                     "statistics are not meaningful.")
    return PerformanceReport(
        start=start, end=end,
        beginning_value=float(values.iloc[0]), ending_value=float(values.iloc[-1]),
        net_contributions=net_contributions,
        gain=float(values.iloc[-1] - values.iloc[0] - net_contributions),
        twr=twr, twr_annualized=twr_annual, mwr=mwr,
        benchmark_return=bench_return, benchmark_symbol=benchmark,
        equity_curve=curve, risk=risk, vs_benchmark=vs_bench, notes=notes,
    )


def contribution_by_position(state: LedgerState, prices: dict[str, float],
                             transactions: list[Transaction] | None = None
                             ) -> pd.DataFrame:
    """Realised plus unrealised profit per symbol, and share of the total.

    Shows where returns actually came from. Most portfolios find that a handful
    of positions produced nearly all of the gain, which is the empirical norm
    (Bessembinder 2018: most individual stocks underperform Treasury bills over
    their lifetime; index returns come from a small minority).
    """
    rows: dict[str, dict] = {}
    # Dividends are only attributable to a symbol when they were recorded
    # against one, so they come from the transaction log rather than the
    # per-account income totals the ledger keeps.
    for txn in (transactions or []):
        if txn.type in (TxnType.DIVIDEND, TxnType.DIVIDEND_REINVEST) and txn.symbol:
            row = rows.setdefault(txn.symbol, {"realized": 0.0, "unrealized": 0.0,
                                               "dividends": 0.0, "cost_basis": 0.0})
            amount = (abs(txn.amount) if txn.amount
                      else abs(txn.quantity * txn.price))
            row["dividends"] += amount

    for gain in state.realized:
        row = rows.setdefault(gain.symbol, {"realized": 0.0, "unrealized": 0.0,
                                            "dividends": 0.0, "cost_basis": 0.0})
        row["realized"] += gain.gain

    for position in state.positions(prices=prices):
        row = rows.setdefault(position.symbol, {"realized": 0.0, "unrealized": 0.0,
                                                "dividends": 0.0, "cost_basis": 0.0})
        row["unrealized"] += position.unrealized_gain if is_num(position.unrealized_gain) else 0.0
        row["cost_basis"] += position.cost_basis
        row["market_value"] = position.market_value

    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).T
    frame["total_gain"] = (frame["realized"].fillna(0.0)
                           + frame["unrealized"].fillna(0.0)
                           + frame["dividends"].fillna(0.0))
    total = frame["total_gain"].sum()
    frame["share_of_gain"] = frame["total_gain"] / total if total else NAN
    frame["return_on_cost"] = frame["total_gain"] / frame["cost_basis"].replace(0.0, np.nan)
    return frame.sort_values("total_gain", ascending=False)
