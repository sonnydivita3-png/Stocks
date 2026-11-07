"""Portfolio-level risk: concentration, exposure, correlation, stress.

The question this answers is not "what is my volatility" but "what am I
actually betting on, and what breaks me". Those are different, and the second
is almost always a concentration or correlation problem rather than a
volatility one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..analysis.risk import (
    beta_alpha, correlation_matrix, diversification_ratio,
    effective_holdings, return_stats, stress_test, to_returns,
)
from ..data.providers.yahoo import fetch_closes, fetch_history
from ..util.mathx import NAN, is_num, safe_div
from .ledger import LedgerState
from .models import Account


@dataclass
class RiskReport:
    total_value: float
    positions: pd.DataFrame
    concentration: dict
    sector_exposure: pd.Series
    account_exposure: pd.Series
    correlation: pd.DataFrame
    portfolio_beta: float
    portfolio_vol: float
    risk_stats: dict
    stress: pd.DataFrame
    notes: list[str] = field(default_factory=list)


def build(state: LedgerState, accounts: dict[int, Account],
          prices: dict[str, float], sectors: dict[str, str] | None = None,
          *, benchmark: str = "SPY", period: str = "2y",
          risk_free: float = 0.0) -> RiskReport:
    """Assemble a full risk report for a replayed ledger."""
    sectors = sectors or {}
    notes: list[str] = []

    positions = state.positions(prices=prices)
    if not positions:
        return RiskReport(0.0, pd.DataFrame(), {}, pd.Series(dtype=float),
                          pd.Series(dtype=float), pd.DataFrame(), NAN, NAN, {},
                          pd.DataFrame(), ["No open positions."])

    cash = state.total_cash()
    rows = []
    for position in positions:
        rows.append({
            "symbol": position.symbol,
            "quantity": position.quantity,
            "price": position.price,
            "market_value": position.market_value,
            "cost_basis": position.cost_basis,
            "unrealized_gain": position.unrealized_gain,
            "unrealized_pct": position.unrealized_pct,
            "sector": sectors.get(position.symbol, "Unknown"),
            "long_term_shares": position.long_term_quantity,
            "short_term_shares": position.short_term_quantity,
        })
    frame = pd.DataFrame(rows).set_index("symbol")
    invested = float(frame["market_value"].sum(skipna=True))
    total_value = invested + max(cash, 0.0)
    frame["weight"] = frame["market_value"] / total_value if total_value else NAN
    frame = frame.sort_values("market_value", ascending=False)

    # ---- concentration ----------------------------------------------------
    weights = frame["weight"].dropna()
    hhi = float((weights**2).sum())
    concentration = {
        "positions": int(len(frame)),
        "cash_weight": safe_div(max(cash, 0.0), total_value),
        "largest_weight": float(weights.max()) if not weights.empty else NAN,
        "top_3_weight": float(weights.head(3).sum()),
        "top_5_weight": float(weights.head(5).sum()),
        "top_10_weight": float(weights.head(10).sum()),
        "herfindahl": hhi,
        # The number of positions the portfolio *behaves* like. Twenty names
        # with one at 40% behaves like about six.
        "effective_holdings": effective_holdings(weights),
    }
    if is_num(concentration["largest_weight"]) and concentration["largest_weight"] > 0.20:
        notes.append(
            f"Largest position is {concentration['largest_weight']:.0%} of the "
            "portfolio. A single-name accounting fraud or failed trial is a "
            "permanent loss, not a drawdown you wait out."
        )
    if is_num(concentration["effective_holdings"]) and len(frame) >= 10:
        if concentration["effective_holdings"] < len(frame) * 0.5:
            notes.append(
                f"{len(frame)} holdings but an effective count of "
                f"{concentration['effective_holdings']:.1f}: the portfolio is far "
                "more concentrated than the position count suggests."
            )

    # ---- exposures --------------------------------------------------------
    sector_exposure = frame.groupby("sector")["weight"].sum().sort_values(ascending=False)
    if not sector_exposure.empty and sector_exposure.iloc[0] > 0.40:
        notes.append(
            f"{sector_exposure.index[0]} is {sector_exposure.iloc[0]:.0%} of the "
            "portfolio. Sector risk is the dominant exposure, not stock selection."
        )

    account_rows = []
    for position in state.positions(prices=prices, by_account=True):
        account = accounts.get(position.account_id) if position.account_id else None
        account_rows.append({
            "account": account.name if account else "unknown",
            "value": position.market_value if is_num(position.market_value) else 0.0,
        })
    account_exposure = (pd.DataFrame(account_rows).groupby("account")["value"].sum()
                        / total_value if account_rows else pd.Series(dtype=float))

    # ---- correlation and market risk -------------------------------------
    symbols = list(frame.index)
    closes = fetch_closes(symbols, period=period, field="adj_close")
    correlation = pd.DataFrame()
    portfolio_beta = portfolio_vol = NAN
    risk_stats: dict = {}
    if not closes.empty and closes.shape[1] >= 1:
        correlation = correlation_matrix(closes)
        returns = closes.pct_change().dropna(how="all")
        available = [s for s in symbols if s in returns.columns]
        if available:
            w = frame.loc[available, "weight"].fillna(0.0)
            w = w / w.sum() if w.sum() > 0 else w
            portfolio_returns = (returns[available] * w).sum(axis=1)
            risk_stats = return_stats(portfolio_returns, risk_free=risk_free)
            portfolio_vol = risk_stats.get("volatility", NAN)

            bench = fetch_history(benchmark, period=period)
            if not bench.empty:
                stats = beta_alpha(portfolio_returns,
                                   to_returns(bench["adj_close"]), risk_free)
                portfolio_beta = stats.get("beta", NAN)
                risk_stats.update({f"vs_{benchmark}_{k}": v for k, v in stats.items()})

            if len(available) > 1:
                pairwise = correlation.loc[available, available]
                upper = pairwise.where(np.triu(np.ones(pairwise.shape), k=1).astype(bool))
                avg_corr = float(upper.stack().mean())
                risk_stats["average_pairwise_correlation"] = avg_corr
                risk_stats["diversification_ratio"] = diversification_ratio(
                    w, returns[available].cov() * 252
                )
                if avg_corr > 0.70:
                    notes.append(
                        f"Average pairwise correlation is {avg_corr:.2f}. These "
                        "positions are largely one bet; the position count is "
                        "not buying diversification."
                    )

    stress = stress_test(total_value, portfolio_beta)
    notes.append(
        "Stress figures scale index drawdowns by beta, which understates the "
        "damage: correlations converge toward one in a crisis and realised "
        "losses typically exceed the beta estimate."
    )
    return RiskReport(
        total_value=total_value, positions=frame, concentration=concentration,
        sector_exposure=sector_exposure, account_exposure=account_exposure,
        correlation=correlation, portfolio_beta=portfolio_beta,
        portfolio_vol=portfolio_vol, risk_stats=risk_stats, stress=stress,
        notes=notes,
    )


def factor_tilt(frame: pd.DataFrame, metrics_by_symbol: dict[str, dict],
                factors: tuple[str, ...] = (
                    "pe", "pb", "roic", "revenue_cagr_5y", "momentum_12_1",
                    "fcf_yield", "net_debt_to_ebitda", "vol63")
                ) -> pd.Series:
    """Value-weighted average of each factor across the portfolio.

    Reveals the implicit bet. A portfolio of individually sensible decisions
    frequently turns out to be a single factor position -- expensive growth, or
    levered value -- which only shows up when the exposures are aggregated.
    """
    if frame.empty:
        return pd.Series(dtype=float)
    weights = frame["weight"].fillna(0.0)
    out: dict[str, float] = {}
    for factor in factors:
        numerator = denominator = 0.0
        for symbol, weight in weights.items():
            value = (metrics_by_symbol.get(symbol) or {}).get(factor)
            if is_num(value) and weight > 0:
                numerator += float(value) * weight
                denominator += weight
        if denominator > 0:
            out[factor] = numerator / denominator
            out[f"{factor}_coverage"] = denominator / float(weights.sum())
    return pd.Series(out, dtype=float)
