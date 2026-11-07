"""Performance statistics for a return series, and honest comparisons.

The summary deliberately leads with drawdown and the Sharpe/Sortino pair rather
than with CAGR. A strategy's CAGR is the number that sells it; its maximum
drawdown is the number that determines whether anyone actually held it long
enough to collect that CAGR.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..analysis.risk import return_stats
from ..util.mathx import NAN, is_num, safe_div


def performance_summary(returns: pd.Series, equity: pd.Series | None = None,
                        risk_free: float = 0.0) -> dict:
    """Full statistics panel for a periodic return series."""
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if len(returns) < 2:
        return {}
    if equity is None:
        equity = (1.0 + returns).cumprod()

    stats = return_stats(returns, risk_free=risk_free)
    stats["drawdown_duration_days"] = longest_drawdown(equity)
    stats["time_underwater"] = time_underwater(equity)
    stats["gain_to_pain"] = gain_to_pain(returns)
    stats["tail_ratio"] = tail_ratio(returns)
    stats["best_year"], stats["worst_year"] = best_worst_year(returns)
    return stats


def longest_drawdown(equity: pd.Series) -> float:
    """Longest run of days spent below a previous peak.

    Usually more decision-relevant than the depth. A 25% drawdown that recovers
    in four months is tolerable; the same 25% taking five years to recover is
    what makes people abandon a strategy at the worst moment.
    """
    equity = pd.to_numeric(equity, errors="coerce").dropna()
    if equity.empty:
        return NAN
    peak = equity.cummax()
    underwater = equity < peak * (1 - 1e-9)
    longest = current = 0
    for flag in underwater:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return float(longest)


def time_underwater(equity: pd.Series) -> float:
    equity = pd.to_numeric(equity, errors="coerce").dropna()
    if equity.empty:
        return NAN
    return float((equity < equity.cummax() * (1 - 1e-9)).mean())


def gain_to_pain(returns: pd.Series) -> float:
    """Sum of returns divided by the sum of losses.

    Harder to game than Sharpe, because it does not reward low volatility
    achieved by sitting in cash.
    """
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    losses = returns[returns < 0].abs().sum()
    return safe_div(float(returns.sum()), float(losses))


def tail_ratio(returns: pd.Series, quantile: float = 0.05) -> float:
    """Size of the right tail against the left.

    Above 1 means the big up moves are larger than the big down moves. Most
    equity strategies sit below 1, and option-selling strategies sit well
    below it while showing excellent Sharpe ratios -- which is precisely the
    distinction Sharpe cannot see.
    """
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if len(returns) < 40:
        return NAN
    right = abs(float(returns.quantile(1 - quantile)))
    left = abs(float(returns.quantile(quantile)))
    return safe_div(right, left)


def best_worst_year(returns: pd.Series) -> tuple[float, float]:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if returns.empty or not isinstance(returns.index, pd.DatetimeIndex):
        return NAN, NAN
    yearly = (1.0 + returns).groupby(returns.index.year).prod() - 1.0
    if yearly.empty:
        return NAN, NAN
    return float(yearly.max()), float(yearly.min())


def annual_table(returns: pd.Series, benchmark: pd.Series | None = None) -> pd.DataFrame:
    """Calendar-year returns, which is how anyone actually experiences a strategy."""
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if returns.empty or not isinstance(returns.index, pd.DatetimeIndex):
        return pd.DataFrame()
    rows = {"strategy": (1.0 + returns).groupby(returns.index.year).prod() - 1.0}
    if benchmark is not None and not benchmark.empty:
        bench = pd.to_numeric(benchmark, errors="coerce").dropna()
        rows["benchmark"] = (1.0 + bench).groupby(bench.index.year).prod() - 1.0
    frame = pd.DataFrame(rows)
    if "benchmark" in frame:
        frame["excess"] = frame["strategy"] - frame["benchmark"]
    return frame


def compare(results: dict[str, "object"]) -> pd.DataFrame:
    """Side-by-side table of several backtests.

    Ordered by Calmar rather than CAGR: return per unit of worst-case pain is
    the comparison that survives contact with an actual investor.
    """
    rows = {}
    for name, result in results.items():
        summary = getattr(result, "summary", None) or {}
        rows[name] = {
            "cagr": summary.get("cagr"),
            "volatility": summary.get("volatility"),
            "sharpe": summary.get("sharpe"),
            "sortino": summary.get("sortino"),
            "max_drawdown": summary.get("max_drawdown"),
            "calmar": summary.get("calmar"),
            "ulcer": summary.get("ulcer_index"),
            "time_underwater": summary.get("time_underwater"),
            "annual_turnover": summary.get("annual_turnover"),
            "gross_cagr": summary.get("gross_cagr"),
        }
    frame = pd.DataFrame(rows).T
    if "calmar" in frame:
        frame = frame.sort_values("calmar", ascending=False)
    return frame


def deflated_sharpe(sharpe: float, trials: int, observations: int) -> float:
    """Sharpe ratio adjusted for the number of strategies tried.

    If you test 50 variants and keep the best, its Sharpe is biased upward by
    selection alone -- with 50 trials on 5 years of data, an expected maximum
    Sharpe near 1.0 arises from pure noise. Bailey & Lopez de Prado's
    correction subtracts that expectation. Any backtest Sharpe quoted without
    disclosing the number of trials behind it is close to meaningless.
    """
    if not (is_num(sharpe) and trials >= 1 and observations > 2):
        return NAN
    if trials == 1:
        return float(sharpe)
    euler = 0.5772156649
    # Expected maximum of `trials` draws from a standard normal.
    expected_max = ((1 - euler) * _norm_ppf(1 - 1.0 / trials)
                    + euler * _norm_ppf(1 - 1.0 / (trials * np.e)))
    return float(sharpe - expected_max / np.sqrt(observations))


def _norm_ppf(p: float) -> float:
    try:
        from scipy.stats import norm
        return float(norm.ppf(p))
    except ImportError:
        # Acklam-style rational approximation, adequate for this correction.
        return float(np.sqrt(2) * _erfinv(2 * p - 1))


def _erfinv(x: float) -> float:
    a = 0.147
    ln = np.log(1 - x * x)
    first = 2 / (np.pi * a) + ln / 2
    return float(np.sign(x) * np.sqrt(np.sqrt(first * first - ln / a) - first))
