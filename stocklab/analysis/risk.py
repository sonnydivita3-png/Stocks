"""Portfolio and single-name risk: dispersion, drawdown, tail loss, beta.

Risk is measured in several incompatible ways on purpose. Volatility treats a
40% gain and a 40% loss as equally bad; drawdown describes what an investor
actually lived through; CVaR describes the tail that ruins a leveraged position.
Reporting one number would hide more than it reveals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import settings
from ..util.mathx import (
    NAN, annualize_return, is_num, max_drawdown, safe_div, ulcer_index,
)


def to_returns(prices: pd.Series) -> pd.Series:
    return pd.to_numeric(prices, errors="coerce").pct_change().dropna()


def return_stats(returns: pd.Series, risk_free: float = 0.0,
                 periods_per_year: int | None = None) -> dict[str, float]:
    """The standard performance/risk panel for a return series."""
    cfg = settings()
    ppy = periods_per_year or cfg.trading_days
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 2:
        return {}

    total = float((1.0 + r).prod() - 1.0)
    years = len(r) / ppy
    ann_return = annualize_return(total, years)
    vol = float(r.std(ddof=1) * np.sqrt(ppy))

    # Downside deviation uses only sub-target returns: the denominator of
    # Sortino, which is the honest version of Sharpe for asymmetric strategies.
    target = risk_free / ppy
    downside = r[r < target] - target
    downside_dev = float(np.sqrt((downside**2).sum() / len(r)) * np.sqrt(ppy)) if len(downside) else 0.0

    equity = (1.0 + r).cumprod()
    mdd = max_drawdown(equity)

    excess_ann = ann_return - risk_free if is_num(ann_return) else NAN
    sorted_r = np.sort(r.to_numpy(dtype=float))
    tail_n = max(1, int(len(sorted_r) * (1.0 - cfg.var_confidence)))

    stats = {
        "total_return": total,
        "cagr": ann_return,
        "volatility": vol,
        "downside_deviation": downside_dev,
        "sharpe": safe_div(excess_ann, vol),
        "sortino": safe_div(excess_ann, downside_dev),
        "calmar": safe_div(excess_ann, abs(mdd)) if is_num(mdd) and mdd < 0 else NAN,
        "max_drawdown": mdd,
        "ulcer_index": ulcer_index(equity),
        "skew": float(r.skew()),
        "excess_kurtosis": float(r.kurtosis()),
        "best_period": float(r.max()),
        "worst_period": float(r.min()),
        "positive_rate": float((r > 0).mean()),
        "observations": float(len(r)),
        "years": years,
        # Historical VaR/CVaR: no normality assumption, which matters because
        # equity returns have fat tails and a Gaussian VaR understates the tail
        # by roughly a third at the 95% level.
        "var_hist": float(-sorted_r[tail_n - 1]),
        "cvar_hist": float(-sorted_r[:tail_n].mean()),
        "var_normal": float(-(r.mean() + r.std(ddof=1) * _z(cfg.var_confidence))),
    }
    stats["recovery_factor"] = safe_div(total, abs(mdd)) if is_num(mdd) and mdd < 0 else NAN
    # Share of the period spent below a previous peak. Often more
    # decision-relevant than the depth: a 25% drawdown that recovers in four
    # months is tolerable, the same 25% taking five years is what makes people
    # abandon a strategy at the worst moment.
    stats["time_underwater"] = float((equity < equity.cummax() * (1 - 1e-9)).mean())
    return stats


def _z(confidence: float) -> float:
    """Inverse standard normal CDF at the lower tail of ``confidence``."""
    try:
        from scipy.stats import norm
        return float(norm.ppf(1.0 - confidence))
    except ImportError:  # keep risk reporting available without scipy
        return {0.90: -1.2816, 0.95: -1.6449, 0.99: -2.3263}.get(round(confidence, 2), -1.6449)


def beta_alpha(returns: pd.Series, benchmark: pd.Series,
               risk_free: float = 0.0, periods_per_year: int | None = None) -> dict[str, float]:
    """Market exposure via OLS, split into up-market and down-market beta.

    The split matters: two stocks with beta 1.0 are completely different
    investments if one has up-beta 1.3 / down-beta 0.7 and the other the
    reverse. A single beta averages that distinction away.
    """
    ppy = periods_per_year or settings().trading_days
    joined = pd.concat([returns.rename("r"), benchmark.rename("b")], axis=1).dropna()
    if len(joined) < 30:
        return {}
    rf_period = risk_free / ppy
    y = joined["r"].to_numpy(dtype=float) - rf_period
    x = joined["b"].to_numpy(dtype=float) - rf_period

    var_x = x.var()
    if var_x == 0:
        return {}
    beta = float(np.cov(x, y, bias=True)[0, 1] / var_x)
    alpha_period = float(y.mean() - beta * x.mean())
    resid = y - (beta * x + alpha_period)
    ss_tot = float(((y - y.mean()) ** 2).sum())

    out = {
        "beta": beta,
        "alpha_annual": float((1.0 + alpha_period) ** ppy - 1.0),
        "r_squared": 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else NAN,
        "tracking_error": float(np.std(joined["r"] - joined["b"], ddof=1) * np.sqrt(ppy)),
        "correlation": float(np.corrcoef(x, y)[0, 1]),
        "observations": float(len(joined)),
    }
    out["information_ratio"] = safe_div(
        float((joined["r"] - joined["b"]).mean() * ppy), out["tracking_error"]
    )

    for label, mask in (("up", x > 0), ("down", x < 0)):
        if mask.sum() >= 15 and x[mask].var() > 0:
            out[f"beta_{label}"] = float(np.cov(x[mask], y[mask], bias=True)[0, 1] / x[mask].var())
    if "beta_up" in out and "beta_down" in out:
        # > 1 means it captures more upside than downside: convexity worth paying for.
        out["beta_asymmetry"] = safe_div(out["beta_up"], out["beta_down"])
    return out


def rolling_beta(returns: pd.Series, benchmark: pd.Series, window: int = 126) -> pd.Series:
    joined = pd.concat([returns.rename("r"), benchmark.rename("b")], axis=1).dropna()
    if len(joined) < window:
        return pd.Series(dtype=float)
    cov = joined["r"].rolling(window).cov(joined["b"])
    var = joined["b"].rolling(window).var()
    return (cov / var.replace(0.0, np.nan)).dropna()


def correlation_matrix(prices: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    returns = prices.pct_change().dropna(how="all")
    return returns.corr(method=method)


def diversification_ratio(weights: pd.Series, cov: pd.DataFrame) -> float:
    """Weighted average volatility divided by portfolio volatility.

    1.0 means the holdings move as one and the diversification is cosmetic;
    above ~1.4 means the positions are genuinely doing different things.
    """
    common = weights.index.intersection(cov.index)
    if len(common) < 2:
        return NAN
    w = weights.loc[common].to_numpy(dtype=float)
    c = cov.loc[common, common].to_numpy(dtype=float)
    port_vol = float(np.sqrt(w @ c @ w))
    weighted_vol = float((np.abs(w) * np.sqrt(np.diag(c))).sum())
    return safe_div(weighted_vol, port_vol)


def effective_holdings(weights: pd.Series) -> float:
    """Inverse Herfindahl: how many positions the portfolio *behaves* like.

    Twenty names with one at 40% behaves like roughly six. This is the number
    to quote when someone says they are diversified across twenty stocks.
    """
    w = pd.to_numeric(weights, errors="coerce").dropna().abs()
    total = float(w.sum())
    if total <= 0:
        return NAN
    shares = w / total
    return float(1.0 / (shares**2).sum())


# Peak-to-trough index moves, used to stress a portfolio by beta. Approximate
# by design -- the purpose is "roughly how much would this have hurt", and
# false precision would imply the next crisis will rhyme exactly.
STRESS_SCENARIOS: dict[str, float] = {
    "1987 Black Monday (1 day)": -0.204,
    "1990 Gulf War / recession": -0.197,
    "2000-02 dot-com unwind": -0.491,
    "2008-09 global financial crisis": -0.565,
    "2011 US downgrade / euro crisis": -0.194,
    "2018 Q4 repricing": -0.198,
    "2020 COVID crash (33 days)": -0.339,
    "2022 rate shock": -0.254,
    "Rates +200bp (duration proxy)": -0.120,
}


def stress_test(portfolio_value: float, beta: float,
                scenarios: dict[str, float] | None = None) -> pd.DataFrame:
    """Beta-scaled loss estimates under historical index drawdowns.

    A first-order approximation only. It assumes beta is stable in a crisis,
    which it is not -- correlations converge toward one and realised losses
    typically exceed the beta estimate. Treat these as optimistic.
    """
    scenarios = scenarios or STRESS_SCENARIOS
    beta = beta if is_num(beta) else 1.0
    rows = []
    for name, index_move in scenarios.items():
        move = index_move * beta
        rows.append({
            "scenario": name,
            "index_move": index_move,
            "portfolio_move": move,
            "value_change": portfolio_value * move,
            "resulting_value": portfolio_value * (1.0 + move),
        })
    return pd.DataFrame(rows)


def kelly_fraction(win_rate: float, win_loss_ratio: float, cap: float = 0.25) -> float:
    """Kelly stake for a binary bet, capped hard.

    Full Kelly maximises long-run log wealth but produces drawdowns no human
    tolerates, and it is acutely sensitive to the win-rate estimate -- a 5pp
    error in a 55% win rate moves the stake by half. Practitioners use a
    fraction of Kelly; the cap here defaults to 25% of capital.
    """
    if not (is_num(win_rate) and is_num(win_loss_ratio)) or win_loss_ratio <= 0:
        return NAN
    win_rate = min(max(win_rate, 0.0), 1.0)
    edge = win_rate - (1.0 - win_rate) / win_loss_ratio
    return float(min(max(edge, 0.0), cap))


def vol_target_weight(asset_vol: float, target_vol: float = 0.15,
                      max_weight: float = 1.0) -> float:
    """Weight that scales a position to a volatility budget."""
    if not is_num(asset_vol) or asset_vol <= 0:
        return NAN
    return float(min(target_vol / asset_vol, max_weight))
