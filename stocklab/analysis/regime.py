"""Market regime: what environment are we actually in, and how much to risk.

The honest case for this module: nobody times markets reliably, and a regime
model that moves you to cash on every wobble will underperform buy-and-hold
after taxes and costs.  What regime signals *do* help with is sizing -- credit
spreads and index trend have real, documented predictive content for drawdown
risk, so they belong in a position-sizing decision rather than an in/out one.

Hence the output: an exposure *multiplier* bounded well away from zero, not a
signal to liquidate.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..data.providers import fred
from ..data.providers.yahoo import fetch_closes, fetch_history
from ..data.universe import BENCHMARKS, SECTOR_ETFS
from ..util.mathx import NAN, clamp, interp_score, is_num, safe_div
from .technicals import sma

log = logging.getLogger(__name__)


def trend_state(symbol: str = "SPY", period: str = "3y") -> dict[str, float]:
    """Index trend: price vs its 200-day line, slope, and distance from highs."""
    hist = fetch_history(symbol, period=period)
    if hist.empty or len(hist) < 220:
        return {}
    close = hist["close"]
    ma50, ma200 = sma(close, 50), sma(close, 200)
    px = hist["adj_close"]
    last = float(close.iloc[-1])

    out = {
        "price": last,
        "above_sma200": float(last > float(ma200.iloc[-1])),
        "above_sma50": float(last > float(ma50.iloc[-1])),
        "sma50_above_sma200": float(float(ma50.iloc[-1]) > float(ma200.iloc[-1])),
        "dist_sma200": safe_div(last - float(ma200.iloc[-1]), float(ma200.iloc[-1])),
        "sma200_slope_3m": safe_div(
            float(ma200.iloc[-1]) - float(ma200.iloc[-63]), float(ma200.iloc[-63])
        ),
        "drawdown_from_high": safe_div(last, float(close.cummax().iloc[-1])) - 1.0,
        "return_12m": safe_div(float(px.iloc[-1]), float(px.iloc[-253])) - 1.0
        if len(px) > 253 else NAN,
        "realized_vol_21d": float(px.pct_change().tail(21).std(ddof=0) * np.sqrt(252)),
        "realized_vol_252d": float(px.pct_change().tail(252).std(ddof=0) * np.sqrt(252)),
    }
    # Days spent above the 200-day over the past year: a trend-persistence read
    # that distinguishes a durable uptrend from a whipsawing one.
    above = (close > ma200).tail(252)
    out["pct_days_above_sma200_1y"] = float(above.mean())
    return out


def breadth(symbols: list[str] | None = None, period: str = "2y",
            max_symbols: int = 120) -> dict[str, float]:
    """Share of a universe above its own 50/200-day averages, and new-high/low spread.

    Breadth divergence -- an index at highs carried by a handful of names -- is
    one of the few genuinely useful warnings available, because a cap-weighted
    index can rise while the median stock falls.
    """
    if symbols is None:
        # Sector ETFs alone give only 11 observations, which makes every breadth
        # ratio jump in 9-point steps. Large-cap constituents are added so the
        # percentages mean something.
        from ..data.universe import MEGA_CAP_SAMPLE
        symbols = (
            MEGA_CAP_SAMPLE
            + list(SECTOR_ETFS.values())
            + [BENCHMARKS[k] for k in ("sp500", "equal_weight_sp500", "small_cap", "mid_cap")]
        )
    symbols = symbols[:max_symbols]
    closes = fetch_closes(symbols, period=period, field="close")
    if closes.empty or len(closes) < 210:
        return {}

    above50 = above200 = counted = 0
    new_highs = new_lows = 0
    for col in closes.columns:
        series = closes[col].dropna()
        if len(series) < 210:
            continue
        counted += 1
        last = float(series.iloc[-1])
        above50 += last > float(series.tail(50).mean())
        above200 += last > float(series.tail(200).mean())
        window = series.tail(252)
        new_highs += last >= float(window.max()) * 0.995
        new_lows += last <= float(window.min()) * 1.005
    if counted == 0:
        return {}

    return {
        "pct_above_sma50": above50 / counted,
        "pct_above_sma200": above200 / counted,
        "pct_at_52w_high": new_highs / counted,
        "pct_at_52w_low": new_lows / counted,
        "net_new_highs": (new_highs - new_lows) / counted,
        "constituents": float(counted),
    }


def equal_weight_divergence(period: str = "1y") -> float:
    """RSP minus SPY over the window: positive means breadth is participating.

    A deeply negative reading says the index is being carried by its largest
    constituents, which historically precedes higher drawdown risk because the
    index becomes a concentrated bet without the holder choosing one.
    """
    closes = fetch_closes([BENCHMARKS["equal_weight_sp500"], BENCHMARKS["sp500"]],
                          period=period, field="adj_close")
    if closes.empty or closes.shape[1] < 2 or len(closes) < 60:
        return NAN
    first, last = closes.iloc[0], closes.iloc[-1]
    rsp = safe_div(float(last.iloc[0]), float(first.iloc[0])) - 1.0
    spy = safe_div(float(last.iloc[1]), float(first.iloc[1])) - 1.0
    return rsp - spy


def credit_state() -> dict[str, float]:
    """High-yield and investment-grade spreads, levels and 3-month changes.

    Credit leads equity. A high-yield OAS widening through 500bp has preceded
    every major equity drawdown of the modern era, usually by weeks.
    """
    out: dict[str, float] = {}
    for key, name in (("hy_oas", "hy_oas"), ("ig_oas", "ig_oas")):
        series = fred.get(name)
        if series.empty:
            continue
        out[key] = float(series.iloc[-1])
        if len(series) > 63:
            out[f"{key}_chg_3m"] = float(series.iloc[-1] - series.iloc[-63])
        if len(series) > 252:
            # Percentile against the last five years: 400bp means different
            # things in 2021 and in 2009.
            window = series.tail(252 * 5)
            out[f"{key}_pctile_5y"] = float((window <= series.iloc[-1]).mean())
    return out


def rates_state() -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("ust_3m", "ust_2y", "ust_10y", "curve_10y2y", "curve_10y3m",
                "fed_funds", "breakeven_10y", "vix", "unemployment", "sahm_rule"):
        value = fred.latest(key)
        if value is not None:
            out[key] = float(value)
    for key in ("curve_10y3m", "unemployment"):
        series = fred.get(key)
        if len(series) > 12:
            out[f"{key}_chg_12m"] = float(series.iloc[-1] - series.iloc[-252 if key.startswith("curve") else -12])
    return out


def assess(benchmark: str = "SPY") -> dict:
    """Composite regime read: 0-100 score, a label, and an exposure multiplier.

    Components are scored independently and blended with fixed weights. Trend
    and credit carry the most weight because they have the strongest evidence
    behind them; the yield curve carries little despite its reputation, because
    its lead time is 12-24 months, which is useless for sizing a position today.
    """
    trend = trend_state(benchmark)
    br = breadth()
    credit = credit_state()
    rates = rates_state()
    ew_div = equal_weight_divergence()

    scores: dict[str, float] = {}

    # --- trend (weight 0.32) ---
    if trend:
        # Graded, not binary. A binary "above the 200-day" test scores 100
        # whether price is 0.4% or 15% above it, which produces a cliff right
        # where the signal is least reliable -- the flat, chopping tape that
        # straddles the average is precisely the one that whipsaws.
        trend_pts = [
            interp_score(trend.get("dist_sma200", NAN),
                         [(-0.12, 0), (-0.03, 25), (0.0, 45), (0.03, 65),
                          (0.08, 88), (0.15, 100)]),
            100.0 * trend.get("sma50_above_sma200", 0.0),
            interp_score(trend.get("sma200_slope_3m", NAN), [(-0.05, 0), (0.0, 50), (0.04, 100)]),
            interp_score(trend.get("drawdown_from_high", NAN),
                         [(-0.25, 0), (-0.10, 40), (-0.03, 85), (0.0, 100)]),
            # Trend persistence: how much of the last year was spent in uptrend.
            interp_score(trend.get("pct_days_above_sma200_1y", NAN),
                         [(0.3, 15), (0.6, 50), (0.9, 85), (1.0, 100)]),
        ]
        valid = [p for p in trend_pts if is_num(p)]
        if valid:
            scores["trend"] = float(np.mean(valid))

    # --- breadth (weight 0.20) ---
    if br:
        breadth_pts = [
            interp_score(br.get("pct_above_sma200", NAN), [(0.2, 0), (0.5, 50), (0.8, 100)]),
            interp_score(br.get("net_new_highs", NAN), [(-0.3, 0), (0.0, 50), (0.3, 100)]),
            interp_score(ew_div, [(-0.15, 10), (-0.05, 40), (0.0, 60), (0.05, 85)]),
        ]
        valid = [p for p in breadth_pts if is_num(p)]
        if valid:
            scores["breadth"] = float(np.mean(valid))

    # --- credit (weight 0.28) ---
    if credit:
        credit_pts = [
            interp_score(credit.get("hy_oas", NAN), [(2.5, 100), (3.5, 75), (5.0, 40), (7.0, 10), (10.0, 0)]),
            interp_score(credit.get("hy_oas_chg_3m", NAN), [(-0.5, 100), (0.0, 70), (0.75, 30), (2.0, 0)]),
            interp_score(credit.get("ig_oas", NAN), [(0.8, 100), (1.2, 70), (2.0, 30), (3.0, 0)]),
        ]
        valid = [p for p in credit_pts if is_num(p)]
        if valid:
            scores["credit"] = float(np.mean(valid))

    # --- volatility (weight 0.12) ---
    vix = rates.get("vix")
    if is_num(vix):
        scores["volatility"] = interp_score(vix, [(12, 100), (18, 75), (25, 45), (35, 15), (50, 0)])

    # --- macro (weight 0.08) ---
    macro_pts = [
        interp_score(rates.get("curve_10y3m", NAN), [(-1.0, 25), (0.0, 55), (1.0, 85), (2.5, 100)]),
        interp_score(rates.get("sahm_rule", NAN), [(0.0, 100), (0.35, 60), (0.5, 30), (1.0, 0)]),
    ]
    valid = [p for p in macro_pts if is_num(p)]
    if valid:
        scores["macro"] = float(np.mean(valid))

    weights = {"trend": 0.32, "credit": 0.28, "breadth": 0.20,
               "volatility": 0.12, "macro": 0.08}
    num = sum(scores[k] * weights[k] for k in scores)
    den = sum(weights[k] for k in scores)
    composite = num / den if den > 0 else NAN

    if not is_num(composite):
        label, exposure = "unknown", 1.0
    elif composite >= 75:
        # Above-normal exposure should require a clearly strong tape, not a
        # merely adequate one, so this band is set deliberately high.
        label, exposure = "risk-on", 1.10
    elif composite >= 60:
        label, exposure = "constructive", 1.00
    elif composite >= 45:
        label, exposure = "neutral", 0.90
    elif composite >= 30:
        label, exposure = "defensive", 0.75
    else:
        label, exposure = "risk-off", 0.60

    return {
        "composite_score": composite,
        "label": label,
        # Deliberately bounded: a regime model that can send exposure to zero
        # will, eventually, and whipsaw costs plus missed recoveries swamp the
        # drawdown it avoided. This scales risk, it does not switch it off.
        "exposure_multiplier": clamp(exposure, 0.60, 1.10),
        "components": scores,
        "weights": {k: weights[k] for k in scores},
        "trend": trend,
        "breadth": br,
        "credit": credit,
        "rates": rates,
        "equal_weight_divergence": ew_div,
        "caveats": [
            "Regime signals are for sizing, not for going to cash: the exposure "
            "multiplier is bounded at 0.60-1.10 by design.",
            "Credit spreads and index trend have the most evidence behind them; "
            "the yield curve's 12-24 month lead makes it near-useless for "
            "position sizing today, hence its 8% weight.",
            "Breadth here uses sector ETFs and index proxies, not full "
            "constituent data, so it is a coarse read.",
        ],
    }
