"""Position sizing: how much, not what.

Sizing decides more of a portfolio's outcome than selection does, and it is
where most individual investors are weakest -- typically holding a 30% position
in whatever worked and 1% positions in everything else, which is a bet on the
past rather than a plan.

Each method here encodes a different definition of "equal":

* ``equal_weight`` -- equal *capital*. Simple, and surprisingly hard to beat,
  but it gives a 70%-volatility biotech the same dollar exposure as a utility.
* ``inverse_volatility`` / ``vol_target`` -- equal *risk contribution in
  isolation*. Usually what people mean by equal weighting once they think
  about it.
* ``risk_parity`` -- equal *marginal* risk contribution, accounting for
  correlation. The only method that notices that three regional banks are one
  position.
* ``kelly`` -- maximises long-run log wealth. Mathematically optimal and
  practically unusable at full size; capped hard, and the reason is documented
  on the function.
* ``atr_risk`` -- sizes from the distance to a stop, so every position risks
  the same fraction of capital if stopped out. The trader's formulation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..util.mathx import NAN, clamp, is_num, safe_div


@dataclass
class SizingResult:
    weights: pd.Series
    method: str
    notes: list[str]

    def to_dollars(self, capital: float) -> pd.Series:
        return (self.weights * capital).round(2)

    def to_shares(self, capital: float, prices: pd.Series,
                  whole_shares: bool = True) -> pd.Series:
        dollars = self.weights * capital
        shares = dollars / prices.reindex(dollars.index)
        return shares.apply(np.floor) if whole_shares else shares


def equal_weight(symbols: list[str], max_weight: float = 1.0) -> SizingResult:
    if not symbols:
        return SizingResult(pd.Series(dtype=float), "equal_weight", ["No symbols."])
    weight = min(1.0 / len(symbols), max_weight)
    weights = pd.Series({s: weight for s in symbols}, dtype=float)
    notes = []
    if weights.sum() < 0.999:
        notes.append(f"Position cap leaves {1 - weights.sum():.1%} in cash.")
    return SizingResult(weights, "equal_weight", notes)


def inverse_volatility(vols: pd.Series, max_weight: float = 0.25) -> SizingResult:
    """Weights proportional to 1/volatility, capped and renormalised."""
    vols = pd.to_numeric(vols, errors="coerce").dropna()
    vols = vols[vols > 0]
    if vols.empty:
        return SizingResult(pd.Series(dtype=float), "inverse_volatility",
                            ["No usable volatility estimates."])
    raw = 1.0 / vols
    weights = raw / raw.sum()
    notes = []
    if (weights > max_weight).any():
        capped = weights.clip(upper=max_weight)
        notes.append(f"{int((weights > max_weight).sum())} position(s) capped at "
                     f"{max_weight:.0%}; the excess is redistributed.")
        weights = capped / capped.sum()
    return SizingResult(weights, "inverse_volatility", notes)


def vol_target(vols: pd.Series, target_vol: float = 0.15,
               max_weight: float = 0.25, max_gross: float = 1.0) -> SizingResult:
    """Size each position so its standalone volatility equals ``target_vol``.

    Unlike inverse-volatility weighting, this does not force full investment:
    if every candidate is more volatile than the target, the result is
    deliberately under-invested rather than levered to fill the gap.
    """
    vols = pd.to_numeric(vols, errors="coerce").dropna()
    vols = vols[vols > 0]
    if vols.empty:
        return SizingResult(pd.Series(dtype=float), "vol_target",
                            ["No usable volatility estimates."])
    weights = (target_vol / vols).clip(upper=max_weight)
    notes = []
    gross = float(weights.sum())
    if gross > max_gross:
        weights = weights * (max_gross / gross)
        notes.append(f"Gross exposure scaled from {gross:.0%} to {max_gross:.0%}.")
    elif gross < 0.95:
        notes.append(f"Only {gross:.0%} invested: holdings are volatile relative "
                     f"to the {target_vol:.0%} target. The remainder is cash.")
    return SizingResult(weights, "vol_target", notes)


def risk_parity(cov: pd.DataFrame, max_weight: float = 0.35,
                iterations: int = 500) -> SizingResult:
    """Equal marginal risk contribution, solved by fixed-point iteration.

    The only method here that sees correlation. Three regional banks at 10%
    each are, in risk terms, close to one 30% position; equal-weighting treats
    them as three independent bets and equal-volatility weighting does too.
    """
    cov = cov.dropna(how="all").dropna(axis=1, how="all")
    if cov.empty or cov.shape[0] != cov.shape[1] or cov.shape[0] < 2:
        return SizingResult(pd.Series(dtype=float), "risk_parity",
                            ["Need a square covariance matrix of 2+ assets."])
    symbols = list(cov.index)
    matrix = cov.to_numpy(dtype=float)
    n = len(symbols)
    weights = np.full(n, 1.0 / n)

    for _ in range(iterations):
        port_vol = float(np.sqrt(weights @ matrix @ weights))
        if port_vol <= 0:
            break
        marginal = matrix @ weights / port_vol
        contribution = weights * marginal
        target = port_vol / n
        # Nudge each weight toward the one that equalises its contribution.
        adjustment = np.divide(target, contribution,
                               out=np.ones_like(contribution), where=contribution > 0)
        updated = weights * np.power(adjustment, 0.1)
        updated = np.clip(updated, 1e-6, None)
        updated /= updated.sum()
        if np.max(np.abs(updated - weights)) < 1e-9:
            weights = updated
            break
        weights = updated

    result = pd.Series(weights, index=symbols)
    notes = []
    if (result > max_weight).any():
        notes.append(f"{int((result > max_weight).sum())} position(s) capped at "
                     f"{max_weight:.0%}.")
        result = result.clip(upper=max_weight)
        result /= result.sum()
    return SizingResult(result, "risk_parity", notes)


def kelly_weights(expected_returns: pd.Series, vols: pd.Series,
                  fraction: float = 0.25, max_weight: float = 0.20
                  ) -> SizingResult:
    """Fractional Kelly: ``f = mu / sigma^2``, scaled down hard.

    Full Kelly maximises the growth rate of log wealth, and is still the wrong
    size for a human to trade. Two reasons:

    1. It is extraordinarily sensitive to the expected-return estimate, which
       for equities is the least knowable quantity in finance. A 2-point error
       in ``mu`` moves the stake by a third.
    2. Full-Kelly drawdowns routinely reach 50%+ by design, since the criterion
       is indifferent to path.

    Quarter-Kelly captures roughly 90% of the growth rate at a fraction of the
    volatility, which is why practitioners use it.
    """
    returns = pd.to_numeric(expected_returns, errors="coerce")
    vols = pd.to_numeric(vols, errors="coerce")
    common = returns.index.intersection(vols.index)
    if len(common) == 0:
        return SizingResult(pd.Series(dtype=float), "kelly",
                            ["No overlapping return and volatility estimates."])
    variance = (vols.loc[common] ** 2).replace(0.0, np.nan)
    raw = (returns.loc[common] / variance).clip(lower=0.0) * fraction
    weights = raw.clip(upper=max_weight).dropna()
    notes = [
        f"Fractional Kelly at {fraction:.0%} of full, capped at {max_weight:.0%} "
        "per position. Full Kelly is correct in theory and untradeable in "
        "practice; it assumes the expected returns are known.",
    ]
    gross = float(weights.sum())
    if gross > 1.0:
        weights = weights / gross
        notes.append(f"Gross scaled from {gross:.0%} to 100% (no leverage).")
    elif gross < 0.95:
        notes.append(f"{1 - gross:.0%} left in cash: the edge does not justify "
                     "full investment on these inputs.")
    return SizingResult(weights, "kelly", notes)


def atr_position_size(capital: float, price: float, atr: float,
                      risk_per_trade: float = 0.01, atr_multiple: float = 2.0,
                      max_weight: float = 0.20) -> dict:
    """Size from the distance to the stop, so each trade risks the same amount.

    If the stop is ``atr_multiple`` ATRs below entry and you are willing to
    lose ``risk_per_trade`` of capital, the share count follows directly. This
    is the formulation that makes risk comparable: a 2-ATR stop on a volatile
    name produces a small position, on a quiet name a large one, and both risk
    1% of the account.
    """
    if not all(is_num(x) for x in (capital, price, atr)) or price <= 0 or atr <= 0:
        return {"shares": 0.0, "weight": NAN,
                "notes": ["Need positive capital, price and ATR."]}
    stop_distance = atr * atr_multiple
    risk_dollars = capital * risk_per_trade
    shares = np.floor(risk_dollars / stop_distance)
    value = shares * price
    weight = safe_div(value, capital)
    notes = []
    if is_num(weight) and weight > max_weight:
        shares = np.floor(capital * max_weight / price)
        value = shares * price
        weight = safe_div(value, capital)
        notes.append(
            f"Size reduced to the {max_weight:.0%} position cap. At a "
            f"{atr_multiple:g}-ATR stop the risk budget alone would have allowed "
            "a larger position, so the cap is binding -- this stop is tight "
            "relative to the name's volatility."
        )
    return {
        "shares": float(shares),
        "value": float(value),
        "weight": weight,
        "stop_price": float(price - stop_distance),
        "stop_distance": float(stop_distance),
        "risk_dollars": float(shares * stop_distance),
        "risk_pct_of_capital": safe_div(shares * stop_distance, capital),
        "notes": notes,
    }


def apply_regime(weights: pd.Series, exposure_multiplier: float) -> pd.Series:
    """Scale a weight vector by a regime exposure multiplier, leaving the rest in cash."""
    if not is_num(exposure_multiplier):
        return weights
    scaled = weights * clamp(exposure_multiplier, 0.0, 1.5)
    gross = float(scaled.sum())
    if gross > 1.0:
        scaled = scaled / gross
    return scaled
