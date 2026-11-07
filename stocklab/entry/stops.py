"""Stop placement and R-multiple planning.

The stop is the only part of a trade you fully control, and it determines
position size, which determines everything. Two principles shape this module:

* **Place the stop where the thesis is wrong, then size to it.** The reverse --
  choosing a size and putting the stop wherever the loss feels tolerable --
  guarantees being stopped out by noise, because the level has nothing to do
  with the stock's volatility.
* **Measure everything in R**, the distance from entry to stop. A system with
  a 40% hit rate and 3R winners is excellent; one with an 80% hit rate and
  0.2R winners is a slow loss. Win rate alone says nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..analysis.technicals import atr, indicator_frame, support_resistance, swing_points
from ..util.mathx import NAN, is_num, safe_div


@dataclass
class StopPlan:
    entry: float
    stop: float
    method: str
    risk_per_share: float
    risk_pct: float
    targets: dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    notes: list[str] = field(default_factory=list)

    def r_multiple(self, price: float) -> float:
        """How many R a given price represents from the entry."""
        if not (is_num(price) and is_num(self.risk_per_share)) or self.risk_per_share <= 0:
            return NAN
        return (price - self.entry) / self.risk_per_share

    def target_at(self, r: float) -> float:
        return self.entry + r * self.risk_per_share


def atr_stop(frame: pd.DataFrame, entry: float | None = None,
             multiple: float = 2.5, period: int = 14) -> StopPlan:
    """Stop at ``multiple`` ATRs below entry.

    Volatility-adjusted, so the same multiple means the same *probability* of
    being hit by noise across a quiet utility and a volatile biotech. Below
    about 2 ATRs the stop sits inside the normal daily range and will be hit
    routinely regardless of the thesis.
    """
    ind = indicator_frame(frame)
    if ind.empty or "atr14" not in ind:
        return StopPlan(NAN, NAN, "atr", NAN, NAN, notes=["No ATR available."])
    entry = float(entry) if is_num(entry) else float(ind["close"].iloc[-1])
    atr_value = float(ind["atr14"].iloc[-1])
    if not is_num(atr_value) or atr_value <= 0:
        return StopPlan(entry, NAN, "atr", NAN, NAN, notes=["ATR is zero or missing."])
    stop = entry - multiple * atr_value
    risk = entry - stop
    notes = []
    if multiple < 2.0:
        notes.append(f"A {multiple:g}-ATR stop sits inside the normal daily range; "
                     "expect to be stopped out by noise.")
    pct = safe_div(risk, entry)
    if is_num(pct) and pct > 0.25:
        notes.append(f"The stop is {pct:.0%} away. This name is volatile enough "
                     "that position size, not the stop, has to carry the risk control.")
    return StopPlan(
        entry=entry, stop=stop, method=f"{multiple:g}x ATR({period})",
        risk_per_share=risk, risk_pct=pct,
        targets={f"{r:g}R": entry + r * risk for r in (1, 2, 3)},
        rationale=f"ATR is ${atr_value:.2f} ({safe_div(atr_value, entry):.1%} of price); "
                  f"{multiple:g}x gives room for normal movement.",
        notes=notes,
    )


def structure_stop(frame: pd.DataFrame, entry: float | None = None,
                   buffer: float = 0.015, lookback: int = 5) -> StopPlan:
    """Stop just below the most recent swing low or support cluster.

    Preferable to a volatility stop when there is a clear level, because being
    stopped out means the *chart* broke, not merely that the price wobbled.
    The weakness is that the level is where everyone else's stops sit too,
    which is exactly where liquidity hunts them.
    """
    ind = indicator_frame(frame)
    if ind.empty:
        return StopPlan(NAN, NAN, "structure", NAN, NAN, notes=["No price data."])
    entry = float(entry) if is_num(entry) else float(ind["close"].iloc[-1])

    levels = support_resistance(frame)
    support = levels["support"][0] if levels["support"] else NAN
    if not is_num(support):
        _, lows = swing_points(ind["high"], ind["low"], lookback)
        recent = lows.dropna().tail(3)
        support = float(recent.min()) if not recent.empty else NAN
    if not is_num(support) or support >= entry:
        return atr_stop(frame, entry)

    stop = support * (1.0 - buffer)
    risk = entry - stop
    return StopPlan(
        entry=entry, stop=stop, method="below structure",
        risk_per_share=risk, risk_pct=safe_div(risk, entry),
        targets={f"{r:g}R": entry + r * risk for r in (1, 2, 3)},
        rationale=f"Support at ${support:.2f} with a {buffer:.1%} buffer; a break "
                  "there invalidates the setup.",
        notes=["Obvious levels attract stop-hunting. Consider a wider buffer or "
               "a close-only stop rather than an intraday trigger."],
    )


def chandelier_exit(frame: pd.DataFrame, multiple: float = 3.0,
                    period: int = 22) -> pd.Series:
    """Trailing stop at ``multiple`` ATRs below the highest high since entry.

    Ratchets up and never down, which is what lets a trend position run while
    capping the give-back. The width is the trade-off: 3 ATRs survives normal
    corrections in a trend, 1.5 does not.
    """
    ind = indicator_frame(frame)
    if ind.empty or "atr14" not in ind:
        return pd.Series(dtype=float)
    highest = ind["high"].rolling(period, min_periods=1).max()
    return (highest - multiple * ind["atr14"]).rename("chandelier")


def pct_stop(entry: float, pct: float = 0.08) -> StopPlan:
    """Fixed percentage stop.

    Included because it is what most people actually use, and flagged because
    it ignores volatility: 8% is a tight stop on a biotech and a loose one on a
    utility, so it silently varies risk across a portfolio.
    """
    stop = entry * (1.0 - pct)
    risk = entry - stop
    return StopPlan(
        entry=entry, stop=stop, method=f"{pct:.0%} below entry",
        risk_per_share=risk, risk_pct=pct,
        targets={f"{r:g}R": entry + r * risk for r in (1, 2, 3)},
        rationale="Fixed percentage.",
        notes=["A fixed percentage ignores the stock's volatility, so the same "
               "number means different risk on different names. Prefer an ATR "
               "or structure stop."],
    )


def expectancy(win_rate: float, avg_win_r: float, avg_loss_r: float = 1.0) -> dict:
    """Expected R per trade, and the break-even win rate for that payoff.

    The useful output is ``breakeven_win_rate``: with 2R winners you need only
    34% to break even, which reframes a 40% hit rate from "mostly wrong" to
    "profitable".
    """
    if not all(is_num(x) for x in (win_rate, avg_win_r, avg_loss_r)):
        return {}
    win_rate = min(max(win_rate, 0.0), 1.0)
    expected = win_rate * avg_win_r - (1.0 - win_rate) * abs(avg_loss_r)
    breakeven = abs(avg_loss_r) / (avg_win_r + abs(avg_loss_r)) if (avg_win_r + abs(avg_loss_r)) else NAN
    return {
        "expectancy_r": expected,
        "breakeven_win_rate": breakeven,
        "edge": win_rate - breakeven if is_num(breakeven) else NAN,
        "payoff_ratio": safe_div(avg_win_r, abs(avg_loss_r)),
    }
