"""Entry setups: where a buy decision becomes a buy *order*.

Valuation says what to own. These say when a price offers a favourable
risk-to-reward *right now*, which is a separate and much weaker claim. The
honest framing is that entry timing mostly controls risk per trade rather than
expected return: a defined stop and a sensible level let you size a position
properly, which matters more than squeezing the last percent out of the entry.

Each setup returns a 0-100 confidence and the conditions that produced it, so a
score can always be interrogated rather than trusted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..analysis.technicals import (
    atr, indicator_frame, rsi, sma, support_resistance, swing_points,
)
from ..util.mathx import NAN, clamp, interp_score, is_num, linreg, safe_div


@dataclass
class Setup:
    name: str
    triggered: bool
    confidence: float          # 0-100
    entry_zone: tuple[float, float]
    conditions: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> list[str]:
        return [k for k, v in self.conditions.items() if v]

    @property
    def failed(self) -> list[str]:
        return [k for k, v in self.conditions.items() if not v]


def _score(conditions: dict[str, bool], weights: dict[str, float] | None = None) -> float:
    if not conditions:
        return NAN
    weights = weights or {}
    total = sum(weights.get(k, 1.0) for k in conditions)
    hit = sum(weights.get(k, 1.0) for k, v in conditions.items() if v)
    return 100.0 * hit / total if total else NAN


def pullback_to_trend(frame: pd.DataFrame, ma_window: int = 50) -> Setup:
    """Buy a dip in an established uptrend.

    The highest-probability discretionary setup there is, because it does not
    require the trend to be called: the trend is already visible, and the dip
    provides a nearby invalidation level. The risk is that "pullback in an
    uptrend" and "start of a downtrend" look identical for the first two weeks,
    which is why the stop matters more than the entry.
    """
    ind = indicator_frame(frame)
    if len(ind) < 220:
        return Setup("Pullback to trend", False, NAN, (NAN, NAN),
                     notes=["Needs about a year of history."])
    last = ind.iloc[-1]
    close = float(last["close"])
    ma = float(last[f"sma{ma_window}"]) if is_num(last.get(f"sma{ma_window}")) else NAN
    ma200 = float(last["sma200"]) if is_num(last.get("sma200")) else NAN
    ma200_series = ind["sma200"].dropna()

    distance = safe_div(close - ma, ma)
    conditions = {
        "price above 200-day": is_num(ma200) and close > ma200,
        "200-day rising": len(ma200_series) > 22 and ma200_series.iloc[-1] > ma200_series.iloc[-22],
        f"within 5% of the {ma_window}-day": is_num(distance) and -0.03 <= distance <= 0.05,
        "RSI between 35 and 55 (reset, not broken)":
            is_num(last.get("rsi14")) and 35 <= float(last["rsi14"]) <= 55,
        "pullback under 15% from the high":
            is_num(ind["close"].tail(252).max())
            and (close / float(ind["close"].tail(252).max()) - 1.0) > -0.15,
        "volume not spiking on the decline":
            not is_num(last.get("rel_volume")) or float(last["rel_volume"]) < 2.0,
    }
    weights = {"price above 200-day": 2.0, "200-day rising": 2.0,
               f"within 5% of the {ma_window}-day": 1.5}
    entry_low = ma * 0.98 if is_num(ma) else NAN
    entry_high = ma * 1.03 if is_num(ma) else NAN
    notes = []
    if conditions["price above 200-day"] and not conditions["200-day rising"]:
        notes.append("Above the 200-day but the 200-day is flat or falling: the "
                     "trend is not confirmed, so treat this as a trade, not an entry.")
    return Setup("Pullback to trend",
                 triggered=all(conditions[k] for k in
                               ("price above 200-day", "200-day rising",
                                f"within 5% of the {ma_window}-day")),
                 confidence=_score(conditions, weights),
                 entry_zone=(entry_low, entry_high),
                 conditions=conditions, notes=notes)


def breakout(frame: pd.DataFrame, window: int = 55) -> Setup:
    """Buy a move out of a consolidation to new highs on expanding volume.

    Volume is the whole condition. A breakout on average volume is where most
    false breakouts live; institutional accumulation shows up as volume well
    above the 50-day norm, and without it there is no evidence anyone large is
    buying.
    """
    ind = indicator_frame(frame)
    if len(ind) < window + 60:
        return Setup("Breakout", False, NAN, (NAN, NAN),
                     notes=["Not enough history for the channel."])
    last = ind.iloc[-1]
    close = float(last["close"])
    channel_high = float(ind["high"].iloc[-window - 1:-1].max())
    channel_low = float(ind["low"].iloc[-window - 1:-1].min())
    channel_width = safe_div(channel_high - channel_low, channel_low)

    # Band width at a multi-month low is the classic pre-breakout squeeze.
    bb_width = ind["bb_width"].dropna()
    squeeze = (len(bb_width) > 126
               and float(bb_width.iloc[-1]) <= float(bb_width.tail(126).quantile(0.30)))

    conditions = {
        f"above the {window}-day high": close >= channel_high * 0.999,
        "volume above 1.5x the 50-day average":
            is_num(last.get("rel_volume")) and float(last["rel_volume"]) >= 1.5,
        "prior range tight (under 25%)": is_num(channel_width) and channel_width <= 0.25,
        "volatility compressed before the move": squeeze,
        "above the 200-day": is_num(last.get("sma200")) and close > float(last["sma200"]),
        "not already extended (<8% above the 20-day)":
            is_num(last.get("sma20")) and (close / float(last["sma20"]) - 1.0) < 0.08,
    }
    weights = {f"above the {window}-day high": 2.0,
               "volume above 1.5x the 50-day average": 2.0}
    notes = []
    if conditions[f"above the {window}-day high"] and not conditions["volume above 1.5x the 50-day average"]:
        notes.append("Price broke out but volume did not confirm. Most failed "
                     "breakouts look exactly like this.")
    return Setup("Breakout",
                 triggered=conditions[f"above the {window}-day high"]
                 and conditions["volume above 1.5x the 50-day average"],
                 confidence=_score(conditions, weights),
                 entry_zone=(channel_high, channel_high * 1.03),
                 conditions=conditions, notes=notes)


def oversold_reversion(frame: pd.DataFrame) -> Setup:
    """Buy a stretched decline in a name that is not actually breaking down.

    Mean reversion works on the short horizon and fails catastrophically on
    falling knives, so the filters here are about *survival* rather than
    oversold-ness: still above the long-term trend, volatility not exploding.
    """
    ind = indicator_frame(frame)
    if len(ind) < 220:
        return Setup("Oversold reversion", False, NAN, (NAN, NAN),
                     notes=["Needs about a year of history."])
    last = ind.iloc[-1]
    close = float(last["close"])
    rsi_now = float(last["rsi14"]) if is_num(last.get("rsi14")) else NAN
    lower_band = float(last["bb_lower"]) if is_num(last.get("bb_lower")) else NAN
    vol21 = float(last["vol21"]) if is_num(last.get("vol21")) else NAN
    vol63 = float(last["vol63"]) if is_num(last.get("vol63")) else NAN

    conditions = {
        "RSI below 32": is_num(rsi_now) and rsi_now < 32,
        "at or below the lower Bollinger band": is_num(lower_band) and close <= lower_band,
        "still above the 200-day (not a breakdown)":
            is_num(last.get("sma200")) and close > float(last["sma200"]),
        "volatility not exploding (21d < 1.6x 63d)":
            is_num(vol21) and is_num(vol63) and vol63 > 0 and vol21 / vol63 < 1.6,
        "no new 52-week low":
            close > float(ind["low"].tail(252).min()) * 1.03,
    }
    weights = {"RSI below 32": 1.5, "still above the 200-day (not a breakdown)": 2.0}
    notes = ["Mean reversion is a short-horizon edge. Define the exit before "
             "entering: these positions are not meant to be held through a "
             "trend change."]
    if not conditions["still above the 200-day (not a breakdown)"]:
        notes.append("Below the 200-day: this is a downtrend, not an oversold "
                     "bounce. Oversold readings persist for months in one.")
    return Setup("Oversold reversion",
                 triggered=conditions["RSI below 32"]
                 and conditions["still above the 200-day (not a breakdown)"],
                 confidence=_score(conditions, weights),
                 entry_zone=(close * 0.97, close * 1.01),
                 conditions=conditions, notes=notes)


def volatility_contraction(frame: pd.DataFrame) -> Setup:
    """Minervini's VCP: successively tighter pullbacks on falling volume.

    The idea is that each contraction represents sellers being exhausted in
    turn. Measured here by comparing the depth of the three most recent
    swing-high-to-low declines and checking volume is drying up. It is a
    pattern-recognition heuristic with no published excess-return evidence of
    its own; it is included because it describes a real supply condition, not
    because it is proven.
    """
    ind = indicator_frame(frame)
    if len(ind) < 150:
        return Setup("Volatility contraction", False, NAN, (NAN, NAN),
                     notes=["Needs about six months of history."])
    window = ind.tail(130)
    highs, lows = swing_points(window["high"], window["low"], lookback=4)
    high_points = highs.dropna()
    low_points = lows.dropna()

    contractions: list[float] = []
    for high_date, high_value in list(high_points.items())[-4:]:
        after = low_points[low_points.index > high_date]
        if not after.empty:
            contractions.append(float(high_value - after.iloc[0]) / float(high_value))

    tightening = (len(contractions) >= 2
                  and all(b < a for a, b in zip(contractions, contractions[1:])))
    volume_drying = False
    if "volume" in window:
        first_half = float(window["volume"].iloc[:65].mean())
        second_half = float(window["volume"].iloc[65:].mean())
        volume_drying = first_half > 0 and second_half < first_half * 0.85

    last = ind.iloc[-1]
    close = float(last["close"])
    pivot = float(high_points.iloc[-1]) if not high_points.empty else NAN

    conditions = {
        "at least two measurable contractions": len(contractions) >= 2,
        "each pullback shallower than the last": tightening,
        "volume drying up": volume_drying,
        "price within 8% of the pivot":
            is_num(pivot) and abs(close / pivot - 1.0) <= 0.08,
        "above the 50-day":
            is_num(last.get("sma50")) and close > float(last["sma50"]),
    }
    return Setup("Volatility contraction",
                 triggered=tightening and conditions["price within 8% of the pivot"],
                 confidence=_score(conditions),
                 entry_zone=(pivot, pivot * 1.03) if is_num(pivot) else (NAN, NAN),
                 conditions=conditions,
                 notes=[f"Contractions measured: "
                        f"{[f'{c:.1%}' for c in contractions]}"] if contractions else [])


ALL_SETUPS = {
    "pullback": pullback_to_trend,
    "breakout": breakout,
    "oversold": oversold_reversion,
    "vcp": volatility_contraction,
}


def evaluate_all(frame: pd.DataFrame) -> dict[str, Setup]:
    out = {}
    for key, func in ALL_SETUPS.items():
        try:
            out[key] = func(frame)
        except Exception as exc:  # noqa: BLE001 - a broken setup must not block the rest
            out[key] = Setup(key, False, NAN, (NAN, NAN),
                             notes=[f"failed: {type(exc).__name__}: {exc}"])
    return out
