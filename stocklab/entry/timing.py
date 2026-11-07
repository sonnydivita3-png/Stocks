"""Combine valuation, setup quality and regime into one actionable trade plan.

The output is deliberately a *plan*, not a signal: entry zone, stop, size,
targets, and the conditions that would invalidate it. A score with no stop and
no size is not a decision anyone can act on or review afterwards.

Weighting reflects the evidence. Valuation carries the most weight because the
case for buying a business cheaply is far better documented than the case for
any chart pattern; the technical setup contributes mainly by defining a stop,
and therefore the size.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..portfolio.sizing import atr_position_size
from ..util.mathx import NAN, interp_score, is_num, safe_div
from .signals import Setup, evaluate_all
from .stops import StopPlan, atr_stop, structure_stop


@dataclass
class TradePlan:
    symbol: str
    price: float
    score: float                       # 0-100 composite
    verdict: str
    best_setup: str
    setups: dict[str, Setup]
    stop: StopPlan
    staged_entries: pd.DataFrame
    size: dict
    components: dict[str, float] = field(default_factory=dict)
    invalidation: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


COMPONENT_WEIGHTS = {
    "valuation": 0.40,     # best-evidenced, and the slowest to change
    "setup": 0.25,         # defines the stop, hence the size
    "trend": 0.20,         # momentum has strong evidence; trend-following less
    "regime": 0.15,        # modest, and bounded: see analysis.regime
}


def staged_entry(price: float, stop: StopPlan, tranches: int = 3,
                 spread: float = 0.05) -> pd.DataFrame:
    """A ladder of limit orders rather than one all-in purchase.

    Scaling in gives up some upside when a position runs immediately, and in
    exchange removes the single worst outcome: committing a full position at
    the exact wrong moment and then having no capital to average into
    weakness. It also keeps the average entry closer to the stop, which raises
    the R-multiple on the same thesis.
    """
    if not is_num(price) or price <= 0 or tranches < 1:
        return pd.DataFrame()

    # Never place a tranche at or below the stop. A limit below the stop would
    # fill into a position that is already invalidated, and it quietly turns a
    # staged entry into averaging down through the exit level. The ladder is
    # compressed to reach at most 80% of the way to the stop.
    if is_num(stop.stop) and stop.stop < price:
        max_spread = 0.80 * (price - stop.stop) / price
        spread = min(spread, max_spread)

    rows = []
    for i in range(tranches):
        # First tranche at market, the rest stepped down.
        limit = price * (1.0 - spread * i / max(tranches - 1, 1))
        weight = 1.0 / tranches
        rows.append({
            "tranche": i + 1,
            "limit_price": round(limit, 2),
            "allocation": weight,
            "distance_from_spot": limit / price - 1.0,
            "r_to_stop": safe_div(limit - stop.stop, stop.risk_per_share)
            if is_num(stop.stop) else NAN,
        })
    frame = pd.DataFrame(rows).set_index("tranche")
    return frame


def build_plan(
    symbol: str,
    frame: pd.DataFrame,
    *,
    margin_of_safety: float | None = None,
    technicals: dict | None = None,
    regime_score: float | None = None,
    capital: float = 100_000.0,
    risk_per_trade: float = 0.01,
    max_weight: float = 0.20,
    stop_style: str = "atr",
) -> TradePlan:
    """Assemble a complete, reviewable trade plan for one symbol."""
    technicals = technicals or {}
    notes: list[str] = []
    if frame.empty or len(frame) < 60:
        return TradePlan(symbol, NAN, NAN, "insufficient price history", "",
                         {}, StopPlan(NAN, NAN, "none", NAN, NAN),
                         pd.DataFrame(), {}, notes=["Need 60+ bars."])

    price = float(frame["close"].iloc[-1])
    setups = evaluate_all(frame)

    triggered = {k: s for k, s in setups.items() if s.triggered and is_num(s.confidence)}
    pool = triggered or {k: s for k, s in setups.items() if is_num(s.confidence)}
    best_key = max(pool, key=lambda k: pool[k].confidence) if pool else ""
    best = setups.get(best_key)
    setup_score = best.confidence if best and is_num(best.confidence) else NAN

    components: dict[str, float] = {}
    if is_num(margin_of_safety):
        # A 30%+ discount scores full marks; paying above fair value scores zero.
        components["valuation"] = interp_score(
            margin_of_safety, [(-0.30, 0), (-0.10, 25), (0.0, 50),
                               (0.15, 70), (0.30, 90), (0.50, 100)])
    if is_num(setup_score):
        components["setup"] = setup_score

    trend_inputs = [
        interp_score(technicals.get("dist_sma200", NAN),
                     [(-0.20, 0), (-0.05, 30), (0.0, 50), (0.10, 80), (0.25, 100)]),
        interp_score(technicals.get("momentum_12_1", NAN),
                     [(-0.30, 0), (0.0, 40), (0.15, 70), (0.40, 100)]),
        interp_score(technicals.get("trend_r2", NAN), [(0.1, 20), (0.5, 60), (0.85, 100)]),
    ]
    valid_trend = [x for x in trend_inputs if is_num(x)]
    if valid_trend:
        components["trend"] = float(sum(valid_trend) / len(valid_trend))
    if is_num(regime_score):
        components["regime"] = float(regime_score)

    numerator = sum(components[k] * COMPONENT_WEIGHTS[k] for k in components)
    denominator = sum(COMPONENT_WEIGHTS[k] for k in components)
    score = numerator / denominator if denominator else NAN

    stop = (structure_stop(frame) if stop_style == "structure" else atr_stop(frame))
    atr_value = NAN
    if is_num(stop.risk_per_share) and stop.risk_per_share > 0:
        atr_value = stop.risk_per_share / 2.5 if stop_style == "atr" else stop.risk_per_share / 2.5
    size = atr_position_size(capital, price,
                             atr_value if is_num(atr_value) else NAN,
                             risk_per_trade=risk_per_trade,
                             atr_multiple=2.5, max_weight=max_weight)

    if not is_num(score):
        verdict = "no usable inputs"
    elif score >= 72:
        verdict = "strong: valuation and setup both support an entry"
    elif score >= 58:
        verdict = "constructive: worth starting a staged position"
    elif score >= 45:
        verdict = "neutral: wait for a better level or a confirmed setup"
    elif score >= 30:
        verdict = "weak: the price is not offering a favourable entry"
    else:
        verdict = "avoid on these inputs"

    invalidation = []
    if is_num(stop.stop):
        invalidation.append(f"Close below ${stop.stop:.2f} ({stop.method}).")
    if is_num(technicals.get("sma200")):
        invalidation.append(f"Loss of the 200-day at ${technicals['sma200']:.2f}.")
    invalidation.append(
        "Thesis break on fundamentals: a revenue or margin miss that moves the "
        "implied growth required to justify the price."
    )

    if "valuation" not in components:
        notes.append("No valuation input: this score is technical only and should "
                     "not be read as a reason to own the business.")
    if triggered and len(triggered) > 1:
        notes.append(f"Multiple setups triggered ({', '.join(triggered)}), which "
                     "usually means a clean technical picture.")
    if not triggered:
        notes.append("No setup is actually triggered; the score reflects the best "
                     "partial match. Entering here is anticipating a setup, not trading one.")
    return TradePlan(
        symbol=symbol.upper(), price=price, score=score, verdict=verdict,
        best_setup=best.name if best else "", setups=setups, stop=stop,
        staged_entries=staged_entry(price, stop), size=size,
        components=components, invalidation=invalidation, notes=notes,
    )
