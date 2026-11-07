"""Dividend discount models, for the companies they actually fit.

A DDM is only honest when dividends are the mechanism by which shareholders
get paid -- mature, payout-stable businesses, utilities, REITs. Applying it to
a company that returns capital through buybacks understates value badly, and
applying it to a non-payer gives zero. :func:`applicability` checks this before
the model is allowed to contribute to a blended valuation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..util.mathx import NAN, clamp, is_num, safe_div


@dataclass
class DDMResult:
    value_per_share: float
    model: str
    dividend_per_share: float
    growth_used: float
    notes: list[str] = field(default_factory=list)

    def upside_vs(self, price: float) -> float:
        return safe_div(self.value_per_share, price) - 1.0


def gordon_growth(dividend_next: float, cost_of_equity: float, growth: float) -> float:
    """``D1 / (r - g)``. Exquisitely sensitive as ``g`` approaches ``r``."""
    if not all(is_num(x) for x in (dividend_next, cost_of_equity, growth)):
        return NAN
    if cost_of_equity <= growth:
        return NAN
    return float(dividend_next / (cost_of_equity - growth))


def two_stage(dividend_now: float, cost_of_equity: float, growth_high: float,
              years_high: int, growth_terminal: float) -> float:
    """High-growth dividends discounted explicitly, then a Gordon tail."""
    if not all(is_num(x) for x in (dividend_now, cost_of_equity, growth_high, growth_terminal)):
        return NAN
    if cost_of_equity <= growth_terminal:
        return NAN
    value = 0.0
    dividend = float(dividend_now)
    for t in range(1, int(years_high) + 1):
        dividend *= (1.0 + growth_high)
        value += dividend / (1.0 + cost_of_equity) ** t
    terminal_dividend = dividend * (1.0 + growth_terminal)
    terminal = terminal_dividend / (cost_of_equity - growth_terminal)
    return float(value + terminal / (1.0 + cost_of_equity) ** int(years_high))


def h_model(dividend_now: float, cost_of_equity: float, growth_short: float,
            growth_long: float, half_life_years: float) -> float:
    """Fuller-Hsia H-model: a linear fade from high to stable growth, closed form.

    ``V = D0[(1+gL) + H(gS - gL)] / (r - gL)``, where H is half the fade period.
    Smoother and more realistic than a two-stage model's abrupt step down, and
    it needs no projection table.
    """
    if not all(is_num(x) for x in (dividend_now, cost_of_equity, growth_short, growth_long)):
        return NAN
    if cost_of_equity <= growth_long:
        return NAN
    h = float(half_life_years)
    return float(dividend_now * ((1.0 + growth_long) + h * (growth_short - growth_long))
                 / (cost_of_equity - growth_long))


def applicability(metrics: dict, is_reit: bool = False) -> tuple[bool, str]:
    """Decide whether a DDM can speak to this company at all."""
    dividend_yield = metrics.get("dividend_yield")
    if not is_num(dividend_yield) or dividend_yield <= 0.001:
        return False, "pays no meaningful dividend"
    buyback = metrics.get("buyback_yield")
    if is_num(buyback) and is_num(dividend_yield) and buyback > dividend_yield * 1.5:
        return False, ("returns capital mainly through buybacks, which a dividend "
                       "model cannot see")

    # Sustainability must be tested against the right earnings measure. A REIT
    # distributing 90% of FFO looks like a 180% payout on GAAP net income,
    # because depreciation on appreciating property sits in between. Judging it
    # on net income rejects the one sector a dividend model fits best.
    if is_reit:
        payout = metrics.get("ffo_payout_ratio")
        basis = "FFO"
        limit = 1.10
    else:
        payout = metrics.get("payout_ratio")
        basis = "earnings"
        limit = 1.20
    if is_num(payout) and payout > limit:
        return False, (f"payout is {payout:.0%} of {basis}; the current dividend "
                       "is not sustainable")
    return True, f"dividend-paying, {basis}-covered payout"


def from_metrics(metrics: dict, cost_of_equity: float,
                 terminal_growth: float = 0.025,
                 is_reit: bool = False) -> DDMResult:
    """Pick the appropriate DDM variant from observed payout behaviour."""
    notes: list[str] = []
    ok, reason = applicability(metrics, is_reit=is_reit)
    if not ok:
        return DDMResult(NAN, "not applicable", NAN, NAN, [reason])

    price = metrics.get("price")
    dividend_yield = metrics.get("dividend_yield")
    dps = (price * dividend_yield) if (is_num(price) and is_num(dividend_yield)) else NAN
    if not is_num(dps) or dps <= 0:
        return DDMResult(NAN, "not applicable", NAN, NAN,
                         ["Dividend per share could not be derived."])

    growth = metrics.get("dividend_cagr_5y")
    if not is_num(growth):
        growth = metrics.get("dividend_cagr_3y")
    if not is_num(growth):
        growth = metrics.get("earnings_cagr_5y")
    if not is_num(growth):
        growth = terminal_growth
        notes.append("No dividend growth history; terminal growth assumed throughout.")
    # Historical dividend growth above ~12% never persists; cap it before it
    # compounds into the terminal value.
    growth = clamp(float(growth), -0.10, 0.12)

    if growth <= terminal_growth + 0.005:
        value = gordon_growth(dps * (1.0 + terminal_growth), cost_of_equity, terminal_growth)
        model = "Gordon growth"
        growth_used = terminal_growth
    else:
        # Fade over a decade: H = 5.
        value = h_model(dps, cost_of_equity, growth, terminal_growth, 5.0)
        model = "H-model (10-year fade)"
        growth_used = growth
        notes.append(f"Dividend growth of {growth:.1%} fades to {terminal_growth:.1%} "
                     "over ten years.")

    if not is_num(value):
        notes.append("Cost of equity does not exceed terminal growth; model undefined.")
    return DDMResult(value_per_share=value, model=model, dividend_per_share=dps,
                     growth_used=growth_used, notes=notes)
