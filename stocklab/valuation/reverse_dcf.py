"""Reverse DCF: what the current price already assumes.

This is the most honest valuation tool available and the one most worth
running first.  A forward DCF asks "what is this worth?", which requires you to
forecast; the answer is then dominated by your forecast and easily dismissed.
A reverse DCF asks "what would have to be true for today's price to be
correct?" -- and that question has a defensible answer, because the price is
observable.

The output is a hurdle, not a value: *revenue must compound at 11.4% for a
decade with margins intact*.  You then judge whether that is plausible, which
is a question about the business rather than about the model.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from ..util.mathx import NAN, is_num
from .dcf import DCFAssumptions, run_dcf


@dataclass
class ImpliedExpectations:
    """What the market price requires, with the robust reading first.

    ``implied_return`` is the headline because it is the only output here that
    does not depend on an assumed equity risk premium: it is simply the IRR of
    the projected cash flows at the market price.  "You are being offered 5.2%
    a year" is a statement an investor can act on without first agreeing about
    what equities should return.

    ``implied_growth`` is far more fragile and must be read with its caveat.
    It solves for near-term growth while holding terminal growth fixed, and
    since terminal value is usually half the total, the near-term knob has to
    move a long way to close any gap.  A large number here means "the gap
    cannot be explained by the forecast period alone" -- not that anyone expects
    35% growth.
    """

    implied_return: float          # IRR at market price: the robust output
    implied_growth: float          # near-term growth, terminal growth held fixed
    implied_growth_joint: float    # growth and terminal growth moved together
    implied_margin: float
    implied_terminal_growth: float
    historical_growth: float
    historical_margin: float
    growth_gap: float              # implied minus historical: the stretch required
    verdict: str
    infeasible: list[str]          # solves with no answer inside their bracket
    notes: list[str]

    @property
    def implied_wacc(self) -> float:
        """Alias: the implied discount rate *is* the implied expected return."""
        return self.implied_return


def _solve(
    a: DCFAssumptions,
    field: str,
    price: float,
    lo: float,
    hi: float,
    tolerance: float = 1e-4,
    max_iter: int = 80,
) -> float:
    """Bisect on one assumption until the model reproduces ``price``.

    Bisection rather than a solver because ``run_dcf`` is cheap, piecewise
    non-smooth (clamps, caps), and occasionally returns ``nan`` -- conditions
    under which Newton-type methods wander off.
    """
    if not is_num(price) or price <= 0:
        return NAN

    def value_at(x: float) -> float:
        return run_dcf(replace(a, **{field: float(x)})).value_per_share

    v_lo, v_hi = value_at(lo), value_at(hi)
    if not (is_num(v_lo) and is_num(v_hi)):
        return NAN
    if (v_lo - price) * (v_hi - price) > 0:
        # Price lies outside what this assumption can produce over the bracket.
        return NAN

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        v_mid = value_at(mid)
        if not is_num(v_mid):
            return NAN
        if abs(v_mid - price) / price < tolerance:
            return float(mid)
        if (v_lo - price) * (v_mid - price) <= 0:
            hi, v_hi = mid, v_mid
        else:
            lo, v_lo = mid, v_mid
    return float((lo + hi) / 2.0)


def implied_growth(a: DCFAssumptions, price: float,
                   lo: float = -0.25, hi: float = 0.60) -> float:
    """Initial revenue growth (fading to terminal) that justifies ``price``."""
    return _solve(a, "revenue_growth_initial", price, lo, hi)


def implied_margin(a: DCFAssumptions, price: float,
                   lo: float = -0.10, hi: float = 0.70) -> float:
    """Target operating margin that justifies ``price``, holding growth fixed."""
    return _solve(a, "operating_margin_target", price, lo, hi)


def implied_discount_rate(a: DCFAssumptions, price: float,
                          lo: float | None = None, hi: float = 0.30) -> float:
    """The discount rate the market is applying -- i.e. the expected return.

    If the market pays today's price for the cash flows you projected, this is
    the IRR it is accepting. Comparing it to your own WACC is often more
    intuitive than comparing values: "the market expects 7.1% from this; I want
    10%" is a cleaner statement than a percentage over- or under-valuation.
    """
    lo = lo if lo is not None else a.terminal_growth + 0.01
    return _solve(a, "wacc", price, hi, lo)  # value falls as WACC rises


def implied_terminal_growth(a: DCFAssumptions, price: float) -> float:
    hi = a.wacc - 0.005
    return _solve(a, "terminal_growth", price, -0.02, hi)


def implied_growth_joint(a: DCFAssumptions, price: float,
                         lo: float = -0.10, hi: float = 0.40) -> float:
    """Growth that justifies ``price`` when near-term *and* terminal move together.

    More interpretable than :func:`implied_growth` -- it answers "what steady
    growth rate would do it?" -- but frequently has no solution, because a
    terminal rate above the risk-free cap is economically incoherent and the
    bracket refuses to go there. That absence is itself the finding: the price
    cannot be explained by growth at any sustainable rate.
    """
    if not is_num(price) or price <= 0:
        return NAN
    cap = min(a.wacc - 0.005, 0.045)

    def value_at(x: float) -> float:
        g = float(x)
        trial = replace(a, revenue_growth_initial=g,
                        revenue_growth_terminal=min(g, cap),
                        terminal_growth=min(g, cap))
        return run_dcf(trial).value_per_share

    v_lo, v_hi = value_at(lo), value_at(hi)
    if not (is_num(v_lo) and is_num(v_hi)) or (v_lo - price) * (v_hi - price) > 0:
        return NAN
    for _ in range(80):
        mid = (lo + hi) / 2.0
        v_mid = value_at(mid)
        if not is_num(v_mid):
            return NAN
        if abs(v_mid - price) / price < 1e-4:
            return float(mid)
        if (v_lo - price) * (v_mid - price) <= 0:
            hi, v_hi = mid, v_mid
        else:
            lo, v_lo = mid, v_mid
    return float((lo + hi) / 2.0)


def analyse(a: DCFAssumptions, price: float, metrics: dict | None = None
            ) -> ImpliedExpectations:
    """Full reverse-DCF read on a single name."""
    metrics = metrics or {}
    notes: list[str] = []
    infeasible: list[str] = []

    r_implied = implied_discount_rate(a, price)
    g_implied = implied_growth(a, price)
    g_joint = implied_growth_joint(a, price)
    m_implied = implied_margin(a, price)
    tg_implied = implied_terminal_growth(a, price)

    for label, value in (("growth", g_implied), ("joint growth", g_joint),
                         ("margin", m_implied), ("terminal growth", tg_implied),
                         ("discount rate", r_implied)):
        if not is_num(value):
            infeasible.append(label)

    hist_growth = metrics.get("revenue_cagr_5y")
    if not is_num(hist_growth):
        hist_growth = metrics.get("revenue_cagr_3y")
    hist_margin = metrics.get("operating_margin_5y_avg")
    if not is_num(hist_margin):
        hist_margin = metrics.get("operating_margin")

    gap = (g_implied - hist_growth) if (is_num(g_implied) and is_num(hist_growth)) else NAN

    # The verdict keys off the implied return, not the implied growth: it is the
    # robust number and it does not depend on an assumed risk premium.
    if is_num(r_implied):
        if r_implied >= a.wacc + 0.02:
            verdict = f"price offers ~{r_implied:.1%} expected return, above the {a.wacc:.1%} hurdle"
        elif r_implied >= a.wacc - 0.01:
            verdict = f"price offers ~{r_implied:.1%} expected return, roughly fair against a {a.wacc:.1%} hurdle"
        elif r_implied >= a.wacc - 0.03:
            verdict = f"price offers ~{r_implied:.1%} expected return, modestly below the {a.wacc:.1%} hurdle"
        else:
            verdict = f"price offers only ~{r_implied:.1%} expected return against a {a.wacc:.1%} hurdle"
    elif is_num(gap):
        verdict = "implied return outside solvable range; judge on implied growth"
    else:
        verdict = "no solvable expectations -- check inputs"

    if is_num(g_implied) and is_num(gap) and gap > 0.06:
        notes.append(
            f"Holding terminal growth at {a.terminal_growth:.1%}, the price needs "
            f"{g_implied:.1%} near-term growth against {hist_growth:.1%} delivered "
            "over five years. Read this as 'the forecast period alone cannot "
            "explain the price', not as a growth forecast -- terminal value is "
            f"{a.years}-year-distant and dominates, so this knob is very sensitive."
        )
    if not is_num(g_joint):
        notes.append(
            "No sustainable steady growth rate (capped at the risk-free rate) "
            "reproduces this price, so the price embeds either a lower discount "
            "rate than estimated or margin expansion beyond anything in the history."
        )
    if is_num(tg_implied) and tg_implied > 0.045:
        notes.append(
            f"The implied {tg_implied:.1%} perpetual growth exceeds plausible "
            "long-run nominal GDP; nothing can outgrow the economy forever."
        )
    if not is_num(m_implied):
        notes.append(
            "No operating margin up to 70% reproduces this price at the assumed "
            "growth, so margin expansion alone cannot justify it."
        )
    return ImpliedExpectations(
        implied_return=r_implied, implied_growth=g_implied,
        implied_growth_joint=g_joint, implied_margin=m_implied,
        implied_terminal_growth=tg_implied,
        historical_growth=hist_growth if is_num(hist_growth) else NAN,
        historical_margin=hist_margin if is_num(hist_margin) else NAN,
        growth_gap=gap, verdict=verdict, infeasible=infeasible, notes=notes,
    )


def required_growth_table(a: DCFAssumptions, prices: list[float]) -> pd.DataFrame:
    """Implied growth across a ladder of prices -- an entry-point map.

    Read it as: "at $180 the market would only need 6% growth, which history
    supports; at $340 it needs 14%, which it does not." That converts a
    valuation into a buy level.
    """
    rows = []
    for px in prices:
        g = implied_growth(a, px)
        rows.append({"price": px, "implied_initial_growth": g,
                     "vs_history": g - a.revenue_growth_initial if is_num(g) else NAN})
    return pd.DataFrame(rows).set_index("price")
