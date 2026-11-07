"""Multi-stage discounted cash flow on free cash flow to the firm.

The model follows Damodaran's structure because it is internally consistent in
the one place most spreadsheet DCFs are not: **terminal reinvestment**.

If a business is assumed to grow at 2.5% forever, it must keep funding that
growth, and the amount it must fund is ``g / ROIC`` of its after-tax operating
profit.  Omitting that term -- projecting perpetual growth with no perpetual
reinvestment -- is free money, and it inflates terminal value (and therefore
fair value) by 20-40% for a typical company.  Every DCF here enforces it.

Three further disciplines:

* Growth **fades** from its current rate to the terminal rate. No company
  sustains 30% growth for a decade, and a constant-growth stage asserts that
  it will.
* Terminal value's share of total value is reported. When it exceeds ~75%,
  the "valuation" is really a statement about year 11 onward and should be
  read as such.
* Terminal growth is capped at the risk-free rate (see :mod:`.wacc`).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from ..config import settings
from ..data.providers.sec import Fundamentals
from ..util.mathx import NAN, clamp, is_num, mean_finite, safe_div
from .wacc import CostOfCapital, terminal_growth_cap


@dataclass
class DCFAssumptions:
    """Every judgement call in the model, in one inspectable object."""

    revenue_base: float
    revenue_growth_initial: float
    revenue_growth_terminal: float
    operating_margin_base: float
    operating_margin_target: float
    sales_to_capital: float
    tax_rate: float
    wacc: float
    terminal_growth: float
    terminal_roic: float
    years: int = 10
    net_debt: float = 0.0
    shares: float = NAN
    mid_year_discounting: bool = True
    margin_fade_years: int | None = None   # defaults to `years`
    non_operating_assets: float = 0.0

    def __post_init__(self) -> None:
        # Growth above the discount rate in perpetuity produces a negative
        # denominator and a nonsensical (negative) terminal value.
        self.terminal_growth = min(self.terminal_growth, self.wacc - 0.005)
        # Returning less than the cost of capital in perpetuity while still
        # growing destroys value; it is a legitimate assumption but a rare one,
        # so the floor sits at the cost of capital.
        self.terminal_roic = max(self.terminal_roic, self.wacc)


@dataclass
class DCFResult:
    enterprise_value: float
    equity_value: float
    value_per_share: float
    pv_explicit: float
    pv_terminal: float
    terminal_value: float
    terminal_share: float
    projection: pd.DataFrame
    assumptions: DCFAssumptions
    notes: list[str] = field(default_factory=list)

    def upside_vs(self, price: float) -> float:
        return safe_div(self.value_per_share, price) - 1.0

    def summary(self) -> dict:
        return {
            "value_per_share": self.value_per_share,
            "equity_value": self.equity_value,
            "enterprise_value": self.enterprise_value,
            "terminal_share": self.terminal_share,
            "wacc": self.assumptions.wacc,
            "terminal_growth": self.assumptions.terminal_growth,
            "terminal_roic": self.assumptions.terminal_roic,
            "growth_initial": self.assumptions.revenue_growth_initial,
            "margin_target": self.assumptions.operating_margin_target,
        }


def _fade(start: float, end: float, years: int) -> np.ndarray:
    """Linear interpolation from ``start`` to ``end`` over ``years`` periods."""
    if years <= 1:
        return np.array([end], dtype=float)
    return np.linspace(start, end, years + 1)[1:]


def run_dcf(a: DCFAssumptions) -> DCFResult:
    """Project, discount, and assemble a per-share value."""
    notes: list[str] = []
    if not all(is_num(x) for x in (a.revenue_base, a.wacc, a.operating_margin_target)):
        return DCFResult(NAN, NAN, NAN, NAN, NAN, NAN, NAN,
                         pd.DataFrame(), a, ["Missing required inputs."])
    if a.revenue_base <= 0:
        return DCFResult(NAN, NAN, NAN, NAN, NAN, NAN, NAN,
                         pd.DataFrame(), a, ["Revenue base must be positive."])
    if a.wacc <= a.terminal_growth:
        return DCFResult(NAN, NAN, NAN, NAN, NAN, NAN, NAN, pd.DataFrame(), a,
                         ["WACC must exceed terminal growth."])

    years = max(int(a.years), 1)
    growth = _fade(a.revenue_growth_initial, a.revenue_growth_terminal, years)
    margin_years = a.margin_fade_years or years
    margin = _fade(a.operating_margin_base, a.operating_margin_target,
                   min(margin_years, years))
    if len(margin) < years:  # hold the target once the fade completes
        margin = np.concatenate([margin, np.full(years - len(margin), a.operating_margin_target)])

    rows = []
    revenue = a.revenue_base
    s2c = a.sales_to_capital if is_num(a.sales_to_capital) and a.sales_to_capital > 0 else NAN
    if not is_num(s2c):
        notes.append(
            "Sales-to-capital could not be estimated, so reinvestment is assumed "
            "zero. Enterprise value is overstated -- treat as a ceiling."
        )

    for t in range(1, years + 1):
        prior_revenue = revenue
        revenue = prior_revenue * (1.0 + growth[t - 1])
        ebit = revenue * margin[t - 1]
        nopat = ebit * (1.0 - a.tax_rate)
        # Reinvestment is driven by the *change* in revenue: growing sales by
        # $1 requires $1/sales-to-capital of new invested capital.
        reinvestment = (revenue - prior_revenue) / s2c if is_num(s2c) else 0.0
        fcff = nopat - reinvestment
        # Mid-year convention: cash arrives through the year, not on 31 December.
        exponent = (t - 0.5) if a.mid_year_discounting else float(t)
        discount = (1.0 + a.wacc) ** exponent
        rows.append({
            "year": t, "revenue": revenue, "revenue_growth": growth[t - 1],
            "operating_margin": margin[t - 1], "ebit": ebit, "nopat": nopat,
            "reinvestment": reinvestment, "fcff": fcff,
            "discount_factor": 1.0 / discount, "pv_fcff": fcff / discount,
        })

    projection = pd.DataFrame(rows).set_index("year")
    pv_explicit = float(projection["pv_fcff"].sum())

    # --- terminal value, with reinvestment enforced -----------------------
    final_ebit = float(projection["ebit"].iloc[-1])
    terminal_nopat = final_ebit * (1.0 + a.terminal_growth) * (1.0 - a.tax_rate)
    # The share of perpetual profit that must be ploughed back to sustain g.
    terminal_reinvest_rate = clamp(a.terminal_growth / a.terminal_roic, 0.0, 0.95)
    terminal_fcff = terminal_nopat * (1.0 - terminal_reinvest_rate)
    terminal_value = terminal_fcff / (a.wacc - a.terminal_growth)

    # Terminal value is the value at the END of the explicit period (it
    # capitalises year N+1 onward), so it discounts a full `years` periods even
    # when the explicit flows use the mid-year convention.
    pv_terminal = terminal_value / (1.0 + a.wacc) ** float(years)

    enterprise_value = pv_explicit + pv_terminal
    equity_value = enterprise_value - a.net_debt + a.non_operating_assets
    value_per_share = safe_div(equity_value, a.shares)
    terminal_share = safe_div(pv_terminal, enterprise_value)

    if is_num(terminal_share) and terminal_share > 0.75:
        notes.append(
            f"{terminal_share:.0%} of value sits in the terminal value, so this is "
            f"mostly a claim about year {years + 1} onward, not about the forecast."
        )
    if equity_value < 0:
        notes.append("Net debt exceeds enterprise value: equity is a call option "
                     "on the assets and DCF per-share output is not meaningful.")
    if terminal_reinvest_rate >= 0.9:
        notes.append("Terminal reinvestment consumes nearly all perpetual profit; "
                     "terminal ROIC is barely above the cost of capital.")

    return DCFResult(
        enterprise_value=enterprise_value, equity_value=equity_value,
        value_per_share=value_per_share, pv_explicit=pv_explicit,
        pv_terminal=pv_terminal, terminal_value=terminal_value,
        terminal_share=terminal_share, projection=projection,
        assumptions=a, notes=notes,
    )


# --------------------------------------------------------------------------- #
# deriving assumptions from filed history
# --------------------------------------------------------------------------- #
def auto_assumptions(
    f: Fundamentals,
    metrics: dict,
    coc: CostOfCapital,
    *,
    years: int = 10,
    growth_override: float | None = None,
    margin_override: float | None = None,
    terminal_growth: float | None = None,
) -> DCFAssumptions:
    """Derive a defensible starting point from the company's own history.

    These are *starting* assumptions, not forecasts. The estimates are
    deliberately shrunk toward the mean:

    * Growth blends the 3-year CAGR with the log-trend fit and is then pulled
      toward nominal GDP, because high growth decays. The empirical half-life
      of excess revenue growth is roughly five years.
    * The margin target blends the current margin with a five-year average,
      since margins mean-revert harder than almost any other series.
    * Terminal ROIC fades halfway from today's ROIC to the cost of capital:
      competition erodes excess returns, and assuming a 90% ROIC persists in
      perpetuity is the single most common way to produce a silly valuation.
    """
    cfg = settings()
    tg = terminal_growth if terminal_growth is not None else terminal_growth_cap(coc.risk_free)

    revenue = metrics.get("revenue")
    if not is_num(revenue):
        revenue = f.get("revenue")

    # --- growth ------------------------------------------------------------
    if growth_override is not None:
        g0 = float(growth_override)
    else:
        candidates = [metrics.get("revenue_cagr_3y"), metrics.get("revenue_trend_growth"),
                      metrics.get("revenue_cagr_5y")]
        historical = mean_finite([c for c in candidates if is_num(c)])
        if not is_num(historical):
            historical = tg
        # Shrink 35% toward terminal growth: a decade of persistence is rare.
        g0 = 0.65 * historical + 0.35 * tg
        # Hard cap. A business can grow 60% for a year; projecting it as the
        # starting point of a ten-year fade still compounds to absurdity.
        g0 = clamp(g0, -0.10, 0.35)

    # --- margins -----------------------------------------------------------
    current_margin = metrics.get("operating_margin")
    avg_margin = metrics.get("operating_margin_5y_avg")
    if margin_override is not None:
        target_margin = float(margin_override)
    else:
        if is_num(current_margin) and is_num(avg_margin):
            target_margin = 0.5 * current_margin + 0.5 * avg_margin
        else:
            target_margin = current_margin if is_num(current_margin) else avg_margin
        if is_num(target_margin):
            target_margin = clamp(target_margin, -0.05, 0.55)
    base_margin = current_margin if is_num(current_margin) else target_margin

    # --- reinvestment efficiency ------------------------------------------
    s2c = _sales_to_capital(f, metrics)

    # --- terminal ROIC -----------------------------------------------------
    roic_now = metrics.get("roic_5y_avg")
    if not is_num(roic_now):
        roic_now = metrics.get("roic")
    if is_num(roic_now) and roic_now > coc.wacc:
        terminal_roic = coc.wacc + 0.5 * (min(roic_now, 0.60) - coc.wacc)
    else:
        terminal_roic = coc.wacc

    shares = metrics.get("shares_outstanding")
    if not is_num(shares):
        shares = metrics.get("shares")

    return DCFAssumptions(
        revenue_base=revenue if is_num(revenue) else NAN,
        revenue_growth_initial=g0,
        revenue_growth_terminal=tg,
        operating_margin_base=base_margin if is_num(base_margin) else NAN,
        operating_margin_target=target_margin if is_num(target_margin) else NAN,
        sales_to_capital=s2c,
        tax_rate=clamp(metrics.get("effective_tax_rate", cfg.default_tax_rate), 0.0, 0.45),
        wacc=coc.wacc,
        terminal_growth=tg,
        terminal_roic=terminal_roic,
        years=years,
        net_debt=metrics.get("net_debt", 0.0) if is_num(metrics.get("net_debt")) else 0.0,
        shares=shares if is_num(shares) else NAN,
    )


def _sales_to_capital(f: Fundamentals, metrics: dict) -> float:
    """Revenue per dollar of invested capital: the reinvestment multiplier.

    Measured from history rather than assumed, because it varies by an order of
    magnitude across industries -- a software company turns capital 4x, a
    utility 0.3x -- and it is what converts a growth assumption into a cash
    cost. Falls back to asset turnover when invested capital is unusable.
    """
    revenue = metrics.get("revenue")
    ic = metrics.get("invested_capital")
    value = safe_div(revenue, ic)
    if is_num(value) and 0.1 <= value <= 20.0:
        return float(value)
    turnover = metrics.get("asset_turnover")
    if is_num(turnover) and turnover > 0:
        return float(clamp(turnover, 0.1, 20.0))
    return NAN


# --------------------------------------------------------------------------- #
# sensitivity
# --------------------------------------------------------------------------- #
def sensitivity(a: DCFAssumptions, x_field: str, x_values, y_field: str, y_values
                ) -> pd.DataFrame:
    """Per-share value across a grid of two assumptions.

    The honest output of a DCF is this table, not the single number in its
    middle cell: it shows how much of the "answer" is the model and how much
    is the input you guessed.
    """
    grid = []
    for y in y_values:
        row = {}
        for x in x_values:
            trial = replace(a, **{x_field: float(x), y_field: float(y)})
            row[round(float(x), 4)] = run_dcf(trial).value_per_share
        grid.append(pd.Series(row, name=round(float(y), 4)))
    out = pd.DataFrame(grid)
    out.index.name = y_field
    out.columns.name = x_field
    return out


def standard_sensitivity(a: DCFAssumptions) -> pd.DataFrame:
    """WACC (+/-200bp) against terminal growth (+/-100bp)."""
    waccs = [a.wacc + d for d in (-0.02, -0.01, 0.0, 0.01, 0.02)]
    growths = [max(0.0, min(a.terminal_growth + d, min(w for w in waccs) - 0.005))
               for d in (-0.01, -0.005, 0.0, 0.005, 0.01)]
    return sensitivity(a, "wacc", waccs, "terminal_growth", sorted(set(growths)))


def scenario_set(a: DCFAssumptions) -> pd.DataFrame:
    """Bear / base / bull built by moving growth, margin and WACC together.

    Scenarios move the drivers *jointly* because they are correlated in
    reality: the world where growth disappoints is also the world where margins
    compress and the discount rate rises. Flexing one at a time understates
    the spread of outcomes.
    """
    scenarios = {
        "bear": {"revenue_growth_initial": a.revenue_growth_initial - 0.05,
                 "operating_margin_target": a.operating_margin_target * 0.80,
                 "wacc": a.wacc + 0.015},
        "base": {},
        "bull": {"revenue_growth_initial": a.revenue_growth_initial + 0.04,
                 "operating_margin_target": a.operating_margin_target * 1.15,
                 "wacc": max(a.wacc - 0.01, a.terminal_growth + 0.01)},
    }
    rows = []
    for name, overrides in scenarios.items():
        trial = replace(a, **overrides) if overrides else a
        result = run_dcf(trial)
        rows.append({
            "scenario": name,
            "growth_initial": trial.revenue_growth_initial,
            "margin_target": trial.operating_margin_target,
            "wacc": trial.wacc,
            "value_per_share": result.value_per_share,
            "terminal_share": result.terminal_share,
        })
    return pd.DataFrame(rows).set_index("scenario")
