"""Residual income (Edwards-Bell-Ohlson): book value plus excess returns.

The structural advantage over a DCF is that most of the value sits in the
*first* term -- today's book value, which is observed, not forecast.  A DCF
routinely puts 60-80% of its answer in a terminal value ten years out; a
residual income model usually puts 50-70% in a number on the balance sheet.
The forecast matters less, so the answer is more stable.

This makes it the right model for **financials**, where the FCFF framework
breaks down entirely: a bank's debt is its raw material, so "free cash flow to
the firm" and "net debt" have no meaning, but book value and ROE are precisely
the terms in which banks are actually analysed and managed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..util.mathx import NAN, clamp, is_num, safe_div


@dataclass
class ResidualIncomeResult:
    value_per_share: float
    book_value_per_share: float
    pv_residual_income: float
    continuing_value: float
    book_value_share: float        # fraction of value already on the balance sheet
    projection: pd.DataFrame
    notes: list[str] = field(default_factory=list)

    def upside_vs(self, price: float) -> float:
        return safe_div(self.value_per_share, price) - 1.0


def residual_income_value(
    book_value_per_share: float,
    roe_initial: float,
    cost_of_equity: float,
    *,
    years: int = 10,
    payout_ratio: float = 0.35,
    terminal_roe: float | None = None,
    persistence: float = 0.60,
) -> ResidualIncomeResult:
    """Value equity as book value plus the present value of excess returns.

    ``persistence`` governs how fast the final year's residual income decays
    once the explicit forecast ends: 0 means competition eliminates excess
    returns immediately, 1 means they last forever. The 0.6 default implies a
    roughly two-year half-life, consistent with the empirical speed at which
    abnormal ROE mean-reverts.

    ``terminal_roe`` defaults to the cost of equity, i.e. excess returns are
    fully competed away by the end of the forecast. That is the neutral
    assumption and it should be moved only for a business with an identifiable
    barrier to entry.
    """
    notes: list[str] = []
    if not all(is_num(x) for x in (book_value_per_share, roe_initial, cost_of_equity)):
        return ResidualIncomeResult(NAN, NAN, NAN, NAN, NAN, pd.DataFrame(),
                                    ["Missing book value, ROE or cost of equity."])
    if book_value_per_share <= 0:
        return ResidualIncomeResult(NAN, book_value_per_share, NAN, NAN, NAN,
                                    pd.DataFrame(),
                                    ["Negative book value: residual income is not "
                                     "applicable (buybacks or accumulated losses)."])
    if cost_of_equity <= 0:
        return ResidualIncomeResult(NAN, book_value_per_share, NAN, NAN, NAN,
                                    pd.DataFrame(), ["Cost of equity must be positive."])

    terminal = terminal_roe if terminal_roe is not None else cost_of_equity
    # Cap the starting ROE: a 90% ROE usually reflects a buyback-shrunken
    # equity base rather than earning power, and compounding it produces
    # nonsense.
    roe_start = clamp(float(roe_initial), -0.50, 0.60)
    if roe_start != roe_initial:
        notes.append(f"Starting ROE clamped from {roe_initial:.1%} to {roe_start:.1%}.")

    roe_path = np.linspace(roe_start, terminal, years + 1)[1:]
    retention = clamp(1.0 - payout_ratio, 0.0, 1.0)

    rows = []
    book = float(book_value_per_share)
    pv_total = 0.0
    last_ri = 0.0
    for t in range(1, years + 1):
        opening = book
        roe_t = float(roe_path[t - 1])
        earnings = opening * roe_t
        residual = (roe_t - cost_of_equity) * opening
        discount = (1.0 + cost_of_equity) ** t
        pv = residual / discount
        pv_total += pv
        # Clean-surplus accounting: closing book = opening + retained earnings.
        book = opening + earnings * retention
        last_ri = residual
        rows.append({"year": t, "opening_book": opening, "roe": roe_t,
                     "earnings": earnings, "residual_income": residual,
                     "pv_residual_income": pv, "closing_book": book})

    projection = pd.DataFrame(rows).set_index("year")

    # Continuing value: the final year's residual income decaying at
    # `persistence`, discounted back from the end of the forecast.
    omega = clamp(persistence, 0.0, 0.98)
    denom = 1.0 + cost_of_equity - omega
    continuing = (last_ri * omega / denom / (1.0 + cost_of_equity) ** years
                  if denom > 0 else 0.0)

    value = float(book_value_per_share) + pv_total + continuing
    return ResidualIncomeResult(
        value_per_share=value,
        book_value_per_share=float(book_value_per_share),
        pv_residual_income=pv_total, continuing_value=continuing,
        book_value_share=safe_div(float(book_value_per_share), value),
        projection=projection, notes=notes,
    )


def implied_cost_of_equity(metrics: dict, price: float, *,
                           low: float = 0.005, high: float = 1.0,
                           tol: float = 1e-4, max_iter: int = 80,
                           **kwargs) -> tuple[float, bool]:
    """The required return at which this model agrees with today's price.

    A reverse DCF asks "what must the business do to justify this price?" and
    answers in a rate. It cannot be run on a bank or an insurer: free cash flow
    to the firm is undefined when debt is the raw material, so those companies
    had no expected-return figure at all while every operating company did.

    The residual income model is the right model for them, and the same
    question can be put to it. Value falls monotonically as the required return
    rises, so the rate where model value meets market price is found by
    bisection. Subtracting the risk-free rate puts a bank on the same footing
    as everything else on the page.

    Returns ``(rate, is_floor)``. ``is_floor`` is True when the price is above
    what the model produces even at ``low`` -- the share is dearer than the
    model can express, and the rate returned is a ceiling on the return, not an
    estimate of it. ``(nan, False)`` when the model cannot be solved at all.
    """
    def value_at(rate: float) -> float:
        try:
            return from_metrics(metrics, rate, **kwargs).value_per_share
        except Exception:
            return NAN

    return solve_implied_rate(value_at, price, low=low, high=high,
                              tol=tol, max_iter=max_iter)


def solve_implied_rate(value_at, price: float, *, low: float = 0.005,
                       high: float = 1.0, tol: float = 1e-4,
                       max_iter: int = 80) -> tuple[float, bool]:
    """Bisect a monotonically decreasing value function for the rate at price.

    ``value_at`` maps a required return to a per-share value. Every discounted
    model falls as the discount rate rises, so one root exists wherever the
    price is bracketed. Shared by the residual income and dividend models so
    both report an expected return on the same basis.
    """
    if not is_num(price) or price <= 0:
        return NAN, False

    lo, hi = float(low), float(high)
    v_lo, v_hi = value_at(lo), value_at(hi)
    if not is_num(v_lo):
        # A Gordon-style model divides by (rate - growth) and blows up below
        # the terminal growth rate, so the nominal floor can be unusable.
        # Walk up to the lowest rate the model will actually answer at.
        for step in range(1, 120):
            probe = float(low) + step * 0.005
            if probe >= hi:
                break
            candidate = value_at(probe)
            if is_num(candidate) and candidate > 0:
                lo, v_lo = probe, candidate
                break
    if not is_num(v_lo) or not is_num(v_hi):
        return NAN, False
    if v_lo < price:
        # Even at a near-zero required return the model says it is worth less
        # than this. Report the boundary and say so rather than invent a rate.
        return lo, True
    if v_hi > price:
        # Worth more than the price at a 100% discount rate: book value or ROE
        # is not believable, not a bargain.
        return NAN, False

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        v = value_at(mid)
        if not is_num(v):
            return NAN, False
        if abs(v - price) <= tol * price:
            return mid, False
        if v > price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0, False


def from_metrics(metrics: dict, cost_of_equity: float, **kwargs) -> ResidualIncomeResult:
    """Convenience wrapper driven by the standard metric dictionary."""
    bvps = metrics.get("book_value_per_share")
    roe = metrics.get("roe")
    payout = metrics.get("payout_ratio")
    if not is_num(payout) or not (0.0 <= payout <= 1.0):
        payout = 0.35
    return residual_income_value(bvps, roe, cost_of_equity,
                                 payout_ratio=float(payout), **kwargs)
