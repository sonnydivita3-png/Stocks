"""Benjamin Graham's screens, implemented with their original thresholds.

These are included because they still work as *filters* and because they are
transparent. They are not included as valuation models, and two deserve an
explicit warning:

* The **Graham formula** ``V = EPS x (8.5 + 2g) x 4.4/Y`` is the most misused
  equation in value investing. Graham published it as a description of how the
  market had historically priced growth, then explicitly warned against using
  it to value growth stocks. Plugging in a 25% growth rate produces a P/E of
  58, which was never the intent.
* **Net-net investing** worked spectacularly in the 1930s-50s and still shows
  excess returns in backtests, but the surviving candidates today are
  overwhelmingly Chinese reverse-mergers, biotech shells and companies whose
  current assets are about to be consumed by losses. The screen finds them;
  it cannot tell you which is which.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data.providers.sec import Fundamentals
from ..util.mathx import NAN, clamp, is_num, safe_div, zero_if_nan


@dataclass
class GrahamResult:
    graham_number: float
    ncav_per_share: float
    net_net_price: float            # two-thirds of NCAV: Graham's buy level
    is_net_net: bool
    graham_formula_value: float
    defensive_criteria: dict[str, bool]
    defensive_score: float
    margin_of_safety: float
    notes: list[str] = field(default_factory=list)


def graham_number(eps: float, book_value_per_share: float) -> float:
    """``sqrt(22.5 x EPS x BVPS)``.

    The 22.5 is Graham's twin caps multiplied: P/E no higher than 15 and P/B no
    higher than 1.5. Undefined for loss-makers or negative book value, which is
    correct -- the rule simply does not apply to them.
    """
    if not (is_num(eps) and is_num(book_value_per_share)):
        return NAN
    if eps <= 0 or book_value_per_share <= 0:
        return NAN
    return float(np.sqrt(22.5 * eps * book_value_per_share))


def net_current_asset_value(f: Fundamentals, shares: float) -> float:
    """``(current assets - total liabilities) / shares``.

    Gives zero credit to every fixed asset: property, plant, brands, patents.
    A liquidation floor by construction.
    """
    ca = f.get("current_assets")
    tl = f.get("total_liabilities")
    if not (is_num(ca) and is_num(tl)):
        return NAN
    preferred = zero_if_nan(f.get("preferred_equity"))
    return safe_div(ca - tl - preferred, shares)


def graham_formula(eps: float, growth_pct: float, aaa_yield_pct: float = 4.4,
                   growth_cap: float = 15.0) -> float:
    """``EPS x (8.5 + 2g) x 4.4 / Y``, with growth capped.

    ``growth_pct`` and ``aaa_yield_pct`` are percentages, not decimals (so 7.5
    means 7.5%). Growth is capped at 15% by default because the formula is
    linear in growth and becomes nonsense without a bound.
    """
    if not is_num(eps) or eps <= 0 or not is_num(growth_pct):
        return NAN
    g = clamp(growth_pct, 0.0, growth_cap)
    y = aaa_yield_pct if is_num(aaa_yield_pct) and aaa_yield_pct > 0 else 4.4
    return float(eps * (8.5 + 2.0 * g) * 4.4 / y)


def defensive_criteria(f: Fundamentals, metrics: dict, price: float) -> dict[str, bool]:
    """Graham's seven tests for the defensive investor (*The Intelligent Investor*).

    Thresholds are his, in nominal terms where he used them; the $100m revenue
    floor replaces his $100m-of-sales test, which in 1973 dollars screened for
    a far larger company than it does now.
    """
    out: dict[str, bool] = {}
    revenue = metrics.get("revenue")
    if is_num(revenue):
        out["adequate_size"] = revenue >= 2e9
    cr = metrics.get("current_ratio")
    if is_num(cr):
        out["current_ratio_over_2"] = cr >= 2.0
    ltd = f.get("long_term_debt")
    wc = (f.get("current_assets") - f.get("current_liabilities")
          if is_num(f.get("current_assets")) and is_num(f.get("current_liabilities")) else NAN)
    if is_num(ltd) and is_num(wc) and wc > 0:
        out["debt_below_working_capital"] = ltd <= wc

    earnings = f.annual_series("net_income", 10)
    if len(earnings) >= 5:
        out["earnings_stability_10y"] = bool((earnings > 0).all())
    divs = f.annual_series("dividends_paid", 10)
    if len(divs) >= 5:
        out["dividend_record"] = bool((divs > 0).all())
    eps_growth = metrics.get("eps_cagr_5y")
    if is_num(eps_growth):
        # Graham wanted a third over a decade: roughly 2.9% a year.
        out["earnings_growth_33pct_10y"] = eps_growth >= 0.029
    pe = metrics.get("pe_5y_avg_earnings")
    if is_num(pe):
        out["pe_under_15"] = pe <= 15.0
    pb = metrics.get("pb")
    if is_num(pb) and is_num(pe):
        # His combined test: P/E x P/B under 22.5.
        out["pe_times_pb_under_22_5"] = (pe * pb) <= 22.5
    return out


def evaluate(f: Fundamentals, metrics: dict, price: float,
             aaa_yield_pct: float | None = None) -> GrahamResult:
    notes: list[str] = []
    shares = metrics.get("shares_outstanding") or metrics.get("shares")
    eps = metrics.get("eps_ttm")
    bvps = metrics.get("book_value_per_share")

    gn = graham_number(eps, bvps)
    ncav = net_current_asset_value(f, shares)
    net_net_price = ncav * (2.0 / 3.0) if is_num(ncav) else NAN
    is_net_net = bool(is_num(net_net_price) and is_num(price) and price < net_net_price)

    growth = metrics.get("eps_cagr_5y")
    if not is_num(growth):
        growth = metrics.get("earnings_cagr_5y")
    gf = graham_formula(eps, growth * 100.0 if is_num(growth) else NAN,
                        aaa_yield_pct if aaa_yield_pct is not None else 4.4)

    criteria = defensive_criteria(f, metrics, price)
    score = (sum(criteria.values()) / len(criteria) * 7.0) if criteria else NAN

    mos = safe_div(gn, price) - 1.0 if is_num(gn) else NAN

    if is_net_net:
        notes.append(
            "Trades below two-thirds of net current asset value. Historically a "
            "powerful signal, but modern survivors of this screen are usually "
            "burning the very current assets being counted -- verify the cash "
            "burn rate before treating the floor as real."
        )
    if is_num(growth) and growth > 0.15:
        notes.append(
            "Growth exceeds the formula's 15% cap, so the Graham formula value "
            "shown is deliberately conservative. Graham warned against applying "
            "it to growth stocks at all."
        )
    if not is_num(gn):
        notes.append("Graham number undefined: requires positive EPS and book value.")
    return GrahamResult(
        graham_number=gn, ncav_per_share=ncav, net_net_price=net_net_price,
        is_net_net=is_net_net, graham_formula_value=gf,
        defensive_criteria=criteria, defensive_score=score,
        margin_of_safety=mos, notes=notes,
    )
