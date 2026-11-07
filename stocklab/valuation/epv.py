"""Greenwald's Earnings Power Value, and the asset reproduction cross-check.

EPV answers a narrower question than a DCF, and is therefore much harder to
fool: *what is this business worth if it never grows again?*  No growth
forecast, no terminal value, no fade schedule -- just normalised earnings
capitalised at the cost of capital.

Its real power is the comparison with **asset reproduction value**: what a
competitor would have to spend to rebuild the business from scratch.

* ``EPV > reproduction value`` means the business earns more than its assets
  should permit, which in a competitive market is only sustainable behind a
  barrier to entry. The gap *is* the franchise value.
* ``EPV < reproduction value`` means management is destroying value: the assets
  would be worth more deployed elsewhere, or liquidated.

That diagnostic is unavailable from a DCF, which assumes the moat it should be
testing for.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..data.providers.sec import Fundamentals
from ..util.mathx import NAN, clamp, is_num, mean_finite, safe_div, zero_if_nan


@dataclass
class EPVResult:
    epv_enterprise: float
    epv_equity: float
    epv_per_share: float
    normalized_ebit: float
    normalized_margin: float
    adjusted_earnings: float
    maintenance_capex: float
    reproduction_value: float
    reproduction_per_share: float
    franchise_value: float
    franchise_ratio: float          # EPV / reproduction value
    interpretation: str
    notes: list[str] = field(default_factory=list)

    def upside_vs(self, price: float) -> float:
        return safe_div(self.epv_per_share, price) - 1.0


def maintenance_capex(f: Fundamentals, metrics: dict) -> tuple[float, list[str]]:
    """Capex needed to stand still, separating out the growth portion.

    Bruce Greenwald's method: capex per dollar of sales, from history, applied
    to the *existing* sales base. Whatever the company spent above that was
    buying growth, and a no-growth valuation should not charge for it.

    Falls back to depreciation, which is the standard proxy and roughly right
    for a business in steady state but understates it under inflation.
    """
    notes: list[str] = []
    revenue = metrics.get("revenue")
    capex_hist = f.annual_series("capex", 6)
    rev_hist = f.annual_series("revenue", 6)

    common = capex_hist.index.intersection(rev_hist.index)
    if len(common) >= 3 and is_num(revenue):
        ratios = (capex_hist.loc[common] / rev_hist.loc[common]).replace(
            [float("inf"), float("-inf")], pd.NA
        ).dropna()
        prior_revenue = float(rev_hist.loc[common].iloc[-2]) if len(common) >= 2 else NAN
        if not ratios.empty and is_num(prior_revenue):
            capex_per_sales = float(ratios.median())
            # Growth capex funds the increase in sales; the rest is maintenance.
            delta_sales = max(float(revenue) - prior_revenue, 0.0)
            growth_capex = capex_per_sales * delta_sales
            total_capex = metrics.get("capex")
            if is_num(total_capex):
                maint = max(float(total_capex) - growth_capex, 0.0)
                return maint, notes

    da = f.get("depreciation_amortization")
    if is_num(da):
        notes.append("Maintenance capex proxied by D&A; understated when "
                     "replacement costs exceed historical cost.")
        return float(da), notes
    capex = metrics.get("capex")
    if is_num(capex):
        notes.append("Maintenance capex set to total capex (conservative).")
        return float(capex), notes
    return NAN, ["Maintenance capex could not be estimated."]


def reproduction_value(f: Fundamentals, metrics: dict,
                       capitalize_intangible_spend: bool = True) -> tuple[float, list[str]]:
    """What it would cost a competitor to rebuild the balance sheet.

    Starts from tangible equity (reported assets less goodwill, less all
    liabilities), then optionally adds back the intangible asset the accounts
    refuse to recognise: years of R&D and brand-building expensed as incurred.
    A pharmaceutical company's real reproduction cost is mostly its research
    base, and tangible book value puts that at zero.
    """
    notes: list[str] = []
    assets = f.get("total_assets")
    liabilities = f.get("total_liabilities")
    if not (is_num(assets) and is_num(liabilities)):
        return NAN, ["Balance sheet incomplete."]

    goodwill = zero_if_nan(f.get("goodwill"))
    value = assets - goodwill - liabilities
    if goodwill > 0:
        notes.append(f"Goodwill of {goodwill/1e9:.1f}bn excluded: it is a record "
                     "of prices paid, not an asset a rival must reproduce.")

    # PP&E is carried at historical cost less depreciation, not at what a rival
    # would pay to build it today. For anything capital-intensive -- refineries,
    # mines, networks, fabs -- that understates reproduction value, which biases
    # the franchise ratio upward. A high ratio for a commodity producer usually
    # means stale asset values, not a moat.
    ppe = f.get("ppe_net")
    if is_num(ppe) and is_num(assets) and assets > 0 and (ppe / assets) > 0.35:
        notes.append(
            "Capital-intensive balance sheet (PP&E is "
            f"{ppe / assets:.0%} of assets): reproduction value is understated "
            "because PP&E sits at historical cost, so the franchise ratio "
            "overstates any competitive advantage."
        )

    if capitalize_intangible_spend:
        # Three years of R&D and a year of selling costs, as a rough stand-in
        # for the intangible base. Crude, and deliberately so -- the point is to
        # avoid valuing a research-driven business at scrap.
        rnd = f.annual_series("rnd", 3)
        if not rnd.empty:
            rnd_capital = float(rnd.sum())
            value += rnd_capital
            notes.append(f"Added {rnd_capital/1e9:.1f}bn of capitalised R&D "
                         "(3 years) to reproduction cost.")
    return value, notes


def earnings_power_value(
    f: Fundamentals,
    metrics: dict,
    wacc: float,
    *,
    margin_years: int = 8,
    tax_rate: float | None = None,
) -> EPVResult:
    """Capitalise normalised, no-growth earnings at the cost of capital."""
    notes: list[str] = []
    revenue = metrics.get("revenue")
    shares = metrics.get("shares_outstanding") or metrics.get("shares")

    # Normalised margin across a full cycle -- the whole point is to avoid
    # capitalising a peak or a trough.
    margin_hist = []
    ebit_hist = f.annual_series("operating_income", margin_years)
    rev_hist = f.annual_series("revenue", margin_years)
    common = ebit_hist.index.intersection(rev_hist.index)
    if len(common) >= 3:
        margin_hist = (ebit_hist.loc[common] / rev_hist.loc[common]).dropna().tolist()
    normalized_margin = mean_finite(margin_hist) if margin_hist else metrics.get("operating_margin")
    if not margin_hist:
        notes.append("Fewer than three years of margin history; using the current "
                     "margin, which defeats the purpose of normalising.")
    if len(common) < margin_years:
        notes.append(f"Margin normalised over {len(common)} years, not {margin_years}; "
                     "a full cycle may not be covered.")

    if not (is_num(revenue) and is_num(normalized_margin) and is_num(wacc) and wacc > 0):
        return EPVResult(NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN,
                         "insufficient data", notes + ["Missing revenue, margin or WACC."])

    t = tax_rate if tax_rate is not None else metrics.get("effective_tax_rate")
    t = clamp(float(t), 0.0, 0.45) if is_num(t) else 0.23

    normalized_ebit = float(revenue) * float(normalized_margin)
    after_tax = normalized_ebit * (1.0 - t)

    maint, maint_notes = maintenance_capex(f, metrics)
    notes.extend(maint_notes)
    da = f.get("depreciation_amortization")
    # Earnings understate cash when depreciation exceeds the spend needed to
    # keep the asset base intact (and overstate it when the reverse is true).
    adjustment = (float(da) - maint) if (is_num(da) and is_num(maint)) else 0.0
    adjusted_earnings = after_tax + adjustment

    epv_ev = adjusted_earnings / wacc
    net_debt = metrics.get("net_debt")
    epv_equity = epv_ev - zero_if_nan(net_debt)
    epv_ps = safe_div(epv_equity, shares)

    repro, repro_notes = reproduction_value(f, metrics)
    notes.extend(repro_notes)
    franchise = (epv_ev - repro) if (is_num(epv_ev) and is_num(repro)) else NAN
    ratio = safe_div(epv_ev, repro) if (is_num(repro) and repro > 0) else NAN

    if not is_num(ratio):
        interpretation = "reproduction value unavailable"
    elif ratio > 1.5:
        interpretation = ("strong franchise: earnings far exceed what the asset base "
                          "should support, implying a real barrier to entry")
    elif ratio > 1.1:
        interpretation = "modest franchise: earnings somewhat above asset-based returns"
    elif ratio > 0.9:
        interpretation = ("commodity economics: earnings roughly equal to the cost of "
                          "reproducing the assets, as competition would predict")
    else:
        interpretation = ("value destruction: the assets would be worth more redeployed "
                          "or liquidated than they are earning in place")

    return EPVResult(
        epv_enterprise=epv_ev, epv_equity=epv_equity, epv_per_share=epv_ps,
        normalized_ebit=normalized_ebit, normalized_margin=float(normalized_margin),
        adjusted_earnings=adjusted_earnings, maintenance_capex=maint,
        reproduction_value=repro, reproduction_per_share=safe_div(repro, shares),
        franchise_value=franchise, franchise_ratio=ratio,
        interpretation=interpretation, notes=notes,
    )
