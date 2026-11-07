"""Blend the valuation models into a range, with honest uncertainty.

A single "fair value" number is the least useful output a valuation can
produce, because it hides the two things that actually matter: how wide the
plausible range is, and whether the models agree.  So this module reports:

* a **range** (not a point) built from the models that apply to this business;
* **model dispersion** -- when DCF, EPV and residual income disagree by 3x,
  that disagreement is the finding, and the blended midpoint is noise;
* the **implied equity risk premium**, which is the one number here that needs
  no assumption about what equities should return. It is the IRR the market
  price offers on the projected cash flows, minus the risk-free rate. "You are
  being paid 0.4% to own this instead of a Treasury" is a complete argument.

Models are weighted by whether they fit the business, not by preference. A
dividend model gets zero weight on a company that does buybacks; a FCFF-based
DCF gets zero weight on a bank.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..data.providers.sec import Fundamentals
from ..util.mathx import NAN, is_num, safe_div
from . import ddm, graham, multiples, residual_income
from .dcf import DCFAssumptions, auto_assumptions, run_dcf, scenario_set
from .epv import earnings_power_value
from .reverse_dcf import analyse as reverse_analyse
from .wacc import CostOfCapital, estimate_cost_of_capital


@dataclass
class ModelEstimate:
    name: str
    value_per_share: float
    weight: float
    applicable: bool
    rationale: str

    @property
    def upside(self) -> float:
        return NAN


@dataclass
class ValuationReport:
    symbol: str
    price: float
    estimates: list[ModelEstimate]
    fair_value: float
    low: float
    high: float
    margin_of_safety: float
    dispersion: float             # (high - low) / midpoint: model disagreement
    confidence: str
    implied_return: float
    implied_equity_premium: float
    risk_free: float
    cost_of_capital: CostOfCapital
    reverse: object
    scenarios: pd.DataFrame
    relative: object
    notes: list[str] = field(default_factory=list)
    # Which model produced ``implied_return``: a reverse DCF for operating
    # companies, residual income for financials and REITs, "" when neither
    # could be solved. ``is_floor`` marks a rate that is a ceiling on the
    # return rather than an estimate of it -- the share is dearer than the
    # model can express.
    implied_return_method: str = ""
    implied_return_is_floor: bool = False

    @property
    def applicable_models(self) -> list[ModelEstimate]:
        return [e for e in self.estimates if e.applicable and is_num(e.value_per_share)]

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for est in self.estimates:
            rows.append({
                "model": est.name,
                "value_per_share": est.value_per_share,
                "upside": safe_div(est.value_per_share, self.price) - 1.0
                if is_num(est.value_per_share) else NAN,
                "weight": est.weight if est.applicable else 0.0,
                "applicable": est.applicable,
                "rationale": est.rationale,
            })
        return pd.DataFrame(rows)


# Base weights. DCF gets the most because it is the only model that handles
# growth explicitly; EPV is next because it is the hardest to fool. Graham's
# number is a sanity floor, not a valuation, hence its small weight.
# How far from the median of the applicable models an estimate may sit before
# it is treated as a broken model rather than a dissenting opinion.
OUTLIER_FACTOR = 5.0

BASE_WEIGHTS = {
    "DCF (FCFF, 10y fade)": 0.30,
    "Earnings Power Value": 0.22,
    "Residual income": 0.18,
    "Multiples (own history)": 0.20,
    "Dividend discount": 0.06,
    "Graham number": 0.04,
}

# Three of the six models are anchored on book value or on the assumption of
# no growth. For an asset-light business earning far more than its cost of
# capital, they are not conservative -- they are measuring the wrong thing,
# and averaging them in at a fixed weight marks the company down by
# construction. Walmart came out at $32.70 against a $125.52 share (earnings
# power $9, book-anchored residual income $21) and Costco at $346 against
# $1,053. The effect was systematic: every recognisable compounder failed the
# "below fair value" test, so the lists filled with asset-heavy cyclicals and
# obscure small caps instead.
#
# Greenwald's own framing settles it. Earnings power value is the value of a
# business with no franchise. Where returns on capital persistently exceed the
# cost of capital there IS a franchise, and EPV is a floor, not an estimate.
BOOK_ANCHORED = ("Earnings Power Value", "Residual income", "Graham number")
FRANCHISE_ROIC_MULTIPLE = 1.5     # returns this far above the cost of capital
FRANCHISE_WEIGHT_FLOOR = 0.25     # is what the book-anchored models keep


def value(
    f: Fundamentals,
    metrics: dict,
    price: float,
    *,
    coc: CostOfCapital | None = None,
    assumptions: DCFAssumptions | None = None,
    years: int = 10,
    include_relative: bool = True,
) -> ValuationReport:
    """Run every applicable model and assemble the range."""
    notes: list[str] = []
    estimates: list[ModelEstimate] = []

    coc = coc or estimate_cost_of_capital(f, metrics.get("market_cap", NAN), metrics)
    assumptions = assumptions or auto_assumptions(f, metrics, coc, years=years)
    risk_free = coc.risk_free

    # ---- DCF --------------------------------------------------------------
    dcf_result = run_dcf(assumptions)
    # A REIT's depreciation is a charge against property that usually
    # appreciates, and its "capex" is mostly acquisition of new assets rather
    # than maintenance of existing ones. FCFF therefore understates a REIT
    # badly -- it valued a healthy triple-net trust at a quarter of its price --
    # so the model is excluded rather than down-weighted.
    dcf_applicable = (not f.is_financial and not f.is_reit
                      and is_num(dcf_result.value_per_share))
    if f.is_financial:
        dcf_reason = ("not applicable to financials: debt is raw material, so "
                      "FCFF and net debt have no meaning")
    elif f.is_reit:
        dcf_reason = ("not applicable to REITs: non-cash depreciation and "
                      "acquisition-led capex make FCFF understate value; use "
                      "FFO-based measures")
    elif not is_num(dcf_result.value_per_share):
        dcf_reason = "could not be computed"
    else:
        dcf_reason = "free cash flow to the firm, growth fading to terminal"
    estimates.append(ModelEstimate(
        "DCF (FCFF, 10y fade)", dcf_result.value_per_share,
        BASE_WEIGHTS["DCF (FCFF, 10y fade)"], dcf_applicable, dcf_reason,
    ))
    notes.extend(dcf_result.notes)

    # ---- EPV --------------------------------------------------------------
    epv_result = earnings_power_value(f, metrics, coc.wacc)
    epv_applicable = (not f.is_financial and not f.is_reit
                      and is_num(epv_result.epv_per_share))
    estimates.append(ModelEstimate(
        "Earnings Power Value", epv_result.epv_per_share,
        BASE_WEIGHTS["Earnings Power Value"], epv_applicable,
        f"no-growth earnings capitalised at {coc.wacc:.1%}; franchise ratio "
        f"{epv_result.franchise_ratio:.2f}x" if epv_applicable
        else "not applicable to financials" if f.is_financial
        else "not applicable to REITs (depreciation distorts normalised margins)"
        if f.is_reit else "insufficient margin history",
    ))

    # ---- residual income --------------------------------------------------
    ri_result = residual_income.from_metrics(metrics, coc.cost_of_equity)
    ri_applicable = is_num(ri_result.value_per_share)
    ri_weight = BASE_WEIGHTS["Residual income"]
    if f.is_financial and ri_applicable:
        # For banks this is the primary model, not a cross-check.
        ri_weight = 0.50
    estimates.append(ModelEstimate(
        "Residual income", ri_result.value_per_share, ri_weight, ri_applicable,
        f"book value plus excess returns; {ri_result.book_value_share:.0%} of "
        f"value is book" if ri_applicable else "negative or missing book value",
    ))

    # ---- relative ---------------------------------------------------------
    relative = None
    rel_value = NAN
    if include_relative:
        relative = multiples.evaluate(f, metrics, price)
        rel_value = relative.implied_price_blend
        notes.extend(relative.notes)
    rel_applicable = is_num(rel_value)
    estimates.append(ModelEstimate(
        "Multiples (own history)", rel_value,
        BASE_WEIGHTS["Multiples (own history)"], rel_applicable,
        f"price at the median of its own multiples "
        f"({relative.composite_percentile:.0%} percentile today)"
        if rel_applicable and relative else "insufficient multiple history",
    ))

    # ---- dividend discount ------------------------------------------------
    ddm_result = ddm.from_metrics(metrics, coc.cost_of_equity, is_reit=f.is_reit)
    ddm_applicable = is_num(ddm_result.value_per_share)
    ddm_weight = BASE_WEIGHTS["Dividend discount"]
    if f.is_reit and ddm_applicable:
        # For a REIT the distribution *is* the return, so the dividend model
        # becomes the primary one rather than a cross-check.
        ddm_weight = 0.45
    estimates.append(ModelEstimate(
        "Dividend discount", ddm_result.value_per_share, ddm_weight, ddm_applicable,
        ddm_result.model if ddm_applicable
        else (ddm_result.notes[0] if ddm_result.notes else "not applicable"),
    ))

    # ---- Graham -----------------------------------------------------------
    graham_result = graham.evaluate(f, metrics, price)
    graham_applicable = is_num(graham_result.graham_number)
    estimates.append(ModelEstimate(
        "Graham number", graham_result.graham_number,
        BASE_WEIGHTS["Graham number"], graham_applicable,
        "sqrt(22.5 x EPS x BVPS): a conservative floor, not a target"
        if graham_applicable else "requires positive EPS and book value",
    ))

    # ---- franchise correction --------------------------------------------
    # Where a business persistently earns well above its cost of capital, the
    # book-anchored models are a floor rather than a valuation, and their full
    # weight drags the blend to a figure nobody would defend. Their weight is
    # cut and what is freed goes to the models that can see a franchise.
    roic_now = metrics.get("roic_5y_avg")
    if not is_num(roic_now):
        roic_now = metrics.get("roic")
    hurdle = coc.wacc if is_num(coc.wacc) and coc.wacc > 0 else NAN
    franchise = (is_num(roic_now) and is_num(hurdle)
                 and roic_now > hurdle * FRANCHISE_ROIC_MULTIPLE)
    if franchise:
        freed = 0.0
        for est in estimates:
            if est.name in BOOK_ANCHORED and est.applicable:
                freed += est.weight * (1.0 - FRANCHISE_WEIGHT_FLOOR)
                est.weight *= FRANCHISE_WEIGHT_FLOOR
        takers = [e for e in estimates
                  if e.applicable and e.name not in BOOK_ANCHORED
                  and is_num(e.value_per_share) and e.value_per_share > 0]
        if takers and freed > 0:
            share = sum(e.weight for e in takers)
            for est in takers:
                est.weight += freed * (est.weight / share if share else 1 / len(takers))
            notes.append(
                f"Returns on capital of {roic_now:.0%} run well above the "
                f"{hurdle:.1%} cost of capital, so this business has a franchise. "
                "The book-value and no-growth models are treated as a floor and "
                "carry a quarter of their usual weight."
            )

    # ---- blend ------------------------------------------------------------
    usable = [e for e in estimates if e.applicable and is_num(e.value_per_share)
              and e.value_per_share > 0]

    # A model whose answer is orders of magnitude away from the others is not
    # contributing to the blend, it is hijacking it. Charter's five estimates
    # ran from $39.91 to $23,127.64 -- a 580-fold spread for a $118 share --
    # and averaging them produced a "fair value" of $5,728 and a margin of
    # safety of 4,751%. The median is robust to exactly this, so estimates
    # more than a factor of OUTLIER_FACTOR from it are set aside and named.
    if len(usable) >= 3:
        ordered = sorted(e.value_per_share for e in usable)
        mid = ordered[len(ordered) // 2] if len(ordered) % 2 else \
            (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2.0
        kept = [e for e in usable
                if mid / OUTLIER_FACTOR <= e.value_per_share <= mid * OUTLIER_FACTOR]
        if kept and len(kept) < len(usable):
            for e in usable:
                if e not in kept:
                    notes.append(
                        f"{e.name} valued it at {e.value_per_share:,.2f} against a "
                        f"median of {mid:,.2f} across the models that applied; that "
                        "is too far out to average in and was set aside."
                    )
            usable = kept

    if usable:
        total_weight = sum(e.weight for e in usable)
        fair = sum(e.value_per_share * e.weight for e in usable) / total_weight
        values = [e.value_per_share for e in usable]
        low, high = float(min(values)), float(max(values))
        midpoint = (low + high) / 2.0
        dispersion = safe_div(high - low, midpoint)
    else:
        fair = low = high = dispersion = NAN

    # One model is not a valuation. Publishing a single uncorroborated figure
    # is how Universe Pharmaceuticals came out "51,122% below fair value" on
    # an earnings-power estimate of $71,061 for a $1.39 share.
    if len(usable) < 2:
        if usable:
            notes.append(
                f"Only one valuation model applied ({usable[0].name}), so nothing "
                "corroborates it and no fair value is published."
            )
        fair = low = high = dispersion = NAN

    # ---- reverse DCF and the premium being offered ------------------------
    reverse = (reverse_analyse(assumptions, price, metrics)
               if not (f.is_financial or f.is_reit) else None)
    implied_return = reverse.implied_return if reverse is not None else NAN
    implied_method = "reverse DCF" if is_num(implied_return) else ""
    implied_is_floor = False
    if not is_num(implied_return):
        # Banks, insurers and REITs get no reverse DCF, which left three
        # quarters of the financial sector with no expected return at all
        # while every operating company had one. Invert whichever model is the
        # primary one for them instead: residual income for a financial, the
        # dividend model for a REIT, where the distribution *is* the return.
        # Inverting the wrong one is worse than inverting none -- residual
        # income on a REIT says it is dear while the blend says it is cheap,
        # because depreciation guts a REIT's book-based return on equity.
        if f.is_reit and ddm_applicable:
            solved, implied_is_floor = residual_income.solve_implied_rate(
                lambda rate: ddm.from_metrics(metrics, rate, is_reit=True).value_per_share,
                price)
            if is_num(solved):
                implied_return, implied_method = solved, "dividend discount"
        elif ri_applicable:
            solved, implied_is_floor = residual_income.implied_cost_of_equity(metrics, price)
            if is_num(solved):
                implied_return, implied_method = solved, "residual income"
    implied_premium = (implied_return - risk_free) if is_num(implied_return) else NAN

    # ---- confidence -------------------------------------------------------
    n_models = len(usable)
    if n_models == 0:
        confidence = "none"
    elif not is_num(dispersion):
        confidence = "low"
    elif dispersion > 1.2 or n_models < 3:
        confidence = "low"
    elif dispersion > 0.6:
        confidence = "moderate"
    else:
        confidence = "reasonable"

    # Where the models disagree six- or eight-fold there is no estimate, only
    # an average of incompatible guesses, and publishing it invites exactly
    # the reliance it cannot bear: Walmart at $43 against a $126 share,
    # Mastercard at $204 against $653. The confidence field already knew; it
    # simply was not acted on. Withholding here means a company we cannot
    # value never clears the "not above fair value" condition -- which is the
    # honest outcome, and the long-standing discipline of not buying what you
    # cannot value.
    if confidence in ("low", "none"):
        notes.append(
            f"The valuation methods ranged from {low:,.2f} to {high:,.2f}, too "
            "wide a spread to publish a single figure, so no fair value is "
            "given for this company."
        )
        fair = NAN


    if is_num(dispersion) and dispersion > 1.0:
        notes.append(
            f"Models span {low:,.2f} to {high:,.2f} -- a {dispersion:.0%} spread. "
            "The disagreement is the result; the blended midpoint should not be "
            "treated as a target price."
        )
    if f.is_reit:
        notes.append(
            f"REIT: valued on FFO and distributions. P/FFO is "
            f"{metrics.get('p_ffo', float('nan')):.1f}x and the payout is "
            f"{metrics.get('ffo_payout_ratio', float('nan')):.0%} of FFO. "
            "FFO here excludes the NAREIT adjustment for property-sale gains, "
            "which are not reliably tagged."
        )
    if metrics.get("years_of_data", 0) < 5:
        notes.append("Under five years of filing history: normalisation and "
                     "trend measures are weak.")
    if is_num(implied_premium) and implied_premium < 0.01:
        notes.append(
            f"At this price the implied equity risk premium is "
            f"{implied_premium:.2%} over the {risk_free:.2%} risk-free rate -- "
            "little compensation for equity risk, on these cash-flow assumptions."
        )

    return ValuationReport(
        symbol=f.symbol, price=price, estimates=estimates,
        fair_value=fair, low=low, high=high,
        margin_of_safety=safe_div(fair, price) - 1.0 if is_num(fair) else NAN,
        dispersion=dispersion, confidence=confidence,
        implied_return=implied_return, implied_equity_premium=implied_premium,
        implied_return_method=implied_method, implied_return_is_floor=implied_is_floor,
        risk_free=risk_free, cost_of_capital=coc, reverse=reverse,
        scenarios=(scenario_set(assumptions)
                   if not (f.is_financial or f.is_reit) else pd.DataFrame()),
        relative=relative, notes=notes,
    )
