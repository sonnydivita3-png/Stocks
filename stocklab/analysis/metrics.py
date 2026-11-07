"""Derived fundamental metrics: profitability, growth, quality, safety, value.

Everything downstream -- screens, valuation models, entry scoring -- reads the
flat ``dict[str, float]`` produced by :func:`compute_metrics`.  Keeping it flat
and nan-tolerant is what lets the screener treat 3,000 companies as a matrix.

Two conventions are load-bearing:

* **Averaged denominators.** Returns on capital divide a flow (a year of
  profit) by a stock (capital at a point in time).  Using the closing balance
  flatters any company that grew its capital base during the year, which is
  most of them.  Beginning/ending averages are used wherever history allows.
* **Sector guards.** ROIC, EV/EBITDA and net debt are meaningless for banks
  (deposits are raw material, not leverage) and distorted for REITs
  (depreciation swamps real earnings).  Those metrics are suppressed rather
  than silently wrong -- see ``applicable`` in the output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import settings
from ..data.providers.sec import Fundamentals
from ..util.mathx import (
    NAN, cagr, first_finite, is_num, log_trend_growth, mean_finite, safe_div, sum_finite,
    zero_if_nan,
)


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
def effective_tax_rate(f: Fundamentals, default: float | None = None) -> float:
    """Cash-ish effective rate, clamped to a believable band.

    A single year's rate swings wildly on one-off items (a repatriation charge,
    a valuation-allowance release), so a three-year average is used and the
    result is clamped to 0-45%: a negative rate is a tax benefit that will not
    recur, and using it in a DCF capitalises a one-off into perpetuity.
    """
    default = settings().default_tax_rate if default is None else default
    pretax = f.annual_series("pretax_income", 3)
    tax = f.annual_series("tax_expense", 3)
    common = pretax.index.intersection(tax.index)
    if len(common) >= 2:
        total_pretax = float(pretax.loc[common].sum())
        total_tax = float(tax.loc[common].sum())
        if total_pretax > 0:
            rate = total_tax / total_pretax
            if 0.0 <= rate <= 0.45:
                return rate
    rate = safe_div(f.get("tax_expense"), f.get("pretax_income"))
    return rate if is_num(rate) and 0.0 <= rate <= 0.45 else default


def total_debt(f: Fundamentals, include_leases: bool = True) -> float:
    """Short-term + long-term debt, optionally plus capitalised operating leases.

    Post-ASC 842 operating lease liabilities are real, contractual, non-cancellable
    obligations; excluding them makes an asset-light retailer look unlevered next
    to a peer that owns its stores.  Included by default.
    """
    debt = sum_finite(f.get("short_term_debt"), f.get("long_term_debt"))
    if include_leases:
        lease = f.get("operating_lease_liability")
        if is_num(lease):
            debt = sum_finite(debt, lease)
    return debt


def liquid_assets(f: Fundamentals) -> float:
    return sum_finite(f.get("cash"), f.get("short_term_investments"))


def net_debt(f: Fundamentals, include_leases: bool = True) -> float:
    debt = total_debt(f, include_leases)
    cash = liquid_assets(f)
    if not is_num(debt):
        return NAN
    return debt - (cash if is_num(cash) else 0.0)


def ebitda(f: Fundamentals) -> float:
    """EBIT + D&A.

    Built up from operating income rather than down from net income: the
    bottom-up route sweeps in non-operating gains, which is how a company with
    a one-off asset sale ends up looking cheap on EV/EBITDA.
    """
    ebit = f.get("operating_income")
    da = f.get("depreciation_amortization")
    if not is_num(ebit):
        return NAN
    return ebit + (da if is_num(da) else 0.0)


def free_cash_flow(f: Fundamentals) -> float:
    cfo, capex = f.get("cfo"), f.get("capex")
    if not is_num(cfo):
        return NAN
    return cfo - (capex if is_num(capex) else 0.0)


def invested_capital(f: Fundamentals, at: pd.Timestamp | None = None) -> float:
    """Debt + equity - excess cash, from the financing side.

    The operating build-up (working capital + fixed assets) is theoretically
    cleaner but needs line items many filers do not tag.  Cash is netted off
    because idle cash earns no operating return and leaving it in understates
    ROIC for cash-rich companies.
    """
    if at is None:
        debt = total_debt(f)
        equity = f.get("equity")
        minority = f.get("minority_interest")
        cash = liquid_assets(f)
    else:
        row = f.annual.loc[at] if at in f.annual.index else None
        if row is None:
            return NAN
        debt = sum_finite(row.get("short_term_debt"), row.get("long_term_debt"),
                          row.get("operating_lease_liability"))
        equity = row.get("equity")
        minority = row.get("minority_interest")
        cash = sum_finite(row.get("cash"), row.get("short_term_investments"))
    if not is_num(equity):
        return NAN
    cap = (debt if is_num(debt) else 0.0) + equity + (minority if is_num(minority) else 0.0)
    cap -= cash if is_num(cash) else 0.0
    return cap if cap > 0 else NAN


def _avg_balance(f: Fundamentals, concept: str) -> float:
    """Average of the two most recent annual balances, else the latest."""
    series = f.annual_series(concept, 2)
    if len(series) >= 2:
        return float(series.mean())
    latest = f.get(concept)
    return latest if is_num(latest) else (float(series.iloc[-1]) if len(series) else NAN)


def _avg_invested_capital(f: Fundamentals) -> float:
    dates = list(f.annual.index[-2:]) if len(f.annual) >= 2 else []
    values = [invested_capital(f, at=d) for d in dates]
    values = [v for v in values if is_num(v)]
    current = invested_capital(f)
    if is_num(current):
        values.append(current)
    return mean_finite(values[-2:]) if values else NAN


# --------------------------------------------------------------------------- #
# growth
# --------------------------------------------------------------------------- #
def growth_block(f: Fundamentals, concept: str, prefix: str) -> dict[str, float]:
    """1y / 3y / 5y growth plus a log-trend fit and its r-squared.

    Endpoint CAGR over a fixed window is hostage to the two years it happens to
    touch -- a cyclical trough as the base year manufactures spectacular growth.
    The log-trend slope uses every observation, and its r-squared is reported as
    a *durability* measure: 0.95 is a compounder, 0.3 is noise with a direction.
    """
    series = f.annual_series(concept)
    out: dict[str, float] = {}
    if series.empty:
        return out

    latest = float(series.iloc[-1])
    out[f"{prefix}_latest"] = latest

    for years in (1, 3, 5, 10):
        if len(series) > years:
            out[f"{prefix}_cagr_{years}y"] = cagr(float(series.iloc[-1 - years]), latest, years)

    growth, r2 = log_trend_growth(series.iloc[-7:])
    out[f"{prefix}_trend_growth"] = growth
    out[f"{prefix}_trend_r2"] = r2

    # Year-over-year dispersion: a proxy for how cyclical the line item is.
    yoy = series.pct_change().dropna()
    if len(yoy) >= 3:
        out[f"{prefix}_yoy_vol"] = float(yoy.iloc[-7:].std())
        out[f"{prefix}_down_years"] = float((yoy.iloc[-7:] < 0).sum())
    return out


# --------------------------------------------------------------------------- #
# composite scores
# --------------------------------------------------------------------------- #
def piotroski_f_score(f: Fundamentals) -> tuple[float, dict[str, float]]:
    """Piotroski (2000) F-Score, 0-9. Nine binary accounting signals.

    Designed for *cheap* stocks: Piotroski's result was that within the highest
    book-to-market quintile, high-F names beat low-F names by ~7.5%/yr.  It is a
    fragility filter, not a growth screen -- expect a quality compounder to score
    6-7 simply because it cannot keep improving margins every year.
    """
    tests: dict[str, float] = {}
    a = f.annual
    if len(a) < 2:
        return NAN, tests

    cur, prev = a.iloc[-1], a.iloc[-2]

    def val(row, key):
        v = row.get(key)
        return float(v) if is_num(v) else NAN

    ni, ni_p = val(cur, "net_income"), val(prev, "net_income")
    cfo, cfo_p = val(cur, "cfo"), val(prev, "cfo")
    ta, ta_p = val(cur, "total_assets"), val(prev, "total_assets")
    ta_pp = val(a.iloc[-3], "total_assets") if len(a) >= 3 else NAN

    roa = safe_div(ni, mean_finite([ta, ta_p]))
    roa_p = safe_div(ni_p, mean_finite([ta_p, ta_pp]))

    # --- profitability ---
    tests["roa_positive"] = float(is_num(roa) and roa > 0)
    tests["cfo_positive"] = float(is_num(cfo) and cfo > 0)
    tests["roa_improving"] = float(is_num(roa) and is_num(roa_p) and roa > roa_p)
    # Accrual test: cash earnings should exceed accounting earnings.
    tests["cfo_exceeds_ni"] = float(is_num(cfo) and is_num(ni) and cfo > ni)

    # --- leverage / liquidity ---
    lev = safe_div(val(cur, "long_term_debt"), mean_finite([ta, ta_p]))
    lev_p = safe_div(val(prev, "long_term_debt"), mean_finite([ta_p, ta_pp]))
    if is_num(lev) and is_num(lev_p):
        tests["leverage_down"] = float(lev <= lev_p)
    elif not is_num(val(cur, "long_term_debt")):
        tests["leverage_down"] = 1.0  # no long-term debt at all passes

    cr = safe_div(val(cur, "current_assets"), val(cur, "current_liabilities"))
    cr_p = safe_div(val(prev, "current_assets"), val(prev, "current_liabilities"))
    if is_num(cr) and is_num(cr_p):
        tests["current_ratio_up"] = float(cr > cr_p)

    sh, sh_p = val(cur, "shares_diluted"), val(prev, "shares_diluted")
    if is_num(sh) and is_num(sh_p):
        # 1% tolerance: routine option-grant drift is not an equity raise.
        tests["no_dilution"] = float(sh <= sh_p * 1.01)

    # --- operating efficiency ---
    gm = safe_div(val(cur, "gross_profit"), val(cur, "revenue"))
    gm_p = safe_div(val(prev, "gross_profit"), val(prev, "revenue"))
    if is_num(gm) and is_num(gm_p):
        tests["gross_margin_up"] = float(gm > gm_p)

    at = safe_div(val(cur, "revenue"), mean_finite([ta, ta_p]))
    at_p = safe_div(val(prev, "revenue"), mean_finite([ta_p, ta_pp]))
    if is_num(at) and is_num(at_p):
        tests["asset_turnover_up"] = float(at > at_p)

    if len(tests) < 6:  # too many untestable components to be meaningful
        return NAN, tests
    # Scale to the full 9 points so a partially testable company stays comparable.
    return float(sum(tests.values()) * 9.0 / len(tests)), tests


def altman_z(f: Fundamentals, market_cap: float) -> tuple[float, str]:
    """Altman Z-Score for bankruptcy risk, with the right variant per company.

    The original 1968 model was fitted on listed *manufacturers*; applying its
    coefficients to a services or retail balance sheet is a common misuse.  The
    Z''-variant (no asset-turnover term, different weights) is used for
    non-manufacturers.  Banks are excluded entirely -- the model has no meaning
    where leverage is the product.
    """
    if f.is_financial:
        return NAN, "not applicable (financial)"

    ta = f.get("total_assets")
    if not is_num(ta) or ta <= 0:
        return NAN, "insufficient data"
    tl = f.get("total_liabilities")
    if not is_num(tl):
        tl = sum_finite(total_debt(f, include_leases=False), f.get("current_liabilities"))
    if not is_num(tl) or tl <= 0:
        return NAN, "insufficient data"

    wc = safe_div(
        sum_finite(f.get("current_assets"), -zero_if_nan(f.get("current_liabilities"))), ta
    )
    re = safe_div(f.get("retained_earnings"), ta)
    ebit_ta = safe_div(f.get("operating_income"), ta)
    mve_tl = safe_div(market_cap, tl)
    sales_ta = safe_div(f.get("revenue"), ta)

    sic = f.sic or 0
    manufacturer = 2000 <= sic <= 3999
    if manufacturer and all(is_num(x) for x in (wc, re, ebit_ta, mve_tl, sales_ta)):
        z = 1.2 * wc + 1.4 * re + 3.3 * ebit_ta + 0.6 * mve_tl + 1.0 * sales_ta
        bands = (1.81, 2.99)
        variant = "Z (manufacturer)"
    elif all(is_num(x) for x in (wc, re, ebit_ta, mve_tl)):
        # Z'' -- Altman's emerging-market/service revision, no sales term.
        z = 6.56 * wc + 3.26 * re + 6.72 * ebit_ta + 1.05 * mve_tl
        bands = (1.1, 2.6)
        variant = "Z'' (non-manufacturer)"
    else:
        return NAN, "insufficient data"

    zone = "distress" if z < bands[0] else ("grey" if z < bands[1] else "safe")
    return float(z), f"{zone} [{variant}]"


def beneish_m_score(f: Fundamentals) -> tuple[float, dict[str, float]]:
    """Beneish M-Score: probability that earnings are being manipulated.

    Above roughly -1.78 flags elevated risk.  It is a *screen for scrutiny*, not
    a verdict -- it fires on fast-growing companies with rising receivables for
    entirely innocent reasons, and it famously flagged Enron two years early.
    Needs two consecutive years of eight line items, so it is often unavailable.
    """
    a = f.annual
    if len(a) < 2:
        return NAN, {}
    cur, prev = a.iloc[-1], a.iloc[-2]

    def v(row, key):
        x = row.get(key)
        return float(x) if is_num(x) else NAN

    rev, rev_p = v(cur, "revenue"), v(prev, "revenue")
    ar, ar_p = v(cur, "receivables"), v(prev, "receivables")
    gp, gp_p = v(cur, "gross_profit"), v(prev, "gross_profit")
    ta, ta_p = v(cur, "total_assets"), v(prev, "total_assets")
    ppe, ppe_p = v(cur, "ppe_net"), v(prev, "ppe_net")
    ca, ca_p = v(cur, "current_assets"), v(prev, "current_assets")
    sec_, sec_p = v(cur, "short_term_investments"), v(prev, "short_term_investments")
    da, da_p = v(cur, "depreciation_amortization"), v(prev, "depreciation_amortization")
    sga, sga_p = v(cur, "sga"), v(prev, "sga")
    tl, tl_p = v(cur, "total_liabilities"), v(prev, "total_liabilities")
    ni, cfo = v(cur, "net_income"), v(cur, "cfo")

    idx: dict[str, float] = {}
    idx["DSRI"] = safe_div(safe_div(ar, rev), safe_div(ar_p, rev_p))
    idx["GMI"] = safe_div(safe_div(gp_p, rev_p), safe_div(gp, rev))
    soft_a = safe_div(ta - sum_finite(ca, ppe, require_any=False), ta)
    soft_p = safe_div(ta_p - sum_finite(ca_p, ppe_p, require_any=False), ta_p)
    idx["AQI"] = safe_div(soft_a, soft_p)
    idx["SGI"] = safe_div(rev, rev_p)
    idx["DEPI"] = safe_div(safe_div(da_p, sum_finite(da_p, ppe_p)),
                           safe_div(da, sum_finite(da, ppe)))
    idx["SGAI"] = safe_div(safe_div(sga, rev), safe_div(sga_p, rev_p))
    idx["LVGI"] = safe_div(safe_div(tl, ta), safe_div(tl_p, ta_p))
    idx["TATA"] = safe_div(ni - cfo, ta)

    needed = ("DSRI", "GMI", "AQI", "SGI", "LVGI", "TATA")
    if sum(is_num(idx[k]) for k in needed) < 5:
        return NAN, idx

    def g(key, fallback=1.0):
        return idx[key] if is_num(idx[key]) else fallback

    m = (
        -4.84
        + 0.920 * g("DSRI")
        + 0.528 * g("GMI")
        + 0.404 * g("AQI")
        + 0.892 * g("SGI")
        + 0.115 * g("DEPI")
        - 0.172 * g("SGAI")
        + 4.679 * g("TATA", 0.0)
        - 0.327 * g("LVGI")
    )
    return float(m), idx


# --------------------------------------------------------------------------- #
# the main entry point
# --------------------------------------------------------------------------- #
def share_count(f: Fundamentals) -> tuple[float, str, list[str]]:
    """Best available total share count, and where it came from.

    Three sources, none individually reliable:

    * the **cover page** (``dei:EntityCommonStockSharesOutstanding``) is the
      most current, but ``companyfacts`` omits dimensional facts, so for a
      multi-class registrant it returns one class or nothing. Mastercard's
      cover page reads 122m against 884m diluted shares, which would value the
      company at a seventh of its market cap;
    * the **diluted weighted average** is a total across classes and is the
      denominator of reported EPS, but it is a trailing period average;
    * **net income divided by diluted EPS** recovers that same denominator even
      when the share count itself is only tagged dimensionally -- which is the
      only route that works for some multi-class filers.

    The cover-page figure is used when it corroborates one of the other two
    (within 15%), since it is the most current; otherwise it is treated as
    incomplete and a total is used instead.
    """
    notes: list[str] = []
    cover = f.get("shares_outstanding")
    diluted = f.get("shares_diluted")
    basic = f.get("shares_basic")
    issued = f.get("shares_issued")

    implied = NAN
    eps = f.get("eps_diluted")
    ni = f.get("net_income")
    if is_num(eps) and eps != 0 and is_num(ni):
        candidate = ni / eps
        if candidate > 0:
            implied = candidate

    reference = first_finite(diluted, implied, basic)
    if is_num(cover) and is_num(reference):
        ratio = cover / reference
        if 0.85 <= ratio <= 1.15:
            return float(cover), "cover page", notes
        notes.append(
            f"Cover-page share count ({cover/1e6:,.1f}m) differs from the total "
            f"implied by diluted EPS ({reference/1e6:,.1f}m) by "
            f"{abs(ratio - 1):.0%}; treating it as a single share class and "
            "using the diluted total instead."
        )
        return float(reference), "diluted total", notes
    if is_num(reference):
        return float(reference), "diluted total", notes
    if is_num(cover):
        notes.append("Only a cover-page share count is available; for a "
                     "multi-class issuer this may cover one class only.")
        return float(cover), "cover page", notes
    value = first_finite(issued)
    if is_num(value):
        return float(value), "shares issued", notes
    return NAN, "unavailable", ["No usable share count could be determined."]


def normalize_split_basis(series: pd.Series, splits: pd.Series | None,
                          tolerance: float = 0.20) -> pd.Series:
    """Put a historical per-share series onto one consistent split basis.

    Filings restate comparatives for a split, but only for the two or three
    years a filing reproduces. Because values are taken from the most recent
    filing, the series silently straddles two bases: Broadcom's diluted share
    count reads 429m for FY2021 and 4,232m for FY2022, which is not a 10x
    issuance but a 10-for-1 split that the FY2022 comparative reflects and the
    FY2021 original does not. Left uncorrected it produced a 63%-a-year
    share-count growth rate.

    Note that the break in the data sits at the *restatement* boundary, not at
    the split date -- Broadcom's split was in 2024 but its data steps between
    FY2021 and FY2022 -- so the split date cannot be used to locate it. Instead
    this walks newest to oldest looking for an adjacent-pair ratio that matches
    a split the company actually did, and rescales everything older. Matching
    against recorded factors (rather than any large jump) keeps a genuine
    10-for-1 share issuance from being "corrected" away.
    """
    if series is None or len(series) < 2:
        return series
    if splits is None or splits.empty:
        return series

    available = [float(f) for f in splits.to_numpy(dtype=float)
                 if is_num(f) and f > 0 and abs(f - 1.0) > 1e-6]
    if not available:
        return series

    out = series.astype(float).copy()
    for i in range(len(out) - 1, 0, -1):
        older, newer = out.iloc[i - 1], out.iloc[i]
        if not (is_num(older) and is_num(newer)) or older <= 0:
            continue
        ratio = newer / older
        if abs(ratio - 1.0) < 0.25:      # ordinary year-to-year drift
            continue
        match, best_error = None, tolerance
        for factor in available:
            for candidate in (factor, 1.0 / factor):
                error = abs(ratio / candidate - 1.0)
                if error < best_error:
                    match, best_error = (factor, candidate), error
        if match is None:
            continue
        factor, candidate = match
        out.iloc[:i] = out.iloc[:i] * candidate
        available.remove(factor)         # each split explains one break
    return out


def compute_metrics(f: Fundamentals, price: float | None = None,
                   dividend_ttm: float | None = None,
                   splits: pd.Series | None = None) -> dict[str, float]:
    """Flatten a company's fundamentals (+ price) into a metric dictionary.

    Valuation multiples are only populated when ``price`` is supplied.  Keys
    absent from the result were not computable -- callers should use
    ``dict.get`` and treat ``None``/``nan`` as "unknown", never as zero.

    ``dividend_ttm`` is trailing twelve-month cash dividends *per share* from
    the exchange record (see :func:`stocklab.data.providers.yahoo.fetch_dividends`).
    Pass it whenever available: XBRL dividend tagging is unreliable, with
    filers using company-specific extensions the standard taxonomy never sees.
    Simon Property Group, for instance, has tagged no standard dividend concept
    since 2019, which makes its yield look like zero.
    """
    m: dict[str, float] = {}
    cfg = settings()

    # ---- scale ------------------------------------------------------------
    shares_out, shares_basis, share_notes = share_count(f)
    # Per-share fundamentals use the diluted weighted average so they line up
    # with the earnings period; market cap uses the best total available.
    shares = first_finite(f.get("shares_diluted"), f.get("shares_basic"), shares_out)
    m["shares"] = shares
    m["shares_outstanding"] = shares_out
    m["share_count_basis"] = shares_basis       # type: ignore[assignment]
    m["share_count_notes"] = share_notes        # type: ignore[assignment]
    # False means only a possibly-single-class cover-page count was available,
    # so market cap and every multiple built on it may be understated. Screens
    # should exclude these names from valuation rankings rather than rank them
    # as cheap, which is exactly what an understated market cap looks like.
    m["share_count_reliable"] = float(
        shares_basis == "diluted total"
        or (shares_basis == "cover page" and not share_notes)
    )
    m["dilution_overhang"] = safe_div(shares, shares_out) - 1.0 if is_num(shares) else NAN

    if is_num(price) and is_num(shares_out) and shares_out > 0:
        market_cap = float(price) * shares_out
    else:
        market_cap = NAN
    m["price"] = float(price) if is_num(price) else NAN
    m["market_cap"] = market_cap

    debt = total_debt(f)
    cash = liquid_assets(f)
    nd = net_debt(f)
    m["total_debt"] = debt
    m["cash_and_investments"] = cash
    m["net_debt"] = nd
    # NB: `x or 0.0` does NOT work here -- float('nan') is truthy, so a missing
    # minority interest would poison the whole expression.
    m["enterprise_value"] = (
        market_cap + zero_if_nan(nd) + zero_if_nan(f.get("minority_interest"))
        if is_num(market_cap) else NAN
    )

    # ---- raw TTM pass-throughs -------------------------------------------
    rev = f.get("revenue")
    ebit = f.get("operating_income")
    ni = f.get("net_income")
    cfo = f.get("cfo")
    capex = f.get("capex")
    fcf = free_cash_flow(f)
    ebitda_v = ebitda(f)
    equity = f.get("equity")

    m.update({
        "revenue": rev, "gross_profit": f.get("gross_profit"), "ebit": ebit,
        "ebitda": ebitda_v, "net_income": ni, "cfo": cfo, "capex": capex,
        "fcf": fcf, "equity": equity, "total_assets": f.get("total_assets"),
        "eps_diluted": f.get("eps_diluted"),
        "stock_comp": f.get("stock_comp"),
        "dividends_paid": f.get("dividends_paid"),
        "buybacks": f.get("buybacks"),
    })
    # TTM EPS recomputed from TTM net income: the reported EPS tag is per-period
    # and never sums to a trailing year once the share count moves.
    m["eps_ttm"] = safe_div(ni, shares)

    # ---- margins ----------------------------------------------------------
    m["gross_margin"] = safe_div(f.get("gross_profit"), rev)
    m["operating_margin"] = safe_div(ebit, rev)
    m["ebitda_margin"] = safe_div(ebitda_v, rev)
    m["net_margin"] = safe_div(ni, rev)
    m["fcf_margin"] = safe_div(fcf, rev)
    m["rnd_intensity"] = safe_div(f.get("rnd"), rev)
    m["sbc_intensity"] = safe_div(f.get("stock_comp"), rev)
    m["capex_intensity"] = safe_div(capex, rev)
    m["capex_to_da"] = safe_div(capex, f.get("depreciation_amortization"))

    # ---- returns on capital ----------------------------------------------
    tax_rate = effective_tax_rate(f)
    m["effective_tax_rate"] = tax_rate
    nopat = ebit * (1.0 - tax_rate) if is_num(ebit) else NAN
    m["nopat"] = nopat

    avg_equity = _avg_balance(f, "equity")
    avg_assets = _avg_balance(f, "total_assets")
    avg_ic = _avg_invested_capital(f)
    m["invested_capital"] = avg_ic

    m["roe"] = safe_div(ni, avg_equity)
    m["roa"] = safe_div(ni, avg_assets)
    m["roic"] = NAN if f.is_financial else safe_div(nopat, avg_ic)
    m["roce"] = NAN if f.is_financial else safe_div(
        ebit, sum_finite(avg_assets, -zero_if_nan(f.get("current_liabilities")))
    )
    # Cash return on invested capital: harder to manage than accounting ROIC.
    m["croic"] = NAN if f.is_financial else safe_div(fcf, avg_ic)
    m["asset_turnover"] = safe_div(rev, avg_assets)
    # Earnings quality: how much of reported profit shows up as cash.
    m["cash_conversion"] = safe_div(cfo, ni)
    m["fcf_conversion"] = safe_div(fcf, ni)
    m["accruals_ratio"] = safe_div(
        (ni - cfo) if (is_num(ni) and is_num(cfo)) else NAN, avg_assets
    )

    # ---- balance sheet strength ------------------------------------------
    m["current_ratio"] = safe_div(f.get("current_assets"), f.get("current_liabilities"))
    m["quick_ratio"] = safe_div(
        sum_finite(f.get("cash"), f.get("short_term_investments"), f.get("receivables")),
        f.get("current_liabilities"),
    )
    m["debt_to_equity"] = safe_div(debt, equity)
    m["debt_to_assets"] = safe_div(debt, f.get("total_assets"))
    m["net_debt_to_ebitda"] = NAN if f.is_financial else safe_div(nd, ebitda_v)
    m["interest_coverage"] = safe_div(ebit, f.get("interest_expense"))
    m["net_cash_per_share"] = safe_div(-nd if is_num(nd) else NAN, shares)
    m["book_value_per_share"] = safe_div(equity, shares)
    tangible_equity = (
        equity - sum_finite(f.get("goodwill"), f.get("intangibles"), require_any=False)
        if is_num(equity) else NAN
    )
    m["tangible_book_value"] = tangible_equity
    m["tangible_bvps"] = safe_div(tangible_equity, shares)

    # ---- growth -----------------------------------------------------------
    for concept, prefix in (
        ("revenue", "revenue"), ("net_income", "earnings"), ("eps_diluted", "eps"),
        ("cfo", "cfo"), ("equity", "book_value"), ("dividends_paid", "dividend"),
    ):
        m.update(growth_block(f, concept, prefix))

    fcf_hist = _annual_fcf(f)
    if not fcf_hist.empty:
        m["fcf_latest_annual"] = float(fcf_hist.iloc[-1])
        for years in (3, 5):
            if len(fcf_hist) > years:
                m[f"fcf_cagr_{years}y"] = cagr(
                    float(fcf_hist.iloc[-1 - years]), float(fcf_hist.iloc[-1]), years
                )
        growth, r2 = log_trend_growth(fcf_hist.iloc[-7:])
        m["fcf_trend_growth"], m["fcf_trend_r2"] = growth, r2
        m["fcf_positive_years"] = float((fcf_hist.iloc[-7:] > 0).sum())
        m["fcf_years_available"] = float(len(fcf_hist))

    # Share-count trend: the dilution that quietly eats per-share growth.
    # Normalised for splits first, or a 10-for-1 reads as a 10x share issuance.
    sh_hist = normalize_split_basis(f.annual_series("shares_diluted"), splits)
    if len(sh_hist) >= 4:
        m["share_count_cagr_3y"] = cagr(float(sh_hist.iloc[-4]), float(sh_hist.iloc[-1]), 3)
    if len(sh_hist) >= 6:
        m["share_count_cagr_5y"] = cagr(float(sh_hist.iloc[-6]), float(sh_hist.iloc[-1]), 5)

    # Margin trajectory: expanding or eroding operating leverage.
    op_margin_hist = _annual_ratio(f, "operating_income", "revenue")
    if len(op_margin_hist) >= 4:
        m["operating_margin_5y_avg"] = float(op_margin_hist.iloc[-5:].mean())
        m["operating_margin_delta_3y"] = float(
            op_margin_hist.iloc[-1] - op_margin_hist.iloc[-4]
        )
    gm_hist = _annual_ratio(f, "gross_profit", "revenue")
    if len(gm_hist) >= 4:
        m["gross_margin_delta_3y"] = float(gm_hist.iloc[-1] - gm_hist.iloc[-4])
        m["gross_margin_stability"] = float(gm_hist.iloc[-7:].std())

    roic_hist = _annual_roic(f)
    if len(roic_hist) >= 3:
        m["roic_5y_avg"] = float(roic_hist.iloc[-5:].mean())
        m["roic_min_5y"] = float(roic_hist.iloc[-5:].min())
        m["roic_stability"] = float(roic_hist.iloc[-5:].std())

    # ---- quality / safety composites -------------------------------------
    f_score, f_tests = piotroski_f_score(f)
    m["piotroski_f"] = f_score
    for key, val in f_tests.items():
        m[f"f_{key}"] = val

    z, z_zone = altman_z(f, market_cap)
    m["altman_z"] = z
    m["altman_zone"] = z_zone  # type: ignore[assignment]  (string, by design)

    mscore, _ = beneish_m_score(f)
    m["beneish_m"] = mscore

    # ---- shareholder returns ---------------------------------------------
    # Prefer the exchange's own dividend record over the filing's cash-flow
    # line: it is per-share, current, and immune to tagging gaps.
    if is_num(dividend_ttm) and is_num(price) and float(price) > 0:
        m["dividend_per_share"] = float(dividend_ttm)
        m["dividend_yield"] = float(dividend_ttm) / float(price)
        m["dividend_source"] = "exchange record"   # type: ignore[assignment]
        dividends_total = (float(dividend_ttm) * shares_out
                           if is_num(shares_out) else NAN)
    else:
        dividends_total = f.get("dividends_paid")
        m["dividend_yield"] = safe_div(dividends_total, market_cap)
        m["dividend_per_share"] = safe_div(dividends_total, shares_out)
        m["dividend_source"] = "XBRL cash flow"    # type: ignore[assignment]
    m["dividends_paid"] = dividends_total
    m["buyback_yield"] = safe_div(f.get("buybacks"), market_cap)
    m["shareholder_yield"] = sum_finite(m["dividend_yield"], m["buyback_yield"],
                                        require_any=False)
    m["payout_ratio"] = safe_div(dividends_total, ni)
    m["fcf_payout_ratio"] = safe_div(dividends_total, fcf)

    # ---- REIT measures ---------------------------------------------------
    # Net income is close to meaningless for a REIT: property depreciation is a
    # large non-cash charge against assets that typically appreciate, so GAAP
    # earnings understate distributable cash by a wide margin and payout ratios
    # computed from them routinely exceed 100% for perfectly healthy trusts.
    # NAREIT-defined FFO adds depreciation back. (Gains on property sales
    # should also be removed; those are not separately tagged often enough to
    # rely on, so this is FFO before that adjustment.)
    da = f.get("depreciation_amortization")
    ffo = (ni + da) if (is_num(ni) and is_num(da)) else NAN
    m["ffo"] = ffo
    m["ffo_per_share"] = safe_div(ffo, shares)
    m["p_ffo"] = safe_div(market_cap, ffo) if (is_num(ffo) and ffo > 0) else NAN
    m["ffo_payout_ratio"] = safe_div(
        float(dividend_ttm) * shares_out
        if (is_num(dividend_ttm) and is_num(shares_out)) else f.get("dividends_paid"),
        ffo,
    )
    m["ffo_yield"] = safe_div(ffo, market_cap)

    # ---- Graham liquidation metrics (used by the net-net screen) ---------
    ncav = (
        f.get("current_assets") - f.get("total_liabilities")
        - zero_if_nan(f.get("preferred_equity"))
        if (is_num(f.get("current_assets")) and is_num(f.get("total_liabilities")))
        else NAN
    )
    m["ncav"] = ncav
    m["ncav_per_share"] = safe_div(ncav, shares)
    if is_num(m["ncav_per_share"]) and is_num(price) and float(price) > 0:
        # Positive means the price sits below Graham's two-thirds-of-NCAV
        # threshold; the magnitude is how far below, as a fraction of price.
        m["ncav_discount"] = (m["ncav_per_share"] * (2.0 / 3.0) - float(price)) / float(price)
    m["net_net_price"] = (m["ncav_per_share"] * (2.0 / 3.0)
                          if is_num(m["ncav_per_share"]) else NAN)

    # ---- valuation multiples ---------------------------------------------
    ev = m["enterprise_value"]
    if is_num(market_cap) and market_cap > 0:
        m["pe"] = safe_div(market_cap, ni) if (is_num(ni) and ni > 0) else NAN
        m["pb"] = safe_div(market_cap, equity) if (is_num(equity) and equity > 0) else NAN
        m["ptbv"] = (safe_div(market_cap, tangible_equity)
                     if (is_num(tangible_equity) and tangible_equity > 0) else NAN)
        m["ps"] = safe_div(market_cap, rev)
        m["p_fcf"] = safe_div(market_cap, fcf) if (is_num(fcf) and fcf > 0) else NAN
        m["p_ocf"] = safe_div(market_cap, cfo) if (is_num(cfo) and cfo > 0) else NAN
        m["earnings_yield"] = safe_div(ni, market_cap)
        m["fcf_yield"] = safe_div(fcf, market_cap)

        # Normalised P/E on 5-year average earnings: strips the cycle out of the
        # denominator, which is the whole point for anything cyclical.
        ni_hist = f.annual_series("net_income", 5)
        if len(ni_hist) >= 3:
            avg_ni = float(ni_hist.mean())
            m["pe_5y_avg_earnings"] = safe_div(market_cap, avg_ni) if avg_ni > 0 else NAN

        if not f.is_financial and is_num(ev) and ev > 0:
            m["ev_ebitda"] = safe_div(ev, ebitda_v) if (is_num(ebitda_v) and ebitda_v > 0) else NAN
            m["ev_ebit"] = safe_div(ev, ebit) if (is_num(ebit) and ebit > 0) else NAN
            m["ev_sales"] = safe_div(ev, rev)
            m["ev_fcf"] = safe_div(ev, fcf) if (is_num(fcf) and fcf > 0) else NAN
            # Greenblatt's earnings yield: EBIT/EV, capital-structure neutral.
            m["ebit_ev_yield"] = safe_div(ebit, ev)

        growth_for_peg = first_finite(m.get("eps_cagr_3y"), m.get("earnings_cagr_3y"),
                                      m.get("revenue_cagr_3y"))
        if is_num(m.get("pe")) and is_num(growth_for_peg) and growth_for_peg > 0:
            m["peg"] = m["pe"] / (growth_for_peg * 100.0)

    # ---- context ----------------------------------------------------------
    m["sector"] = f.sector            # type: ignore[assignment]
    m["industry"] = f.industry        # type: ignore[assignment]
    m["is_financial"] = float(f.is_financial)
    m["is_reit"] = float(f.is_reit)
    m["years_of_data"] = float(len(f.annual))
    m["ttm_basis"] = f.ttm.get("ttm_basis", 0.0)
    m["size_premium"] = cfg.size_premium(market_cap if is_num(market_cap) else None)
    return m


# --------------------------------------------------------------------------- #
# annual helper series
# --------------------------------------------------------------------------- #
def _annual_fcf(f: Fundamentals) -> pd.Series:
    if "cfo" not in f.annual:
        return pd.Series(dtype=float)
    cfo = f.annual["cfo"]
    capex = f.annual["capex"] if "capex" in f.annual else pd.Series(0.0, index=cfo.index)
    return (cfo - capex.reindex(cfo.index).fillna(0.0)).dropna()


def _annual_ratio(f: Fundamentals, num: str, den: str) -> pd.Series:
    if num not in f.annual or den not in f.annual:
        return pd.Series(dtype=float)
    out = f.annual[num] / f.annual[den].replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def _annual_roic(f: Fundamentals) -> pd.Series:
    """Historical ROIC, so stability and worst-year can be measured."""
    if f.is_financial or "operating_income" not in f.annual:
        return pd.Series(dtype=float)
    tax_rate = effective_tax_rate(f)
    out: dict[pd.Timestamp, float] = {}
    for date in f.annual.index:
        ebit = f.annual["operating_income"].get(date)
        ic = invested_capital(f, at=date)
        val = safe_div(ebit * (1.0 - tax_rate) if is_num(ebit) else NAN, ic)
        if is_num(val):
            out[date] = val
    return pd.Series(out, dtype=float).sort_index()
