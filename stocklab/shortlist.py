"""Which companies are worth owning, as distinct from which are good.

The four-pillar grade is a weighted average, and an average is
*compensatory*: strength in one pillar cancels weakness in another. That is
the right shape for "is this a sound business" and the wrong shape for
"should I buy this, at this price". On a 515-company run, 62% of the
companies graded A or B traded above their own fair value estimate and 34%
were priced to return less than a US Treasury. The grade was not broken --
it was answering a different question, and a good business bought at a bad
price is still a bad investment.

This module answers the other question with **conjunctive screens**: lists of
conditions, every one of which must hold. Nothing compensates for anything.
Survivors are ranked by their *weakest* pillar rather than their average,
which is the same idea applied to the ordering: a company scoring 70/72/75/71
is a better thing to own for ten years than one scoring 98/95/92/40.

**There are three screens, not one.** A single set of conditions was a value
screen wearing a general-purpose label. Run over 515 companies it passed one
growth company out of 58 and none of 116 speculative ones, because the
conditions that make a mature business worth buying -- repeated profits, a
discount to a conservative valuation, an expected return above a Treasury --
are conditions a fast-growing company fails by being fast-growing, and a
speculative one fails by definition. Demanding a discount to a no-growth
valuation is how you never buy a compounder.

So each kind of opportunity is judged on what actually makes it work:

* **Value** -- a sound, dull, repeatedly profitable business at a discount.
* **Growth** -- sustained growth funded by the business rather than by issuing
  stock, with unit economics that improve as it scales, at a price that does
  not already assume faster growth than it has ever delivered.
* **Speculative** -- unproven, so the test is survival and improving economics,
  not value: can it fund itself long enough for the thesis to be settled.

Categories are separate from, and orthogonal to, the screens. They say what
kind of thing a company is, not whether to buy it.
"""
from __future__ import annotations

from dataclasses import dataclass

from .util.mathx import NAN, is_num

CATEGORY_WORDS = {
    "growth": "Growing fast",
    "compounder": "Quality compounder",
    "income": "Pays an income",
    "steady": "Steady",
    "shrinking": "Shrinking",
    "speculative": "Speculative",
    "hard_to_value": "Hard to value",
    "unclear": "Cannot be sized",
}

# Thresholds. Each is a round number chosen to match how the word is
# ordinarily used, not tuned to produce a particular count.
FAST_GROWTH = 0.20          # sales compounding at a fifth a year
STEADY_GROWTH = 0.05
HIGH_RETURN = 0.15          # return on capital a durable franchise earns
INCOME_YIELD = 0.03         # roughly a Treasury's yield, in dividends
MIN_COVERAGE_TO_VALUE = 10


def categorise(m: dict) -> str:
    """One label per company, decided in order. Describes the business only."""
    if not m.get("share_count_reliable"):
        return "unclear"
    if not m.get("profitable"):
        return "speculative"
    if not is_num(m.get("fair_value")) or (m.get("coverage") or 0) < MIN_COVERAGE_TO_VALUE:
        # Not the same thing as speculative, and the distinction matters to a
        # reader without a finance background. Apollo and First Bancorp are
        # large, profitable and long-established; what is missing is a
        # valuation we are willing to stand behind, not the earnings.
        return "hard_to_value"

    growth = m.get("revenue_cagr_3y")
    roic = m.get("roic_5y_avg") if is_num(m.get("roic_5y_avg")) else m.get("roic")
    if not is_num(roic):
        roic = m.get("roe")
    quality = m.get("quality")

    if is_num(growth) and growth >= FAST_GROWTH:
        return "growth"
    if ((is_num(roic) and roic >= HIGH_RETURN) or (is_num(quality) and quality >= 70)) \
            and is_num(growth) and growth >= STEADY_GROWTH:
        return "compounder"
    if (m.get("dividend_yield") or 0) >= INCOME_YIELD:
        return "income"
    if is_num(growth) and growth < 0:
        return "shrinking"
    return "steady"


@dataclass(frozen=True)
class Gate:
    key: str
    label: str
    why: str


# ---- thresholds -----------------------------------------------------------
# Round numbers chosen to match how each phrase is ordinarily used, not tuned
# to produce a particular count.
MIN_PROFITABLE_YEARS = 4          # of the last five
MAX_ABOVE_OWN_AVERAGE = 2.0       # latest operating profit vs its own 5y average
MIN_COVERAGE = 14
MIN_QUALITY = 60

# The safety bar rises as companies get smaller, because the universe does.
# Across 515 companies the share losing money runs 2% of mega caps, 9% of
# large, 28% of mid, 38% of small and 83% of micro, and median financial
# safety falls 74 -> 69 -> 64 -> 54 -> 40 across the same bands. A small
# company has less access to capital when a year goes wrong, so the same
# leverage is a materially worse risk. This is the one place where size
# genuinely changes the test rather than just the odds.
SAFETY_BY_SIZE = {"mega": 55, "large": 55, "mid": 60, "small": 65, "micro": 70}
MIN_SAFETY = 55

# You cannot build or leave a position in something that barely trades, and
# the thinnest names are also where filing errors and manipulation live.
LIQUIDITY_BY_SIZE = {"mega": 1e6, "large": 1e6, "mid": 1e6,
                     "small": 2e6, "micro": 2e6}


def safety_bar(m: dict) -> float:
    return SAFETY_BY_SIZE.get(str(m.get("size") or ""), MIN_SAFETY)


def liquidity_bar(m: dict) -> float:
    return LIQUIDITY_BY_SIZE.get(str(m.get("size") or ""), 1e6)
MIN_GROWTH_SCORE = 45
MAX_MARGIN_OF_SAFETY = 1.0
AGREED = ("moderate", "reasonable")

GROWTH_3Y = 0.15                  # sustained, not one good year
GROWTH_5Y = 0.10
DECELERATION_FLOOR = 0.6          # 3y growth against 5y growth
MIN_TREND_R2 = 0.70               # a climb, not a single jump
MIN_GROSS_MARGIN = 0.25
MAX_GROSS_MARGIN_EROSION = -0.05  # over three years
MAX_DILUTION = 0.05               # share count growth a year
RULE_OF_40 = 40.0

SPEC_GROWTH_3Y = 0.25
SPEC_MIN_GROSS_MARGIN = 0.20
SPEC_MIN_RUNWAY_YEARS = 2.0
SPEC_MAX_DILUTION = 0.15
SPEC_MAX_EV_SALES = 15.0
SPEC_MIN_CURRENT_RATIO = 1.2


def _n(m: dict, key: str, default=None):
    value = m.get(key)
    return value if is_num(value) else default


# ---- shared conditions ----------------------------------------------------
INTEGRITY = (
    Gate("share_count", "We can tell how many shares exist",
         "No two filings agreed on the share count, so every per-share figure "
         "is unreliable. An understated count looks exactly like a bargain."),
    Gate("revenue_known", "Its sales figure survives its own arithmetic",
         "Reported revenue was smaller than reported profit, or otherwise "
         "could not be reconciled, so every margin and multiple is withheld."),
)


def _integrity(m: dict) -> dict[str, bool]:
    return {
        "share_count": bool(m.get("share_count_reliable")),
        "revenue_known": m.get("revenue_reliable") is None
        or bool(m.get("revenue_reliable")),
    }


# ---- value ----------------------------------------------------------------
VALUE_GATES = INTEGRITY + (
    Gate("profitable", "It makes a profit",
         "A business with no earnings cannot be valued on its earnings."),
    Gate("profits_repeat", "It has made a profit most years, not just lately",
         "Profitable in fewer than four of the last five years. Every model "
         "here is anchored on recent earnings, so one good year after several "
         "bad ones is read as the normal state of the business."),
    Gate("not_a_windfall", "This year's profit is not a one-off",
         "Operating profit is more than double its own five-year average "
         "without the sales to explain it, so the valuation rests on a "
         "windfall or a cyclical peak rather than on what it ordinarily earns."),
    Gate("covered", "Its filings cover most of what we measure",
         "Scored on fewer than fourteen inputs, which is not enough to judge "
         "a company you intend to hold for years."),
    Gate("quality", "It is a good business",
         "Earns too little on the money put into it."),
    Gate("safety", "It could survive a bad year",
         "Carries more debt, or thinner cover, than a company of its size "
         "should. The bar rises as companies get smaller, because a small "
         "company has less access to capital when a year goes wrong."),
    Gate("tradeable", "You could actually buy and sell it",
         "Fewer than a million dollars of stock change hands a day, so a "
         "position is hard to build and harder to leave."),
    Gate("growing", "It is not shrinking",
         "Sales are flat or falling; time works against you."),
    Gate("beats_treasury", "It is priced to beat a government bond",
         "At today's price the expected return is below a 10-year Treasury, "
         "which you can have without the risk."),
    Gate("below_fair", "It is not above its fair value estimate",
         "You would be paying more than the business looks worth."),
    Gate("models_agree", "The valuation methods agree with each other",
         "The methods disagree too widely for the estimate to mean much."),
    Gate("estimate_sane", "The estimate is not wilder than the market",
         "The model says it is worth more than twice its price, which is more "
         "likely a bad estimate than a bargain."),
)


def _value(m: dict) -> dict[str, bool]:
    mos, erp = m.get("margin_of_safety"), m.get("implied_equity_premium")
    spike = m.get("earnings_vs_own_average")
    sales_growth = _n(m, "revenue_cagr_3y", 0.0)
    return {
        **_integrity(m),
        "profitable": bool(m.get("profitable")),
        "profits_repeat": (_n(m, "profitable_years", 0)
                           >= min(MIN_PROFITABLE_YEARS, _n(m, "profitable_years_of", 0))
                           and _n(m, "profitable_years_of", 0) >= 3),
        # A fast-growing company *should* earn more than its own five-year
        # average; that is what growth is. A windfall is a profit spike with
        # no sales behind it, so the sales have to be checked before the
        # spike is held against the company.
        "not_a_windfall": (not is_num(spike) or spike <= MAX_ABOVE_OWN_AVERAGE
                           or sales_growth >= GROWTH_3Y),
        "covered": _n(m, "coverage", 0) >= MIN_COVERAGE,
        "quality": _n(m, "quality", -1) >= MIN_QUALITY,
        "safety": _n(m, "safety", -1) >= safety_bar(m),
        "tradeable": _n(m, "dollar_volume_50d", 0) >= liquidity_bar(m),
        "growing": _n(m, "growth", -1) >= MIN_GROWTH_SCORE,
        "beats_treasury": is_num(erp) and erp > 0
        and not m.get("implied_return_is_floor"),
        "below_fair": is_num(mos) and mos >= 0.0,
        "models_agree": m.get("confidence") in AGREED,
        "estimate_sane": is_num(mos) and mos <= MAX_MARGIN_OF_SAFETY,
    }


# ---- growth ---------------------------------------------------------------
GROWTH_GATES = INTEGRITY + (
    Gate("sustained", "It has grown for years, not for one year",
         "Sales growth over three and five years is not both high. One fast "
         "year is a comparison against a weak one as often as it is a trend."),
    Gate("not_decelerating", "Growth is not falling away",
         "Three-year growth has dropped well below the five-year rate. The "
         "price of a growth company assumes the growth continues."),
    Gate("steady_climb", "The growth is a climb, not a single jump",
         "Sales do not fit a trend closely, so the total came from one or two "
         "events -- often an acquisition -- rather than from the business "
         "compounding."),
    Gate("unit_economics", "It makes money on what it sells",
         "Gross margin is thin or falling as it grows, which means scale is "
         "not making it more profitable."),
    Gate("economics_improving", "Scale is reaching the bottom line",
         "Neither profitable at the operating line nor visibly closing the "
         "gap. Growth that never becomes profit is just spending."),
    Gate("not_bought_with_stock", "The growth is not paid for by issuing shares",
         "The share count is rising more than 5% a year, so the growth per "
         "share you own is far smaller than the growth in the business."),
    Gate("funds_itself", "It can pay for its own growth",
         "Growth plus cash margin falls short of the conventional bar, and it "
         "does not generate cash or hold enough to fund itself for three years."),
    Gate("survivable", "A bad year would not finish it",
         "Too much debt, or too little short-term cover, for a company whose "
         "value depends on reaching a larger future."),
    Gate("tradeable", "You could actually buy and sell it",
         "Too little stock changes hands each day to build a position in, or "
         "to leave when the growth stops."),
    Gate("priced_within_record", "The price does not assume more than it has ever done",
         "The share price already implies faster growth than the company has "
         "actually delivered. You would be paying for an acceleration that "
         "has not happened."),
)


def _growth(m: dict) -> dict[str, bool]:
    g3, g5 = m.get("revenue_cagr_3y"), m.get("revenue_cagr_5y")
    implied, historical = m.get("implied_growth"), m.get("historical_growth")
    net_debt_ebitda = m.get("net_debt_to_ebitda")
    runway, fcf = m.get("cash_runway_years"), m.get("fcf")
    return {
        **_integrity(m),
        "sustained": is_num(g3) and is_num(g5) and g3 >= GROWTH_3Y and g5 >= GROWTH_5Y,
        "not_decelerating": is_num(g3) and is_num(g5)
        and (g5 <= 0 or g3 >= g5 * DECELERATION_FLOOR),
        "steady_climb": _n(m, "revenue_trend_r2", -1) >= MIN_TREND_R2,
        "unit_economics": _n(m, "gross_margin", -1) >= MIN_GROSS_MARGIN
        and _n(m, "gross_margin_delta_3y", 0.0) >= MAX_GROSS_MARGIN_EROSION,
        "economics_improving": _n(m, "operating_margin", -1) > 0
        or _n(m, "operating_margin_delta_3y", -1) > 0.02,
        "not_bought_with_stock": _n(m, "share_count_cagr_3y", 0.0) <= MAX_DILUTION,
        "funds_itself": _n(m, "rule_of_40", -999) >= RULE_OF_40
        or (is_num(fcf) and fcf >= 0)
        or _n(m, "cash_runway_years", -1) >= 3.0,
        "survivable": (not is_num(net_debt_ebitda) or net_debt_ebitda <= 4.0)
        and _n(m, "current_ratio", 99) >= 1.0,
        "tradeable": _n(m, "dollar_volume_50d", 0) >= liquidity_bar(m),
        # The reverse DCF says what growth the price requires. Requiring more
        # than the company has delivered is the definition of paying up for
        # hope, and it is the one price test that suits a growth company --
        # demanding a discount to a no-growth valuation never passes one.
        "priced_within_record": is_num(implied) and is_num(historical)
        and implied <= historical,
    }


# ---- speculative ----------------------------------------------------------
SPECULATIVE_GATES = INTEGRITY + (
    # Without this the screen is nonsense in the other direction: its
    # conditions are deliberately looser than the value screen's, so Nvidia
    # and Duolingo sailed through a list headed "speculative".
    Gate("unproven", "It is genuinely unproven",
         "This company is already profitable and established. Judge it on the "
         "value or growth screen, where the conditions are stricter."),
    Gate("has_customers", "People are already buying from it",
         "Sales are absent or not growing quickly. Without revenue there is "
         "nothing to judge but the story."),
    Gate("unit_economics", "It makes money on each sale",
         "Gross margin is too thin for scale to ever produce a profit. "
         "Selling more of this would lose more money."),
    Gate("economics_improving", "The losses are narrowing",
         "The operating margin is not improving, so growth is not bringing it "
         "closer to funding itself."),
    Gate("runway", "It can fund itself for two more years",
         "At the current burn, the cash on hand does not last two years. It "
         "will have to raise money, probably from a weaker position."),
    Gate("dilution_bounded", "It is not issuing shares faster than it grows",
         "The share count is rising more than 15% a year, so each raise takes "
         "a further bite out of whatever you own."),
    Gate("not_priced_as_a_success", "It is not already priced as if it worked",
         "The price is more than fifteen times sales, which already assumes "
         "the outcome you are speculating on."),
    Gate("solvent", "It can pay its bills this year",
         "Short-term assets do not comfortably cover short-term liabilities."),
    Gate("tradeable", "You could actually get out",
         "Too little stock changes hands each day. In the one place where "
         "you are most likely to want to sell quickly, you could not."),
)


def _speculative(m: dict) -> dict[str, bool]:
    return {
        **_integrity(m),
        "unproven": categorise(m) == "speculative",
        "has_customers": _n(m, "revenue", 0) > 0
        and _n(m, "revenue_cagr_3y", -1) >= SPEC_GROWTH_3Y,
        "unit_economics": _n(m, "gross_margin", -1) >= SPEC_MIN_GROSS_MARGIN,
        "economics_improving": _n(m, "operating_margin_delta_3y", -1) > 0,
        "runway": _n(m, "cash_runway_years", -1) >= SPEC_MIN_RUNWAY_YEARS
        or _n(m, "fcf", -1) >= 0,
        "dilution_bounded": _n(m, "share_count_cagr_3y", 0.0) <= SPEC_MAX_DILUTION,
        "not_priced_as_a_success": _n(m, "ev_sales", 0.0) <= SPEC_MAX_EV_SALES,
        "solvent": _n(m, "current_ratio", -1) >= SPEC_MIN_CURRENT_RATIO,
        "tradeable": _n(m, "dollar_volume_50d", 0) >= liquidity_bar(m),
    }


SCREENS: dict[str, tuple] = {
    "value": VALUE_GATES,
    "growth": GROWTH_GATES,
    "speculative": SPECULATIVE_GATES,
}
_RUNNERS = {"value": _value, "growth": _growth, "speculative": _speculative}

SCREEN_WORDS = {
    "value": "A sound business at a discount",
    "growth": "Growing, funding its own growth, not priced beyond its record",
    "speculative": "Unproven, but with the economics and the runway to find out",
}


def gate_results(m: dict, screen: str = "value") -> dict[str, bool]:
    """Which conditions this company meets on the named screen."""
    return _RUNNERS[screen](m)


def weakest_pillar(m: dict) -> float:
    """The lowest of the four scores. Ranks a long hold better than the mean:
    70/72/75/71 is a steadier thing to own than 98/95/92/40."""
    parts = [m.get(k) for k in ("quality", "growth", "safety", "price")]
    usable = [p for p in parts if is_num(p)]
    return min(usable) if len(usable) == 4 else NAN


def assess(m: dict) -> dict:
    """Category, and the verdict of each of the three screens.

    A company is offered against every screen, not only the one its category
    suggests, because the categories describe the business and the screens
    describe the case for buying it. A compounder can pass the growth screen;
    a company labelled growth can be cheap enough for the value one.
    """
    out: dict = {"category": categorise(m), "weakest_pillar": weakest_pillar(m)}
    passing = []
    for name, gates in SCREENS.items():
        results = gate_results(m, name)
        ok = all(results.values())
        out[name] = {
            "gates": results,
            "passed": sum(1 for v in results.values() if v),
            "total": len(results),
            "qualifies": ok,
            "failed": [g.key for g in gates if not results[g.key]],
        }
        if ok:
            passing.append(name)
    out["passes"] = passing
    # Kept for the page and the stored snapshot: "qualifies" has always meant
    # the value screen, which is the one that answers "buy and hold this".
    out["qualifies"] = "value" in passing
    out["gates"] = out["value"]["gates"]
    out["gates_passed"] = out["value"]["passed"]
    out["gates_total"] = out["value"]["total"]
    return out


# The value screen is the default, so its gate list keeps the bare name for
# the builder and the page, which were written before there were three.
GATES = VALUE_GATES
