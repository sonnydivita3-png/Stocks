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
    "compounder": "Quietly compounding",
    "income": "Pays you to wait",
    "steady": "Slow and steady",
    "shrinking": "Going backwards",
    "speculative": "A long shot",
    "hard_to_value": "Hard to pin down",
    "unclear": "Cannot be measured",
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


# What kind of thing a condition tests. Missing one is not one thing:
# a company that fails only on price is a business you would want at a
# lower price, and the page can say which price. A company that fails
# because we cannot measure it is not evidence against the company at
# all -- it is a limit of the tools. Only a business failure is evidence.
GATE_KINDS = {
    "integrity": "whether the numbers can be trusted",
    "business": "whether the business is any good",
    "price": "whether today's price is fair",
    "measurement": "whether we can judge it at all",
}


@dataclass(frozen=True)
class Gate:
    key: str
    label: str
    why: str
    kind: str = "business" 


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
    Gate("share_count", "We know how big the company is",
         "The company's own paperwork disagrees with itself about how many "
         "shares exist, so we cannot work out what a single share is worth. "
         "Too few shares makes anything look like a steal.", "integrity"),
    Gate("revenue_known", "Its sales figure actually adds up",
         "The sales it reported are smaller than the profit it reported, which "
         "cannot happen. Something is mislabelled, so we have left the "
         "figures out rather than guess.", "integrity"),
)


def _integrity(m: dict) -> dict[str, bool]:
    return {
        "share_count": bool(m.get("share_count_reliable")),
        "revenue_known": m.get("revenue_reliable") is None
        or bool(m.get("revenue_reliable")),
    }


# ---- value ----------------------------------------------------------------
VALUE_GATES = INTEGRITY + (
    Gate("profitable", "It actually makes money",
         "It is spending more than it earns. There are no profits to put a "
         "price on, so its value rests on a future that has not happened.", "business"),
    Gate("profits_repeat", "It has made money most years, not just the last one",
         "It turned a profit in fewer than four of the last five years. One "
         "good year after a run of bad ones flatters everything we measure, "
         "because the sums assume the good year is the normal one.", "business"),
    Gate("not_a_windfall", "This year was not a fluke",
         "Profit is more than double its own five-year average and sales did "
         "not grow to match, so something one-off is behind it. Valuing the "
         "company on a lucky year is how you overpay.", "business"),
    Gate("covered", "We can see enough of its numbers",
         "Too few of its figures are available to judge a company you mean to "
         "hold for years. Not a mark against it, just not enough to go on.", "measurement"),
    Gate("quality", "It is good at making money",
         "It earns too little on the money put into it. A business that turns "
         "a dollar into pennies has to grow enormously to be worth owning.", "business"),
    Gate("safety", "It could ride out a bad year",
         "It owes more, or holds less in reserve, than a company its size "
         "safely can. Smaller companies are held to a higher bar here: when "
         "trouble comes, they find it much harder to borrow their way out.", "business"),
    Gate("tradeable", "You could actually buy and sell it",
         "Hardly any of its shares trade on a normal day. Buying would push "
         "the price up against you, and selling in a hurry could be ugly.", "measurement"),
    Gate("growing", "It is not going backwards",
         "Sales are flat or falling. You can buy a shrinking company cheaply "
         "and still lose, because the thing you bought keeps getting smaller.", "business"),
    Gate("beats_treasury", "It should beat a government bond",
         "At today's price you would earn less than lending money to the US "
         "government, which pays about 4.8% and cannot go bust. If a share "
         "cannot beat that, the risk is buying you nothing.", "price"),
    Gate("below_fair", "It is not more expensive than it looks worth",
         "The share costs more than our estimate of what the business is worth. "
         "A fine company at too high a price is still a poor purchase.", "price"),
    Gate("models_agree", "Our different ways of pricing it agree",
         "We price every company six different ways. For this one the answers "
         "are wildly apart, which means we do not really know what it is "
         "worth, so we will not pretend otherwise.", "measurement"),
    Gate("estimate_sane", "Our estimate is not wildly optimistic",
         "Our sums say it is worth more than twice what it costs. A gap that "
         "big is usually our mistake rather than the market's.", "measurement"),
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
    Gate("sustained", "It has been growing for years, not months",
         "One fast year is often just a comparison against a bad one. We want "
         "sales climbing over three years and five, not a single good jump.", "business"),
    Gate("not_decelerating", "The growth is not fizzling out",
         "It is growing more slowly than it used to. You pay a high price for "
         "a growing company precisely because you expect it to keep going.", "business"),
    Gate("steady_climb", "It climbed steadily rather than jumping once",
         "The sales line is lumpy rather than a steady climb, which usually "
         "means it bought another company rather than grew its own.", "business"),
    Gate("unit_economics", "It makes money on each thing it sells",
         "The profit on each sale is thin, or getting thinner as it grows. "
         "Selling more of something barely profitable does not fix it.", "business"),
    Gate("economics_improving", "Getting bigger is making it richer",
         "It is not yet profitable and not visibly getting closer. Growth that "
         "never turns into profit is just expensive.", "business"),
    Gate("not_bought_with_stock", "It is not printing shares to pay for it",
         "It is creating new shares faster than 5% a year. The company grows, "
         "but your slice of it shrinks, so you get much less than the "
         "headline growth suggests.", "business"),
    Gate("funds_itself", "It can pay for its own growth",
         "It cannot fund its own expansion out of what it earns, and has less "
         "than three years of cash to fall back on. Sooner or later it has "
         "to ask someone for money, on whatever terms are going then.", "business"),
    Gate("survivable", "A bad year would not finish it",
         "Too much owed, or too little in hand, for a company whose whole case "
         "rests on reaching a bigger future.", "business"),
    Gate("tradeable", "You could actually buy and sell it",
         "Barely any of its shares trade each day. Easy to buy into, hard to "
         "get out of on the day the growth stops.", "measurement"),
    Gate("priced_within_record", "You are not paying for growth it has never managed",
         "Work backwards from the share price and it demands faster growth "
         "than this company has ever produced. Everything has to go better "
         "than it ever has just for you to break even.", "price"),
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
    Gate("unproven", "It really is an unproven company",
         "This one already makes money and has a track record. Look at it on "
         "the other two lists, where the bar is set higher.", "measurement"),
    Gate("has_customers", "Real people are already paying it",
         "It has little in the way of sales, or they are not growing fast. "
         "Without customers there is nothing to judge but the story.", "business"),
    Gate("unit_economics", "It does not lose money on every sale",
         "It makes too little on each sale for growth to ever add up to a "
         "profit. Selling twice as much would just lose twice as much.", "business"),
    Gate("economics_improving", "The losses are getting smaller",
         "It is not losing less as it grows, so getting bigger is not bringing "
         "it any closer to paying for itself.", "business"),
    Gate("runway", "It has two years of cash left",
         "At the rate it is spending, the money runs out inside two years. It "
         "will have to go asking for more, and nobody asking from weakness "
         "gets a good deal.", "business"),
    Gate("dilution_bounded", "It is not flooding the market with shares",
         "New shares are appearing faster than 15% a year. Every time it "
         "raises money, the piece you own gets smaller.", "business"),
    Gate("not_priced_as_a_success", "It is not already priced as a success",
         "It already costs more than fifteen years of its entire sales. The "
         "happy ending you are betting on is in the price before you start.", "price"),
    Gate("solvent", "It can pay this year's bills",
         "What it owes in the next year is uncomfortably close to what it has "
         "to pay with.", "business"),
    Gate("tradeable", "You could actually get out",
         "Hardly any shares trade each day. This is the group you are most "
         "likely to need to sell in a hurry, and the one where you could not.", "measurement"),
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
    "value": "A solid business going cheap",
    "growth": "Growing fast, paying its own way, not priced for a miracle",
    "speculative": "Unproven, but with the cash and the economics to find out",
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

    # What a single miss means, which is not one thing. A company failing
    # only on price is a business worth owning at a lower price, and the
    # price that would convert it can be named. One failing only because we
    # cannot measure it is not evidence against the company. Only a business
    # failure is evidence.
    for name, gates in SCREENS.items():
        verdict = out[name]
        failed = verdict["failed"]
        by_key = {g.key: g for g in gates}
        kinds = {by_key[k].kind for k in failed if k in by_key}
        verdict["miss_kind"] = (
            "" if not failed
            else "business" if "business" in kinds
            else "integrity" if "integrity" in kinds
            else "price" if kinds == {"price"}
            else "measurement")
        # For a value miss on price alone, the fair value estimate IS the
        # price at which it would pass.
        trigger = NAN
        if name == "value" and failed == ["below_fair"]:
            fair = m.get("fair_value")
            if is_num(fair):
                trigger = float(fair)
        verdict["trigger_price"] = trigger

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
