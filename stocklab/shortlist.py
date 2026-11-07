"""Which companies are worth owning, as distinct from which are good.

The four-pillar grade is a weighted average, and an average is
*compensatory*: strength in one pillar cancels weakness in another. That is
the right shape for "is this a sound business" and the wrong shape for
"should I buy this, at this price". On a 515-company run, 62% of the
companies graded A or B traded above their own fair value estimate and 34%
were priced to return less than a US Treasury. The grade was not broken --
it was answering a different question, and a good business bought at a bad
price is still a bad investment.

This module answers the other question with a **conjunctive screen**: a list
of conditions, every one of which must hold. Nothing compensates for
anything. Survivors are then ranked by their *weakest* pillar rather than
their average, which is the same idea applied to the ordering: a company
scoring 70/72/75/71 is a better thing to own for ten years than one scoring
98/95/92/40.

Categories are separate from, and orthogonal to, the screen. They say what
kind of thing a company is, not whether to buy it.
"""
from __future__ import annotations

from dataclasses import dataclass

from .util.mathx import is_num

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


GATES: tuple[Gate, ...] = (
    Gate("share_count", "We can tell how many shares exist",
         "No two filings agreed on the share count, so every per-share figure "
         "is unreliable. An understated count looks exactly like a bargain."),
    Gate("profitable", "It makes a profit",
         "A business with no earnings cannot be valued on its earnings, and "
         "its price rests on a future that has not happened."),
    Gate("covered", "Its filings cover most of what we measure",
         "Scored on fewer than fourteen of the inputs, which is not enough to "
         "judge a company you intend to hold for years."),
    Gate("quality", "It is a good business",
         "Earns too little on the money put into it."),
    Gate("safety", "It could survive a bad year",
         "Carries more debt, or thinner cover, than a long hold should."),
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
         "The model says it is worth more than twice its price. Ranking by an "
         "estimate selects for estimation error, and a gap that large is more "
         "likely a bad estimate than a bargain."),
)

MIN_COVERAGE = 14
MIN_QUALITY = 60
MIN_SAFETY = 55
MIN_GROWTH = 45
MAX_MARGIN_OF_SAFETY = 1.0
AGREED = ("moderate", "reasonable")


def gate_results(m: dict) -> dict[str, bool]:
    """Which conditions this company meets. Every one must hold to qualify."""
    mos = m.get("margin_of_safety")
    erp = m.get("implied_equity_premium")
    return {
        "share_count": bool(m.get("share_count_reliable")),
        "profitable": bool(m.get("profitable")),
        "covered": (m.get("coverage") or 0) >= MIN_COVERAGE,
        "quality": is_num(m.get("quality")) and m["quality"] >= MIN_QUALITY,
        "safety": is_num(m.get("safety")) and m["safety"] >= MIN_SAFETY,
        "growing": is_num(m.get("growth")) and m["growth"] >= MIN_GROWTH,
        "beats_treasury": is_num(erp) and erp > 0 and not m.get("implied_return_is_floor"),
        "below_fair": is_num(mos) and mos >= 0.0,
        "models_agree": m.get("confidence") in AGREED,
        "estimate_sane": is_num(mos) and mos <= MAX_MARGIN_OF_SAFETY,
    }


def weakest_pillar(m: dict) -> float:
    """The lowest of the four scores. Ranks a long hold better than the mean:
    70/72/75/71 is a steadier thing to own than 98/95/92/40."""
    parts = [m.get(k) for k in ("quality", "growth", "safety", "price")]
    usable = [p for p in parts if is_num(p)]
    return min(usable) if len(usable) == 4 else float("nan")


def assess(m: dict) -> dict:
    """Category, gate results and rank key for one company."""
    gates = gate_results(m)
    return {
        "category": categorise(m),
        "gates": gates,
        "gates_passed": sum(1 for v in gates.values() if v),
        "gates_total": len(gates),
        "qualifies": all(gates.values()),
        "weakest_pillar": weakest_pillar(m),
    }
