"""Turn the metric dictionary into grades and sentences a non-specialist reads.

The analysis modules speak in ROIC, EV/EBIT and implied equity premia. Most
people do not, and a page of ratios is not an answer to "is this any good".
This module is the translation layer: four scores out of 100, one letter grade,
and a plain sentence for each score.

Two decisions worth stating, because they shape every number here:

* **Momentum is excluded from the overall score.** What a share price has been
  doing lately is a trading signal, not a statement about whether a business is
  sound or sensibly priced. It is computed and reported separately so it cannot
  quietly inflate a grade.
* **A missing metric is not a zero, but it is not free either.** Each score
  averages the sub-scores that could actually be computed, so a company that
  never reports one line item is judged on the rest rather than penalised for
  the gap. But averaging only the survivors meant a thinly measured company
  could score higher than a fully measured one: two favourable readings out of
  five averaged to 95, while five honest readings averaged to 80. Scores built
  on fewer inputs than expected are therefore shrunk toward 50 in proportion to
  what is missing, and where too little exists to judge at all the score is
  ``None`` and the grade is withheld.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .util.mathx import NAN, interp_score, is_num

# Weights for the overall score. Quality leads because the evidence for buying
# good businesses is the strongest of the four; safety trails because it mostly
# screens out disasters rather than picking winners.
WEIGHTS = {"quality": 0.30, "price": 0.25, "growth": 0.25, "safety": 0.20}

# A grade needs most of the weight present, or it would rest on one pillar.
MIN_WEIGHT_FOR_GRADE = 0.60

GRADE_CUTS = ((80, "A"), (68, "B"), (54, "C"), (40, "D"))

SIZE_BANDS = ((2e11, "mega"), (1e10, "large"), (2e9, "mid"), (3e8, "small"))
SIZE_WORDS = {"mega": "Mega", "large": "Large", "mid": "Mid",
              "small": "Small", "micro": "Micro", "unknown": "Unknown"}


@dataclass
class ScoreCard:
    quality: float = NAN
    growth: float = NAN
    safety: float = NAN
    price: float = NAN
    momentum: float = NAN
    overall: float = NAN
    grade: str = "-"
    coverage: int = 0
    says: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"quality": self.quality, "growth": self.growth,
                "safety": self.safety, "price": self.price,
                "momentum": self.momentum, "overall": self.overall,
                "grade": self.grade, "coverage": self.coverage,
                **{f"say_{k}": v for k, v in self.says.items()}}


# The score a thinly measured company is pulled toward. 50 is the middle of the
# scale: "we do not know" should read as average, not as good and not as bad.
NEUTRAL = 50.0


def _pillar(parts, expect: int | None = None) -> tuple[float, int]:
    """Average the sub-scores that exist, shrunk toward neutral when thin.

    ``expect`` is how many sub-scores a company of this kind ought to produce.
    Each missing one counts as a pseudo-observation at ``NEUTRAL``, so a pillar
    measured on two of five inputs lands roughly halfway between its own
    average and 50. With everything present the shrinkage term is zero and the
    result is the plain mean. ``nan`` when nothing could be computed.
    """
    values = [p for p in parts if is_num(p)]
    if not values:
        return NAN, 0
    mean = sum(values) / len(values)
    if expect is None:
        return mean, len(values)
    missing = max(0, expect - len(values))
    if not missing:
        return mean, len(values)
    return (len(values) * mean + missing * NEUTRAL) / (len(values) + missing), len(values)


def size_band(market_cap) -> str:
    if not is_num(market_cap):
        return "unknown"
    for floor, name in SIZE_BANDS:
        if market_cap >= floor:
            return name
    return "micro"


def grade_for(score) -> str:
    if not is_num(score):
        return "-"
    for cut, letter in GRADE_CUTS:
        if score >= cut:
            return letter
    return "F"


# Expected sub-score counts, used for the shrinkage in ``_pillar``. Financials
# legitimately produce fewer, so they are held to a lower bar rather than
# penalised for an accounting convention they did not choose.
EXPECT_QUALITY, EXPECT_QUALITY_FIN = 4, 3
EXPECT_GROWTH = 3
# Four, not three, on purpose. An operating company produces five or six of
# the six safety measures. An insurer produces exactly three, and all three
# look excellent for a well-run one -- while the thing that actually sinks an
# insurer, reserves set too low against catastrophe exposure, appears in none
# of them. Expecting four means those three are shrunk a quarter of the way to
# neutral: strong on what can be seen, not a clean bill of health.
EXPECT_SAFETY = 4
EXPECT_PRICE = 3

# A quality score resting on a single ratio is not a judgement either.
MIN_QUALITY_PARTS = 2


def quality_score(m: dict) -> tuple[float, int]:
    """How much profit the business earns on the money put into it.

    For a bank or insurer, return on *invested capital* is not defined in any
    useful way and free cash flow is dominated by float and reserve movements:
    premiums arrive years before claims are paid, so operating cash flow looks
    enormous and says nothing about the quality of the underwriting. Scoring
    insurers on those lines put seven of them in the top fourteen. Financials
    are scored on return on equity instead, and their cash-flow lines are
    dropped rather than flattered.
    """
    is_financial = bool(m.get("is_financial"))
    roic = m.get("roic_5y_avg") if is_num(m.get("roic_5y_avg")) else m.get("roic")
    parts = [
        interp_score(m.get("operating_margin"), [(-0.15, 0), (0.0, 35), (0.10, 60), (0.22, 85), (0.35, 100)]),
        interp_score(m.get("piotroski_f"), [(2, 5), (4, 35), (6, 65), (8, 90), (9, 100)]),
    ]
    if is_financial:
        # Return on equity, banded tighter: 15% on equity is a good bank, not a
        # merely adequate one, and leverage means the tail is not as long.
        parts.append(interp_score(m.get("roe"),
                     [(0.0, 5), (0.06, 30), (0.11, 58), (0.17, 82), (0.28, 100)]))
        expect = EXPECT_QUALITY_FIN
    else:
        parts.append(interp_score(roic,
                     [(-0.05, 0), (0.05, 30), (0.12, 60), (0.20, 82), (0.35, 100)]))
        parts.append(interp_score(m.get("fcf_margin"),
                     [(-0.15, 0), (0.0, 35), (0.08, 60), (0.20, 88), (0.32, 100)]))
        # Cash conversion catches profit that never turns into money.
        parts.append(interp_score(m.get("cash_conversion"),
                     [(0.2, 10), (0.7, 50), (1.0, 80), (1.4, 100)]))
        expect = EXPECT_QUALITY
    value, count = _pillar(parts, expect)
    if count < MIN_QUALITY_PARTS:
        return NAN, count
    return value, count


def growth_score(m: dict) -> tuple[float, int]:
    """How fast sales are rising, and how steadily rather than in one lucky year."""
    return _pillar([
        interp_score(m.get("revenue_cagr_3y"), [(-0.10, 0), (0.0, 25), (0.08, 50), (0.20, 78), (0.40, 100)]),
        interp_score(m.get("revenue_cagr_5y"), [(-0.10, 0), (0.0, 25), (0.08, 50), (0.20, 78), (0.40, 100)]),
        # r-squared of a log-trend fit: a steady climb scores well, the same
        # total growth delivered in one jump does not.
        interp_score(m.get("revenue_trend_r2"), [(0.2, 15), (0.5, 45), (0.75, 75), (0.95, 100)]),
        interp_score(m.get("fcf_cagr_5y"), [(-0.15, 0), (0.0, 35), (0.12, 70), (0.30, 100)]),
    ], EXPECT_GROWTH)


# A safety score built from one or two surviving sub-scores is not a judgement,
# it is an absence of one. Below this many, no score is produced.
MIN_SAFETY_PARTS = 3


def safety_score(m: dict) -> tuple[float, int]:
    """Whether it could survive a bad year.

    Three of the five general measures -- net debt to profit, Altman Z and the
    current ratio -- are meaningless for a bank or insurer and are suppressed
    upstream for them. Scoring only the survivors treated that suppression as a
    pass: insurers came out at 100 on two favourable sub-scores and swept the A
    grades. So financials are scored on equity as a share of assets, which is
    the measure that does apply to them, and any company with fewer than
    ``MIN_SAFETY_PARTS`` usable measures gets no safety score at all.
    """
    is_financial = bool(m.get("is_financial"))
    equity_band = ([(0.04, 5), (0.07, 35), (0.10, 60), (0.15, 85), (0.25, 100)]
                   if is_financial else
                   [(0.10, 5), (0.25, 35), (0.40, 65), (0.60, 90), (0.80, 100)])
    value, count = _pillar([
        interp_score(m.get("net_debt_to_ebitda"),
                     [(-1.0, 100), (0.0, 90), (1.5, 70), (3.0, 45), (5.0, 15), (8.0, 0)]),
        interp_score(m.get("interest_coverage"), [(0.5, 0), (2.0, 35), (5.0, 70), (12.0, 95), (25.0, 100)]),
        interp_score(m.get("altman_z"), [(1.0, 5), (1.8, 30), (3.0, 70), (5.0, 92), (8.0, 100)]),
        interp_score(m.get("current_ratio"), [(0.6, 10), (1.0, 40), (1.6, 75), (2.5, 95)]),
        interp_score(m.get("fcf_positive_years"), [(0, 5), (3, 45), (5, 80), (7, 100)]),
        interp_score(m.get("equity_to_assets"), equity_band),
    ], EXPECT_SAFETY)
    if count < MIN_SAFETY_PARTS:
        return NAN, count
    return value, count


def price_score(m: dict, report=None) -> tuple[float, int]:
    """Whether today's price is sensible for what the business earns."""
    erp = getattr(report, "implied_equity_premium", None)
    mos = getattr(report, "margin_of_safety", None)
    parts = [
        interp_score(erp, [(-0.02, 0), (0.0, 25), (0.02, 50), (0.05, 78), (0.10, 100)]),
        interp_score(m.get("fcf_yield"), [(-0.02, 5), (0.0, 20), (0.03, 45), (0.06, 72), (0.12, 100)]),
        interp_score(mos, [(-0.60, 0), (-0.30, 25), (0.0, 55), (0.30, 85), (0.60, 100)]),
    ]
    score, count = _pillar(parts, EXPECT_PRICE)
    relative = getattr(report, "relative", None)
    percentile = getattr(relative, "composite_percentile", None) if relative else None
    if is_num(percentile):
        # Inverted: cheap against the company's own ten-year range scores well.
        own_history = interp_score(1.0 - percentile,
                                   [(0.0, 5), (0.3, 35), (0.5, 55), (0.75, 82), (1.0, 100)])
        score, count = _pillar([score, own_history])
    return score, count


def momentum_score(t: dict) -> tuple[float, int]:
    """What the share price has been doing. Reported, never folded into the grade."""
    return _pillar([
        interp_score(t.get("momentum_12_1"), [(-0.40, 0), (-0.10, 25), (0.10, 55), (0.35, 85), (0.80, 100)]),
        interp_score(t.get("dist_sma200"), [(-0.25, 0), (-0.05, 35), (0.05, 60), (0.20, 88), (0.40, 100)]),
        interp_score(t.get("trend_r2"), [(0.1, 20), (0.5, 55), (0.85, 95)]),
    ])


def plain_language(m: dict, report=None) -> dict[str, str]:
    """One sentence per score, in words that need no finance background."""
    says: dict[str, str] = {}

    roic = m.get("roic_5y_avg") if is_num(m.get("roic_5y_avg")) else m.get("roic")
    operating = m.get("operating_margin")
    roe = m.get("roe")
    if bool(m.get("is_financial")) and is_num(roe) and roe > 0:
        says["quality"] = (f"Earns about ${roe * 100:.0f} a year for every $100 of "
                           "shareholders' money in the business.")
    elif is_num(roic) and roic > 0:
        says["quality"] = (f"Turns every $100 invested in the business into "
                           f"${roic * 100:.0f} of profit a year.")
    elif is_num(operating) and operating < 0:
        says["quality"] = "Still loses money on its core operations."
    else:
        says["quality"] = "Not enough history to judge how profitably it operates."

    growth = m.get("revenue_cagr_3y")
    if not is_num(growth):
        says["growth"] = "No usable three-year sales history."
    elif growth > 0.25:
        says["growth"] = f"Sales growing fast, about {growth * 100:.0f}% a year over three years."
    elif growth > 0.08:
        says["growth"] = f"Sales growing steadily, about {growth * 100:.0f}% a year over three years."
    elif growth > 0:
        says["growth"] = f"Sales barely growing, about {growth * 100:.0f}% a year."
    else:
        says["growth"] = f"Sales shrinking, about {growth * 100:.0f}% a year."

    debt = m.get("net_debt_to_ebitda")
    equity = m.get("equity_to_assets")
    if bool(m.get("is_financial")) and is_num(equity):
        says["safety"] = (f"Holds ${equity * 100:.0f} of its own money behind every $100 "
                          "of assets. Whether it has set aside enough for future claims "
                          "is not visible in these figures.")
    elif not is_num(debt):
        says["safety"] = "Debt load cannot be judged from the filings."
    elif debt < 0:
        says["safety"] = "Holds more cash than debt."
    elif debt < 2:
        says["safety"] = f"Modest borrowing: about {debt:.1f} years of profit."
    elif debt < 4:
        says["safety"] = f"Noticeable borrowing: about {debt:.1f} years of profit."
    else:
        says["safety"] = f"Heavy borrowing: about {debt:.1f} years of profit."

    erp = getattr(report, "implied_equity_premium", None)
    if not is_num(erp):
        says["price"] = "Too little profit history to work out what the price assumes."
    elif erp >= 0.05:
        says["price"] = (f"At this price you would earn roughly {erp * 100:.1f}% a year more "
                         "than a government bond, if the business performs as its history suggests.")
    elif erp >= 0.02:
        says["price"] = f"Priced to pay about {erp * 100:.1f}% a year more than a government bond."
    elif erp >= 0:
        says["price"] = f"Priced to pay only about {erp * 100:.1f}% a year more than a government bond."
    else:
        says["price"] = "Priced to earn less than a government bond, on its own history."
    return says


def score(metrics: dict, technicals: dict | None = None, report=None) -> ScoreCard:
    """Full scorecard for one company."""
    technicals = technicals or {}
    quality, q_n = quality_score(metrics)
    growth, g_n = growth_score(metrics)
    safety, s_n = safety_score(metrics)
    price, p_n = price_score(metrics, report)
    momentum, _ = momentum_score(technicals)

    parts = {"quality": quality, "growth": growth, "safety": safety, "price": price}
    numerator = denominator = 0.0
    for name, weight in WEIGHTS.items():
        if is_num(parts[name]):
            numerator += parts[name] * weight
            denominator += weight
    overall = (numerator / denominator) if denominator >= MIN_WEIGHT_FOR_GRADE else NAN

    return ScoreCard(
        quality=quality, growth=growth, safety=safety, price=price,
        momentum=momentum, overall=overall, grade=grade_for(overall),
        coverage=q_n + g_n + s_n + p_n, says=plain_language(metrics, report),
    )
