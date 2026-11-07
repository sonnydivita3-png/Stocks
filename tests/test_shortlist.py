"""The screen that answers "should I own this", as against "is this good"."""
from __future__ import annotations

import math

import pytest

from stocklab.shortlist import (GATES, assess, categorise, gate_results,
                                weakest_pillar)

# A company that clears everything: profitable, well covered, good, safe,
# growing, cheap enough, and the methods agree.
QUALIFIER = {
    "share_count_reliable": 1.0, "profitable": True, "coverage": 16,
    "quality": 80.0, "growth": 70.0, "safety": 75.0, "price": 72.0,
    "fair_value": 120.0, "margin_of_safety": 0.15,
    "implied_equity_premium": 0.04, "implied_return_is_floor": False,
    "confidence": "moderate", "revenue_cagr_3y": 0.10, "roic_5y_avg": 0.22,
    "dividend_yield": 0.01,
}


class TestEveryConditionMustHold:
    """The point of a conjunctive screen: nothing compensates for anything."""

    def test_the_reference_company_qualifies(self):
        assert assess(QUALIFIER)["qualifies"]

    @pytest.mark.parametrize("field,value", [
        ("share_count_reliable", 0.0),
        ("profitable", False),
        ("coverage", 11),
        ("quality", 55.0),
        ("safety", 40.0),
        ("growth", 20.0),
        ("implied_equity_premium", -0.01),
        ("margin_of_safety", -0.05),
        ("confidence", "low"),
    ])
    def test_failing_any_one_condition_disqualifies(self, field, value):
        assert not assess({**QUALIFIER, field: value})["qualifies"]

    def test_excellence_elsewhere_cannot_buy_off_a_bad_price(self):
        """The whole reason this exists: 62% of A and B graded companies in a
        515-name run traded above their own fair value estimate."""
        superb = {**QUALIFIER, "quality": 100.0, "growth": 100.0,
                  "safety": 100.0, "margin_of_safety": -0.30}
        assert not assess(superb)["qualifies"]
        assert assess(superb)["gates_passed"] == len(GATES) - 1

    def test_a_bounded_return_is_not_a_positive_one(self):
        """Where a price exceeds what the model can justify at any sensible
        rate, the premium reported is a ceiling, not a measurement."""
        floored = {**QUALIFIER, "implied_equity_premium": 0.01,
                   "implied_return_is_floor": True}
        assert not gate_results(floored)["beats_treasury"]

    def test_an_estimate_wilder_than_the_market_is_refused(self):
        """Comcast came out 248% below fair value. Ranking by an estimate
        selects for estimation error."""
        wild = {**QUALIFIER, "margin_of_safety": 2.48}
        assert not gate_results(wild)["estimate_sane"]
        assert gate_results({**QUALIFIER, "margin_of_safety": 0.9})["estimate_sane"]

    def test_every_gate_carries_a_reason_a_reader_can_act_on(self):
        for gate in GATES:
            assert gate.label and gate.why
            assert gate.key in gate_results(QUALIFIER)


class TestRankingByTheWeakestPillar:
    def test_a_balanced_company_outranks_a_lopsided_one(self):
        balanced = {"quality": 70.0, "growth": 72.0, "safety": 75.0, "price": 71.0}
        lopsided = {"quality": 98.0, "growth": 95.0, "safety": 92.0, "price": 40.0}
        assert weakest_pillar(balanced) > weakest_pillar(lopsided)

    def test_a_missing_pillar_leaves_no_rank(self):
        assert math.isnan(weakest_pillar({"quality": 70.0, "growth": 72.0,
                                          "safety": 75.0}))


class TestCategories:
    def test_fast_sales_growth_is_growth(self):
        assert categorise({**QUALIFIER, "revenue_cagr_3y": 0.35}) == "growth"

    def test_high_returns_and_steady_growth_is_a_compounder(self):
        assert categorise({**QUALIFIER, "revenue_cagr_3y": 0.09,
                           "roic_5y_avg": 0.25}) == "compounder"

    def test_a_payer_with_little_growth_is_income(self):
        assert categorise({**QUALIFIER, "revenue_cagr_3y": 0.01,
                           "roic_5y_avg": 0.04, "quality": 50.0,
                           "dividend_yield": 0.045}) == "income"

    def test_falling_sales_is_shrinking(self):
        assert categorise({**QUALIFIER, "revenue_cagr_3y": -0.06,
                           "roic_5y_avg": 0.04, "quality": 50.0,
                           "dividend_yield": 0.0}) == "shrinking"

    def test_a_loss_maker_is_speculative_however_fast_it_grows(self):
        assert categorise({**QUALIFIER, "profitable": False,
                           "revenue_cagr_3y": 1.5}) == "speculative"

    def test_a_company_we_cannot_size_is_its_own_label(self):
        """Berkshire's share count could not be established; calling it
        speculative would be wrong, and calling it anything else dishonest."""
        assert categorise({**QUALIFIER, "share_count_reliable": 0.0}) == "unclear"

    def test_category_is_about_the_business_not_the_price(self):
        dear = {**QUALIFIER, "margin_of_safety": -0.5, "revenue_cagr_3y": 0.35}
        assert categorise(dear) == "growth"
        assert not assess(dear)["qualifies"]
