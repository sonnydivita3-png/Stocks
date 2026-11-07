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
    # Earnings that repeat: five profitable years, this one in line with them.
    "profitable_years": 5.0, "profitable_years_of": 5.0,
    "earnings_vs_own_average": 1.2,
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

    def test_profitable_but_unvaluable_is_not_called_speculative(self):
        """Apollo and First Bancorp are large, profitable and long
        established. What is missing is a valuation worth standing behind,
        not the earnings, and a reader without a finance background should
        not be told those are the same thing."""
        assert categorise({**QUALIFIER, "fair_value": float("nan")}) == "hard_to_value"
        assert categorise({**QUALIFIER, "coverage": 4}) == "hard_to_value"

    def test_a_company_we_cannot_size_is_its_own_label(self):
        """Berkshire's share count could not be established; calling it
        speculative would be wrong, and calling it anything else dishonest."""
        assert categorise({**QUALIFIER, "share_count_reliable": 0.0}) == "unclear"

    def test_category_is_about_the_business_not_the_price(self):
        dear = {**QUALIFIER, "margin_of_safety": -0.5, "revenue_cagr_3y": 0.35}
        assert categorise(dear) == "growth"
        assert not assess(dear)["qualifies"]


class TestTheBuilderImportsDoNotCollide:
    """stocklab.analysis.regime and stocklab.shortlist both export `assess`.
    Importing both under one name shadowed the regime's, and every snapshot
    after that silently shipped without a market backdrop -- three missing
    files and a caught "'str' object has no attribute 'get'"."""

    def test_the_two_assess_functions_are_distinct(self):
        from stocklab.analysis.regime import assess as regime_assess
        from stocklab.shortlist import assess as shortlist_assess
        assert regime_assess is not shortlist_assess

    def test_the_builder_does_not_bind_both_to_one_name(self):
        import ast
        import pathlib
        source = pathlib.Path("scripts/build_desk_snapshot.py").read_text()
        bound: dict[str, str] = {}
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.ImportFrom):
                continue
            for alias in node.names:
                name = alias.asname or alias.name
                assert name not in bound, (
                    f"{node.module}.{alias.name} is imported as {name!r}, which "
                    f"{bound.get(name)} already binds")
                bound[name] = node.module or ""


class TestEarningsThatDoNotRepeat:
    """Every model here is anchored on recent earnings, so a single good year
    after several bad ones is read as the normal state of the business.
    Compugen lost money in four of its last five years; one licensing payment
    turned a negative five-year average into a profit, scored as a 591%
    return on capital, and put it on a list headed "worth buying now"."""

    BASE = {
        "share_count_reliable": 1.0, "profitable": True, "coverage": 16,
        "quality": 80.0, "growth": 70.0, "safety": 75.0, "price": 72.0,
        "fair_value": 120.0, "margin_of_safety": 0.15,
        "implied_equity_premium": 0.04, "confidence": "moderate",
        "profitable_years": 5.0, "profitable_years_of": 5.0,
        "earnings_vs_own_average": 1.2,
    }

    def test_one_good_year_in_five_does_not_qualify(self):
        assert not assess({**self.BASE, "profitable_years": 1.0})["qualifies"]

    def test_four_of_five_is_enough(self):
        assert assess({**self.BASE, "profitable_years": 4.0})["qualifies"]

    def test_a_year_far_above_its_own_average_does_not_qualify(self):
        """Nutex earned ten times its own five-year average, on arbitration
        awards rather than operations. The filings cannot say that; the
        arithmetic can say the year is not representative."""
        assert not assess({**self.BASE, "earnings_vs_own_average": 10.0})["qualifies"]
        assert not gate_results({**self.BASE,
                                 "earnings_vs_own_average": 2.5})["not_a_windfall"]

    def test_ordinary_growth_is_not_a_windfall(self):
        assert gate_results({**self.BASE,
                             "earnings_vs_own_average": 1.9})["not_a_windfall"]

    def test_a_short_history_cannot_demonstrate_repetition(self):
        assert not gate_results({**self.BASE, "profitable_years": 2.0,
                                 "profitable_years_of": 2.0})["profits_repeat"]

    def test_an_unmeasurable_peak_does_not_block_on_its_own(self):
        """A negative five-year average leaves no ratio to test; the
        profitable-years gate is what catches those."""
        assert gate_results({**self.BASE,
                             "earnings_vs_own_average": float("nan")})["not_a_windfall"]


class TestThreeScreensNotOne:
    """A single condition set was a value screen wearing a general-purpose
    label. Over 515 companies it passed one growth company out of 58 and none
    of 116 speculative ones -- because demanding repeated profits, a discount
    to a conservative valuation and a return above a Treasury rules out a
    fast-growing company for being fast-growing, and an unproven one by
    definition."""

    GROWER = {
        "share_count_reliable": 1.0, "revenue_reliable": 1.0,
        "revenue_cagr_3y": 0.30, "revenue_cagr_5y": 0.28, "revenue_trend_r2": 0.92,
        "gross_margin": 0.62, "gross_margin_delta_3y": 0.02,
        "operating_margin": 0.14, "operating_margin_delta_3y": 0.05,
        "share_count_cagr_3y": 0.01, "rule_of_40": 48.0, "fcf": 500.0,
        "net_debt_to_ebitda": 0.4, "current_ratio": 2.4,
        "implied_growth": 0.22, "historical_growth": 0.30,
    }
    PUNT = {
        "share_count_reliable": 1.0, "revenue_reliable": 1.0, "profitable": False,
        "revenue": 180.0, "revenue_cagr_3y": 0.55, "gross_margin": 0.38,
        "operating_margin_delta_3y": 0.08, "cash_runway_years": 3.4,
        "share_count_cagr_3y": 0.08, "ev_sales": 6.0, "current_ratio": 2.9,
    }

    def test_a_grower_passes_the_growth_screen(self):
        assert assess(self.GROWER)["growth"]["qualifies"]

    def test_the_same_grower_need_not_pass_the_value_screen(self):
        """It has no fair value, no premium over a Treasury and no five-year
        profit record. On one screen that was a rejection; it is now simply a
        different question."""
        assert not assess(self.GROWER)["value"]["qualifies"]

    def test_paying_for_growth_never_delivered_fails(self):
        """The reverse DCF says what the price assumes. Requiring more than
        the company has done is the definition of paying up for hope."""
        dear = {**self.GROWER, "implied_growth": 0.45, "historical_growth": 0.30}
        assert not gate_results(dear, "growth")["priced_within_record"]

    def test_growth_bought_by_issuing_stock_fails(self):
        diluted = {**self.GROWER, "share_count_cagr_3y": 0.12}
        assert not gate_results(diluted, "growth")["not_bought_with_stock"]

    def test_growth_on_collapsing_margins_fails(self):
        thin = {**self.GROWER, "gross_margin_delta_3y": -0.11}
        assert not gate_results(thin, "growth")["unit_economics"]

    def test_one_jump_is_not_a_trend(self):
        lumpy = {**self.GROWER, "revenue_trend_r2": 0.35}
        assert not gate_results(lumpy, "growth")["steady_climb"]

    def test_a_punt_with_runway_and_economics_passes_the_punt_screen(self):
        assert assess(self.PUNT)["speculative"]["qualifies"]

    def test_a_punt_without_runway_does_not(self):
        """The only question that matters for something unproven is whether
        it survives long enough for the thesis to be settled."""
        assert not gate_results({**self.PUNT, "cash_runway_years": 0.8,
                                 "fcf": -50.0}, "speculative")["runway"]

    def test_an_established_company_is_not_offered_as_a_punt(self):
        """The speculative conditions are looser by design, so without this
        Nvidia and Duolingo passed a list headed 'speculative'."""
        established = {**self.GROWER, "profitable": True, "fair_value": 100.0,
                       "coverage": 16}
        assert not gate_results(established, "speculative")["unproven"]
        assert not assess(established)["speculative"]["qualifies"]

    def test_fast_growth_is_no_longer_mistaken_for_a_windfall(self):
        """A grower should earn more than its own five-year average; that is
        what growth is. A windfall is a profit spike with no sales behind it."""
        grew = {"earnings_vs_own_average": 3.0, "revenue_cagr_3y": 0.30}
        spiked = {"earnings_vs_own_average": 3.0, "revenue_cagr_3y": 0.01}
        assert gate_results(grew, "value")["not_a_windfall"]
        assert not gate_results(spiked, "value")["not_a_windfall"]
