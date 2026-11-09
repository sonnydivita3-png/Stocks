"""The translation layer from metrics to grades and plain sentences."""
from __future__ import annotations

import math

import pytest

from stocklab.scoring import (
    MIN_WEIGHT_FOR_GRADE, WEIGHTS, ScoreCard, grade_for, momentum_score,
    plain_language, price_score, quality_score, safety_score, score, size_band,
)


class FakeReport:
    def __init__(self, erp=None, mos=None, percentile=None):
        self.implied_equity_premium = erp
        self.margin_of_safety = mos
        self.relative = type("R", (), {"composite_percentile": percentile})() if percentile is not None else None


STRONG = {
    "roic_5y_avg": 0.28, "operating_margin": 0.30, "fcf_margin": 0.25,
    "cash_conversion": 1.1, "piotroski_f": 8, "revenue_cagr_3y": 0.18,
    "revenue_cagr_5y": 0.16, "revenue_trend_r2": 0.93, "fcf_cagr_5y": 0.19,
    "net_debt_to_ebitda": 0.3, "interest_coverage": 18, "altman_z": 6.0,
    "current_ratio": 1.9, "fcf_positive_years": 7, "fcf_yield": 0.05,
}
WEAK = {
    "roic_5y_avg": -0.08, "operating_margin": -0.25, "fcf_margin": -0.30,
    "cash_conversion": 0.1, "piotroski_f": 2, "revenue_cagr_3y": -0.05,
    "revenue_cagr_5y": -0.03, "revenue_trend_r2": 0.2, "net_debt_to_ebitda": 7.0,
    "interest_coverage": 0.6, "altman_z": 1.1, "current_ratio": 0.7,
    "fcf_positive_years": 0, "fcf_yield": -0.06,
}


class TestGrades:
    @pytest.mark.parametrize("value,expected", [
        (95, "A"), (80, "A"), (79, "B"), (68, "B"), (60, "C"),
        (54, "C"), (45, "D"), (40, "D"), (10, "F"), (0, "F"),
    ])
    def test_bands(self, value, expected):
        assert grade_for(value) == expected

    def test_missing_score_has_no_grade(self):
        assert grade_for(None) == "-"
        assert grade_for(float("nan")) == "-"


class TestSizeBands:
    @pytest.mark.parametrize("cap,expected", [
        (5e11, "mega"), (2e11, "mega"), (5e10, "large"), (1e10, "large"),
        (5e9, "mid"), (2e9, "mid"), (5e8, "small"), (3e8, "small"),
        (5e7, "micro"), (None, "unknown"),
    ])
    def test_bands(self, cap, expected):
        assert size_band(cap) == expected


class TestPillars:
    def test_strong_outscores_weak_on_every_pillar(self):
        for fn in (quality_score, safety_score):
            assert fn(STRONG)[0] > fn(WEAK)[0]

    def test_a_missing_metric_is_not_a_zero(self):
        """Judged on what exists, rather than penalised for a gap.

        A company that never reports one line item must not be dragged down as
        though that line item were zero.
        """
        full = quality_score(STRONG)[0]
        partial = dict(STRONG)
        del partial["piotroski_f"]
        del partial["cash_conversion"]
        assert quality_score(partial)[1] == 3
        assert abs(quality_score(partial)[0] - full) < 20   # close, not collapsed

    def test_no_data_gives_no_score(self):
        value, count = quality_score({})
        assert math.isnan(value) and count == 0

    def test_price_uses_own_history_when_available(self):
        cheap = price_score({"fcf_yield": 0.05}, FakeReport(erp=0.02, mos=0.0, percentile=0.05))
        rich = price_score({"fcf_yield": 0.05}, FakeReport(erp=0.02, mos=0.0, percentile=0.95))
        assert cheap[0] > rich[0]

    def test_price_without_a_report_still_scores_on_cash_yield(self):
        value, count = price_score({"fcf_yield": 0.08}, None)
        assert value > 0 and count >= 1


class TestOverall:
    def test_strong_company_grades_well(self):
        card = score(STRONG, {"momentum_12_1": 0.2, "dist_sma200": 0.08, "trend_r2": 0.7},
                     FakeReport(erp=0.03, mos=0.10))
        assert card.grade in ("A", "B")
        assert card.overall > 68

    def test_weak_company_grades_badly(self):
        card = score(WEAK, {}, FakeReport(erp=-0.01, mos=-0.4))
        assert card.grade in ("D", "F")

    def test_momentum_is_excluded_from_the_overall_score(self):
        """Price action must not inflate a grade.

        Two identical businesses, one with a soaring share price, must receive
        the same overall score; momentum is reported on its own.
        """
        flat = score(STRONG, {"momentum_12_1": -0.3, "dist_sma200": -0.2, "trend_r2": 0.2},
                     FakeReport(erp=0.03))
        soaring = score(STRONG, {"momentum_12_1": 0.8, "dist_sma200": 0.4, "trend_r2": 0.9},
                        FakeReport(erp=0.03))
        assert flat.overall == pytest.approx(soaring.overall)
        assert soaring.momentum > flat.momentum

    def test_grade_withheld_when_too_little_is_known(self):
        card = score({"current_ratio": 1.5}, {}, None)   # safety alone: 0.20 weight
        assert math.isnan(card.overall) and card.grade == "-"

    def test_weights_sum_to_one(self):
        assert sum(WEIGHTS.values()) == pytest.approx(1.0)
        assert MIN_WEIGHT_FOR_GRADE < 1.0

    def test_scorecard_serialises_flat(self):
        card = score(STRONG, {}, FakeReport(erp=0.03))
        out = card.as_dict()
        for key in ("quality", "growth", "safety", "price", "overall", "grade",
                    "say_quality", "say_price"):
            assert key in out


class TestPlainLanguage:
    def test_profitable_company_gets_a_dollar_sentence(self):
        says = plain_language({"roic_5y_avg": 0.22}, FakeReport(erp=0.03))
        assert "$22" in says["quality"]          # the figure, not the phrasing

    def test_loss_maker_is_said_plainly(self):
        says = plain_language({"operating_margin": -0.2}, None)
        assert "loses money" in says["quality"]

    def test_net_cash_is_said_plainly(self):
        assert "cash" in plain_language({"net_debt_to_ebitda": -1.5}, None)["safety"]
        assert "debt" in plain_language({"net_debt_to_ebitda": -1.5}, None)["safety"]

    def test_shrinking_sales_are_not_dressed_up(self):
        assert "backwards" in plain_language({"revenue_cagr_3y": -0.08}, None)["growth"]

    def test_expensive_price_is_said_plainly(self):
        says = plain_language({}, FakeReport(erp=-0.01))
        assert "government" in says["price"]     # the comparison, not the phrasing

    def test_every_pillar_always_gets_a_sentence(self):
        for metrics, report in (({}, None), (STRONG, FakeReport(erp=0.06)), (WEAK, None)):
            says = plain_language(metrics, report)
            assert set(says) == {"quality", "growth", "safety", "price"}
            assert all(v and v[0].isupper() and v.endswith(".") for v in says.values())


class TestSafetyEvidence:
    """Suppressed metrics must not read as a pass.

    Net debt, Altman Z and the current ratio are meaningless for a bank or
    insurer and are suppressed upstream. Scoring only the survivors gave
    insurers a perfect safety score off two favourable measures, and they swept
    the top grades on the strength of tests that had not been run.
    """

    FINANCIAL = {"is_financial": 1.0, "interest_coverage": 50.0,
                 "fcf_positive_years": 7, "equity_to_assets": 0.12}

    def test_financials_are_scored_on_equity_to_assets(self):
        from stocklab.scoring import safety_score
        strong = dict(self.FINANCIAL, equity_to_assets=0.25)
        thin = dict(self.FINANCIAL, equity_to_assets=0.05)
        assert safety_score(strong)[0] > safety_score(thin)[0]

    def test_a_thinly_capitalised_bank_does_not_score_perfectly(self):
        from stocklab.scoring import safety_score
        value, count = safety_score(dict(self.FINANCIAL, equity_to_assets=0.05))
        assert count >= 3
        assert value < 80

    def test_too_few_measures_yields_no_safety_score(self):
        from stocklab.scoring import MIN_SAFETY_PARTS, safety_score
        value, count = safety_score({"interest_coverage": 50.0, "fcf_positive_years": 7})
        assert count < MIN_SAFETY_PARTS
        assert math.isnan(value)

    def test_no_safety_score_still_allows_a_grade_from_the_others(self):
        """Safety carries 0.20, so the remaining 0.80 can still grade."""
        card = score({"roic_5y_avg": 0.25, "operating_margin": 0.25,
                      "revenue_cagr_3y": 0.15, "revenue_cagr_5y": 0.14,
                      "interest_coverage": 50.0, "fcf_yield": 0.05},
                     {}, FakeReport(erp=0.03))
        assert math.isnan(card.safety)
        assert not math.isnan(card.overall)

    def test_equity_bands_differ_for_financials(self):
        """12% equity/assets is sound for a bank and alarming for a manufacturer."""
        from stocklab.scoring import safety_score
        base = {"interest_coverage": 8.0, "fcf_positive_years": 5, "equity_to_assets": 0.12}
        bank = safety_score(dict(base, is_financial=1.0))[0]
        maker = safety_score(dict(base, is_financial=0.0))[0]
        assert bank > maker


class TestThinCoverageIsNotRewarded:
    """A company we barely measure must not outrank one we measure fully.

    Averaging only the sub-scores that exist meant two flattering readings beat
    five honest ones. Pillars are now shrunk toward 50 by whatever is missing.
    """

    def test_shrinkage_pulls_a_thin_pillar_toward_neutral(self):
        from stocklab.scoring import NEUTRAL, _pillar
        thin, n = _pillar([100.0, 100.0], expect=4)
        assert n == 2
        assert thin == pytest.approx((2 * 100.0 + 2 * NEUTRAL) / 4)
        assert thin < 100.0

    def test_full_coverage_is_the_plain_mean(self):
        from stocklab.scoring import _pillar
        value, n = _pillar([90.0, 70.0, 80.0, 60.0], expect=4)
        assert n == 4
        assert value == pytest.approx(75.0)

    def test_extra_sub_scores_are_never_penalised(self):
        from stocklab.scoring import _pillar
        value, n = _pillar([80.0, 80.0, 80.0, 80.0, 80.0], expect=3)
        assert n == 5
        assert value == pytest.approx(80.0)

    def test_fully_measured_beats_thinly_measured_at_equal_readings(self):
        from stocklab.scoring import quality_score
        thin = quality_score({"operating_margin": 0.40, "piotroski_f": 8})
        full = quality_score({"operating_margin": 0.40, "piotroski_f": 8,
                              "roic_5y_avg": 0.30, "fcf_margin": 0.28,
                              "cash_conversion": 1.2})
        assert full[1] > thin[1]
        assert full[0] > thin[0]

    def test_one_ratio_is_not_a_quality_score(self):
        from stocklab.scoring import MIN_QUALITY_PARTS, quality_score
        value, count = quality_score({"operating_margin": 0.45})
        assert count < MIN_QUALITY_PARTS
        assert math.isnan(value)


class TestInsurersAreNotFlatteredByFloat:
    """Premiums arrive years before claims, so an insurer's operating cash flow
    is float, not earning power. Scoring it as free cash flow put seven
    insurers in the top fourteen of a 515-company run."""

    INSURER = {"is_financial": 1.0, "operating_margin": 0.50, "piotroski_f": 7,
               "roe": 0.18, "fcf_margin": 0.95, "cash_conversion": 3.0}

    def test_cash_flow_lines_are_ignored_for_financials(self):
        from stocklab.scoring import quality_score
        with_cash = quality_score(self.INSURER)
        without = quality_score({k: v for k, v in self.INSURER.items()
                                 if k not in ("fcf_margin", "cash_conversion")})
        assert with_cash == without

    def test_return_on_equity_drives_the_financial_score(self):
        from stocklab.scoring import quality_score
        good = quality_score(dict(self.INSURER, roe=0.25))[0]
        poor = quality_score(dict(self.INSURER, roe=0.04))[0]
        assert good > poor + 15

    def test_an_insurer_no_longer_scores_near_perfect_on_two_lines(self):
        from stocklab.scoring import quality_score
        value, _ = quality_score({"is_financial": 1.0, "operating_margin": 0.50})
        assert math.isnan(value) or value < 85

    def test_the_plain_sentence_uses_equity_not_invested_capital(self):
        from stocklab.scoring import plain_language
        says = plain_language(self.INSURER)
        assert "owners" in says["quality"]       # equity, said plainly
        assert "Not enough history" not in says["quality"]


class TestInsurerSafetyIsNotACleanBillOfHealth:
    """Three of the six safety measures are meaningless for an insurer, and
    the three that remain all look excellent for a well-run one. The risk that
    actually sinks insurers -- reserves set too low against catastrophe
    exposure -- is in none of them, so three measures must not read as 100."""

    STRONG_INSURER = {"is_financial": 1.0, "equity_to_assets": 0.30,
                      "interest_coverage": 40.0, "fcf_positive_years": 7}

    def test_three_perfect_measures_do_not_make_a_perfect_score(self):
        from stocklab.scoring import safety_score
        value, count = safety_score(self.STRONG_INSURER)
        assert count == 3
        assert value < 95

    def test_a_fully_measured_company_is_not_shrunk(self):
        """Six measures is more than expected, so the result is the plain mean
        of the sub-scores with no pull toward neutral."""
        from stocklab.scoring import safety_score
        value, count = safety_score({
            "net_debt_to_ebitda": -1.0, "interest_coverage": 40.0, "altman_z": 9.0,
            "current_ratio": 3.0, "fcf_positive_years": 7, "equity_to_assets": 0.85})
        assert count == 6
        # Five sub-scores max out; a current ratio of 3.0 is worth 95, not 100.
        assert value == pytest.approx((100 * 5 + 95) / 6)
        assert value > 99

    def test_a_weak_insurer_still_scores_below_a_strong_one(self):
        from stocklab.scoring import safety_score
        weak = safety_score(dict(self.STRONG_INSURER, equity_to_assets=0.05))[0]
        assert weak < safety_score(self.STRONG_INSURER)[0] - 15

    def test_the_sentence_says_what_the_score_cannot_see(self):
        from stocklab.scoring import plain_language
        says = plain_language(self.STRONG_INSURER)
        assert "future claims" in says["safety"]
        assert "cannot be judged" not in says["safety"]
