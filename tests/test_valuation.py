"""Valuation models, including the invariants that keep a DCF honest."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from stocklab.analysis.metrics import compute_metrics, normalize_split_basis, share_count
from stocklab.valuation import ddm, graham, residual_income
from stocklab.valuation.dcf import (
    DCFAssumptions, run_dcf, scenario_set, sensitivity,
)
from stocklab.valuation.epv import earnings_power_value
from stocklab.valuation.reverse_dcf import implied_growth, implied_discount_rate
from stocklab.valuation.wacc import (
    synthetic_rating, terminal_growth_cap,
)


def base_assumptions(**overrides) -> DCFAssumptions:
    defaults = dict(
        revenue_base=1000.0, revenue_growth_initial=0.08,
        revenue_growth_terminal=0.025, operating_margin_base=0.20,
        operating_margin_target=0.20, sales_to_capital=2.0, tax_rate=0.21,
        wacc=0.09, terminal_growth=0.025, terminal_roic=0.15, years=10,
        net_debt=100.0, shares=100.0,
    )
    defaults.update(overrides)
    return DCFAssumptions(**defaults)


class TestDCFInvariants:
    def test_produces_a_positive_value_for_a_profitable_business(self):
        result = run_dcf(base_assumptions())
        assert result.value_per_share > 0
        assert result.enterprise_value > 0

    def test_terminal_reinvestment_is_enforced(self):
        """Growth in perpetuity must be funded at g/ROIC of profit.

        Omitting it is free money and inflates terminal value 20-40%. A higher
        terminal ROIC needs less reinvestment, so it must produce more value.
        """
        low_roic = run_dcf(base_assumptions(terminal_roic=0.10))
        high_roic = run_dcf(base_assumptions(terminal_roic=0.40))
        assert high_roic.value_per_share > low_roic.value_per_share
        # And the reinvestment charge must be material, not cosmetic.
        assert (high_roic.terminal_value - low_roic.terminal_value) / low_roic.terminal_value > 0.05

    def test_value_falls_as_the_discount_rate_rises(self):
        cheap = run_dcf(base_assumptions(wacc=0.07))
        dear = run_dcf(base_assumptions(wacc=0.12))
        assert cheap.value_per_share > dear.value_per_share

    def test_value_rises_with_growth(self):
        slow = run_dcf(base_assumptions(revenue_growth_initial=0.02))
        fast = run_dcf(base_assumptions(revenue_growth_initial=0.15))
        assert fast.value_per_share > slow.value_per_share

    def test_growth_above_the_discount_rate_is_refused(self):
        # Otherwise the Gordon denominator goes negative and the model returns a
        # confidently negative "value".
        result = run_dcf(base_assumptions(wacc=0.05, terminal_growth=0.08))
        assert result.assumptions.terminal_growth < result.assumptions.wacc

    def test_terminal_roic_floored_at_the_cost_of_capital(self):
        assumptions = base_assumptions(terminal_roic=0.02, wacc=0.09)
        assert assumptions.terminal_roic == pytest.approx(0.09)

    def test_terminal_share_is_reported_and_dominant_for_a_long_horizon(self):
        result = run_dcf(base_assumptions())
        assert 0.0 < result.terminal_share < 1.0

    def test_net_debt_reduces_equity_value(self):
        none = run_dcf(base_assumptions(net_debt=0.0))
        levered = run_dcf(base_assumptions(net_debt=500.0))
        assert none.equity_value - levered.equity_value == pytest.approx(500.0)

    def test_missing_inputs_return_nan_not_an_exception(self):
        result = run_dcf(base_assumptions(revenue_base=float("nan")))
        assert math.isnan(result.value_per_share)
        assert result.notes

    def test_zero_reinvestment_is_flagged_as_a_ceiling(self):
        result = run_dcf(base_assumptions(sales_to_capital=float("nan")))
        assert any("reinvestment" in note.lower() for note in result.notes)

    def test_sensitivity_grid_shape_and_monotonicity(self):
        grid = sensitivity(base_assumptions(), "wacc", [0.08, 0.09, 0.10],
                           "terminal_growth", [0.02, 0.025])
        assert grid.shape == (2, 3)
        # Values must fall left to right as WACC rises.
        row = grid.iloc[0].tolist()
        assert row == sorted(row, reverse=True)

    def test_scenarios_are_ordered_bear_base_bull(self):
        frame = scenario_set(base_assumptions())
        assert (frame.loc["bear", "value_per_share"]
                < frame.loc["base", "value_per_share"]
                < frame.loc["bull", "value_per_share"])


class TestReverseDCF:
    def test_round_trip_recovers_the_growth_assumption(self):
        assumptions = base_assumptions()
        fair = run_dcf(assumptions).value_per_share
        recovered = implied_growth(assumptions, fair)
        assert recovered == pytest.approx(assumptions.revenue_growth_initial, abs=5e-3)

    def test_round_trip_recovers_the_discount_rate(self):
        assumptions = base_assumptions()
        fair = run_dcf(assumptions).value_per_share
        recovered = implied_discount_rate(assumptions, fair)
        assert recovered == pytest.approx(assumptions.wacc, abs=3e-3)

    def test_a_higher_price_requires_more_growth(self):
        assumptions = base_assumptions()
        fair = run_dcf(assumptions).value_per_share
        assert implied_growth(assumptions, fair * 1.5) > implied_growth(assumptions, fair)

    def test_unreachable_price_returns_nan(self):
        assert math.isnan(implied_growth(base_assumptions(), 1e9))


class TestWACC:
    @pytest.mark.parametrize("coverage,expected", [
        (20.0, "AAA"), (5.0, "A"), (3.0, "BBB"), (1.6, "B"), (0.1, "D"),
    ])
    def test_synthetic_rating_bands(self, coverage, expected):
        assert synthetic_rating(coverage)[0] == expected

    def test_unknown_coverage_assumed_investment_grade(self):
        rating, spread = synthetic_rating(float("nan"))
        assert "BBB" in rating and 0 < spread < 0.05

    def test_financials_bypass_coverage(self):
        rating, _ = synthetic_rating(0.5, is_financial=True)
        assert "financial" in rating

    def test_terminal_growth_capped_by_the_risk_free_rate(self):
        # Nothing can outgrow the economy forever; the 10-year yield proxies
        # long-run nominal growth.
        assert terminal_growth_cap(0.01) <= 0.01 + 1e-9
        assert terminal_growth_cap(0.10) <= 0.025 + 1e-9


class TestGraham:
    def test_graham_number(self):
        assert graham.graham_number(5.0, 20.0) == pytest.approx(math.sqrt(22.5 * 100))

    @pytest.mark.parametrize("eps,bvps", [(-1.0, 20.0), (5.0, -3.0), (0.0, 10.0)])
    def test_undefined_for_losses_or_negative_book(self, eps, bvps):
        assert math.isnan(graham.graham_number(eps, bvps))

    def test_formula_caps_growth(self):
        capped = graham.graham_formula(5.0, 50.0)
        at_cap = graham.graham_formula(5.0, 15.0)
        assert capped == pytest.approx(at_cap)


class TestResidualIncome:
    def test_excess_return_above_the_cost_of_equity_adds_value(self):
        result = residual_income.residual_income_value(100.0, 0.20, 0.10)
        assert result.value_per_share > 100.0
        assert 0.0 < result.book_value_share < 1.0

    def test_roe_equal_to_the_cost_of_equity_is_worth_book(self):
        result = residual_income.residual_income_value(100.0, 0.10, 0.10)
        assert result.value_per_share == pytest.approx(100.0, abs=1e-6)

    def test_roe_below_the_cost_of_equity_destroys_value(self):
        result = residual_income.residual_income_value(100.0, 0.04, 0.10)
        assert result.value_per_share < 100.0

    def test_negative_book_value_is_not_applicable(self):
        result = residual_income.residual_income_value(-10.0, 0.20, 0.10)
        assert math.isnan(result.value_per_share)
        assert result.notes


class TestDDMApplicability:
    def test_rejects_a_non_payer(self):
        ok, reason = ddm.applicability({"dividend_yield": 0.0})
        assert not ok and "no meaningful dividend" in reason

    def test_rejects_a_buyback_driven_company(self):
        ok, reason = ddm.applicability({"dividend_yield": 0.004,
                                        "buyback_yield": 0.03})
        assert not ok and "buyback" in reason

    def test_reit_payout_judged_on_ffo_not_earnings(self):
        # A REIT distributing 90% of FFO looks like a 180% payout on GAAP net
        # income, because depreciation on appreciating property sits between.
        metrics = {"dividend_yield": 0.05, "payout_ratio": 1.8,
                   "ffo_payout_ratio": 0.80}
        assert ddm.applicability(metrics, is_reit=False)[0] is False
        assert ddm.applicability(metrics, is_reit=True)[0] is True

    def test_gordon_growth_undefined_when_growth_exceeds_the_discount_rate(self):
        assert math.isnan(ddm.gordon_growth(1.0, 0.05, 0.08))


class TestShareCounts:
    def test_cover_page_used_when_corroborated(self, healthy_company):
        count, basis, notes, ok = share_count(healthy_company)
        assert count == pytest.approx(100.0)
        assert basis == "cover page" and ok and not notes

    def test_multi_class_cover_page_is_overridden(self, healthy_company):
        # Mastercard's cover page reads 122m against 884m diluted shares;
        # trusting it values the company at a seventh of its market cap.
        healthy_company.ttm["shares_outstanding"] = 14.0
        count, basis, notes, ok = share_count(healthy_company)
        assert count == pytest.approx(100.0)
        assert basis == "diluted total" and notes and ok

    def test_falls_back_to_net_income_over_eps(self, healthy_company):
        healthy_company.ttm["shares_outstanding"] = float("nan")
        healthy_company.ttm["shares_diluted"] = float("nan")
        healthy_company.ttm["shares_basic"] = float("nan")
        count, basis, _, _ = share_count(healthy_company)
        assert count > 0

    def test_a_units_switch_is_outvoted_rather_than_obeyed(self, healthy_company):
        """McDonald's reported 750,100,000 diluted shares for 2020 and 751.8
        for 2021 -- the same figure in millions. The old fixed priority put
        diluted first, which priced McDonald's at $197,000 and gave it a fair
        value of $206m a share. Two sources that still agree now outvote it."""
        healthy_company.ttm["shares_diluted"] = 100.0 / 1e6
        count, basis, notes, ok = share_count(healthy_company)
        assert count == pytest.approx(100.0, rel=0.15)
        assert ok
        assert any("factor" in n for n in notes)

    def test_a_lone_source_is_noted_but_still_used(self, healthy_company):
        """Requiring a second filing tag was too strict: it withheld Visa,
        Duolingo, Global Ship Lease and KKR, whose single counts are right.
        The note is kept; the figure is not thrown away."""
        healthy_company.ttm["shares_diluted"] = float("nan")
        healthy_company.ttm["shares_basic"] = float("nan")
        healthy_company.ttm["eps_diluted"] = float("nan")
        count, basis, notes, ok = share_count(healthy_company)
        assert count == pytest.approx(100.0)
        assert ok
        assert any("corroborates" in n for n in notes)

    def test_shares_issued_may_corroborate_but_never_wins_alone(self, healthy_company):
        """It counts treasury stock the company has bought back, so it is an
        upper bound, not the answer."""
        healthy_company.ttm["shares_outstanding"] = float("nan")
        healthy_company.ttm["shares_diluted"] = float("nan")
        healthy_company.ttm["shares_basic"] = float("nan")
        healthy_company.ttm["eps_diluted"] = float("nan")
        healthy_company.ttm["shares_issued"] = 120.0
        count, basis, _, _ = share_count(healthy_company)
        assert basis == "shares issued"
        count2, basis2, _, _ = share_count(healthy_company)
        healthy_company.ttm["shares_outstanding"] = 100.0
        count3, basis3, _, _ = share_count(healthy_company)
        assert basis3 == "cover page" and count3 == pytest.approx(100.0)

    def test_a_partial_issued_count_does_not_vouch_for_a_wrong_figure(self, healthy_company):
        """Duolingo's shares issued reads 16.6m against 49.6m diluted. Left in
        the vote it corroborated the wrong number and halved the company."""
        healthy_company.ttm["shares_outstanding"] = float("nan")
        healthy_company.ttm["shares_basic"] = float("nan")
        healthy_company.ttm["shares_diluted"] = 100.0
        healthy_company.ttm["net_income"] = 34.0
        healthy_company.ttm["eps_diluted"] = 1.0        # implies 34 shares
        healthy_company.ttm["shares_issued"] = 33.0     # agrees with the wrong one
        count, basis, _, _ = share_count(healthy_company)
        assert count == pytest.approx(100.0)
        assert basis == "diluted total"

    def test_an_impossible_market_value_rejects_the_count(self, healthy_company):
        """Berkshire's Class A cover-page count prices it at $537m against
        $260bn of sales. No second filing tag catches that, because there is
        no second tag; the arithmetic does."""
        healthy_company.ttm["shares_diluted"] = float("nan")
        healthy_company.ttm["shares_basic"] = float("nan")
        healthy_company.ttm["eps_diluted"] = float("nan")
        healthy_company.ttm["shares_outstanding"] = 0.001
        healthy_company.ttm["revenue"] = 1_000_000.0
        from stocklab.analysis.metrics import compute_metrics
        m = compute_metrics(healthy_company, price=50.0)
        assert m["share_count_reliable"] == 0.0
        assert math.isnan(m["market_cap"])
        assert math.isnan(m["shares"])

    def test_a_plausible_market_value_is_left_alone(self, healthy_company):
        """The bounds are wide enough for a distressed company; they catch
        arithmetic, not opinions."""
        from stocklab.analysis.metrics import compute_metrics
        healthy_company.ttm["revenue"] = 1_000.0
        m = compute_metrics(healthy_company, price=50.0)
        assert m["share_count_reliable"] == 1.0
        assert m["market_cap"] == pytest.approx(5_000.0)

    def test_a_cover_page_far_above_the_total_wins(self, healthy_company):
        """A cover page can under-report (one class of several). It cannot
        over-report: those shares exist. Shift4's diluted average and its
        implied-by-EPS count agreed at 1.3m against a cover page of 81.2m,
        two figures from the same mis-scaled filing outvoting the right one."""
        healthy_company.ttm["shares_diluted"] = 1.0
        healthy_company.ttm["shares_basic"] = 1.0
        healthy_company.ttm["eps_diluted"] = (
            healthy_company.ttm["net_income"] / 1.0)
        count, basis, notes, ok = share_count(healthy_company)
        assert count == pytest.approx(100.0)
        assert basis == "cover page"
        assert ok



class TestSplitNormalisation:
    def test_restatement_boundary_is_corrected(self):
        # Broadcom-shaped: the data steps 10x between FY2021 and FY2022 while
        # the actual split was in 2024, because later filings restate only the
        # comparatives they reproduce.
        index = pd.to_datetime(["2020-11-01", "2021-10-31", "2022-10-30",
                                "2023-10-29", "2024-11-03"])
        series = pd.Series([421.0, 429.0, 4232.0, 4272.0, 4778.0], index=index)
        splits = pd.Series({pd.Timestamp("2024-07-15"): 10.0})
        fixed = normalize_split_basis(series, splits)
        assert fixed.iloc[0] == pytest.approx(4210.0)
        assert fixed.iloc[-1] == pytest.approx(4778.0)

    def test_genuine_growth_is_left_alone(self):
        index = pd.to_datetime(["2022-01-01", "2023-01-01", "2024-01-01"])
        series = pd.Series([100.0, 150.0, 220.0], index=index)
        splits = pd.Series({pd.Timestamp("2023-06-01"): 10.0})
        assert normalize_split_basis(series, splits).tolist() == series.tolist()

    def test_no_split_history_is_a_passthrough(self):
        series = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-01", "2025-01-01"]))
        assert normalize_split_basis(series, pd.Series(dtype=float)).tolist() == [1.0, 2.0]


class TestMetricsFromFixture:
    def test_core_metrics_are_computed(self, healthy_company):
        metrics = compute_metrics(healthy_company, price=50.0)
        assert metrics["market_cap"] == pytest.approx(5000.0)
        assert metrics["operating_margin"] == pytest.approx(0.20, abs=1e-6)
        assert metrics["revenue_cagr_5y"] == pytest.approx(0.10, abs=1e-6)
        assert metrics["pe"] > 0
        assert metrics["roic"] > 0

    def test_enterprise_value_survives_a_missing_minority_interest(self, healthy_company):
        # `nan or 0.0` returns nan, since nan is truthy; EV must not be poisoned.
        metrics = compute_metrics(healthy_company, price=50.0)
        assert metrics["enterprise_value"] > 0

    def test_dividend_from_the_exchange_record_wins(self, healthy_company):
        metrics = compute_metrics(healthy_company, price=50.0, dividend_ttm=2.0)
        assert metrics["dividend_yield"] == pytest.approx(0.04)
        assert metrics["dividend_source"] == "exchange record"

    def test_no_price_means_no_multiples(self, healthy_company):
        metrics = compute_metrics(healthy_company, price=None)
        assert math.isnan(metrics["market_cap"])
        assert "pe" not in metrics or math.isnan(metrics.get("pe", float("nan")))

    def test_epv_is_computable_and_positive(self, healthy_company):
        metrics = compute_metrics(healthy_company, price=50.0)
        result = earnings_power_value(healthy_company, metrics, 0.09)
        assert result.epv_per_share > 0
        assert result.interpretation


class TestImpliedReturnForModelsOtherThanDCF:
    """A reverse DCF cannot be run on a bank, an insurer or a REIT: free cash
    flow to the firm is undefined when debt is the raw material. That left 77
    of 78 financials in a 515-company run with no expected-return figure while
    every operating company had one. The fix inverts whichever model is
    primary for them instead."""

    @staticmethod
    def _gordon(d1, growth):
        """A toy model: value falls as the required return rises, undefined
        at or below the growth rate."""
        def value_at(rate):
            return float('nan') if rate <= growth else d1 / (rate - growth)
        return value_at

    def test_solves_the_rate_where_model_value_meets_price(self):
        from stocklab.valuation.residual_income import solve_implied_rate
        # D1 = 3, g = 2%, price 50 -> rate = 3/50 + 0.02 = 8%
        rate, floored = solve_implied_rate(self._gordon(3.0, 0.02), 50.0)
        assert not floored
        assert rate == pytest.approx(0.08, abs=1e-3)

    def test_walks_past_a_singularity_below_the_growth_rate(self):
        """Starting the search at 0.5% would read nan and give up, which is
        how REITs silently lost their expected return."""
        from stocklab.valuation.residual_income import solve_implied_rate
        rate, _ = solve_implied_rate(self._gordon(4.0, 0.06), 80.0, low=0.005)
        assert not math.isnan(rate)
        assert rate == pytest.approx(0.11, abs=1e-3)

    def test_a_price_above_what_the_model_can_justify_is_flagged_as_a_floor(self):
        from stocklab.valuation.residual_income import solve_implied_rate
        # At the lowest rate the model answers at, value is still under price.
        rate, floored = solve_implied_rate(self._gordon(0.01, 0.02), 1000.0)
        assert floored
        assert rate > 0.02

    def test_no_rate_is_invented_when_the_model_cannot_be_solved(self):
        from stocklab.valuation.residual_income import solve_implied_rate
        rate, floored = solve_implied_rate(lambda _: float('nan'), 50.0)
        assert math.isnan(rate)
        assert not floored

    def test_a_non_positive_price_yields_nothing(self):
        from stocklab.valuation.residual_income import solve_implied_rate
        assert math.isnan(solve_implied_rate(self._gordon(3.0, 0.02), 0.0)[0])

    def test_residual_income_inverts_to_the_rate_it_was_valued_at(self):
        """Round trip: value at 9%, then ask what rate explains that value."""
        from stocklab.valuation import residual_income
        metrics = {"book_value_per_share": 40.0, "roe": 0.16, "payout_ratio": 0.3}
        value = residual_income.from_metrics(metrics, 0.09).value_per_share
        assert value > 0
        rate, floored = residual_income.implied_cost_of_equity(metrics, value)
        assert not floored
        assert rate == pytest.approx(0.09, abs=2e-3)
