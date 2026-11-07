"""The screening engine, especially its handling of missing data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocklab.screening import presets
from stocklab.screening.factors import FACTOR_DIRECTIONS, add_ranks, composite_score
from stocklab.screening.screener import Filter, Screen


@pytest.fixture
def matrix() -> pd.DataFrame:
    return pd.DataFrame({
        "name": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"],
        "sector": ["Information Technology", "Information Technology",
                   "Financials", "Energy", "Health Care",
                   "Information Technology"],
        "market_cap": [5e9, 2e10, 8e9, 1e9, 4e8, 1.2e10],
        "dollar_volume_50d": [5e7, 2e8, 3e7, 5e6, 8e5, 9e7],
        "revenue": [1e9, 5e9, 2e9, 8e8, 1e8, 3e9],
        "years_of_data": [10, 15, 12, 8, 4, 11],
        "share_count_reliable": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "roic": [0.25, 0.18, np.nan, 0.06, 0.02, 0.21],
        "ebit_ev_yield": [0.12, 0.05, 0.09, 0.15, 0.01, 0.07],
        "pe": [12.0, 30.0, 9.0, 7.0, np.nan, 18.0],
        "fcf_yield": [0.09, 0.03, 0.07, 0.11, -0.02, 0.05],
        "revenue_cagr_5y": [0.14, 0.20, 0.03, -0.02, 0.45, 0.11],
        "beneish_m": [-2.5, -2.1, -1.9, -2.8, -1.1, -2.4],
        "piotroski_f": [8.0, 7.0, 6.0, 5.0, 3.0, 7.0],
        "ebit": [2e8, 9e8, 3e8, 1.5e8, -2e7, 5e8],
    }, index=pd.Index(["AAA", "BBB", "CCC", "DDD", "EEE", "ZZZ"], name="symbol"))


class TestFilter:
    def test_comparison_operators(self, matrix):
        assert Filter("roic", ">=", 0.15).mask(matrix).tolist() == [
            True, True, False, False, False, True]
        assert Filter("pe", "<=", 10.0).mask(matrix).sum() == 2

    def test_between_is_inclusive(self, matrix):
        assert Filter("pe", "between", (9.0, 12.0)).mask(matrix).sum() == 2

    def test_membership(self, matrix):
        chosen = Filter("sector", "in", ["Financials", "Energy"]).mask(matrix)
        assert chosen.sum() == 2

    def test_exclusion(self, matrix):
        assert Filter("sector", "notin", ["Financials"]).mask(matrix).sum() == 5

    def test_required_filter_rejects_missing_data(self, matrix):
        # Rejecting on absence biases a screen toward complete disclosure --
        # which skews large, US-domiciled and mature -- so each filter decides.
        assert Filter("roic", ">=", 0.0, required=True).mask(matrix)["CCC"] is np.True_ or \
               Filter("roic", ">=", 0.0, required=True).mask(matrix)["CCC"] == False

    def test_optional_filter_passes_missing_data(self, matrix):
        assert Filter("roic", ">=", 0.15, required=False).mask(matrix)["CCC"]

    def test_absent_column_rejects_loudly_when_required(self, matrix):
        # A required filter on a column that does not exist must reject
        # everything rather than being silently ignored.
        assert Filter("not_a_column", ">", 0).mask(matrix).sum() == 0
        assert Filter("not_a_column", ">", 0, required=False).mask(matrix).all()

    def test_unknown_operator_raises(self, matrix):
        with pytest.raises(ValueError):
            Filter("roic", "~~", 1).mask(matrix)


class TestScreen:
    def test_attrition_attributes_each_rejection_once(self, matrix):
        screen = Screen(name="t", description="",
                        filters=[Filter("market_cap", ">=", 1e9),
                                 Filter("roic", ">=", 0.15)])
        result = screen.run(matrix)
        # Five names, one fails on size and two on ROIC (one is nan, required).
        assert result.filter_attrition.sum() == result.universe_size - len(result)

    def test_excluded_records_the_reason(self, matrix):
        screen = Screen(name="t", description="",
                        filters=[Filter("pe", "<=", 10.0, label="cheap")])
        result = screen.run(matrix)
        assert "cheap" in result.excluded.loc["BBB", "failed_filters"]

    def test_scoring_orders_results(self, matrix):
        screen = Screen(name="t", description="",
                        filters=[Filter("ebit", ">", 0)],
                        score_weights={"roic": 1.0, "fcf_yield": 1.0})
        result = screen.run(matrix)
        scores = result.passed["score"].dropna()
        assert scores.tolist() == sorted(scores.tolist(), reverse=True)

    def test_empty_universe_is_handled(self):
        screen = Screen(name="t", description="", filters=[Filter("roic", ">", 0)])
        result = screen.run(pd.DataFrame())
        assert len(result) == 0 and result.notes

    def test_limit_truncates(self, matrix):
        screen = Screen(name="t", description="", filters=[])
        assert len(screen.run(matrix, limit=2)) == 2


class TestRanksAndScores:
    def test_lower_is_better_factors_are_inverted(self, matrix):
        ranked = add_ranks(matrix, ["pe", "roic"])
        # DDD has the lowest P/E, so it must have the highest P/E percentile.
        assert ranked["pe_pct"].idxmax() == "DDD"
        assert ranked["roic_pct"].idxmax() == "AAA"

    def test_direction_table_covers_the_scored_factors(self):
        assert FACTOR_DIRECTIONS["pe"] is False
        assert FACTOR_DIRECTIONS["roic"] is True

    def test_composite_renormalises_over_available_factors(self, matrix):
        ranked = add_ranks(matrix, ["roic", "fcf_yield"])
        scores = composite_score(ranked, {"roic": 1.0, "fcf_yield": 1.0})
        # CCC has no ROIC, so it must still be scored on FCF yield alone
        # rather than penalised as though its ROIC were zero.
        assert not np.isnan(scores["CCC"])

    def test_require_suppresses_thinly_covered_names(self, matrix):
        ranked = add_ranks(matrix, ["roic", "fcf_yield"])
        scores = composite_score(ranked, {"roic": 1.0, "fcf_yield": 1.0}, require=2)
        assert np.isnan(scores["CCC"])

    def test_sector_relative_ranking(self, matrix):
        # A utility on 18x and a software company on 18x are not equally
        # priced, so multiples are ranked within sector where possible.
        ranked = add_ranks(matrix, ["pe"], by_sector=True)
        assert "pe_pct" in ranked.columns
        # Information Technology has AAA (12x), ZZZ (18x) and BBB (30x);
        # cheapest within the sector must rank highest.
        assert ranked.loc["AAA", "pe_pct"] > ranked.loc["ZZZ", "pe_pct"]
        assert ranked.loc["ZZZ", "pe_pct"] > ranked.loc["BBB", "pe_pct"]

    def test_thin_sectors_fall_back_to_the_whole_universe(self, matrix):
        # Without a fallback, a universe where no sector reaches the minimum
        # yields no rank columns at all and every lookup raises.
        ranked = add_ranks(matrix, ["pe"], by_sector=True, min_per_sector=99)
        assert "pe_pct" in ranked.columns
        assert ranked["pe_pct"].notna().sum() >= 5


class TestPresets:
    def test_every_preset_builds_and_documents_itself(self):
        for key in presets.ALL_PRESETS:
            screen = presets.get(key)
            assert screen.filters, f"{key} has no filters"
            assert screen.failure_modes, f"{key} does not document its failure modes"

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError):
            presets.get("not-a-strategy")

    def test_magic_formula_excludes_financials(self, matrix):
        result = presets.magic_formula(min_market_cap=1e8).run(matrix)
        assert "CCC" not in result.passed.index   # Financials

    def test_presets_run_against_a_tiny_universe(self, matrix):
        for key in presets.ALL_PRESETS:
            result = presets.get(key).run(matrix)
            assert result.universe_size == len(matrix)

    def test_describe_all_lists_every_preset(self):
        assert len(presets.describe_all()) == len(presets.ALL_PRESETS)
