"""The numerical primitives must be total: nan in, nan out, never an exception."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from stocklab.util.mathx import (
    blend, cagr, clamp, first_finite, interp_score, irr_from_values, is_num,
    linreg, log_trend_growth, max_drawdown, pct_rank, safe_div, sum_finite,
    ulcer_index, winsorize, xirr, zero_if_nan, zscore,
)


class TestSafeDiv:
    def test_normal(self):
        assert safe_div(10, 4) == 2.5

    @pytest.mark.parametrize("num,den", [(1, 0), (None, 2), (2, None),
                                         (float("nan"), 2), (1, float("inf"))])
    def test_degenerate_returns_nan(self, num, den):
        assert math.isnan(safe_div(num, den))

    def test_explicit_default(self):
        assert safe_div(1, 0, default=0.0) == 0.0


class TestCagr:
    def test_known_value(self):
        assert cagr(100, 200, 5) == pytest.approx(0.148698, abs=1e-6)

    def test_non_positive_base_is_undefined(self):
        # Going from a loss to a profit has no meaningful growth *rate*;
        # returning one (e.g. via absolute values) ranks garbage to the top.
        assert math.isnan(cagr(-10, 100, 3))
        assert math.isnan(cagr(0, 100, 3))

    def test_zero_years(self):
        assert math.isnan(cagr(100, 200, 0))


class TestXirr:
    def test_doubling_over_five_years(self):
        flows = [(pd.Timestamp("2020-01-01"), -1000.0),
                 (pd.Timestamp("2025-01-01"), 2000.0)]
        assert xirr(flows) == pytest.approx(0.1487, abs=2e-3)

    def test_requires_a_sign_change(self):
        flows = [(pd.Timestamp("2020-01-01"), -100.0),
                 (pd.Timestamp("2021-01-01"), -100.0)]
        assert math.isnan(xirr(flows))

    def test_irregular_flows(self):
        flows = [(pd.Timestamp("2020-01-01"), -1000.0),
                 (pd.Timestamp("2020-07-01"), -500.0),
                 (pd.Timestamp("2024-01-01"), 2000.0)]
        rate = xirr(flows)
        assert is_num(rate) and 0.0 < rate < 0.3

    def test_periodic_helper(self):
        # NPV of [-100, 50, 60, 20] is zero at about 16.35%; the small shortfall
        # against that is the 365-day day-count used inside xirr.
        assert irr_from_values([-100, 50, 60, 20]) == pytest.approx(0.1635, abs=5e-3)


class TestCrossSectional:
    def test_winsorize_clips_outliers(self):
        series = pd.Series([1, 2, 3, 4, 5, 9000.0])
        assert winsorize(series).max() < 9000.0

    def test_zscore_centres_and_scales(self):
        z = zscore(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert z.mean() == pytest.approx(0.0, abs=1e-9)

    def test_robust_zscore_resists_an_outlier(self):
        base = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        polluted = pd.Series([1.0, 2.0, 3.0, 4.0, 5000.0])
        # The median/MAD version should barely move for the unaffected points.
        assert abs(zscore(polluted, robust=True).iloc[0]
                   - zscore(base, robust=True).iloc[0]) < 0.6

    def test_zscore_all_nan(self):
        assert zscore(pd.Series([np.nan, np.nan])).isna().all()

    def test_pct_rank_direction(self):
        series = pd.Series({"cheap": 5.0, "mid": 10.0, "rich": 20.0})
        ascending = pct_rank(series, ascending=True)
        assert ascending["cheap"] < ascending["rich"]


class TestDrawdown:
    def test_max_drawdown(self):
        equity = pd.Series([1.0, 1.2, 0.9, 1.1, 0.8])
        assert max_drawdown(equity) == pytest.approx(-1 / 3, abs=1e-9)

    def test_ulcer_penalises_duration(self):
        quick = pd.Series([1.0, 0.8, 1.0, 1.0, 1.0, 1.0])
        prolonged = pd.Series([1.0, 0.8, 0.8, 0.8, 0.8, 1.0])
        assert ulcer_index(prolonged) > ulcer_index(quick)

    def test_empty(self):
        assert math.isnan(max_drawdown(pd.Series(dtype=float)))


class TestTrend:
    def test_log_trend_on_clean_growth(self):
        growth, r2 = log_trend_growth(pd.Series([100 * 1.1**i for i in range(8)]))
        assert growth == pytest.approx(0.10, abs=1e-6)
        assert r2 == pytest.approx(1.0, abs=1e-9)

    def test_r2_falls_with_noise(self):
        _, clean = log_trend_growth(pd.Series([100 * 1.1**i for i in range(8)]))
        noisy = pd.Series([100, 180, 90, 200, 110, 240, 130, 260], dtype=float)
        _, messy = log_trend_growth(noisy)
        assert messy < clean

    def test_needs_positive_values(self):
        growth, _ = log_trend_growth(pd.Series([-1.0, -2.0, -3.0]))
        assert math.isnan(growth)

    def test_linreg_slope(self):
        slope, intercept, r2 = linreg([1.0, 2.0, 3.0, 4.0])
        assert slope == pytest.approx(1.0)
        assert r2 == pytest.approx(1.0)


class TestHelpers:
    def test_interp_score_endpoints_and_middle(self):
        points = [(0.0, 0.0), (0.10, 50.0), (0.25, 100.0)]
        assert interp_score(-1.0, points) == 0.0
        assert interp_score(1.0, points) == 100.0
        assert interp_score(0.10, points) == pytest.approx(50.0)
        assert interp_score(0.175, points) == pytest.approx(75.0, abs=1e-6)

    def test_blend_renormalises_over_available_inputs(self):
        # A missing model must not drag the blend toward zero.
        assert blend({"a": 10.0, "b": float("nan"), "c": 20.0},
                     {"a": 1.0, "b": 5.0, "c": 1.0}) == pytest.approx(15.0)

    def test_first_and_sum_finite(self):
        assert first_finite(float("nan"), None, 7.0) == 7.0
        assert math.isnan(first_finite(float("nan"), None))
        assert sum_finite(1.0, float("nan"), 2.0) == 3.0
        assert math.isnan(sum_finite(float("nan")))
        assert sum_finite(float("nan"), require_any=False) == 0.0

    def test_zero_if_nan_exists_because_or_fails_on_nan(self):
        assert (float("nan") or 0.0) != 0.0      # the trap this guards against
        assert zero_if_nan(float("nan")) == 0.0
        assert zero_if_nan(3.0) == 3.0

    def test_clamp(self):
        assert clamp(5.0, 0.0, 1.0) == 1.0
        assert math.isnan(clamp(float("nan"), 0.0, 1.0))
