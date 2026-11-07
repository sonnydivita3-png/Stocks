"""Indicators must match their original definitions, not an approximation."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from stocklab.analysis.risk import (
    beta_alpha, effective_holdings, kelly_fraction, return_stats, stress_test,
    to_returns, vol_target_weight,
)
from stocklab.analysis.technicals import (
    adx, atr, bollinger, indicator_frame, rsi, sma, technical_snapshot,
    true_range, wilder,
)


class TestWilderSmoothing:
    def test_wilder_equals_an_ema_with_span_2n_minus_1(self):
        """RSI, ATR and ADX use alpha=1/n, equivalent to span=2n-1.

        Using span=n makes every one of them roughly twice as twitchy as the
        indicator practitioners believe they are reading.
        """
        series = pd.Series(np.random.default_rng(0).normal(size=200))
        period = 14
        theirs = wilder(series, period)
        equivalent = series.ewm(span=2 * period - 1, adjust=False,
                                min_periods=period).mean()
        pd.testing.assert_series_equal(theirs, equivalent, check_names=False)

    def test_differs_materially_from_span_n(self):
        series = pd.Series(np.random.default_rng(1).normal(size=300))
        wrong = series.ewm(span=14, adjust=False, min_periods=14).mean()
        assert (wilder(series, 14) - wrong).abs().max() > 0.05


class TestRSI:
    def test_bounded_zero_to_hundred(self, price_history):
        values = rsi(price_history["close"], 14).dropna()
        assert values.min() >= 0.0 and values.max() <= 100.0

    def test_monotonically_rising_prices_give_rsi_100(self):
        rising = pd.Series(np.arange(1.0, 101.0))
        assert rsi(rising, 14).dropna().iloc[-1] == pytest.approx(100.0)

    def test_monotonically_falling_prices_give_rsi_zero(self):
        falling = pd.Series(np.arange(100.0, 0.0, -1.0))
        assert rsi(falling, 14).dropna().iloc[-1] == pytest.approx(0.0, abs=1e-9)


class TestATR:
    def test_true_range_uses_the_wider_of_three_measures(self):
        frame = pd.DataFrame({"high": [11.0, 12.0], "low": [9.0, 10.0],
                              "close": [10.0, 20.0]})
        tr = true_range(frame["high"], frame["low"], frame["close"])
        # Second bar gapped: |low - previous close| = 0, |high - prev close| = 2,
        # high-low = 2, so the range is 2.
        assert tr.iloc[1] == pytest.approx(2.0)

    def test_atr_is_positive(self, price_history):
        values = atr(price_history["high"], price_history["low"],
                     price_history["close"], 14).dropna()
        assert (values > 0).all()


class TestOtherIndicators:
    def test_bollinger_bands_are_ordered(self, price_history):
        lower, mid, upper = bollinger(price_history["close"])
        frame = pd.concat([lower, mid, upper], axis=1).dropna()
        assert (frame.iloc[:, 0] <= frame.iloc[:, 1]).all()
        assert (frame.iloc[:, 1] <= frame.iloc[:, 2]).all()

    def test_adx_bounded_and_direction_free(self, price_history):
        values, plus, minus = adx(price_history["high"], price_history["low"],
                                  price_history["close"], 14)
        clean = values.dropna()
        assert clean.min() >= 0.0 and clean.max() <= 100.0

    def test_sma_needs_a_full_window(self):
        series = pd.Series(np.arange(10.0))
        assert sma(series, 5).isna().sum() == 4

    def test_indicator_frame_adds_the_standard_set(self, price_history):
        frame = indicator_frame(price_history)
        for column in ("sma50", "sma200", "rsi14", "atr14", "adx14", "vol21",
                       "macd", "bb_width", "rel_volume"):
            assert column in frame


class TestSnapshot:
    def test_snapshot_is_populated(self, price_history):
        snap = technical_snapshot(price_history)
        for key in ("close", "rsi14", "atr_pct", "dist_sma200",
                    "pct_from_52w_high", "range_position_52w", "return_12m",
                    "momentum_12_1", "trend_template_score"):
            assert key in snap

    def test_short_history_returns_empty_rather_than_garbage(self):
        tiny = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0],
                             "close": [1.0], "volume": [1.0], "adj_close": [1.0]},
                            index=pd.to_datetime(["2025-01-01"]))
        assert technical_snapshot(tiny) == {}

    def test_trend_template_is_scored_out_of_eight(self, price_history):
        score = technical_snapshot(price_history)["trend_template_score"]
        assert 0.0 <= score <= 8.0

    def test_relative_strength_against_a_benchmark(self, price_history):
        benchmark = price_history["adj_close"] * 0.5
        snap = technical_snapshot(price_history, benchmark)
        # Identical shape means zero excess return, by construction.
        assert snap["excess_12m"] == pytest.approx(0.0, abs=1e-9)


class TestRiskStats:
    def test_panel_is_complete(self, price_history):
        stats = return_stats(to_returns(price_history["adj_close"]), risk_free=0.03)
        for key in ("cagr", "volatility", "sharpe", "sortino", "max_drawdown",
                    "var_hist", "cvar_hist", "ulcer_index"):
            assert key in stats

    def test_cvar_is_at_least_as_severe_as_var(self, price_history):
        stats = return_stats(to_returns(price_history["adj_close"]))
        assert stats["cvar_hist"] >= stats["var_hist"]

    def test_beta_of_an_asset_against_itself_is_one(self, price_history):
        returns = to_returns(price_history["adj_close"])
        stats = beta_alpha(returns, returns)
        assert stats["beta"] == pytest.approx(1.0, abs=1e-9)
        assert stats["r_squared"] == pytest.approx(1.0, abs=1e-9)

    def test_doubled_exposure_doubles_beta(self, price_history):
        returns = to_returns(price_history["adj_close"])
        assert beta_alpha(returns * 2.0, returns)["beta"] == pytest.approx(2.0, abs=1e-9)

    def test_effective_holdings_sees_through_a_position_count(self):
        # Twenty names with one at 40% does not behave like twenty positions.
        weights = pd.Series([0.40] + [0.6 / 19] * 19)
        assert effective_holdings(weights) < 8.0
        assert effective_holdings(pd.Series([0.1] * 10)) == pytest.approx(10.0)

    def test_stress_scales_with_beta(self):
        low = stress_test(100_000, 0.5)
        high = stress_test(100_000, 2.0)
        assert high["value_change"].min() < low["value_change"].min()

    def test_kelly_is_capped_and_floored(self):
        assert kelly_fraction(0.9, 5.0, cap=0.25) == pytest.approx(0.25)
        assert kelly_fraction(0.2, 1.0) == pytest.approx(0.0)
        assert math.isnan(kelly_fraction(float("nan"), 2.0))

    def test_vol_target_weight(self):
        assert vol_target_weight(0.30, 0.15) == pytest.approx(0.5)
        assert vol_target_weight(0.10, 0.15, max_weight=1.0) == pytest.approx(1.0)

    def test_empty_input_returns_empty_stats(self):
        assert return_stats(pd.Series(dtype=float)) == {}
