"""The backtester's guard rails are the feature; these test them directly."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocklab.backtest import engine, metrics


@pytest.fixture
def panel() -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=1000)
    rng = np.random.default_rng(7)
    data = {}
    for i, symbol in enumerate(["A", "B", "C", "D", "E"]):
        steps = rng.normal(0.0003 + i * 0.0001, 0.013, len(index))
        data[symbol] = 100.0 * np.exp(np.cumsum(steps))
    return pd.DataFrame(data, index=index)


class TestGrossExposure:
    @pytest.mark.parametrize("top_n", [1, 2, 3, 5])
    def test_exposure_never_exceeds_one(self, panel, top_n):
        """Weights must not accumulate between rebalances.

        Carrying them with `replace(0, nan).ffill()` fills per column, so a name
        selected once keeps its weight forever and exposure ratchets toward the
        size of the universe -- silent leverage that inflated a ten-stock
        momentum backtest to a 50% annual return.
        """
        scores = panel.pct_change(21)
        weights = engine.weights_from_ranks(scores, top_n=top_n, frequency="M")
        gross = weights.sum(axis=1)
        assert gross.max() == pytest.approx(1.0, abs=1e-9)
        invested = gross[gross > 0]
        assert invested.min() == pytest.approx(1.0, abs=1e-9)

    def test_position_count_matches_top_n(self, panel):
        weights = engine.weights_from_ranks(panel.pct_change(21), top_n=2,
                                            frequency="M")
        held = (weights > 0).sum(axis=1)
        assert set(held.unique()) <= {0, 2}

    def test_max_weight_cap_is_respected(self, panel):
        weights = engine.weights_from_ranks(panel.pct_change(21), top_n=5,
                                            frequency="M", max_weight=0.10)
        assert weights.max().max() <= 0.10 + 1e-9


class TestLookAhead:
    def test_the_shift_destroys_same_bar_foresight(self, panel):
        """A rule ranking on the day's own return must not be profitable.

        Applied unshifted it is clairvoyant; through the engine it must collapse,
        because a weight derived from the close of t can only earn the return
        from t to t+1.
        """
        same_day = panel.pct_change()
        weights = engine.weights_from_ranks(same_day, top_n=2, frequency="D")
        honest = engine.run(panel, weights, slippage_bps=0, commission_per_trade=0)
        unshifted = (panel.pct_change().fillna(0.0) * weights).sum(axis=1)
        cheating_cagr = (1 + unshifted).prod() ** (252 / len(unshifted)) - 1
        assert cheating_cagr > 5.0                      # clairvoyant
        assert honest.summary["cagr"] < 0.5             # foresight removed

    def test_weights_are_shifted_by_exactly_one_bar(self, panel):
        weights = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
        weights.iloc[10, 0] = 1.0
        result = engine.run(panel, weights)
        assert result.weights.iloc[10, 0] == pytest.approx(0.0)
        assert result.weights.iloc[11, 0] == pytest.approx(1.0)


class TestCosts:
    def test_turnover_is_charged(self, panel):
        weights = engine.weights_from_ranks(panel.pct_change(5), top_n=2,
                                            frequency="D")
        free = engine.run(panel, weights, slippage_bps=0, commission_per_trade=0)
        costly = engine.run(panel, weights, slippage_bps=25, commission_per_trade=0)
        assert costly.summary["cagr"] < free.summary["cagr"]

    def test_gross_and_net_are_both_reported(self, panel):
        weights = engine.weights_from_ranks(panel.pct_change(5), top_n=2,
                                            frequency="D")
        result = engine.run(panel, weights, slippage_bps=30)
        assert result.summary["gross_cagr"] > result.summary["cagr"]

    def test_buy_and_hold_has_almost_no_turnover(self, panel):
        result = engine.run(panel, engine.buy_and_hold_weights(panel))
        assert result.summary["annual_turnover"] < 0.6


class TestWarnings:
    def test_survivorship_is_always_disclosed(self, panel):
        result = engine.run(panel, engine.buy_and_hold_weights(panel))
        assert any("urvivorship" in w for w in result.warnings)

    def test_high_turnover_is_flagged(self, panel):
        weights = engine.weights_from_ranks(panel.pct_change(2), top_n=1,
                                            frequency="D")
        result = engine.run(panel, weights)
        assert any("turnover" in w.lower() for w in result.warnings)

    def test_short_history_is_flagged(self, panel):
        short = panel.iloc[:200]
        result = engine.run(short, engine.buy_and_hold_weights(short))
        assert any("years of data" in w for w in result.warnings)


class TestMetrics:
    def test_longest_drawdown_counts_days_underwater(self):
        equity = pd.Series([1.0, 0.9, 0.9, 0.9, 1.1, 1.2])
        assert metrics.longest_drawdown(equity) == pytest.approx(3.0)

    def test_time_underwater_is_a_fraction(self):
        equity = pd.Series([1.0, 0.9, 1.1, 1.2])
        assert 0.0 < metrics.time_underwater(equity) < 1.0

    def test_tail_ratio_above_one_for_a_right_skewed_series(self):
        rng = np.random.default_rng(3)
        right_skewed = pd.Series(np.concatenate([rng.normal(0, 0.01, 200),
                                                 rng.normal(0.08, 0.01, 20)]))
        assert metrics.tail_ratio(right_skewed) > 1.0

    def test_deflated_sharpe_penalises_more_trials(self):
        single = metrics.deflated_sharpe(1.5, 1, 1000)
        many = metrics.deflated_sharpe(1.5, 100, 1000)
        assert single == pytest.approx(1.5)
        assert many < single

    def test_annual_table_computes_excess(self, panel):
        returns = panel["A"].pct_change().dropna()
        table = metrics.annual_table(returns, panel["B"].pct_change().dropna())
        assert "excess" in table
        assert len(table) >= 3

    def test_compare_is_sorted_by_calmar(self, panel):
        a = engine.run(panel, engine.buy_and_hold_weights(panel))
        b = engine.run(panel, engine.weights_from_ranks(panel.pct_change(21),
                                                        top_n=1, frequency="M"))
        frame = metrics.compare({"hold": a, "momentum": b})
        calmars = frame["calmar"].dropna().tolist()
        assert calmars == sorted(calmars, reverse=True)
