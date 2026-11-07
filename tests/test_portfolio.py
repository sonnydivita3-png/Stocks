"""Lot accounting, tax treatment, rebalancing and sizing."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from stocklab.portfolio import rebalance, sizing, tax
from stocklab.portfolio.ledger import replay, select_lots
from stocklab.portfolio.models import (
    Account, AccountType, Lot, Transaction, TxnType,
)
from stocklab.portfolio.store import PortfolioStore


def accounts(*specs):
    out = {}
    for index, (name, kind) in enumerate(specs, start=1):
        out[index] = Account(name=name, type=kind, id=index)
    return out


class TestAccountTypeRules:
    def test_tax_classification(self):
        assert AccountType.TRADITIONAL_IRA.tax_deferred
        assert AccountType.ROTH_IRA.tax_free
        assert AccountType.TAXABLE.taxable_now
        assert AccountType.HSA.sheltered

    def test_only_taxable_accounts_allow_loss_harvesting(self):
        # Selling at a loss in an IRA achieves nothing: the loss is not
        # deductible and basis is irrelevant to how withdrawals are taxed.
        assert AccountType.TAXABLE.allows_loss_harvest
        assert not AccountType.ROTH_IRA.allows_loss_harvest
        assert not AccountType.TRADITIONAL_401K.allows_loss_harvest


class TestLotSelection:
    def _lots(self):
        return [
            Lot(1, "X", date(2020, 1, 1), 100, 10.0, id=1),
            Lot(1, "X", date(2022, 1, 1), 100, 50.0, id=2),
            Lot(1, "X", date(2023, 1, 1), 100, 30.0, id=3),
        ]

    def test_fifo_takes_the_oldest(self):
        picks = select_lots(self._lots(), 150, "fifo")
        assert [lot.id for lot, _ in picks] == [1, 2]

    def test_lifo_takes_the_newest(self):
        picks = select_lots(self._lots(), 150, "lifo")
        assert [lot.id for lot, _ in picks] == [3, 2]

    def test_hifo_minimises_the_realised_gain(self):
        fifo = sum(q * lot.cost_per_share for lot, q in select_lots(self._lots(), 150, "fifo"))
        hifo = sum(q * lot.cost_per_share for lot, q in select_lots(self._lots(), 150, "hifo"))
        # A higher basis means a smaller gain; FIFO is the broker default and
        # usually the worst choice in an appreciated position.
        assert hifo > fifo

    def test_specific_lots_respect_the_requested_order(self):
        picks = select_lots(self._lots(), 150, "specific", specific_ids=(3, 1))
        assert [lot.id for lot, _ in picks] == [3, 1]

    def test_cannot_oversell(self):
        picks = select_lots(self._lots(), 1000, "fifo")
        assert sum(q for _, q in picks) == pytest.approx(300.0)


class TestReplay:
    def test_buy_then_sell_realises_the_right_gain(self):
        acct = accounts(("Taxable", AccountType.TAXABLE))
        state = replay([
            Transaction(1, date(2023, 1, 10), TxnType.BUY, "AAA", 100, 10.0, id=1),
            Transaction(1, date(2025, 2, 1), TxnType.SELL, "AAA", 40, 25.0, id=2),
        ], acct)
        assert len(state.realized) == 1
        assert state.realized[0].gain == pytest.approx(600.0)
        assert state.realized[0].is_long_term
        assert state.open_lots()[0].remaining == pytest.approx(60.0)

    def test_commission_enters_the_cost_basis(self):
        acct = accounts(("Taxable", AccountType.TAXABLE))
        state = replay([
            Transaction(1, date(2023, 1, 10), TxnType.BUY, "AAA", 100, 10.0,
                        fees=20.0, id=1),
        ], acct)
        assert state.open_lots()[0].cost_per_share == pytest.approx(10.20)

    def test_split_preserves_total_basis(self):
        acct = accounts(("Taxable", AccountType.TAXABLE))
        state = replay([
            Transaction(1, date(2020, 1, 2), TxnType.BUY, "AAA", 100, 25.0, id=1),
            Transaction(1, date(2020, 8, 31), TxnType.SPLIT, "AAA", price=4.0, id=2),
        ], acct)
        lot = state.open_lots()[0]
        assert lot.quantity == pytest.approx(400.0)
        assert lot.cost_per_share == pytest.approx(6.25)
        assert lot.cost_basis == pytest.approx(2500.0)

    def test_holding_period_boundary_is_strictly_more_than_a_year(self):
        acct = accounts(("Taxable", AccountType.TAXABLE))
        open_date = date(2024, 1, 1)
        state = replay([
            Transaction(1, open_date, TxnType.BUY, "AAA", 10, 10.0, id=1),
            Transaction(1, open_date + timedelta(days=365), TxnType.SELL,
                        "AAA", 10, 12.0, id=2),
        ], acct)
        # Exactly 365 days is still short-term: the rule is *more than* 12 months.
        assert state.realized[0].term == "short"

    def test_overselling_is_clipped_and_warned(self):
        acct = accounts(("Taxable", AccountType.TAXABLE))
        state = replay([
            Transaction(1, date(2024, 1, 1), TxnType.BUY, "AAA", 10, 10.0, id=1),
            Transaction(1, date(2024, 6, 1), TxnType.SELL, "AAA", 50, 12.0, id=2),
        ], acct)
        assert state.warnings
        assert sum(g.quantity for g in state.realized) == pytest.approx(10.0)

    def test_dividends_and_fees_move_cash(self):
        acct = accounts(("Taxable", AccountType.TAXABLE))
        state = replay([
            Transaction(1, date(2024, 1, 1), TxnType.DEPOSIT, amount=1000.0, id=1),
            Transaction(1, date(2024, 3, 1), TxnType.DIVIDEND, "AAA", amount=50.0, id=2),
            Transaction(1, date(2024, 4, 1), TxnType.FEE, amount=10.0, id=3),
        ], acct)
        assert state.total_cash(1) == pytest.approx(1040.0)
        assert state.income[1] == pytest.approx(50.0)

    def test_return_of_capital_reduces_basis(self):
        acct = accounts(("Taxable", AccountType.TAXABLE))
        state = replay([
            Transaction(1, date(2024, 1, 1), TxnType.BUY, "AAA", 100, 10.0, id=1),
            Transaction(1, date(2024, 6, 1), TxnType.RETURN_OF_CAPITAL, "AAA",
                        amount=200.0, id=2),
        ], acct)
        assert state.open_lots()[0].cost_per_share == pytest.approx(8.0)

    def test_external_flows_are_recorded_for_return_calculations(self):
        acct = accounts(("Taxable", AccountType.TAXABLE))
        state = replay([
            Transaction(1, date(2024, 1, 1), TxnType.DEPOSIT, amount=1000.0, id=1),
            Transaction(1, date(2024, 6, 1), TxnType.WITHDRAWAL, amount=200.0, id=2),
        ], acct)
        assert [amount for _, amount in state.external_flows] == [1000.0, -200.0]


class TestWashSales:
    def test_repurchase_within_thirty_days_disallows_the_loss(self):
        acct = accounts(("Taxable", AccountType.TAXABLE))
        state = replay([
            Transaction(1, date(2026, 1, 5), TxnType.BUY, "XYZ", 100, 50.0, id=1),
            Transaction(1, date(2026, 3, 2), TxnType.SELL, "XYZ", 100, 30.0, id=2),
            Transaction(1, date(2026, 3, 10), TxnType.BUY, "XYZ", 100, 31.0, id=3),
        ], acct)
        loss = state.realized[0]
        assert loss.gain == pytest.approx(-2000.0)
        assert loss.wash_sale_disallowed == pytest.approx(2000.0)
        assert loss.taxable_gain == pytest.approx(0.0)

    def test_deferred_loss_is_added_to_the_replacement_basis(self):
        acct = accounts(("Taxable", AccountType.TAXABLE))
        state = replay([
            Transaction(1, date(2026, 1, 5), TxnType.BUY, "XYZ", 100, 50.0, id=1),
            Transaction(1, date(2026, 3, 2), TxnType.SELL, "XYZ", 100, 30.0, id=2),
            Transaction(1, date(2026, 3, 10), TxnType.BUY, "XYZ", 100, 31.0, id=3),
        ], acct)
        replacement = [lot for lot in state.lots if lot.open_date == date(2026, 3, 10)][0]
        assert replacement.wash_sale_adjustment == pytest.approx(2000.0)

    def test_replacement_in_a_retirement_account_destroys_the_loss(self):
        # Rev. Rul. 2008-5: no taxable basis exists in a sheltered account for
        # the adjustment to attach to, so the loss is gone for good.
        acct = accounts(("Brokerage", AccountType.TAXABLE), ("Roth", AccountType.ROTH_IRA))
        state = replay([
            Transaction(1, date(2026, 1, 5), TxnType.BUY, "XYZ", 100, 50.0, id=1),
            Transaction(1, date(2026, 3, 2), TxnType.SELL, "XYZ", 100, 30.0, id=2),
            Transaction(2, date(2026, 3, 10), TxnType.BUY, "XYZ", 100, 31.0, id=3),
        ], acct)
        loss = state.realized[0]
        assert loss.wash_sale_disallowed == pytest.approx(2000.0)
        assert "PERMANENTLY" in loss.notes

    def test_repurchase_outside_the_window_is_clean(self):
        acct = accounts(("Taxable", AccountType.TAXABLE))
        state = replay([
            Transaction(1, date(2026, 1, 5), TxnType.BUY, "XYZ", 100, 50.0, id=1),
            Transaction(1, date(2026, 3, 2), TxnType.SELL, "XYZ", 100, 30.0, id=2),
            Transaction(1, date(2026, 5, 1), TxnType.BUY, "XYZ", 100, 31.0, id=3),
        ], acct)
        assert state.realized[0].wash_sale_disallowed == pytest.approx(0.0)

    def test_a_gain_is_never_a_wash_sale(self):
        acct = accounts(("Taxable", AccountType.TAXABLE))
        state = replay([
            Transaction(1, date(2026, 1, 5), TxnType.BUY, "XYZ", 100, 30.0, id=1),
            Transaction(1, date(2026, 3, 2), TxnType.SELL, "XYZ", 100, 50.0, id=2),
            Transaction(1, date(2026, 3, 10), TxnType.BUY, "XYZ", 100, 51.0, id=3),
        ], acct)
        assert state.realized[0].wash_sale_disallowed == pytest.approx(0.0)


class TestTax:
    def _state(self):
        acct = accounts(("Taxable", AccountType.TAXABLE), ("Roth", AccountType.ROTH_IRA))
        txns = [
            Transaction(1, date(2023, 1, 10), TxnType.BUY, "AAA", 100, 100.0, id=1),
            Transaction(1, date(2026, 2, 1), TxnType.SELL, "AAA", 100, 160.0, id=2),
            Transaction(1, date(2026, 3, 1), TxnType.BUY, "BBB", 200, 50.0, id=3),
            Transaction(1, date(2026, 9, 1), TxnType.SELL, "BBB", 200, 30.0, id=4),
            Transaction(1, date(2025, 6, 1), TxnType.BUY, "CCC", 100, 80.0, id=5),
            Transaction(2, date(2025, 6, 1), TxnType.BUY, "DDD", 100, 40.0, id=6),
        ]
        return replay(txns, acct), acct

    def test_short_and_long_gains_net_against_each_other(self):
        state, acct = self._state()
        report = tax.annual_report(state, acct, 2026)
        # $6,000 long-term gain less a $4,000 short-term loss leaves $2,000 long.
        assert report.long_term_gains == pytest.approx(2000.0)
        assert report.short_term_gains == pytest.approx(0.0)
        assert report.estimated_tax == pytest.approx(300.0)

    def test_net_loss_respects_the_three_thousand_limit(self):
        acct = accounts(("Taxable", AccountType.TAXABLE))
        state = replay([
            Transaction(1, date(2024, 1, 10), TxnType.BUY, "AAA", 100, 100.0, id=1),
            Transaction(1, date(2026, 2, 1), TxnType.SELL, "AAA", 100, 20.0, id=2),
        ], acct)
        report = tax.annual_report(state, acct, 2026)
        assert report.deductible_against_income == pytest.approx(3000.0)
        assert report.loss_carryforward == pytest.approx(5000.0)

    def test_sheltered_accounts_are_excluded_from_the_report(self):
        acct = accounts(("Roth", AccountType.ROTH_IRA))
        state = replay([
            Transaction(1, date(2024, 1, 10), TxnType.BUY, "AAA", 100, 10.0, id=1),
            Transaction(1, date(2026, 2, 1), TxnType.SELL, "AAA", 100, 50.0, id=2),
        ], acct)
        assert tax.annual_report(state, acct, 2026).realized_detail.empty

    def test_harvest_candidates_skip_sheltered_accounts(self):
        state, acct = self._state()
        candidates = tax.harvest_candidates(state, acct, {"CCC": 55.0, "DDD": 25.0},
                                            min_loss=100)
        symbols = {c.symbol for c in candidates}
        assert "CCC" in symbols      # taxable
        assert "DDD" not in symbols  # Roth: the loss is not deductible

    def test_asset_location_flags_both_directions(self):
        state, acct = self._state()
        review = tax.asset_location_review(
            state, acct, {"CCC": 55.0, "DDD": 25.0},
            {"CCC": "reit", "DDD": "growth_equity"})
        verdicts = dict(zip(review["symbol"], review["verdict"]))
        assert "misplaced" in verdicts["CCC"]     # heavily taxed, held taxable
        assert "suboptimal" in verdicts["DDD"]    # tax-efficient, held sheltered

    def test_municipal_bonds_in_a_sheltered_account_are_flagged(self):
        acct = accounts(("Roth", AccountType.ROTH_IRA))
        state = replay([Transaction(1, date(2024, 1, 1), TxnType.BUY, "MUB",
                                    100, 100.0, id=1)], acct)
        review = tax.asset_location_review(state, acct, {"MUB": 100.0},
                                           {"MUB": "municipal_bond"})
        assert "wasted" in review["verdict"].iloc[0]

    def test_days_to_long_term_is_actionable(self):
        acct = accounts(("Taxable", AccountType.TAXABLE))
        open_date = date.today() - timedelta(days=340)
        state = replay([Transaction(1, open_date, TxnType.BUY, "AAA", 10, 10.0, id=1)], acct)
        frame = tax.unrealized_frame(state, acct, {"AAA": 12.0})
        assert frame["days_to_long_term"].iloc[0] == pytest.approx(26)
        assert frame["term"].iloc[0] == "short"


class TestRebalance:
    def _state(self):
        acct = accounts(("Taxable", AccountType.TAXABLE), ("Roth", AccountType.ROTH_IRA))
        txns = [
            Transaction(1, date(2020, 1, 2), TxnType.BUY, "AAA", 500, 20.0, id=1),
            Transaction(2, date(2020, 1, 2), TxnType.BUY, "AAA", 300, 20.0, id=2),
            Transaction(1, date(2024, 1, 2), TxnType.BUY, "BBB", 200, 50.0, id=3),
            Transaction(1, date(2020, 1, 2), TxnType.DEPOSIT, amount=30000.0, id=4),
            Transaction(2, date(2020, 1, 2), TxnType.DEPOSIT, amount=10000.0, id=5),
        ]
        return replay(txns, acct), acct

    def test_drift_band_requires_both_breaches(self):
        weights = pd.Series({"AAA": 0.42, "BBB": 0.58})
        targets = pd.Series({"AAA": 0.40, "BBB": 0.60})
        frame = rebalance.drift_table(weights, targets,
                                      relative_band=0.25, absolute_band=0.05)
        # 2 points of drift is 5% relative: inside both bands, so no trade.
        assert not frame["out_of_band"].any()

    def test_large_drift_triggers_a_trade(self):
        state, acct = self._state()
        plan = rebalance.plan(state, acct, {"AAA": 100.0, "BBB": 40.0},
                              pd.Series({"AAA": 0.40, "BBB": 0.40, "CASH": 0.20}))
        assert plan.trades

    def test_sells_are_routed_to_the_sheltered_account_first(self):
        state, acct = self._state()
        plan = rebalance.plan(state, acct, {"AAA": 100.0, "BBB": 40.0},
                              pd.Series({"AAA": 0.40, "BBB": 0.40, "CASH": 0.20}))
        sells = [t for t in plan.trades if t.side == "sell"]
        assert sells and sells[0].account_name == "Roth"
        assert sells[0].estimated_tax == pytest.approx(0.0)

    def test_tax_budget_blocks_an_expensive_sell(self):
        state, acct = self._state()
        plan = rebalance.plan(state, acct, {"AAA": 100.0, "BBB": 40.0},
                              pd.Series({"AAA": 0.40, "BBB": 0.40, "CASH": 0.20}),
                              max_tax_budget=0.0)
        assert plan.estimated_tax == pytest.approx(0.0)
        assert any("budget" in note for note in plan.notes)

    def test_no_targets_is_an_error_not_a_silent_no_op(self):
        state, acct = self._state()
        plan = rebalance.plan(state, acct, {}, pd.Series(dtype=float))
        assert not plan.trades and plan.notes

    def test_weights_are_normalised_when_they_do_not_sum_to_one(self):
        state, acct = self._state()
        plan = rebalance.plan(state, acct, {"AAA": 100.0, "BBB": 40.0},
                              pd.Series({"AAA": 0.8, "BBB": 0.8}))
        assert any("normalised" in note for note in plan.notes)


class TestSizing:
    def test_equal_weight_sums_to_one(self):
        result = sizing.equal_weight(["A", "B", "C", "D"])
        assert result.weights.sum() == pytest.approx(1.0)

    def test_inverse_volatility_favours_the_quiet_asset(self):
        vols = pd.Series({"QUIET": 0.10, "WILD": 0.40})
        weights = sizing.inverse_volatility(vols).weights
        assert weights["QUIET"] > weights["WILD"]

    def test_risk_parity_equalises_marginal_risk_contributions(self):
        symbols = ["B1", "B2", "T"]
        import numpy as np
        corr = np.array([[1, .9, .2], [.9, 1, .2], [.2, .2, 1]])
        sd = np.array([.25, .25, .30])
        cov = pd.DataFrame(corr * np.outer(sd, sd), index=symbols, columns=symbols)
        weights = sizing.risk_parity(cov).weights
        matrix = cov.to_numpy()
        w = weights.to_numpy()
        port_vol = float(np.sqrt(w @ matrix @ w))
        contributions = w * (matrix @ w) / port_vol
        assert contributions.max() - contributions.min() < 1e-6
        # The diversifier earns a larger weight than the correlated pair.
        assert weights["T"] > weights["B1"]

    def test_atr_sizing_risks_exactly_the_budget(self):
        result = sizing.atr_position_size(100_000, 150.0, 4.5,
                                          risk_per_trade=0.01, atr_multiple=2.0)
        assert result["risk_pct_of_capital"] == pytest.approx(0.01, abs=5e-4)
        assert result["stop_price"] == pytest.approx(141.0)

    def test_atr_sizing_respects_the_position_cap(self):
        result = sizing.atr_position_size(100_000, 150.0, 0.5,
                                          risk_per_trade=0.01, max_weight=0.10)
        assert result["weight"] <= 0.10 + 1e-9
        assert result["notes"]

    def test_kelly_is_capped(self):
        result = sizing.kelly_weights(pd.Series({"A": 0.50}), pd.Series({"A": 0.20}),
                                       fraction=0.25, max_weight=0.20)
        assert result.weights["A"] <= 0.20 + 1e-9

    def test_regime_scaling_leaves_the_remainder_in_cash(self):
        weights = pd.Series({"A": 0.5, "B": 0.5})
        scaled = sizing.apply_regime(weights, 0.6)
        assert scaled.sum() == pytest.approx(0.6)


class TestStore:
    def test_round_trip(self, tmp_path):
        store = PortfolioStore(tmp_path / "p.sqlite")
        account_id = store.add_account(Account(name="Main", type=AccountType.TAXABLE))
        store.add_transaction(Transaction(account_id, date(2025, 1, 2), TxnType.BUY,
                                          "AAA", 10, 100.0))
        assert len(store.accounts()) == 1
        assert store.get_account("Main").type is AccountType.TAXABLE
        assert len(store.transactions()) == 1
        assert store.symbols() == ["AAA"]

    def test_transactions_are_ordered_deterministically(self, tmp_path):
        store = PortfolioStore(tmp_path / "p.sqlite")
        account_id = store.add_account(Account(name="Main", type=AccountType.TAXABLE))
        for price in (10.0, 20.0, 30.0):
            store.add_transaction(Transaction(account_id, date(2025, 1, 2),
                                              TxnType.BUY, "AAA", 1, price))
        # Same-day trades must replay in a fixed order or basis depends on
        # query order.
        assert [t.price for t in store.transactions()] == [10.0, 20.0, 30.0]

    def test_targets_and_watchlist(self, tmp_path):
        store = PortfolioStore(tmp_path / "p.sqlite")
        store.set_target("core", "AAA", 0.6)
        store.set_target("core", "BBB", 0.4)
        assert store.targets("core").sum() == pytest.approx(1.0)
        store.set_target("core", "AAA", 0.5)        # upsert, not duplicate
        assert store.targets("core")["AAA"] == pytest.approx(0.5)
        store.watch("CCC", thesis="cheap")
        assert "CCC" in list(store.watchlist()["symbol"])
        store.unwatch("CCC")
        assert store.watchlist().empty


class TestStagedEntry:
    def test_no_tranche_is_placed_below_the_stop(self):
        """A limit below the stop would fill into an invalidated position.

        It also quietly turns a staged entry into averaging down through the
        exit level, which is the opposite of what the stop is for.
        """
        from stocklab.entry.stops import StopPlan
        from stocklab.entry.timing import staged_entry

        # A tight stop, 3% away, against a ladder that would reach 5% down.
        stop = StopPlan(entry=100.0, stop=97.0, method="test",
                        risk_per_share=3.0, risk_pct=0.03)
        ladder = staged_entry(100.0, stop, tranches=3, spread=0.05)
        assert (ladder["limit_price"] > stop.stop).all()
        assert (ladder["r_to_stop"] > 0).all()

    def test_a_wide_stop_leaves_the_ladder_untouched(self):
        from stocklab.entry.stops import StopPlan
        from stocklab.entry.timing import staged_entry

        stop = StopPlan(entry=100.0, stop=80.0, method="test",
                        risk_per_share=20.0, risk_pct=0.20)
        ladder = staged_entry(100.0, stop, tranches=3, spread=0.05)
        assert ladder["limit_price"].iloc[-1] == pytest.approx(95.0)

    def test_allocations_sum_to_one(self):
        from stocklab.entry.stops import StopPlan
        from stocklab.entry.timing import staged_entry

        stop = StopPlan(entry=100.0, stop=90.0, method="test",
                        risk_per_share=10.0, risk_pct=0.10)
        assert staged_entry(100.0, stop, tranches=4)["allocation"].sum() == pytest.approx(1.0)
