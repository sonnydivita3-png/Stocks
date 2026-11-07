"""Replay a transaction log into lots, positions, cash and realised gains.

The whole module is one deterministic fold over the transaction list. Nothing
is cached and nothing is mutated in place in the database, so the output always
matches the inputs.

Three things here are easy to get subtly wrong and are therefore spelled out:

* **Lot selection on a sale** materially changes the tax bill. FIFO is the
  broker default and usually the *worst* choice in an appreciating position,
  because the oldest lot has the lowest basis. HIFO minimises the realised
  gain. The method is a per-account setting.
* **Splits adjust basis per share, not total basis.** A 4-for-1 split
  quadruples quantity and quarters cost per share; total basis is unchanged.
* **Wash sales apply across every account you own**, including IRAs, and a
  wash triggered by a purchase inside an IRA destroys the loss permanently
  rather than deferring it (Rev. Rul. 2008-5).
"""
from __future__ import annotations

import itertools
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta


from ..util.mathx import NAN, is_num, safe_div
from .models import (
    Account, CostBasisMethod, Lot, Position, RealizedGain,
    Transaction, TxnType,
)

log = logging.getLogger(__name__)

WASH_SALE_WINDOW_DAYS = 30
_EPS = 1e-9


@dataclass
class LedgerState:
    lots: list[Lot] = field(default_factory=list)
    cash: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    realized: list[RealizedGain] = field(default_factory=list)
    external_flows: list[tuple[date, float]] = field(default_factory=list)
    income: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    fees: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    warnings: list[str] = field(default_factory=list)

    def open_lots(self, account_id: int | None = None,
                  symbol: str | None = None) -> list[Lot]:
        out = [lot for lot in self.lots if lot.is_open]
        if account_id is not None:
            out = [lot for lot in out if lot.account_id == account_id]
        if symbol:
            sym = symbol.upper()
            out = [lot for lot in out if lot.symbol == sym]
        return out

    def positions(self, account_id: int | None = None,
                  prices: dict[str, float] | None = None,
                  by_account: bool = False) -> list[Position]:
        """Aggregate open lots into positions.

        ``by_account=False`` combines the same ticker held in several accounts
        into one position, which is the right view for portfolio risk. Keep
        them separate for anything tax-related.
        """
        prices = prices or {}
        groups: dict[tuple, list[Lot]] = defaultdict(list)
        for lot in self.open_lots(account_id):
            key = (lot.symbol, lot.account_id) if by_account else (lot.symbol, None)
            groups[key].append(lot)

        out = []
        for (symbol, acct), lots in sorted(groups.items()):
            quantity = sum(lot.remaining for lot in lots)
            if quantity <= _EPS:
                continue
            basis = sum(lot.cost_basis for lot in lots)
            out.append(Position(
                symbol=symbol, quantity=quantity, cost_basis=basis,
                price=prices.get(symbol, NAN), lots=lots, account_id=acct,
            ))
        return out

    def total_cash(self, account_id: int | None = None) -> float:
        if account_id is not None:
            return float(self.cash.get(account_id, 0.0))
        return float(sum(self.cash.values()))


# --------------------------------------------------------------------------- #
# lot selection
# --------------------------------------------------------------------------- #
def select_lots(lots: list[Lot], quantity: float, method: CostBasisMethod = "fifo",
                specific_ids: tuple[int, ...] = (), as_of: date | None = None
                ) -> list[tuple[Lot, float]]:
    """Choose which lots a sale consumes. Returns ``[(lot, quantity_used)]``.

    ``hifo`` is highest-cost-first, which minimises the realised gain. For a
    held-up position that is usually a better outcome than the broker's FIFO
    default, though it also consumes long-term lots last, so it can convert
    what would have been long-term gains into short-term ones. There is no
    universally correct method; there is only the one you chose deliberately.
    """
    available = [lot for lot in lots if lot.is_open]
    if not available or quantity <= _EPS:
        return []

    if method == "specific" and specific_ids:
        order = {lot_id: index for index, lot_id in enumerate(specific_ids)}
        available = [lot for lot in available if lot.id in order]
        available.sort(key=lambda lot: order.get(lot.id, 1 << 30))
    elif method == "lifo":
        available.sort(key=lambda lot: (lot.open_date, lot.id or 0), reverse=True)
    elif method == "hifo":
        available.sort(key=lambda lot: -lot.cost_per_share)
    elif method == "average":
        # One synthetic lot at the weighted average cost. Permitted for mutual
        # funds and DRIPs; not for individual equities under US rules, which is
        # why it is never the default here.
        total_qty = sum(lot.remaining for lot in available)
        total_cost = sum(lot.cost_basis for lot in available)
        avg = safe_div(total_cost, total_qty)
        for lot in available:
            lot.cost_per_share = avg if is_num(avg) else lot.cost_per_share
            lot.wash_sale_adjustment = 0.0
        available.sort(key=lambda lot: (lot.open_date, lot.id or 0))
    else:  # fifo
        available.sort(key=lambda lot: (lot.open_date, lot.id or 0))

    picks: list[tuple[Lot, float]] = []
    remaining = quantity
    for lot in available:
        if remaining <= _EPS:
            break
        used = min(lot.remaining, remaining)
        if used > _EPS:
            picks.append((lot, used))
            remaining -= used
    return picks


# --------------------------------------------------------------------------- #
# replay
# --------------------------------------------------------------------------- #
def replay(transactions: list[Transaction], accounts: dict[int, Account] | None = None,
           detect_wash_sales: bool = True) -> LedgerState:
    """Fold a transaction log into a :class:`LedgerState`."""
    accounts = accounts or {}
    state = LedgerState()
    lot_counter = itertools.count(1)
    ordered = sorted(transactions, key=lambda t: (t.date, t.id or 0))

    for txn in ordered:
        account = accounts.get(txn.account_id)
        method: CostBasisMethod = account.cost_basis_method if account else "fifo"

        if txn.type in (TxnType.BUY, TxnType.TRANSFER_IN, TxnType.DIVIDEND_REINVEST):
            if txn.quantity <= 0:
                state.warnings.append(
                    f"{txn.date} {txn.type.value} {txn.symbol}: non-positive quantity, skipped")
                continue
            # Commissions are part of the cost basis, not a separate expense.
            total_cost = txn.quantity * txn.price + txn.fees
            state.lots.append(Lot(
                id=next(lot_counter), account_id=txn.account_id, symbol=txn.symbol,
                open_date=txn.date, quantity=txn.quantity,
                cost_per_share=safe_div(total_cost, txn.quantity),
                source_txn_id=txn.id,
            ))
            state.cash[txn.account_id] += txn.amount if txn.amount else -total_cost
            if txn.type is TxnType.DIVIDEND_REINVEST:
                # A DRIP is income then a purchase; the cash nets to zero but
                # the dividend is still taxable in a taxable account.
                state.income[txn.account_id] += txn.quantity * txn.price
                state.cash[txn.account_id] += txn.quantity * txn.price
            if txn.type is TxnType.TRANSFER_IN:
                state.cash[txn.account_id] += total_cost  # no cash actually moved
                state.external_flows.append((txn.date, total_cost))

        elif txn.type in (TxnType.SELL, TxnType.TRANSFER_OUT):
            lots = state.open_lots(txn.account_id, txn.symbol)
            held = sum(lot.remaining for lot in lots)
            quantity = txn.quantity
            if quantity - held > 1e-6:
                state.warnings.append(
                    f"{txn.date} sell {txn.quantity:g} {txn.symbol} in account "
                    f"{txn.account_id} but only {held:g} held -- short sales and "
                    "untracked openings are not modelled; clipped to holdings."
                )
                quantity = held
            picks = select_lots(lots, quantity, method, txn.lot_ids, txn.date)
            gross = quantity * txn.price
            for lot, used in picks:
                share = safe_div(used, quantity) if quantity else 0.0
                # Selling costs reduce proceeds; they do not touch basis.
                proceeds = gross * share - txn.fees * share
                basis = used * lot.cost_per_share
                if lot.wash_sale_adjustment and lot.quantity > 0:
                    basis += lot.wash_sale_adjustment * (used / lot.quantity)
                lot.closed_quantity += used
                state.realized.append(RealizedGain(
                    account_id=txn.account_id, symbol=txn.symbol,
                    sell_date=txn.date, quantity=used, proceeds=proceeds,
                    cost_basis=basis, open_date=lot.open_date, lot_id=lot.id,
                ))
            state.cash[txn.account_id] += txn.amount if txn.amount else (gross - txn.fees)
            if txn.type is TxnType.TRANSFER_OUT:
                state.cash[txn.account_id] -= gross - txn.fees
                state.external_flows.append((txn.date, -(gross - txn.fees)))

        elif txn.type is TxnType.SPLIT:
            # `price` carries the ratio: 4.0 for a 4-for-1, 0.1 for a 1-for-10.
            ratio = txn.price if txn.price > 0 else (txn.quantity if txn.quantity > 0 else 0)
            if ratio <= 0:
                state.warnings.append(f"{txn.date} split {txn.symbol}: no ratio given")
                continue
            for lot in state.open_lots(txn.account_id, txn.symbol):
                lot.quantity *= ratio
                lot.closed_quantity *= ratio
                lot.cost_per_share /= ratio   # total basis unchanged

        elif txn.type is TxnType.RETURN_OF_CAPITAL:
            # Reduces basis rather than counting as income; taxable only once
            # basis reaches zero.
            lots = state.open_lots(txn.account_id, txn.symbol)
            total_qty = sum(lot.remaining for lot in lots)
            if total_qty > _EPS:
                per_share = safe_div(abs(txn.amount), total_qty)
                for lot in lots:
                    lot.cost_per_share = max(0.0, lot.cost_per_share - per_share)
            state.cash[txn.account_id] += abs(txn.amount)

        elif txn.type in (TxnType.DIVIDEND, TxnType.INTEREST):
            state.cash[txn.account_id] += abs(txn.amount)
            state.income[txn.account_id] += abs(txn.amount)

        elif txn.type is TxnType.FEE:
            state.cash[txn.account_id] -= abs(txn.amount)
            state.fees[txn.account_id] += abs(txn.amount)

        elif txn.type is TxnType.DEPOSIT:
            state.cash[txn.account_id] += abs(txn.amount)
            state.external_flows.append((txn.date, abs(txn.amount)))

        elif txn.type is TxnType.WITHDRAWAL:
            state.cash[txn.account_id] -= abs(txn.amount)
            state.external_flows.append((txn.date, -abs(txn.amount)))

        elif txn.type is TxnType.SPINOFF:
            # Proper treatment allocates basis between parent and spun-off
            # entity by relative fair value, which requires the company's own
            # Form 8937. Here the new shares arrive at the stated price and the
            # parent's basis is left alone, which is wrong for tax purposes.
            state.lots.append(Lot(
                id=next(lot_counter), account_id=txn.account_id, symbol=txn.symbol,
                open_date=txn.date, quantity=txn.quantity,
                cost_per_share=txn.price, source_txn_id=txn.id,
            ))
            state.warnings.append(
                f"{txn.date} spinoff {txn.symbol}: basis not reallocated from the "
                "parent. Use the issuer's Form 8937 allocation and record it as "
                "an explicit basis adjustment for tax reporting."
            )

    if detect_wash_sales:
        apply_wash_sales(state, ordered, accounts)
    return state


# --------------------------------------------------------------------------- #
# wash sales
# --------------------------------------------------------------------------- #
def apply_wash_sales(state: LedgerState, transactions: list[Transaction],
                     accounts: dict[int, Account] | None = None) -> None:
    """Flag realised losses disallowed by the 30-day wash-sale rule.

    A loss is disallowed if substantially identical securities were bought
    within 30 days *before or after* the sale -- a 61-day window centred on the
    sale. Two consequences are commonly missed:

    * The rule spans **every account the taxpayer controls**, including
      retirement accounts and a spouse's accounts. Selling at a loss in a
      brokerage account while a 401(k) buys the same fund triggers it.
    * When the replacement shares sit in an IRA or 401(k), the loss is
      **permanently** disallowed, not deferred: there is no taxable basis in a
      sheltered account for the adjustment to attach to (Rev. Rul. 2008-5).

    Limitations: "substantially identical" is a judgement the IRS has never
    fully specified. Options, convertibles and two S&P 500 funds from different
    issuers are all live questions this implementation does not attempt. It
    flags the unambiguous same-ticker case. Not tax advice.
    """
    accounts = accounts or {}
    purchases: dict[str, list[tuple[date, float, Transaction]]] = defaultdict(list)
    for txn in transactions:
        if txn.type in (TxnType.BUY, TxnType.DIVIDEND_REINVEST, TxnType.TRANSFER_IN):
            if txn.quantity > 0:
                purchases[txn.symbol].append((txn.date, txn.quantity, txn))

    # Each replacement share can absorb only one loss, so consumption is tracked.
    consumed: dict[int, float] = defaultdict(float)
    losses = [g for g in state.realized if g.gain < -_EPS]
    losses.sort(key=lambda g: g.sell_date)

    for loss in losses:
        account = accounts.get(loss.account_id)
        if account is not None and not account.type.allows_loss_harvest:
            # A loss inside a sheltered account was never deductible, so there
            # is nothing for the wash-sale rule to disallow.
            loss.notes = (f"loss not deductible: {account.type.label} is not a "
                          "taxable account")
            continue

        window_start = loss.sell_date - timedelta(days=WASH_SALE_WINDOW_DAYS)
        window_end = loss.sell_date + timedelta(days=WASH_SALE_WINDOW_DAYS)
        matched = 0.0
        permanent = False
        where: list[str] = []

        for buy_date, buy_qty, buy_txn in purchases.get(loss.symbol, []):
            if not (window_start <= buy_date <= window_end):
                continue
            # A purchase made in the same transaction chain as the sale itself
            # cannot be its own replacement.
            key = id(buy_txn)
            free = buy_qty - consumed[key]
            if free <= _EPS:
                continue
            take = min(free, loss.quantity - matched)
            if take <= _EPS:
                continue
            consumed[key] += take
            matched += take
            buy_account = accounts.get(buy_txn.account_id)
            if buy_account is not None and buy_account.type.sheltered:
                permanent = True
                where.append(f"{buy_account.type.label} on {buy_date}")
            else:
                where.append(f"account {buy_txn.account_id} on {buy_date}")
            if matched >= loss.quantity - _EPS:
                break

        if matched <= _EPS:
            continue

        fraction = min(matched / loss.quantity, 1.0) if loss.quantity else 0.0
        disallowed = abs(loss.gain) * fraction
        loss.wash_sale_disallowed = disallowed
        loss.notes = (
            f"wash sale: {matched:g} of {loss.quantity:g} shares replaced within "
            f"30 days ({'; '.join(where[:3])}). ${disallowed:,.2f} of the loss is "
            + ("PERMANENTLY disallowed -- the replacement is in a sheltered "
               "account, so there is no taxable basis to carry it forward to."
               if permanent else
               "disallowed now and added to the replacement lot's basis.")
        )

        if not permanent:
            # Defer the loss by raising the replacement lot's basis.
            for lot in state.lots:
                if (lot.symbol == loss.symbol
                        and window_start <= lot.open_date <= window_end
                        and lot.open_date >= loss.sell_date):
                    lot.wash_sale_adjustment += disallowed
                    break


def build(store, account_id: int | None = None, as_of: date | None = None,
          detect_wash_sales: bool = True) -> tuple[LedgerState, dict[int, Account]]:
    """Load from a store and replay. The normal entry point."""
    accounts = {a.id: a for a in store.accounts() if a.id is not None}
    txns = store.transactions(account_id=account_id, end=as_of)
    return replay(txns, accounts, detect_wash_sales=detect_wash_sales), accounts
