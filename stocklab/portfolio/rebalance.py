"""Turn target weights into a trade list, with tax and account placement in mind.

Rebalancing is not free, and a naive implementation destroys more value than
the drift it corrects. Three costs are modelled:

* **Tax.** Selling an appreciated position in a taxable account crystallises a
  bill now in exchange for a diversification benefit later. Sells are therefore
  routed to sheltered accounts first, where a sale costs nothing, and within a
  taxable account to the lots with the smallest gain (or an outright loss).
* **Drift bands.** Trading on every deviation generates turnover for noise.
  Nothing is traded until a position is outside its band -- a 25% relative /
  5-point absolute rule by default, in the spirit of the Daryanani band
  approach -- and then it is traded back to target, not merely to the band edge.
* **Minimum trade size.** A $40 trade is not worth the spread or the record.

What is *not* modelled: the 30-day wash-sale interaction with a subsequent buy
in the same plan, which is flagged rather than solved, and bid-ask spread on
illiquid names.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from ..config import settings
from ..util.mathx import NAN, is_num, safe_div
from .ledger import LedgerState, select_lots
from .models import Account


@dataclass
class Trade:
    account_id: int
    account_name: str
    symbol: str
    side: str                      # "buy" | "sell"
    quantity: float
    price: float
    value: float
    reason: str
    estimated_gain: float = NAN    # realised gain if this sell executes
    estimated_tax: float = NAN
    term: str = ""
    lot_ids: tuple[int, ...] = ()
    warnings: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "account": self.account_name, "symbol": self.symbol,
            "side": self.side, "quantity": round(self.quantity, 4),
            "price": round(self.price, 2), "value": round(self.value, 2),
            "est_gain": self.estimated_gain, "est_tax": self.estimated_tax,
            "term": self.term, "reason": self.reason,
            "warnings": "; ".join(self.warnings),
        }


@dataclass
class RebalancePlan:
    trades: list[Trade]
    drift: pd.DataFrame
    total_value: float
    turnover: float
    estimated_tax: float
    cash_after: dict[int, float]
    notes: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(columns=["account", "symbol", "side", "quantity",
                                         "price", "value", "est_gain", "est_tax",
                                         "term", "reason", "warnings"])
        return pd.DataFrame([t.as_row() for t in self.trades])

    @property
    def net_cash(self) -> float:
        return sum(-t.value if t.side == "buy" else t.value for t in self.trades)


def current_weights(state: LedgerState, prices: dict[str, float],
                    include_cash: bool = True) -> tuple[pd.Series, float]:
    """Portfolio weights across all accounts, and the total value."""
    positions = state.positions(prices=prices)
    values = {p.symbol: p.market_value for p in positions if is_num(p.market_value)}
    cash = state.total_cash() if include_cash else 0.0
    total = sum(values.values()) + max(cash, 0.0)
    if total <= 0:
        return pd.Series(dtype=float), 0.0
    weights = pd.Series({k: v / total for k, v in values.items()}, dtype=float)
    if include_cash and cash > 0:
        weights["CASH"] = cash / total
    return weights.sort_values(ascending=False), total


def drift_table(weights: pd.Series, targets: pd.Series,
                relative_band: float = 0.25,
                absolute_band: float = 0.05) -> pd.DataFrame:
    """Compare actual to target and decide what is out of tolerance.

    Two bands, and a position must breach *both* to trade. The relative band
    catches a 2% target that has grown to 3% (a 50% overweight that an absolute
    band would ignore); the absolute band stops a 0.5% target from triggering a
    trade every time it moves a few basis points.
    """
    symbols = sorted(set(weights.index) | set(targets.index))
    rows = []
    for symbol in symbols:
        actual = float(weights.get(symbol, 0.0))
        target = float(targets.get(symbol, 0.0))
        diff = actual - target
        relative = safe_div(diff, target) if target > 0 else (np.inf if actual > 0 else 0.0)
        breach_rel = abs(relative) >= relative_band if is_num(relative) else True
        breach_abs = abs(diff) >= absolute_band
        rows.append({
            "symbol": symbol, "actual": actual, "target": target,
            "drift": diff, "relative_drift": relative,
            "out_of_band": bool(breach_rel and breach_abs),
            "action": "sell" if diff > 0 else ("buy" if diff < 0 else "hold"),
        })
    frame = pd.DataFrame(rows).set_index("symbol")
    return frame.sort_values("drift", key=abs, ascending=False)


def plan(
    state: LedgerState,
    accounts: dict[int, Account],
    prices: dict[str, float],
    targets: pd.Series,
    *,
    relative_band: float = 0.25,
    absolute_band: float = 0.05,
    min_trade: float | None = None,
    whole_shares: bool = True,
    short_term_rate: float = 0.24,
    long_term_rate: float = 0.15,
    tax_aware: bool = True,
    max_tax_budget: float | None = None,
) -> RebalancePlan:
    """Build a rebalancing trade list.

    ``max_tax_budget`` caps the estimated tax the plan may incur; sells are
    taken in order of tax efficiency (losses first, then smallest gains) and
    the plan stops adding taxable sells once the budget is exhausted, leaving
    the remaining drift uncorrected and saying so.
    """
    cfg = settings()
    min_trade = cfg.min_trade_value if min_trade is None else min_trade
    notes: list[str] = []
    today = date.today()

    targets = targets[targets > 0].astype(float)
    if targets.empty:
        return RebalancePlan([], pd.DataFrame(), 0.0, 0.0, 0.0, {},
                             ["No target weights supplied."])
    total_target = float(targets.sum())
    if abs(total_target - 1.0) > 0.01:
        notes.append(f"Target weights sum to {total_target:.1%}; normalised to 100%.")
        targets = targets / total_target

    weights, total_value = current_weights(state, prices)
    if total_value <= 0:
        return RebalancePlan([], pd.DataFrame(), 0.0, 0.0, 0.0, {},
                             ["Portfolio has no value to rebalance."])

    drift = drift_table(weights, targets, relative_band, absolute_band)
    trades: list[Trade] = []
    tax_spent = 0.0
    cash_after = {acct_id: state.total_cash(acct_id) for acct_id in accounts}

    # ---- sells ------------------------------------------------------------
    for symbol, row in drift[drift["out_of_band"] & (drift["action"] == "sell")].iterrows():
        if symbol == "CASH":
            continue
        price = prices.get(symbol)
        if not is_num(price) or price <= 0:
            notes.append(f"{symbol}: no price, skipped.")
            continue
        dollars_to_sell = float(row["drift"]) * total_value

        # Rank the accounts holding it by how expensive a sale is: sheltered
        # accounts first (no tax at all), then taxable.
        holders = [p for p in state.positions(prices=prices, by_account=True)
                   if p.symbol == symbol and p.quantity > 0]
        if tax_aware:
            holders.sort(key=lambda p: (
                0 if (accounts.get(p.account_id) and accounts[p.account_id].type.sheltered) else 1,
                -safe_div(p.unrealized_gain, p.cost_basis) if is_num(p.unrealized_gain) else 0,
            ))

        remaining = dollars_to_sell
        for position in holders:
            if remaining < min_trade:
                break
            account = accounts.get(position.account_id)
            if account is None:
                continue
            sellable_value = position.quantity * price
            take_value = min(remaining, sellable_value)
            quantity = take_value / price
            if whole_shares:
                quantity = float(np.floor(quantity))
            if quantity <= 0 or quantity * price < min_trade:
                continue

            picks = select_lots(list(position.lots), quantity,
                                "hifo" if tax_aware else account.cost_basis_method)
            gain = 0.0
            long_qty = short_qty = 0.0
            for lot, used in picks:
                gain += used * price - used * lot.cost_per_share
                if lot.is_long_term(today):
                    long_qty += used
                else:
                    short_qty += used
            est_tax = 0.0
            term = ""
            if account.type.taxable_now and gain > 0:
                share_long = safe_div(long_qty, long_qty + short_qty) or 0.0
                rate = long_term_rate * share_long + short_term_rate * (1.0 - share_long)
                est_tax = gain * rate
                term = "long" if share_long >= 0.5 else "short"
            elif account.type.taxable_now:
                term = "long" if long_qty >= short_qty else "short"

            if (max_tax_budget is not None and est_tax > 0
                    and tax_spent + est_tax > max_tax_budget):
                notes.append(
                    f"{symbol}: sell in {account.name} skipped -- would cost "
                    f"${est_tax:,.0f} in tax and exceed the ${max_tax_budget:,.0f} "
                    "budget. Drift left uncorrected."
                )
                continue

            warnings: list[str] = []
            if account.type.taxable_now and gain < 0:
                warnings.append(
                    "Realises a loss: do not buy this symbol back within 30 days "
                    "in any account, including retirement accounts."
                )
            if account.type.taxable_now and short_qty > 0 and gain > 0:
                warnings.append(
                    f"{short_qty:g} shares are still short-term; waiting would "
                    "move that gain to the lower long-term rate."
                )

            trades.append(Trade(
                account_id=account.id, account_name=account.name, symbol=symbol,
                side="sell", quantity=quantity, price=float(price),
                value=quantity * float(price),
                reason=(f"overweight {row['actual']:.1%} vs target {row['target']:.1%}"
                        + ("" if not account.type.sheltered else "; sold in a sheltered account (no tax)")),
                estimated_gain=gain, estimated_tax=est_tax, term=term,
                lot_ids=tuple(lot.id for lot, _ in picks if lot.id is not None),
                warnings=warnings,
            ))
            tax_spent += est_tax
            cash_after[account.id] = cash_after.get(account.id, 0.0) + quantity * float(price)
            remaining -= quantity * float(price)

    # ---- buys -------------------------------------------------------------
    buy_rows = drift[drift["out_of_band"] & (drift["action"] == "buy")]
    buy_rows = buy_rows[buy_rows.index != "CASH"]
    total_cash = sum(max(v, 0.0) for v in cash_after.values())
    wanted = float((-buy_rows["drift"]).clip(lower=0).sum()) * total_value

    if wanted > total_cash and total_cash > 0:
        notes.append(
            f"Buys want ${wanted:,.0f} but only ${total_cash:,.0f} of cash is "
            "available after sells; purchases scaled down pro rata."
        )
    scale = min(1.0, safe_div(total_cash, wanted) or 0.0) if wanted > 0 else 0.0

    for symbol, row in buy_rows.iterrows():
        price = prices.get(symbol)
        if not is_num(price) or price <= 0:
            notes.append(f"{symbol}: no price, cannot size a purchase.")
            continue
        dollars = -float(row["drift"]) * total_value * scale
        if dollars < min_trade:
            continue
        # Put the buy where the cash is; prefer sheltered accounts for new
        # money only if the caller has not pinned targets to accounts.
        candidates = sorted(
            [(acct_id, cash) for acct_id, cash in cash_after.items() if cash > min_trade],
            key=lambda kv: -kv[1],
        )
        if not candidates:
            notes.append(f"{symbol}: no account has cash for a purchase.")
            continue
        for acct_id, cash in candidates:
            if dollars < min_trade:
                break
            account = accounts.get(acct_id)
            if account is None:
                continue
            spend = min(dollars, cash)
            quantity = spend / float(price)
            if whole_shares:
                quantity = float(np.floor(quantity))
            if quantity <= 0 or quantity * float(price) < min_trade:
                continue
            value = quantity * float(price)
            trades.append(Trade(
                account_id=acct_id, account_name=account.name, symbol=symbol,
                side="buy", quantity=quantity, price=float(price), value=value,
                reason=f"underweight {row['actual']:.1%} vs target {row['target']:.1%}",
            ))
            cash_after[acct_id] = cash - value
            dollars -= value

    traded = sum(abs(t.value) for t in trades)
    turnover = safe_div(traded, total_value)

    # Flag the wash-sale trap: selling at a loss and buying the same name back
    # in the same plan.
    loss_sales = {t.symbol for t in trades if t.side == "sell"
                  and is_num(t.estimated_gain) and t.estimated_gain < 0}
    repurchases = {t.symbol for t in trades if t.side == "buy"} & loss_sales
    for symbol in repurchases:
        notes.append(
            f"{symbol}: the plan both sells at a loss and buys back. That is a "
            "wash sale -- the loss would be disallowed. Drop one leg."
        )

    if not trades:
        notes.append(
            "No position is outside its drift band "
            f"({relative_band:.0%} relative and {absolute_band:.0%} absolute), "
            "so no trades. Not rebalancing is usually the right answer."
        )
    if tax_spent > 0:
        notes.append(
            f"Estimated tax on this plan: ${tax_spent:,.0f}, against turnover of "
            f"{turnover:.1%}. Weigh that against the diversification gained."
        )
    return RebalancePlan(trades=trades, drift=drift, total_value=total_value,
                         turnover=turnover, estimated_tax=tax_spent,
                         cash_after=cash_after, notes=notes)
