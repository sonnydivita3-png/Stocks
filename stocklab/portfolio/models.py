"""Domain model for multi-account portfolio tracking.

Transactions are the source of truth; lots and positions are *derived* by
replaying them. That costs a little performance and buys idempotency, a full
audit trail, and the ability to correct a mistyped trade from five years ago
and have every downstream figure -- cost basis, realised gains, wash-sale
flags, performance -- recompute correctly.

Account type is not cosmetic. It determines whether a gain is taxable, whether
a loss can be harvested, whether a wash sale is merely deferred or permanently
destroyed, and which assets belong where. Those rules live on the enum.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Literal

import pandas as pd

from ..util.mathx import NAN, is_num, safe_div


class AccountType(str, Enum):
    TAXABLE = "taxable"
    TRADITIONAL_IRA = "traditional_ira"
    ROTH_IRA = "roth_ira"
    SEP_IRA = "sep_ira"
    SIMPLE_IRA = "simple_ira"
    TRADITIONAL_401K = "traditional_401k"
    ROTH_401K = "roth_401k"
    HSA = "hsa"
    PLAN_529 = "529"
    TRUST = "trust"
    CUSTODIAL = "custodial"

    @property
    def tax_deferred(self) -> bool:
        """Contributions pre-tax, withdrawals taxed as ordinary income."""
        return self in {
            AccountType.TRADITIONAL_IRA, AccountType.SEP_IRA,
            AccountType.SIMPLE_IRA, AccountType.TRADITIONAL_401K,
        }

    @property
    def tax_free(self) -> bool:
        """Qualified withdrawals are untaxed (Roth, HSA, 529)."""
        return self in {
            AccountType.ROTH_IRA, AccountType.ROTH_401K,
            AccountType.HSA, AccountType.PLAN_529,
        }

    @property
    def taxable_now(self) -> bool:
        """Dividends and realised gains hit this year's return."""
        return self in {AccountType.TAXABLE, AccountType.TRUST,
                        AccountType.CUSTODIAL}

    @property
    def allows_loss_harvest(self) -> bool:
        """Only a taxable account can turn a loss into a deduction.

        Selling at a loss inside an IRA or 401(k) achieves nothing: the loss is
        not deductible, and the basis is irrelevant because withdrawals are
        taxed on the full amount (or not at all, for a Roth).
        """
        return self.taxable_now

    @property
    def sheltered(self) -> bool:
        return self.tax_deferred or self.tax_free

    @property
    def label(self) -> str:
        return {
            AccountType.TAXABLE: "Taxable brokerage",
            AccountType.TRADITIONAL_IRA: "Traditional IRA",
            AccountType.ROTH_IRA: "Roth IRA",
            AccountType.SEP_IRA: "SEP IRA",
            AccountType.SIMPLE_IRA: "SIMPLE IRA",
            AccountType.TRADITIONAL_401K: "Traditional 401(k)",
            AccountType.ROTH_401K: "Roth 401(k)",
            AccountType.HSA: "HSA",
            AccountType.PLAN_529: "529 plan",
            AccountType.TRUST: "Trust",
            AccountType.CUSTODIAL: "Custodial (UTMA/UGMA)",
        }[self]


class TxnType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    DIVIDEND = "dividend"              # cash dividend received
    DIVIDEND_REINVEST = "reinvest"     # DRIP: dividend immediately buys shares
    SPLIT = "split"                    # quantity adjusts, basis per share adjusts
    SPINOFF = "spinoff"
    TRANSFER_IN = "transfer_in"        # shares arriving with an existing basis
    TRANSFER_OUT = "transfer_out"
    DEPOSIT = "deposit"                # external cash in
    WITHDRAWAL = "withdrawal"          # external cash out
    FEE = "fee"
    INTEREST = "interest"
    RETURN_OF_CAPITAL = "return_of_capital"   # reduces basis, not income

    @property
    def is_external_flow(self) -> bool:
        """Cash crossing the portfolio boundary.

        The distinction money-weighted return depends on: a deposit is not
        performance, and counting it as one is the single most common error in
        home-made return calculations.
        """
        return self in {TxnType.DEPOSIT, TxnType.WITHDRAWAL,
                        TxnType.TRANSFER_IN, TxnType.TRANSFER_OUT}

    @property
    def affects_shares(self) -> bool:
        return self in {TxnType.BUY, TxnType.SELL, TxnType.DIVIDEND_REINVEST,
                        TxnType.SPLIT, TxnType.TRANSFER_IN, TxnType.TRANSFER_OUT,
                        TxnType.SPINOFF}


CostBasisMethod = Literal["fifo", "lifo", "hifo", "average", "specific"]


@dataclass
class Account:
    name: str
    type: AccountType = AccountType.TAXABLE
    broker: str = ""
    owner: str = ""
    cash: float = 0.0
    currency: str = "USD"
    cost_basis_method: CostBasisMethod = "fifo"
    id: int | None = None
    notes: str = ""

    @property
    def is_sheltered(self) -> bool:
        return self.type.sheltered


@dataclass
class Transaction:
    account_id: int
    date: date
    type: TxnType
    symbol: str = ""
    quantity: float = 0.0
    price: float = 0.0
    amount: float = 0.0        # signed cash effect; derived when not supplied
    fees: float = 0.0
    lot_ids: tuple[int, ...] = ()   # for `specific` cost basis
    notes: str = ""
    id: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.date, (datetime, pd.Timestamp)):
            self.date = pd.Timestamp(self.date).date()
        elif isinstance(self.date, str):
            self.date = pd.Timestamp(self.date).date()
        self.symbol = (self.symbol or "").upper().strip()
        if isinstance(self.type, str):
            self.type = TxnType(self.type)
        if self.amount == 0.0 and self.type in (TxnType.BUY, TxnType.SELL):
            gross = self.quantity * self.price
            # Buys consume cash and fees add to the outflow; sells release cash
            # net of fees. Getting this sign convention wrong silently corrupts
            # every cash balance downstream.
            self.amount = -(gross + self.fees) if self.type is TxnType.BUY else (gross - self.fees)


@dataclass
class Lot:
    """One tax lot: a purchase that has not been fully sold."""

    account_id: int
    symbol: str
    open_date: date
    quantity: float
    cost_per_share: float
    id: int | None = None
    closed_quantity: float = 0.0
    # Disallowed wash-sale losses are added to the replacement lot's basis.
    wash_sale_adjustment: float = 0.0
    source_txn_id: int | None = None

    @property
    def remaining(self) -> float:
        return self.quantity - self.closed_quantity

    @property
    def is_open(self) -> bool:
        return self.remaining > 1e-9

    @property
    def cost_basis(self) -> float:
        return self.remaining * self.cost_per_share + self.wash_sale_adjustment

    def holding_days(self, as_of: date | None = None) -> int:
        as_of = as_of or date.today()
        return (as_of - self.open_date).days

    def is_long_term(self, as_of: date | None = None) -> bool:
        """Held more than one year.

        US rule: *more than* 12 months, so exactly 365 days is still
        short-term. The off-by-one matters -- it is the difference between
        ordinary income rates and 0/15/20%.
        """
        return self.holding_days(as_of) > 365


@dataclass
class RealizedGain:
    account_id: int
    symbol: str
    sell_date: date
    quantity: float
    proceeds: float
    cost_basis: float
    open_date: date
    lot_id: int | None = None
    wash_sale_disallowed: float = 0.0
    notes: str = ""

    @property
    def gain(self) -> float:
        return self.proceeds - self.cost_basis

    @property
    def taxable_gain(self) -> float:
        """Gain after adding back any disallowed wash-sale loss."""
        return self.gain + self.wash_sale_disallowed

    @property
    def holding_days(self) -> int:
        return (self.sell_date - self.open_date).days

    @property
    def is_long_term(self) -> bool:
        return self.holding_days > 365

    @property
    def term(self) -> str:
        return "long" if self.is_long_term else "short"


@dataclass
class Position:
    """An aggregated holding, optionally spanning accounts."""

    symbol: str
    quantity: float
    cost_basis: float
    price: float = NAN
    lots: list[Lot] = field(default_factory=list)
    account_id: int | None = None
    account_name: str = ""

    @property
    def market_value(self) -> float:
        return self.quantity * self.price if is_num(self.price) else NAN

    @property
    def cost_per_share(self) -> float:
        return safe_div(self.cost_basis, self.quantity)

    @property
    def unrealized_gain(self) -> float:
        mv = self.market_value
        return mv - self.cost_basis if is_num(mv) else NAN

    @property
    def unrealized_pct(self) -> float:
        return safe_div(self.unrealized_gain, self.cost_basis)

    @property
    def long_term_quantity(self) -> float:
        return sum(lot.remaining for lot in self.lots if lot.is_long_term())

    @property
    def short_term_quantity(self) -> float:
        return sum(lot.remaining for lot in self.lots if not lot.is_long_term())
