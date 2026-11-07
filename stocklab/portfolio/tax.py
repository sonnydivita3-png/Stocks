"""Tax reporting and loss-harvesting analysis.

**Not tax advice, and not a substitute for a 1099-B.** Cost basis depends on
elections made at the broker, corporate-action adjustments the issuer reports
on Form 8937, and rules this module does not model (constructive sales,
straddles, Section 1256 contracts, qualified vs ordinary dividend
classification, state treatment). Use it to find candidates and understand
magnitudes, then reconcile against broker records before filing.

What it does model, because these drive most of the decisions:

* short- versus long-term classification on the *more than* 12 months rule;
* the wash-sale window, including the cross-account case that makes a loss
  permanently rather than temporarily disallowed;
* the $3,000 annual limit on net capital losses against ordinary income, and
  the indefinite carry-forward of the remainder;
* why harvesting inside a retirement account accomplishes nothing at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..util.mathx import NAN, is_num, safe_div
from .ledger import WASH_SALE_WINDOW_DAYS, LedgerState
from .models import Account, Lot, Position, RealizedGain

# 2026 federal rates. Long-term brackets are indexed annually, so these are
# defaults to override, not constants. State tax is not modelled: Wisconsin,
# for instance, taxes capital gains as ordinary income with a 30% exclusion for
# assets held over a year, which no federal-only estimate will capture.
DEFAULT_SHORT_TERM_RATE = 0.24      # ordinary income bracket
DEFAULT_LONG_TERM_RATE = 0.15       # 0 / 15 / 20 schedule
DEFAULT_NIIT_RATE = 0.038           # net investment income tax above thresholds
CAPITAL_LOSS_ORDINARY_LIMIT = 3000.0


@dataclass
class HarvestCandidate:
    symbol: str
    account_id: int
    account_name: str
    quantity: float
    cost_basis: float
    market_value: float
    loss: float
    term: str
    estimated_tax_saving: float
    wash_sale_risk: bool
    blocked_until: date | None
    notes: list[str] = field(default_factory=list)


@dataclass
class TaxReport:
    year: int
    short_term_gains: float
    long_term_gains: float
    net_gain: float
    disallowed_wash_sales: float
    deductible_against_income: float
    loss_carryforward: float
    estimated_tax: float
    by_symbol: pd.DataFrame
    realized_detail: pd.DataFrame
    notes: list[str] = field(default_factory=list)


def realized_frame(state: LedgerState, accounts: dict[int, Account],
                   year: int | None = None, taxable_only: bool = True
                   ) -> pd.DataFrame:
    """Every realised disposal as a row, ready for review against a 1099-B."""
    rows = []
    for gain in state.realized:
        account = accounts.get(gain.account_id)
        if taxable_only and account is not None and not account.type.taxable_now:
            continue
        if year is not None and gain.sell_date.year != year:
            continue
        rows.append({
            "symbol": gain.symbol,
            "account": account.name if account else gain.account_id,
            "account_type": account.type.value if account else "",
            "open_date": gain.open_date,
            "sell_date": gain.sell_date,
            "quantity": gain.quantity,
            "proceeds": gain.proceeds,
            "cost_basis": gain.cost_basis,
            "gain": gain.gain,
            "holding_days": gain.holding_days,
            "term": gain.term,
            "wash_sale_disallowed": gain.wash_sale_disallowed,
            "taxable_gain": gain.taxable_gain,
            "notes": gain.notes,
        })
    if not rows:
        return pd.DataFrame(columns=[
            "symbol", "account", "account_type", "open_date", "sell_date",
            "quantity", "proceeds", "cost_basis", "gain", "holding_days",
            "term", "wash_sale_disallowed", "taxable_gain", "notes"])
    return pd.DataFrame(rows).sort_values("sell_date")


def annual_report(state: LedgerState, accounts: dict[int, Account],
                  year: int | None = None,
                  short_term_rate: float = DEFAULT_SHORT_TERM_RATE,
                  long_term_rate: float = DEFAULT_LONG_TERM_RATE,
                  niit: bool = False) -> TaxReport:
    """Realised gain/loss summary and an estimated federal liability."""
    year = year or date.today().year
    detail = realized_frame(state, accounts, year=year, taxable_only=True)
    notes: list[str] = []

    if detail.empty:
        return TaxReport(year, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         pd.DataFrame(), detail,
                         notes=["No taxable disposals in this year."])

    short = float(detail.loc[detail["term"] == "short", "taxable_gain"].sum())
    long = float(detail.loc[detail["term"] == "long", "taxable_gain"].sum())
    disallowed = float(detail["wash_sale_disallowed"].sum())
    net = short + long

    # Netting order under US rules: short against short, long against long,
    # then the two net against each other.
    if short < 0 and long > 0:
        offset = min(-short, long)
        short += offset
        long -= offset
    elif long < 0 and short > 0:
        offset = min(-long, short)
        long += offset
        short -= offset

    deductible = carryforward = 0.0
    if net < 0:
        deductible = min(-net, CAPITAL_LOSS_ORDINARY_LIMIT)
        carryforward = max(0.0, -net - CAPITAL_LOSS_ORDINARY_LIMIT)
        notes.append(
            f"Net capital loss of ${-net:,.0f}. ${deductible:,.0f} is deductible "
            f"against ordinary income this year (the $3,000 annual limit) and "
            f"${carryforward:,.0f} carries forward indefinitely."
        )

    rate_niit = DEFAULT_NIIT_RATE if niit else 0.0
    estimated = (max(short, 0.0) * (short_term_rate + rate_niit)
                 + max(long, 0.0) * (long_term_rate + rate_niit))

    if disallowed > 0:
        notes.append(
            f"${disallowed:,.0f} of losses disallowed by the wash-sale rule and "
            "added back to taxable gains. See the per-row notes for which "
            "replacements triggered it."
        )
    if short > 0 and long >= 0:
        notes.append(
            f"${short:,.0f} of gains are short-term, taxed at ordinary rates "
            f"({short_term_rate:.0%}) rather than {long_term_rate:.0%}. Holding "
            "past the one-year mark would have saved roughly "
            f"${short * (short_term_rate - long_term_rate):,.0f}."
        )
    notes.append("Federal estimate only; state tax is not modelled.")

    by_symbol = detail.groupby("symbol").agg(
        proceeds=("proceeds", "sum"), cost_basis=("cost_basis", "sum"),
        gain=("gain", "sum"), taxable_gain=("taxable_gain", "sum"),
        disposals=("quantity", "count"),
    ).sort_values("taxable_gain")

    return TaxReport(
        year=year, short_term_gains=short, long_term_gains=long, net_gain=net,
        disallowed_wash_sales=disallowed, deductible_against_income=deductible,
        loss_carryforward=carryforward, estimated_tax=estimated,
        by_symbol=by_symbol, realized_detail=detail, notes=notes,
    )


def unrealized_frame(state: LedgerState, accounts: dict[int, Account],
                     prices: dict[str, float]) -> pd.DataFrame:
    """Open lots with gain, holding period, and days until long-term treatment.

    ``days_to_long_term`` is the actionable column: a lot 20 days short of the
    one-year mark is usually worth waiting on, since the rate difference is
    typically 9 percentage points.
    """
    rows = []
    today = date.today()
    for lot in state.open_lots():
        account = accounts.get(lot.account_id)
        price = prices.get(lot.symbol, NAN)
        market_value = lot.remaining * price if is_num(price) else NAN
        basis = lot.cost_basis
        gain = market_value - basis if is_num(market_value) else NAN
        days = lot.holding_days(today)
        rows.append({
            "symbol": lot.symbol,
            "account": account.name if account else lot.account_id,
            "account_type": account.type.value if account else "",
            "sheltered": bool(account.type.sheltered) if account else False,
            "lot_id": lot.id,
            "open_date": lot.open_date,
            "quantity": lot.remaining,
            "cost_per_share": lot.cost_per_share,
            "cost_basis": basis,
            "price": price,
            "market_value": market_value,
            "unrealized_gain": gain,
            "unrealized_pct": safe_div(gain, basis),
            "holding_days": days,
            "term": "long" if days > 365 else "short",
            "days_to_long_term": max(0, 366 - days),
            "wash_sale_basis_adjustment": lot.wash_sale_adjustment,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("unrealized_gain")


def harvest_candidates(state: LedgerState, accounts: dict[int, Account],
                       prices: dict[str, float],
                       min_loss: float = 500.0,
                       short_term_rate: float = DEFAULT_SHORT_TERM_RATE,
                       long_term_rate: float = DEFAULT_LONG_TERM_RATE
                       ) -> list[HarvestCandidate]:
    """Lots sitting at a loss that could be sold to offset gains.

    Only taxable accounts are considered. Harvesting inside an IRA or 401(k)
    does nothing: the loss is not deductible and basis is irrelevant to how the
    eventual withdrawal is taxed. That is worth stating explicitly, because
    "sell the losers" is advice people apply to the wrong account.

    Each candidate is checked for wash-sale risk from purchases in the last 30
    days *in any account*, since a repurchase anywhere -- a 401(k)
    contribution buying the same fund, for example -- disallows the loss.
    """
    today = date.today()
    recent_buys: dict[str, date] = {}
    for lot in state.lots:
        if (today - lot.open_date).days <= WASH_SALE_WINDOW_DAYS:
            prior = recent_buys.get(lot.symbol)
            if prior is None or lot.open_date > prior:
                recent_buys[lot.symbol] = lot.open_date

    out: list[HarvestCandidate] = []
    for lot in state.open_lots():
        account = accounts.get(lot.account_id)
        if account is None or not account.type.allows_loss_harvest:
            continue
        price = prices.get(lot.symbol)
        if not is_num(price):
            continue
        market_value = lot.remaining * price
        loss = market_value - lot.cost_basis
        if loss > -min_loss:
            continue

        long_term = lot.is_long_term(today)
        rate = long_term_rate if long_term else short_term_rate
        notes: list[str] = []

        bought_on = recent_buys.get(lot.symbol)
        wash_risk = bought_on is not None
        blocked_until = None
        if wash_risk:
            blocked_until = bought_on + timedelta(days=WASH_SALE_WINDOW_DAYS + 1)
            notes.append(
                f"Shares of {lot.symbol} were bought on {bought_on}; selling at a "
                f"loss now triggers a wash sale. Clear after {blocked_until}."
            )
        if not long_term:
            notes.append(
                "Short-term loss: offsets short-term gains first, which are "
                "taxed at the higher rate, so this is the more valuable offset."
            )
        notes.append(
            "Repurchasing the same security within 30 days forfeits the "
            "deduction. A similar-but-not-identical holding keeps the exposure; "
            "two S&P 500 funds from different issuers is a grey area the IRS "
            "has never resolved."
        )

        out.append(HarvestCandidate(
            symbol=lot.symbol, account_id=lot.account_id,
            account_name=account.name, quantity=lot.remaining,
            cost_basis=lot.cost_basis, market_value=market_value, loss=loss,
            term="long" if long_term else "short",
            estimated_tax_saving=abs(loss) * rate,
            wash_sale_risk=wash_risk, blocked_until=blocked_until, notes=notes,
        ))
    return sorted(out, key=lambda c: c.loss)


# --------------------------------------------------------------------------- #
# asset location
# --------------------------------------------------------------------------- #
# Tax drag per dollar held, highest first. The ordering is what matters:
# interest and non-qualified distributions are taxed annually at ordinary
# rates, while a low-turnover equity position defers almost everything.
TAX_INEFFICIENCY = {
    "high_yield_bond": 1.00,
    "taxable_bond": 0.95,
    "reit": 0.90,          # distributions are largely non-qualified
    "high_turnover_equity": 0.60,
    "high_dividend_equity": 0.45,
    "broad_equity": 0.20,
    "growth_equity": 0.10,  # little or no dividend, gains deferred
    "municipal_bond": 0.00,  # already tax-exempt: wasted in a sheltered account
}


def asset_location_review(state: LedgerState, accounts: dict[int, Account],
                          prices: dict[str, float],
                          classifications: dict[str, str] | None = None
                          ) -> pd.DataFrame:
    """Flag holdings that would be better off in a different account type.

    The principle: put the most heavily taxed assets where they are shielded,
    and hold tax-efficient assets in the taxable account. Two corrections the
    naive version of this rule gets wrong and this one does not:

    * **Municipal bonds in an IRA** waste their exemption entirely -- they pay a
      lower yield precisely because the income is already untaxed.
    * **A taxable account gives you the loss-harvesting option and the
      step-up in basis at death**, neither of which exists in a sheltered
      account. Equities are the asset whose gains most benefit from both.

    The quantitative benefit also depends on the holding period, which this
    does not attempt to forecast; it ranks mismatches rather than pricing them.
    """
    classifications = classifications or {}
    rows = []
    for position in state.positions(prices=prices, by_account=True):
        account = accounts.get(position.account_id) if position.account_id else None
        if account is None:
            continue
        kind = classifications.get(position.symbol, "broad_equity")
        drag = TAX_INEFFICIENCY.get(kind, 0.20)
        sheltered = account.type.sheltered

        if kind == "municipal_bond" and sheltered:
            verdict = "misplaced: tax-exempt income wasted in a sheltered account"
        elif drag >= 0.60 and not sheltered:
            verdict = "misplaced: heavily taxed asset held in a taxable account"
        elif drag <= 0.20 and sheltered:
            verdict = ("suboptimal: tax-efficient asset occupies sheltered space, "
                       "and forgoes loss harvesting and the basis step-up")
        else:
            verdict = "appropriate"

        rows.append({
            "symbol": position.symbol,
            "account": account.name,
            "account_type": account.type.label,
            "sheltered": sheltered,
            "classification": kind,
            "tax_inefficiency": drag,
            "market_value": position.market_value,
            "verdict": verdict,
        })
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return frame.sort_values(["verdict", "market_value"], ascending=[True, False])
