"""Cost of capital: CAPM equity cost, synthetic-rating debt cost, and WACC.

The discount rate is the most consequential number in any valuation and the
least examined.  A 1-point change in WACC moves a terminal-value-heavy DCF by
15-25%, which is larger than almost any disagreement about growth.  So every
input here is explicit, bounded, and reported alongside the output.

Design choices worth defending:

* **Betas are shrunk toward 1.0** (Blume). Raw regression betas are noisy
  estimates with standard errors often above 0.2, and they mean-revert. Using a
  raw beta of 2.4 from a 2-year window is a statement about the past, not a
  forecast of the future.
* **Cost of debt comes from a synthetic rating**, not the actual coupon.
  A company's existing debt was issued at past rates; what matters for
  valuation is what it would pay to borrow *now*.
* **Terminal growth is capped at the risk-free rate.** A company growing
  faster than the economy forever eventually becomes the economy. The
  risk-free rate is a usable proxy for long-run nominal growth.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import settings
from ..data.providers import fred
from ..data.providers.sec import Fundamentals
from ..data.providers.yahoo import fetch_history
from ..util.mathx import NAN, clamp, is_num, safe_div

# Damodaran-style synthetic ratings: interest-coverage band -> credit spread.
# Spreads are approximate and drift with the credit cycle; override via
# `Settings` or pass `cost_of_debt` explicitly when you have a real quote.
SYNTHETIC_RATING_TABLE: tuple[tuple[float, str, float], ...] = (
    (8.50, "AAA", 0.0069),
    (6.50, "AA", 0.0085),
    (5.50, "A+", 0.0107),
    (4.25, "A", 0.0118),
    (3.00, "A-", 0.0133),
    (2.50, "BBB", 0.0171),
    (2.25, "BB+", 0.0231),
    (2.00, "BB", 0.0277),
    (1.75, "B+", 0.0396),
    (1.50, "B", 0.0462),
    (1.25, "B-", 0.0578),
    (0.80, "CCC", 0.0808),
    (0.65, "CC", 0.0866),
    (0.20, "C", 0.1178),
    (-1e9, "D", 0.1556),
)

BLUME_WEIGHT = 0.67  # weight on the raw regression beta; remainder pulls toward 1.0


@dataclass
class CostOfCapital:
    """A WACC estimate with every input it was built from kept visible."""

    wacc: float
    cost_of_equity: float
    cost_of_debt_pretax: float
    cost_of_debt_after_tax: float
    risk_free: float
    equity_risk_premium: float
    beta_raw: float
    beta_adjusted: float
    size_premium: float
    tax_rate: float
    weight_equity: float
    weight_debt: float
    synthetic_rating: str
    interest_coverage: float
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "notes"} | {
            "notes": list(self.notes)
        }

    def __str__(self) -> str:
        return (
            f"WACC {self.wacc:.2%} = {self.weight_equity:.0%} equity @ "
            f"{self.cost_of_equity:.2%} + {self.weight_debt:.0%} debt @ "
            f"{self.cost_of_debt_after_tax:.2%} "
            f"(beta {self.beta_adjusted:.2f}, rating {self.synthetic_rating})"
        )


def estimate_beta(symbol: str, benchmark: str = "SPY", years: int = 5,
                  frequency: str = "1wk") -> tuple[float, float, int]:
    """Regression beta against a benchmark, plus its Blume-adjusted value.

    Weekly data over five years is the default: daily returns over the same
    span pick up bid-ask bounce and non-synchronous trading, which biases betas
    for anything less than highly liquid, while monthly data leaves too few
    observations to estimate anything.

    Returns ``(raw_beta, adjusted_beta, observations)``.
    """
    period = f"{max(years, 2)}y"
    joined = pd.DataFrame()
    # Weekly preferred; fall back to daily if the weekly series is unavailable
    # or too short, since a noisy beta beats silently defaulting to 1.0.
    for interval, min_obs in ((frequency, 40), ("1d", 120)):
        try:
            stock = fetch_history(symbol, period=period, interval=interval)
            bench = fetch_history(benchmark, period=period, interval=interval)
        except Exception:  # noqa: BLE001 - try the next interval
            continue
        if stock.empty or bench.empty:
            continue
        candidate = pd.concat(
            [stock["adj_close"].pct_change().rename("s"),
             bench["adj_close"].pct_change().rename("b")], axis=1
        ).dropna()
        if len(candidate) >= min_obs:
            joined = candidate
            break
    if joined.empty:
        return NAN, 1.0, 0

    x = joined["b"].to_numpy(dtype=float)
    y = joined["s"].to_numpy(dtype=float)
    if x.var() == 0:
        return NAN, 1.0, len(joined)
    raw = float(np.cov(x, y, bias=True)[0, 1] / x.var())
    # Shrink toward the market. Blume (1975) showed betas mean-revert; this is
    # the standard correction and it materially improves out-of-sample fit.
    adjusted = BLUME_WEIGHT * raw + (1.0 - BLUME_WEIGHT) * 1.0
    return raw, clamp(adjusted, 0.3, 2.5), len(joined)


def synthetic_rating(interest_coverage: float, is_financial: bool = False
                     ) -> tuple[str, float]:
    """Map interest coverage to a letter rating and a credit spread.

    Returns ``(rating, spread)``. Unknown coverage is treated as investment
    grade rather than distressed: most companies that fail to tag interest
    expense do so because it is immaterial, not because they cannot pay it.
    """
    if not is_num(interest_coverage):
        return "BBB (assumed)", 0.0171
    if is_financial:
        # Coverage ratios do not describe a bank's solvency; assume mid-IG.
        return "A- (financial, assumed)", 0.0133
    for threshold, rating, spread in SYNTHETIC_RATING_TABLE:
        if interest_coverage > threshold:
            return rating, spread
    return "D", 0.1556


def estimate_cost_of_capital(
    f: Fundamentals,
    market_cap: float,
    metrics: dict | None = None,
    *,
    risk_free: float | None = None,
    equity_risk_premium: float | None = None,
    beta_override: float | None = None,
    cost_of_debt_override: float | None = None,
    tax_rate: float | None = None,
    benchmark: str = "SPY",
) -> CostOfCapital:
    """Build a WACC for one company from its filings and market data."""
    cfg = settings()
    notes: list[str] = []
    metrics = metrics or {}

    rf = risk_free if risk_free is not None else fred.risk_free_rate()
    erp = equity_risk_premium if equity_risk_premium is not None else cfg.equity_risk_premium

    if beta_override is not None:
        beta_raw = beta_adj = float(beta_override)
        obs = 0
        notes.append(f"Beta overridden to {beta_adj:.2f}.")
    else:
        beta_raw, beta_adj, obs = estimate_beta(f.symbol, benchmark)
        if not is_num(beta_raw):
            notes.append("Beta could not be estimated from price history; using 1.0.")
        elif obs < 100:
            notes.append(
                f"Beta estimated from only {obs} observations -- wide standard error."
            )

    size_prem = cfg.size_premium(market_cap if is_num(market_cap) else None)
    if size_prem > 0:
        notes.append(f"Size premium of {size_prem:.2%} applied for small/micro cap.")

    cost_equity = rf + beta_adj * erp + size_prem

    # --- debt -------------------------------------------------------------
    coverage = metrics.get("interest_coverage")
    if not is_num(coverage):
        coverage = safe_div(f.get("operating_income"), f.get("interest_expense"))
    rating, spread = synthetic_rating(coverage, f.is_financial)
    if cost_of_debt_override is not None:
        kd_pretax = float(cost_of_debt_override)
        notes.append(f"Pre-tax cost of debt overridden to {kd_pretax:.2%}.")
    else:
        kd_pretax = rf + spread
        if not is_num(coverage):
            notes.append(
                "Interest expense not reported, so the credit spread is assumed at "
                "BBB. Check this for any leveraged name."
            )

    t = tax_rate if tax_rate is not None else metrics.get("effective_tax_rate")
    if not is_num(t):
        t = cfg.default_tax_rate
    t = clamp(float(t), 0.0, 0.45)
    kd_after_tax = kd_pretax * (1.0 - t)

    # --- weights ----------------------------------------------------------
    # Market value of equity against book value of debt. Using book debt is the
    # conventional compromise: the market value of most corporate debt is not
    # observable, and for investment-grade issuers it trades near par anyway.
    debt = metrics.get("total_debt")
    if not is_num(debt):
        debt = 0.0
    if not is_num(market_cap) or market_cap <= 0:
        notes.append("Market cap unavailable; WACC falls back to cost of equity.")
        return CostOfCapital(
            wacc=clamp(cost_equity, cfg.min_wacc, cfg.max_wacc),
            cost_of_equity=cost_equity, cost_of_debt_pretax=kd_pretax,
            cost_of_debt_after_tax=kd_after_tax, risk_free=rf,
            equity_risk_premium=erp, beta_raw=beta_raw, beta_adjusted=beta_adj,
            size_premium=size_prem, tax_rate=t, weight_equity=1.0, weight_debt=0.0,
            synthetic_rating=rating,
            interest_coverage=coverage if is_num(coverage) else NAN, notes=notes,
        )

    total_capital = market_cap + debt
    we = market_cap / total_capital
    wd = debt / total_capital

    wacc_raw = we * cost_equity + wd * kd_after_tax
    wacc = clamp(wacc_raw, cfg.min_wacc, cfg.max_wacc)
    if abs(wacc - wacc_raw) > 1e-9:
        notes.append(
            f"WACC clamped from {wacc_raw:.2%} to {wacc:.2%} "
            f"(bounds {cfg.min_wacc:.0%}-{cfg.max_wacc:.0%})."
        )

    return CostOfCapital(
        wacc=wacc, cost_of_equity=cost_equity, cost_of_debt_pretax=kd_pretax,
        cost_of_debt_after_tax=kd_after_tax, risk_free=rf, equity_risk_premium=erp,
        beta_raw=beta_raw, beta_adjusted=beta_adj, size_premium=size_prem,
        tax_rate=t, weight_equity=we, weight_debt=wd, synthetic_rating=rating,
        interest_coverage=coverage if is_num(coverage) else NAN, notes=notes,
    )


def terminal_growth_cap(risk_free: float | None = None) -> float:
    """Upper bound on perpetual growth.

    Capped at the risk-free rate, following Damodaran: the ten-year yield is a
    market-implied estimate of long-run nominal growth, and nothing can outgrow
    the economy in perpetuity without becoming it. The configured default is
    used when it is lower, which it usually should be.
    """
    cfg = settings()
    rf = risk_free if risk_free is not None else fred.risk_free_rate()
    return float(min(cfg.terminal_growth, max(rf, 0.005)))
