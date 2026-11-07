"""Relative valuation: against the company's own history, and against peers.

The own-history comparison is built **point-in-time**. Each trailing-twelve-month
figure is stamped with the date it was actually filed, not the date the period
ended, so the multiple on 1 February uses the numbers an investor could have
seen on 1 February. Using period-end dates instead backdates roughly six weeks
of hindsight into every observation, which flatters any strategy built on it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.providers.sec import Fundamentals
from ..data.providers.yahoo import fetch_history, fetch_splits
from ..util.mathx import NAN, is_num, median_finite, safe_div, zero_if_nan

# Multiples worth tracking, and whether they price equity or the whole firm.
MULTIPLE_SPECS: dict[str, dict] = {
    "pe": {"concept": "net_income", "basis": "equity", "label": "P/E"},
    "ps": {"concept": "revenue", "basis": "equity", "label": "P/S"},
    "p_fcf": {"concept": "fcf", "basis": "equity", "label": "P/FCF"},
    "p_ocf": {"concept": "cfo", "basis": "equity", "label": "P/OCF"},
    "pb": {"concept": "equity", "basis": "equity", "label": "P/B", "stock": True},
    "ev_ebit": {"concept": "operating_income", "basis": "enterprise", "label": "EV/EBIT"},
    "ev_sales": {"concept": "revenue", "basis": "enterprise", "label": "EV/Sales"},
}


@dataclass
class MultipleBand:
    name: str
    current: float
    median: float
    p25: float
    p75: float
    p10: float
    p90: float
    percentile: float          # where the current reading sits in its own history
    observations: int
    implied_price_at_median: float
    years_covered: float

    @property
    def rich_or_cheap(self) -> str:
        if not is_num(self.percentile):
            return "unknown"
        if self.percentile >= 0.85:
            return "rich vs own history"
        if self.percentile >= 0.60:
            return "above own median"
        if self.percentile >= 0.40:
            return "near own median"
        if self.percentile >= 0.15:
            return "below own median"
        return "cheap vs own history"


@dataclass
class RelativeValuation:
    bands: dict[str, MultipleBand]
    composite_percentile: float
    implied_price_blend: float
    notes: list[str] = field(default_factory=list)


def _known_at(f: Fundamentals, concept: str) -> pd.Series:
    """TTM series for ``concept`` re-indexed to the date it became public.

    Each fiscal period is shifted forward to its filing date. Where a filing
    date is missing, the SEC's filing deadline (45 days after a quarter end) is
    used as the conservative stand-in -- never the period end itself.
    """
    series = f.ttm_series(concept)
    if series.empty:
        return series
    filed = f.filed_dates
    out: dict[pd.Timestamp, float] = {}
    for period_end, value in series.items():
        stamp = filed.get(period_end)
        try:
            known = pd.Timestamp(stamp) if stamp else period_end + pd.Timedelta(days=45)
        except (ValueError, TypeError):
            known = period_end + pd.Timedelta(days=45)
        out[known] = float(value)
    return pd.Series(out, dtype=float).sort_index()


def _instant_known_at(f: Fundamentals, concept: str) -> pd.Series:
    """Same idea for a balance-sheet item (no trailing sum)."""
    if concept not in f.quarterly:
        return pd.Series(dtype=float)
    series = f.quarterly[concept].dropna()
    filed = f.filed_dates
    out: dict[pd.Timestamp, float] = {}
    for period_end, value in series.items():
        stamp = filed.get(period_end)
        try:
            known = pd.Timestamp(stamp) if stamp else period_end + pd.Timedelta(days=45)
        except (ValueError, TypeError):
            known = period_end + pd.Timedelta(days=45)
        out[known] = float(value)
    return pd.Series(out, dtype=float).sort_index()


def split_adjusted_shares(symbol: str, shares: pd.Series,
                          splits: pd.Series | None = None) -> pd.Series:
    """Restate historical share counts onto today's split basis.

    Yahoo's ``close`` is split-adjusted; SEC share counts are as-filed. Multiply
    one by the other and every pre-split observation is wrong by the split
    factor -- Apple's 4:1 split in 2020 made its 2016 market cap look like
    $107bn instead of $600bn, which dragged the 10th percentile of its P/E band
    down to 4x. A company has never been cheap at 4x; the data was.

    Each historical count is scaled by the product of every split that happened
    *after* it.
    """
    if shares.empty:
        return shares
    events = fetch_splits(symbol) if splits is None else splits
    if events is None or events.empty:
        return shares

    out = shares.copy().astype(float)
    for date, value in shares.items():
        later = events[events.index > pd.Timestamp(date)]
        factor = float(later.prod()) if not later.empty else 1.0
        out.loc[date] = float(value) * factor
    return out


def multiple_history(f: Fundamentals, name: str, price: pd.Series,
                     shares: pd.Series, net_debt: pd.Series | None = None
                     ) -> pd.Series:
    """Daily history of one multiple, using only data known at each date."""
    spec = MULTIPLE_SPECS.get(name)
    if spec is None:
        return pd.Series(dtype=float)

    concept = spec["concept"]
    if concept == "fcf":
        cfo = _known_at(f, "cfo")
        capex = _known_at(f, "capex")
        if cfo.empty:
            return pd.Series(dtype=float)
        fundamental = cfo.subtract(capex.reindex(cfo.index).ffill().fillna(0.0))
    elif spec.get("stock"):
        fundamental = _instant_known_at(f, concept)
    else:
        fundamental = _known_at(f, concept)
    if fundamental.empty:
        return pd.Series(dtype=float)

    aligned = fundamental.reindex(price.index.union(fundamental.index)).ffill()
    aligned = aligned.reindex(price.index)
    share_count = shares.reindex(price.index).ffill()

    market_cap = price * share_count
    if spec["basis"] == "enterprise":
        nd = (net_debt.reindex(price.index).ffill() if net_debt is not None
              else pd.Series(0.0, index=price.index))
        numerator = market_cap + nd.fillna(0.0)
    else:
        numerator = market_cap

    # A multiple on a negative denominator is meaningless, not negative.
    denominator = aligned.where(aligned > 0)
    out = numerator / denominator
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def band(name: str, history: pd.Series, current_price: float,
         current_fundamental: float, shares: float,
         net_debt: float = 0.0) -> MultipleBand | None:
    """Percentile position of the current multiple within its own history."""
    history = history.dropna()
    if len(history) < 120:  # under ~6 months of daily observations
        return None
    # Trim the fat tail: a multiple spikes to 900x whenever earnings dip through
    # zero, and those points would otherwise dominate the percentile scale.
    trimmed = history[history <= history.quantile(0.99)]
    if trimmed.empty:
        return None

    basis = MULTIPLE_SPECS[name]["basis"]
    if not (is_num(current_fundamental) and current_fundamental > 0 and is_num(shares)):
        return None
    numerator = current_price * shares + (zero_if_nan(net_debt) if basis == "enterprise" else 0.0)
    current = numerator / current_fundamental

    median = float(trimmed.median())
    implied_equity = median * current_fundamental - (
        zero_if_nan(net_debt) if basis == "enterprise" else 0.0
    )
    return MultipleBand(
        name=name,
        current=float(current),
        median=median,
        p25=float(trimmed.quantile(0.25)), p75=float(trimmed.quantile(0.75)),
        p10=float(trimmed.quantile(0.10)), p90=float(trimmed.quantile(0.90)),
        percentile=float((trimmed <= current).mean()),
        observations=int(len(trimmed)),
        implied_price_at_median=safe_div(implied_equity, shares),
        years_covered=float((trimmed.index[-1] - trimmed.index[0]).days / 365.25),
    )


def evaluate(f: Fundamentals, metrics: dict, price: float,
             lookback: str = "10y") -> RelativeValuation:
    """Build own-history bands for every multiple the company supports."""
    notes: list[str] = []
    hist_full = fetch_history(f.symbol, period=lookback, include_actions=True)
    hist = hist_full
    if hist.empty:
        return RelativeValuation({}, NAN, NAN, ["No price history available."])

    px = hist["close"]
    shares_now = metrics.get("shares_outstanding") or metrics.get("shares")
    share_series = split_adjusted_shares(
        f.symbol, _instant_known_at(f, "shares_outstanding")
    )
    if share_series.empty and is_num(shares_now):
        share_series = pd.Series({px.index[0]: float(shares_now)})
        notes.append("No share-count history; current count applied across the "
                     "window, which distorts older multiples for any company "
                     "that has bought back stock.")

    debt_series = _instant_known_at(f, "long_term_debt")
    cash_series = _instant_known_at(f, "cash")
    if not debt_series.empty:
        idx = debt_series.index.union(cash_series.index)
        nd_series = (debt_series.reindex(idx).ffill().fillna(0.0)
                     - cash_series.reindex(idx).ffill().fillna(0.0))
    else:
        nd_series = None

    bands: dict[str, MultipleBand] = {}
    for name, spec in MULTIPLE_SPECS.items():
        series = multiple_history(f, name, px, share_series, nd_series)
        if series.empty:
            continue
        concept = spec["concept"]
        if concept == "fcf":
            current_fundamental = metrics.get("fcf")
        elif concept == "operating_income":
            current_fundamental = metrics.get("ebit")
        else:
            current_fundamental = metrics.get(concept)
        result = band(name, series, price, current_fundamental,
                      shares_now, metrics.get("net_debt", 0.0))
        if result is not None:
            bands[name] = result

    if not bands:
        return RelativeValuation({}, NAN, NAN, notes + ["No multiple had enough history."])

    composite = float(np.mean([b.percentile for b in bands.values() if is_num(b.percentile)]))
    implied = median_finite([b.implied_price_at_median for b in bands.values()])

    short = [b.name for b in bands.values() if b.years_covered < 5]
    if short:
        notes.append(f"Under five years of history for: {', '.join(short)}. "
                     "A percentile from a short window mostly reflects the "
                     "market regime of that window.")
    return RelativeValuation(bands=bands, composite_percentile=composite,
                             implied_price_blend=implied, notes=notes)


def peer_comparison(rows: list[dict], metrics_keys: tuple[str, ...] = (
        "pe", "ev_ebitda", "ps", "pb", "fcf_yield", "roic", "revenue_cagr_3y",
        "operating_margin", "net_debt_to_ebitda")) -> pd.DataFrame:
    """Rank a peer set on valuation and quality side by side.

    Each ``row`` is ``{"symbol": ..., **metrics}``. Percentile ranks are added
    so a name can be read as "cheapest quartile, best-quality quartile" rather
    than requiring the reader to hold nine raw numbers in their head.
    """
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).set_index("symbol")
    keep = [k for k in metrics_keys if k in frame.columns]
    out = frame[keep].copy()
    # Lower is better for valuation multiples and leverage; higher for the rest.
    lower_better = {"pe", "ev_ebitda", "ps", "pb", "net_debt_to_ebitda"}
    for col in keep:
        out[f"{col}_rank"] = out[col].rank(pct=True, ascending=col in lower_better)
    return out
