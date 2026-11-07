"""FRED series via the public CSV endpoint (no API key required).

Macro data earns its place here for two reasons: the risk-free rate is a direct
input to every discount rate in :mod:`stocklab.valuation`, and the yield curve
plus credit spreads are the most reliable regime signals available for free.
"""
from __future__ import annotations

import io
import logging

import pandas as pd

from ...config import settings
from ..cache import cache, make_key
from ..http import OfflineError, get_text

log = logging.getLogger(__name__)

CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# Curated series set. Keys are stable names the rest of the codebase uses, so a
# FRED discontinuation only needs fixing here.
SERIES = {
    "ust_3m": "DGS3MO",
    "ust_2y": "DGS2",
    "ust_10y": "DGS10",
    "ust_30y": "DGS30",
    "curve_10y2y": "T10Y2Y",
    "curve_10y3m": "T10Y3M",
    "hy_oas": "BAMLH0A0HYM2",       # high-yield option-adjusted spread
    "ig_oas": "BAMLC0A0CM",         # investment-grade OAS
    "vix": "VIXCLS",
    "fed_funds": "DFF",
    "breakeven_10y": "T10YIE",
    "breakeven_5y": "T5YIE",
    "unemployment": "UNRATE",
    "cpi": "CPIAUCSL",
    "real_gdp": "GDPC1",
    "industrial_production": "INDPRO",
    "initial_claims": "ICSA",
    "recession_indicator": "USREC",  # NBER-dated, published with a long lag
    "dollar_index": "DTWEXBGS",
    "sahm_rule": "SAHMREALTIME",
}


def fetch_series(series_id: str, ttl: int | None = None) -> pd.Series:
    """One FRED series as a date-indexed float Series (missing values dropped)."""
    cfg = settings()
    key = make_key("fred", series_id)
    ttl = cfg.ttl_macro if ttl is None else ttl
    cached = cache().get_df(key, ttl)
    if cached is not None and not cached.empty:
        return cached.iloc[:, 0]

    try:
        text = get_text(CSV_URL, {"id": series_id})
    except OfflineError:
        stale = cache().get_df(key, None)
        return stale.iloc[:, 0] if stale is not None and not stale.empty else pd.Series(dtype=float)
    if not text:
        return pd.Series(dtype=float)

    frame = pd.read_csv(io.StringIO(text))
    if frame.shape[1] < 2:
        return pd.Series(dtype=float)
    date_col = frame.columns[0]
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    # FRED writes "." for a missing observation (holidays, unpublished periods).
    frame[frame.columns[1]] = pd.to_numeric(frame[frame.columns[1]], errors="coerce")
    frame = frame.dropna().set_index(date_col)
    frame.index.name = "date"
    if frame.empty:
        return pd.Series(dtype=float)
    cache().set_df(key, frame)
    return frame.iloc[:, 0]


def get(name: str, ttl: int | None = None) -> pd.Series:
    """Fetch by friendly name from :data:`SERIES` (or pass a raw FRED id)."""
    return fetch_series(SERIES.get(name, name), ttl=ttl)


def latest(name: str, default: float | None = None) -> float | None:
    series = get(name)
    if series.empty:
        return default
    return float(series.iloc[-1])


def risk_free_rate(tenor: str = "ust_10y") -> float:
    """Current risk-free rate as a decimal.

    The 10-year is the right default for equity valuation: it roughly matches
    the duration of the cash flows being discounted. Using the 3-month bill
    instead makes every DCF a bet on the front end of the curve.
    """
    cfg = settings()
    value = latest(tenor)
    if value is None or not (0.0 <= value <= 25.0):
        log.info("fred: falling back to configured risk-free rate")
        return cfg.risk_free_fallback
    return value / 100.0


def macro_snapshot() -> dict[str, float]:
    """Current readings for the regime model, with as-of dates."""
    out: dict[str, float] = {}
    for name in ("ust_3m", "ust_2y", "ust_10y", "curve_10y2y", "curve_10y3m",
                 "hy_oas", "ig_oas", "vix", "fed_funds", "breakeven_10y",
                 "unemployment", "initial_claims", "sahm_rule", "dollar_index"):
        series = get(name)
        if series.empty:
            continue
        out[name] = float(series.iloc[-1])
        out[f"{name}_asof"] = float(pd.Timestamp(series.index[-1]).value // 10**9)
        if len(series) > 60:
            out[f"{name}_chg_3m"] = float(series.iloc[-1] - series.iloc[-60])
    return out
