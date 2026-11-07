"""Shared fixtures. Every test here runs offline against synthetic data."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stocklab.config import configure          # noqa: E402
from stocklab.data.cache import reset_cache              # noqa: E402
from stocklab.data.providers.sec import Fundamentals     # noqa: E402


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):
    """Point every test at a throwaway data directory and forbid network calls."""
    configure(root=tmp_path, offline=True)
    reset_cache(tmp_path / "cache.sqlite")
    # Any accidental HTTP use should fail loudly rather than reach the internet.
    import stocklab.data.http as http_module
    monkeypatch.setattr(
        http_module, "session",
        lambda: pytest.fail("a test attempted a network request"),
    )
    yield
    configure(offline=False)


def _annual_frame() -> pd.DataFrame:
    """Eight years of a steadily compounding, profitable business."""
    years = pd.to_datetime([f"{y}-12-31" for y in range(2019, 2027)])
    n = len(years)
    revenue = np.array([1000.0 * (1.10**i) for i in range(n)])
    frame = pd.DataFrame({
        "revenue": revenue,
        "cost_of_revenue": revenue * 0.55,
        "gross_profit": revenue * 0.45,
        "operating_income": revenue * 0.20,
        "pretax_income": revenue * 0.185,
        "tax_expense": revenue * 0.185 * 0.21,
        "net_income": revenue * 0.185 * 0.79,
        "eps_diluted": revenue * 0.185 * 0.79 / 100.0,
        "shares_diluted": np.full(n, 100.0),
        "interest_expense": revenue * 0.015,
        "depreciation_amortization": revenue * 0.05,
        "cfo": revenue * 0.22,
        "capex": revenue * 0.06,
        "dividends_paid": revenue * 0.03,
        "buybacks": revenue * 0.02,
        "total_assets": revenue * 1.5,
        "current_assets": revenue * 0.6,
        "current_liabilities": revenue * 0.3,
        "total_liabilities": revenue * 0.8,
        "equity": revenue * 0.7,
        "retained_earnings": revenue * 0.5,
        "cash": revenue * 0.2,
        "short_term_investments": revenue * 0.05,
        "receivables": revenue * 0.15,
        "inventory": revenue * 0.1,
        "ppe_net": revenue * 0.4,
        "goodwill": np.full(n, 50.0),
        "short_term_debt": revenue * 0.05,
        "long_term_debt": revenue * 0.25,
        "shares_outstanding": np.full(n, 100.0),
        "sga": revenue * 0.15,
        "rnd": revenue * 0.08,
        "stock_comp": revenue * 0.02,
    }, index=years)
    frame.index.name = "period_end"
    return frame


@pytest.fixture
def healthy_company() -> Fundamentals:
    annual = _annual_frame()
    quarters = []
    for stamp in annual.index:
        for q in range(4):
            quarters.append(stamp - pd.DateOffset(months=9 - 3 * q))
    quarterly = pd.DataFrame(
        np.repeat(annual.to_numpy() / 4.0, 4, axis=0),
        index=pd.DatetimeIndex(sorted(quarters)), columns=annual.columns,
    )
    # Stock items are point-in-time, not quarterly slices.
    for column in ("total_assets", "equity", "cash", "shares_outstanding",
                   "current_assets", "current_liabilities", "total_liabilities",
                   "short_term_debt", "long_term_debt", "goodwill", "ppe_net",
                   "receivables", "inventory", "short_term_investments",
                   "retained_earnings", "shares_diluted", "eps_diluted"):
        quarterly[column] = quarterly[column] * 4.0
    quarterly.index.name = "period_end"

    ttm = quarterly.iloc[-4:].sum()
    for column in ("total_assets", "equity", "cash", "shares_outstanding",
                   "current_assets", "current_liabilities", "total_liabilities",
                   "short_term_debt", "long_term_debt", "goodwill", "ppe_net",
                   "receivables", "inventory", "short_term_investments",
                   "retained_earnings"):
        ttm[column] = quarterly[column].iloc[-1]
    ttm["shares_diluted"] = 100.0
    ttm["shares_outstanding"] = 100.0
    ttm["eps_diluted"] = float(annual["eps_diluted"].iloc[-1])
    ttm["ttm_basis"] = 2.0

    filed = pd.Series(
        {stamp: (stamp + pd.Timedelta(days=40)).date().isoformat()
         for stamp in quarterly.index}, dtype="object")

    return Fundamentals(
        symbol="TEST", cik="0000000001",
        profile={"name": "Test Industries Inc.", "sic": 3571,
                 "sic_description": "Electronic Computers"},
        annual=annual, quarterly=quarterly, ttm=ttm, filed_dates=filed,
        sources={"revenue": ["Revenues"]},
    )


@pytest.fixture
def price_history() -> pd.DataFrame:
    """Three years of daily bars with a mild uptrend and realistic noise."""
    index = pd.bdate_range("2024-01-01", periods=740)
    rng = np.random.default_rng(42)
    steps = rng.normal(0.0004, 0.014, len(index))
    close = 100.0 * np.exp(np.cumsum(steps))
    high = close * (1 + np.abs(rng.normal(0, 0.006, len(index))))
    low = close * (1 - np.abs(rng.normal(0, 0.006, len(index))))
    frame = pd.DataFrame({
        "open": np.r_[close[0], close[:-1]],
        "high": np.maximum(high, close),
        "low": np.minimum(low, close),
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, len(index)).astype(float),
        "adj_close": close,
    }, index=index)
    frame.index.name = "date"
    return frame
