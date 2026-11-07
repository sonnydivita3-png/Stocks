"""The :class:`Company` facade: one object that wires the data layers together.

Everything the analysis, valuation and reporting code needs for a single name,
loaded once and cached. Exists so callers never have to remember that dividend
yield comes from the exchange record while revenue comes from EDGAR.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from .analysis.metrics import compute_metrics
from .analysis.technicals import technical_snapshot
from .data.providers.sec import Fundamentals, load_fundamentals
from .data.providers.yahoo import fetch_dividends, fetch_history, fetch_meta
from .util.mathx import NAN, is_num

log = logging.getLogger(__name__)


@dataclass
class Company:
    symbol: str
    fundamentals: Fundamentals
    quote: dict = field(default_factory=dict)
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: dict = field(default_factory=dict)
    technicals: dict = field(default_factory=dict)
    dividend_ttm: float = NAN

    @property
    def price(self) -> float:
        value = self.quote.get("price")
        if is_num(value):
            return float(value)
        if not self.history.empty:
            return float(self.history["close"].iloc[-1])
        return NAN

    @property
    def name(self) -> str:
        return self.fundamentals.name or self.quote.get("long_name") or self.symbol

    @property
    def sector(self) -> str:
        return self.fundamentals.sector

    @property
    def has_data(self) -> bool:
        return self.fundamentals.has_data and is_num(self.price)

    @property
    def notes(self) -> list[str]:
        return list(self.fundamentals.notes)

    @classmethod
    def load(cls, symbol: str, *, period: str = "5y",
             benchmark: str = "SPY", with_technicals: bool = True) -> "Company":
        symbol = symbol.upper().strip()
        fundamentals = load_fundamentals(symbol)
        quote = fetch_meta(symbol)
        history = fetch_history(symbol, period=period)

        # Trailing twelve months of cash dividends per share, from the exchange
        # record rather than the filing: XBRL dividend tagging has real gaps.
        dividend_ttm = NAN
        dividends = fetch_dividends(symbol)
        if not dividends.empty:
            cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=365)
            recent = dividends[dividends.index >= cutoff]
            if not recent.empty:
                dividend_ttm = float(recent.sum())
            elif not dividends.empty:
                # Paid historically but nothing in the last year: the dividend
                # has been suspended, which is a zero yield, not unknown.
                dividend_ttm = 0.0

        price = quote.get("price")
        if not is_num(price) and not history.empty:
            price = float(history["close"].iloc[-1])

        metrics = compute_metrics(fundamentals, price, dividend_ttm=dividend_ttm)

        technicals: dict = {}
        if with_technicals and not history.empty and len(history) > 60:
            bench = fetch_history(benchmark, period=period)
            technicals = technical_snapshot(
                history, bench["adj_close"] if not bench.empty else None
            )

        return cls(symbol=symbol, fundamentals=fundamentals, quote=quote,
                   history=history, metrics=metrics, technicals=technicals,
                   dividend_ttm=dividend_ttm)

    def metric(self, key: str, default: float = NAN) -> float:
        value = self.metrics.get(key, default)
        return float(value) if is_num(value) else default
