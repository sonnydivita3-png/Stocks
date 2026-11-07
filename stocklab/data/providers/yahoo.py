"""Yahoo Finance chart endpoint: split/dividend-adjusted OHLCV history.

Only the public ``/v8/finance/chart`` endpoint is used.  It needs no cookie or
crumb (unlike ``quoteSummary``/``v7/quote``, which now gate on both), and it
returns the full corporate-action history alongside prices, which is what makes
a correct total-return series possible.

Yahoo is an unofficial, undocumented, occasionally wrong source.  It is used
here for *prices* only; fundamentals come from SEC XBRL, which is auditable.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from ...config import settings
from ..cache import cache, make_key
from ..http import OfflineError, get_json

log = logging.getLogger(__name__)

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

OHLCV = ["open", "high", "low", "close", "volume", "adj_close"]


def _empty_history() -> pd.DataFrame:
    """An empty OHLCV frame that still carries a DatetimeIndex.

    Returning a default RangeIndex here poisons any later index union: joining
    an empty int-indexed frame with real date-indexed ones produces an object
    index, and callers that expect timestamps get floats instead.
    """
    return pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], name="date"))

_VALID_INTERVALS = {"1d", "1wk", "1mo"}


def _to_frame(result: dict) -> pd.DataFrame:
    """Flatten one Yahoo chart ``result`` block into a tidy OHLCV frame."""
    stamps = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    adj = (result.get("indicators", {}).get("adjclose") or [{}])[0]

    if not stamps:
        return _empty_history()

    # Yahoo timestamps are exchange-local market opens in UTC epoch seconds.
    # Normalising to midnight keeps daily bars joinable across exchanges.
    tz = (result.get("meta") or {}).get("exchangeTimezoneName") or "UTC"
    index = pd.to_datetime(stamps, unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()

    frame = pd.DataFrame(
        {
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
            "adj_close": adj.get("adjclose", quote.get("close")),
        },
        index=index,
    )
    frame.index.name = "date"
    # Yahoo emits a provisional bar for the session in progress with null OHLC.
    frame = frame[frame["close"].notna()]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame.astype(float)


def _actions(result: dict) -> tuple[pd.Series, pd.Series]:
    """Dividend and split series from the chart payload's ``events`` block."""
    events = result.get("events") or {}

    divs = {
        pd.to_datetime(int(v["date"]), unit="s").normalize(): float(v.get("amount", 0.0))
        for v in (events.get("dividends") or {}).values()
        if v.get("date") is not None
    }
    splits = {}
    for v in (events.get("splits") or {}).values():
        if v.get("date") is None:
            continue
        num, den = v.get("numerator"), v.get("denominator")
        ratio = v.get("splitRatio")
        if num and den:
            factor = float(num) / float(den)
        elif isinstance(ratio, str) and ":" in ratio:
            a, b = ratio.split(":")
            factor = float(a) / float(b)
        else:
            continue
        splits[pd.to_datetime(int(v["date"]), unit="s").normalize()] = factor

    return (
        pd.Series(divs, dtype=float).sort_index(),
        pd.Series(splits, dtype=float).sort_index(),
    )


def _align_events(events: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Place each event on the first bar at or after its date."""
    out = pd.Series(np.nan, index=index, dtype=float)
    if events.empty or len(index) == 0:
        return out
    positions = index.searchsorted(events.index, side="left")
    for pos, value in zip(positions, events.to_numpy(dtype=float)):
        if pos < len(index):
            prior = out.iloc[pos]
            # Two events landing on one bar compound (e.g. two splits in a month).
            out.iloc[pos] = value if pd.isna(prior) else prior * value
    return out


def fetch_history(
    symbol: str,
    *,
    period: str = "10y",
    interval: str = "1d",
    ttl: int | None = None,
    include_actions: bool = False,
) -> pd.DataFrame:
    """Daily (or weekly/monthly) history for one symbol.

    Returns a frame indexed by date with ``open/high/low/close/volume/adj_close``.
    ``adj_close`` is Yahoo's total-return series (splits *and* dividends);
    ``close`` is split-adjusted only.  Use ``adj_close`` for returns and
    ``close`` for anything compared against a quoted price or a stop level.
    """
    if interval not in _VALID_INTERVALS:
        raise ValueError(f"interval must be one of {sorted(_VALID_INTERVALS)}")

    cfg = settings()
    # `include_actions` is part of the key: a frame cached without the
    # dividend/split columns must not satisfy a request that needs them, or the
    # caller silently gets a frame with no `split` column and skips corporate
    # actions entirely.
    key = make_key("yahoo:hist", symbol.upper(), period, interval,
                   "act" if include_actions else "raw")
    ttl = cfg.ttl_prices if ttl is None else ttl

    cached = cache().get_df(key, ttl)
    if cached is not None and not cached.empty:
        return cached

    try:
        payload = get_json(
            CHART_URL.format(symbol=symbol.upper()),
            params={
                "range": period,
                "interval": interval,
                "events": "div,split",
                "includeAdjustedClose": "true",
            },
        )
    except OfflineError:
        stale = cache().get_df(key, None)  # offline: stale beats nothing
        if stale is not None:
            return stale
        raise
    except Exception as exc:  # noqa: BLE001
        # Yahoo is an unofficial endpoint that throttles without warning. One
        # unavailable price series must degrade a report, not abort a screen
        # over hundreds of names, so fall back to stale cache then to empty.
        stale = cache().get_df(key, None)
        if stale is not None and not stale.empty:
            log.warning("yahoo: %s unavailable (%s); using cached data", symbol, exc)
            return stale
        log.warning("yahoo: %s unavailable (%s); no cached data", symbol, exc)
        return _empty_history()

    chart = (payload or {}).get("chart") or {}
    results = chart.get("result") or []
    if chart.get("error") or not results:
        log.debug("yahoo: no data for %s (%s)", symbol, chart.get("error"))
        return _empty_history()

    result = results[0]
    frame = _to_frame(result)
    if frame.empty:
        return frame

    if include_actions:
        divs, splits = _actions(result)
        # Snap each event to the nearest bar rather than requiring an exact
        # index match: event timestamps are normalised from UTC while bars are
        # normalised in the exchange timezone, so they can differ by a day, and
        # on a weekly or monthly grid they will essentially never coincide.
        frame["dividend"] = _align_events(divs, frame.index).fillna(0.0)
        frame["split"] = _align_events(splits, frame.index).fillna(1.0)

    cache().set_df(key, frame)
    return frame


def fetch_meta(symbol: str, ttl: int | None = None) -> dict:
    """Exchange/currency metadata plus the latest quote from the chart payload.

    This is the only quote source that still works without an authenticated
    crumb, so it is also how we get a current price for valuation.
    """
    cfg = settings()
    key = make_key("yahoo:meta", symbol.upper())
    ttl = cfg.ttl_prices if ttl is None else ttl
    cached = cache().get_json(key, ttl)
    if cached is not None:
        return cached

    try:
        payload = get_json(
            CHART_URL.format(symbol=symbol.upper()),
            params={"range": "5d", "interval": "1d"},
        )
    except OfflineError:
        return cache().get_json(key, None) or {}
    except Exception as exc:  # noqa: BLE001 - see fetch_history
        cached_stale = cache().get_json(key, None)
        if cached_stale:
            log.warning("yahoo: quote for %s unavailable (%s); using cache", symbol, exc)
            return cached_stale
        log.warning("yahoo: quote for %s unavailable (%s)", symbol, exc)
        return {}

    results = ((payload or {}).get("chart") or {}).get("result") or []
    meta = (results[0].get("meta") if results else None) or {}
    out = {
        "symbol": meta.get("symbol", symbol.upper()),
        "currency": meta.get("currency"),
        "exchange": meta.get("fullExchangeName") or meta.get("exchangeName"),
        "instrument_type": meta.get("instrumentType"),
        "price": meta.get("regularMarketPrice"),
        "previous_close": meta.get("chartPreviousClose") or meta.get("previousClose"),
        "fifty_two_week_high": meta.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low": meta.get("fiftyTwoWeekLow"),
        "long_name": meta.get("longName") or meta.get("shortName"),
        "timezone": meta.get("exchangeTimezoneName"),
    }
    if out.get("price") is not None:
        cache().set_json(key, out)
    return out


def fetch_corporate_actions(symbol: str, ttl: int | None = None
                            ) -> tuple[pd.Series, pd.Series]:
    """Full dividend and split history, independent of any price grid.

    Deliberately separate from :func:`fetch_history`. Yahoo answers
    ``range=max`` with monthly bars, so aligning event dates onto that index
    drops every split -- which made a company that had split 28-for-1 look as
    though it never had.

    The two event types need different windows, because Yahoo caps the events
    list at roughly 170 entries per response and drops the middle of a long
    history to fit. Coca-Cola has paid quarterly dividends since 1962, so a
    ``range=max`` request returns its 1962-2004 payments and then jumps
    straight to the present, reporting a single dividend in the trailing year
    and a 0.5% yield. Dividends therefore come from a bounded ten-year daily
    window (complete, and all a trailing yield or five-year growth rate needs),
    while splits -- rare enough to survive the cap -- come from the full history.

    Returns ``(dividends, splits)`` indexed by event date.
    """
    key = make_key("yahoo:actions", symbol.upper())
    # Corporate actions are immutable once they happen, so this can be cached
    # for a long time; a month is conservative.
    ttl = (60 * 60 * 24 * 30) if ttl is None else ttl
    cached = cache().get_df(key, ttl)
    if cached is not None:
        divs = cached["dividend"].dropna() if "dividend" in cached else pd.Series(dtype=float)
        splits = cached["split"].dropna() if "split" in cached else pd.Series(dtype=float)
        return divs, splits

    def events_for(period: str, interval: str) -> tuple[pd.Series, pd.Series]:
        try:
            payload = get_json(
                CHART_URL.format(symbol=symbol.upper()),
                params={"range": period, "interval": interval,
                        "events": "div,split", "includeAdjustedClose": "true"},
            )
        except OfflineError:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        except Exception as exc:  # noqa: BLE001 - see fetch_history
            log.warning("yahoo: corporate actions for %s unavailable (%s)", symbol, exc)
            return pd.Series(dtype=float), pd.Series(dtype=float)
        results = ((payload or {}).get("chart") or {}).get("result") or []
        if not results:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        return _actions(results[0])

    divs, _ = events_for("10y", "1d")
    _, splits = events_for("max", "1mo")

    frame = pd.concat([divs.rename("dividend"), splits.rename("split")], axis=1).sort_index()
    if not frame.empty:
        frame.index.name = "date"
        cache().set_df(key, frame)
    return divs, splits


def fetch_splits(symbol: str, ttl: int | None = None) -> pd.Series:
    """Every split this symbol has had, as ``{date: factor}``."""
    _, splits = fetch_corporate_actions(symbol, ttl=ttl)
    return splits


def fetch_dividends(symbol: str, ttl: int | None = None) -> pd.Series:
    """Cash dividends per share by ex-date, over the last ten years."""
    divs, _ = fetch_corporate_actions(symbol, ttl=ttl)
    return divs


def fetch_closes(
    symbols: Iterable[str],
    *,
    period: str = "5y",
    interval: str = "1d",
    field: str = "adj_close",
    ttl: int | None = None,
) -> pd.DataFrame:
    """Wide frame of one price field across many symbols, outer-joined on date.

    Missing symbols are skipped rather than fatal; a screen over 500 tickers
    will always contain a few delistings and typos.
    """
    series: dict[str, pd.Series] = {}
    for sym in symbols:
        sym = sym.upper()
        try:
            hist = fetch_history(sym, period=period, interval=interval, ttl=ttl)
        except Exception as exc:  # noqa: BLE001 - one bad symbol must not stop a batch
            log.warning("yahoo: %s failed: %s", sym, exc)
            continue
        if not hist.empty and field in hist and hist[field].notna().any():
            series[sym] = hist[field]
    if not series:
        return pd.DataFrame()
    out = pd.DataFrame(series).sort_index()
    out.index.name = "date"
    return out
