"""SEC EDGAR XBRL: authoritative fundamentals, assembled into clean statements.

Why SEC rather than a convenience API: these are the numbers the company
actually filed, with the filing date attached.  That lets us (a) avoid
look-ahead bias in backtests by asking "what was known on date X", and (b) see
restatements, which vendor APIs silently overwrite.

Three endpoints are used:
  * ``company_tickers.json``  ticker -> CIK
  * ``submissions/CIK*.json`` entity profile (name, SIC, exchange, filings)
  * ``companyfacts/CIK*.json`` every XBRL fact the company ever reported

The hard part is not fetching, it is assembly.  XBRL facts are a bag of
``(tag, start, end, value, filed)`` tuples with no period grid, duplicated
across filings, and with Q4 almost never reported directly.  The private
helpers below rebuild a coherent fiscal-period grid from that bag.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from ...config import settings
from ...util.mathx import NAN, is_num, safe_div
from ..cache import cache, make_key
from ..concepts import ALL_CONCEPTS, BY_NAME, Concept, is_financial, is_reit, sector_for_sic
from ..http import OfflineError, get, get_json

log = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
FRAME_URL = "https://data.sec.gov/api/xbrl/frames/{tax}/{tag}/{unit}/{period}.json"

_DAYS_PER_QUARTER = 91.3
# A "quarter" in practice runs 80-100 days; a fiscal year 340-380.  52/53-week
# retail calendars sit at the edges of both, hence the generous windows.
_MIN_PERIOD_DAYS = 55
_MAX_PERIOD_DAYS = 400
_CLUSTER_TOL_DAYS = 20   # two end dates this close are the same fiscal period
_BOUNDARY_TOL_DAYS = 12  # slack when testing whether sub-periods tile a period


# --------------------------------------------------------------------------- #
# ticker <-> CIK
# --------------------------------------------------------------------------- #
def fetch_ticker_map(ttl: int | None = None) -> dict[str, str]:
    """``{TICKER: zero-padded CIK}`` for every SEC registrant with a ticker."""
    cfg = settings()
    key = make_key("sec:tickers")
    ttl = cfg.ttl_universe if ttl is None else ttl
    cached = cache().get_json(key, ttl)
    if cached:
        return cached
    try:
        payload = get_json(TICKERS_URL)
    except OfflineError:
        return cache().get_json(key, None) or {}
    out: dict[str, str] = {}
    for row in (payload or {}).values():
        ticker = str(row.get("ticker", "")).upper().strip()
        cik = row.get("cik_str")
        if ticker and cik is not None:
            out[ticker] = str(int(cik)).zfill(10)
    if out:
        cache().set_json(key, out)
    return out


def ticker_to_cik(ticker: str) -> str | None:
    """Resolve a ticker to a 10-digit CIK.

    Handles the class-share mismatch between exchanges and EDGAR: Yahoo writes
    ``BRK-B``, EDGAR writes ``BRK-B`` too, but many feeds use ``BRK.B``.
    """
    mapping = fetch_ticker_map()
    raw = ticker.upper().strip()
    for candidate in (raw, raw.replace(".", "-"), raw.replace("-", "."), raw.split(".")[0]):
        if candidate in mapping:
            return mapping[candidate]
    return None


# --------------------------------------------------------------------------- #
# raw endpoint wrappers
# --------------------------------------------------------------------------- #
def fetch_submissions(cik: str, ttl: int | None = None) -> dict:
    cfg = settings()
    cik = str(cik).zfill(10)
    key = make_key("sec:sub", cik)
    ttl = cfg.ttl_profile if ttl is None else ttl
    cached = cache().get_json(key, ttl)
    if cached is not None:
        return cached
    try:
        payload = get_json(SUBMISSIONS_URL.format(cik=cik))
    except OfflineError:
        return cache().get_json(key, None) or {}
    if not payload:
        return {}
    recent = (payload.get("filings") or {}).get("recent") or {}
    # Keep only what we use; the full document is ~160 KB of filing index.
    trimmed = {
        "cik": cik,
        "name": payload.get("name"),
        "sic": payload.get("sic"),
        "sic_description": payload.get("sicDescription"),
        "tickers": payload.get("tickers") or [],
        "exchanges": payload.get("exchanges") or [],
        "fiscal_year_end": payload.get("fiscalYearEnd"),
        "entity_type": payload.get("entityType"),
        "state_of_incorporation": payload.get("stateOfIncorporation"),
        "category": payload.get("category"),
        "recent_filings": [
            {
                "form": f,
                "filed": d,
                "period": p,
                "accession": a,
            }
            for f, d, p, a in zip(
                recent.get("form", [])[:60],
                recent.get("filingDate", [])[:60],
                recent.get("reportDate", [])[:60],
                recent.get("accessionNumber", [])[:60],
            )
        ],
    }
    cache().set_json(key, trimmed)
    return trimmed


def fetch_companyfacts(cik: str, ttl: int | None = None) -> dict:
    """Every XBRL fact for a company (~2-6 MB raw, ~400 KB gzipped in cache)."""
    cfg = settings()
    cik = str(cik).zfill(10)
    key = make_key("sec:facts", cik)
    ttl = cfg.ttl_fundamentals if ttl is None else ttl
    cached = cache().get_json(key, ttl)
    if cached is not None:
        return cached
    try:
        payload = get_json(FACTS_URL.format(cik=cik))
    except OfflineError:
        return cache().get_json(key, None) or {}
    if not payload:
        return {}
    cache().set_json(key, payload)
    return payload


def fetch_frame(tag: str, period: str, unit: str = "USD", taxonomy: str = "us-gaap",
                ttl: int | None = None) -> pd.DataFrame:
    """One concept, one period, *every* filer -- the cheap path for screening.

    ``period`` is a frame label such as ``CY2025`` (annual), ``CY2025Q1``
    (quarterly duration) or ``CY2025Q1I`` (instant).  One request replaces
    thousands of companyfacts downloads, so factor matrices are built from
    frames and only shortlisted names get the full treatment.
    """
    cfg = settings()
    key = make_key("sec:frame", taxonomy, tag, unit, period)
    ttl = cfg.ttl_fundamentals if ttl is None else ttl
    cached = cache().get_df(key, ttl)
    if cached is not None:
        return _normalise_cik_index(cached)
    try:
        payload = get_json(FRAME_URL.format(tax=taxonomy, tag=tag, unit=unit, period=period))
    except OfflineError:
        stale = cache().get_df(key, None)
        return _normalise_cik_index(stale) if stale is not None else pd.DataFrame()
    data = (payload or {}).get("data") or []
    if not data:
        return pd.DataFrame()
    frame = pd.DataFrame(data)
    keep = [c for c in ("cik", "entityName", "val", "start", "end", "accn") if c in frame]
    frame = frame[keep].copy()
    frame = frame.set_index("cik")
    cache().set_df(key, frame)
    return _normalise_cik_index(frame)


def _normalise_cik_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Force the index to 10-digit zero-padded CIK strings.

    The cache stores frames as CSV, and a column of zero-padded digits reads
    back as int64 -- "0000320193" returns as 320193. Every lookup against the
    ticker map then misses silently, which made a whole-market screen match a
    few hundred companies instead of several thousand. Applied on both the
    fresh and cached paths so the two cannot disagree.
    """
    if frame is None or frame.empty:
        return frame if frame is not None else pd.DataFrame()
    out = frame.copy()
    out.index = pd.Index(
        [str(int(value)).zfill(10) if str(value).strip().lstrip("-").isdigit()
         else str(value) for value in out.index],
        name="cik",
    )
    return out


# --------------------------------------------------------------------------- #
# fact-bag -> period grid
# --------------------------------------------------------------------------- #
def _iter_unit_rows(facts: dict, concept: Concept) -> Iterable[tuple[str, dict]]:
    """Yield ``(tag, row)`` for every row of every candidate tag, in priority order."""
    taxonomy_block = (facts.get("facts") or {}).get(concept.taxonomy) or {}
    for tag in concept.tags:
        entry = taxonomy_block.get(tag)
        if not entry:
            continue
        units = entry.get("units") or {}
        rows = units.get(concept.unit)
        if rows is None:
            # Unit labels drift (``USD/shares`` vs ``usdPerShare``); if there is
            # exactly one unit, trust it rather than dropping the tag.
            if len(units) == 1:
                rows = next(iter(units.values()))
            else:
                continue
        for row in rows:
            yield tag, row


def _dedupe_latest_filed(rows: Iterable[dict], key_fields: tuple[str, ...]) -> dict[tuple, dict]:
    """Collapse duplicate facts for the same period, keeping the newest filing.

    The same fiscal year appears in the original 10-K and again as a comparative
    in later filings.  Taking ``max(filed)`` means we use restated figures,
    which is what you want for analysis (and what you must *not* do for
    point-in-time backtests -- see :func:`as_of_filter`).
    """
    best: dict[tuple, dict] = {}
    for row in rows:
        try:
            k = tuple(row.get(f) for f in key_fields)
        except TypeError:
            continue
        if None in k:
            continue
        prior = best.get(k)
        if prior is None or str(row.get("filed", "")) > str(prior.get("filed", "")):
            best[k] = row
    return best


def _n_quarters(start: pd.Timestamp, end: pd.Timestamp) -> int | None:
    days = (end - start).days + 1
    if days < _MIN_PERIOD_DAYS or days > _MAX_PERIOD_DAYS:
        return None
    n = int(round(days / _DAYS_PER_QUARTER))
    return n if 1 <= n <= 4 else None


def _derive_quarters(
    durations: list[tuple[pd.Timestamp, pd.Timestamp, float]]
) -> dict[pd.Timestamp, float]:
    """Rebuild a discrete quarterly series from a mix of discrete and YTD facts.

    US filers report Q1, then H1, then 9M cumulative, then the full year -- so
    Q2, Q3 and Q4 must be recovered by subtraction.  The algorithm is interval
    arithmetic: for any multi-quarter fact, if the quarters already known tile
    all of it but one contiguous gap, that gap's value is the remainder.
    Iterating lets an annual figure yield Q4 once Q1-Q3 are themselves derived.

    Returns ``{period_end: value}``.
    """
    known: dict[tuple[pd.Timestamp, pd.Timestamp], float] = {}
    longs: list[tuple[pd.Timestamp, pd.Timestamp, float, int]] = []

    for start, end, val in durations:
        n = _n_quarters(start, end)
        if n is None:
            continue
        if n == 1:
            known.setdefault((start, end), val)
        else:
            longs.append((start, end, val, n))

    longs.sort(key=lambda t: t[3])  # shortest spans first: they constrain most

    for _ in range(4):  # convergence is fast; 4 passes is ample for 4 quarters
        progress = False
        for start, end, val, n in longs:
            inside = [
                (qs, qe, qv)
                for (qs, qe), qv in known.items()
                if qs >= start - pd.Timedelta(days=_BOUNDARY_TOL_DAYS)
                and qe <= end + pd.Timedelta(days=_BOUNDARY_TOL_DAYS)
            ]
            if len(inside) != n - 1:
                continue
            inside.sort(key=lambda t: t[0])

            # Walk the span and collect uncovered gaps.
            gaps: list[tuple[pd.Timestamp, pd.Timestamp]] = []
            cursor = start
            for qs, qe, _ in inside:
                if (qs - cursor).days > _BOUNDARY_TOL_DAYS:
                    gaps.append((cursor, qs - pd.Timedelta(days=1)))
                cursor = max(cursor, qe + pd.Timedelta(days=1))
            if (end - cursor).days > _BOUNDARY_TOL_DAYS:
                gaps.append((cursor, end))

            if len(gaps) != 1:
                continue
            gap_start, gap_end = gaps[0]
            if _n_quarters(gap_start, gap_end) != 1:
                continue
            if (gap_start, gap_end) in known:
                continue
            known[(gap_start, gap_end)] = val - sum(qv for _, _, qv in inside)
            progress = True
        if not progress:
            break

    return {end: val for (_, end), val in known.items()}


def _cluster(dates: Iterable[pd.Timestamp], tol_days: int = _CLUSTER_TOL_DAYS) -> dict:
    """Single-linkage clustering of period-end dates onto a canonical grid.

    Different tags can close the same fiscal period on dates a few days apart
    (52/53-week calendars, or a filer tagging 2025-09-30 where the 10-K says
    2025-09-27).  Without this, a DataFrame built by joining on raw end dates
    comes out as a staircase of half-empty rows.  Each cluster is labelled by
    its latest date.
    """
    uniq = sorted({pd.Timestamp(d).normalize() for d in dates})
    if not uniq:
        return {}
    mapping: dict[pd.Timestamp, pd.Timestamp] = {}
    group = [uniq[0]]
    for d in uniq[1:]:
        if (d - group[-1]).days <= tol_days:
            group.append(d)
        else:
            label = group[-1]
            mapping.update({g: label for g in group})
            group = [d]
    mapping.update({g: group[-1] for g in group})
    return mapping


# An instant can be dated after the period it describes: `dei`'s share count is
# the *cover page* figure, stamped when the filing went out, typically 2-4 weeks
# past fiscal year end.  So look back further than forward.
_SNAP_BACK_DAYS = 70
_SNAP_FORWARD_DAYS = 20


def _snap(instant: pd.Timestamp, grid: list[pd.Timestamp]) -> pd.Timestamp | None:
    """Attach an instant to the fiscal period it belongs to, or drop it.

    Returning ``None`` is the point: a Q3 balance sheet must not be grafted onto
    the annual frame just because it is the nearest row.
    """
    back = [g for g in grid if 0 <= (instant - g).days <= _SNAP_BACK_DAYS]
    if back:
        return max(back)
    forward = [g for g in grid if 0 < (g - instant).days <= _SNAP_FORWARD_DAYS]
    if forward:
        return min(forward)
    return None


# --------------------------------------------------------------------------- #
# assembled fundamentals
# --------------------------------------------------------------------------- #
@dataclass
class Fundamentals:
    """Normalised financial history for one company.

    ``annual`` and ``quarterly`` are DataFrames indexed by fiscal period end,
    with one column per concept in :mod:`stocklab.data.concepts`.  ``ttm`` is a
    trailing-twelve-month Series for flow items plus the latest reported value
    for balance-sheet items -- the mixed-basis snapshot valuation needs.
    """

    symbol: str
    cik: str
    profile: dict = field(default_factory=dict)
    annual: pd.DataFrame = field(default_factory=pd.DataFrame)
    quarterly: pd.DataFrame = field(default_factory=pd.DataFrame)
    ttm: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    filed_dates: pd.Series = field(default_factory=lambda: pd.Series(dtype="object"))
    # Which XBRL tag(s) produced each concept -- every number here is traceable
    # back to a filed tag, which is the whole reason for using EDGAR directly.
    sources: dict[str, list[str]] = field(default_factory=dict)
    # Year-to-date figures scaled to a full year.  NOT a TTM: it assumes the
    # remaining quarters look like those reported.  Exposed separately, and
    # never folded into `ttm`, so a seasonal business cannot be misread.
    run_rate: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    # Extra CIKs whose filings were stitched in (corporate reorganisations).
    # Non-empty means this history spans more than one registrant.
    chained_ciks: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # -- profile shortcuts -------------------------------------------------
    @property
    def name(self) -> str:
        return self.profile.get("name") or self.symbol

    @property
    def sic(self) -> int | None:
        try:
            return int(self.profile.get("sic"))
        except (TypeError, ValueError):
            return None

    @property
    def sector(self) -> str:
        return sector_for_sic(self.sic)

    @property
    def industry(self) -> str:
        return self.profile.get("sic_description") or "Unknown"

    @property
    def is_financial(self) -> bool:
        return is_financial(self.sic)

    @property
    def is_reit(self) -> bool:
        return is_reit(self.sic)

    @property
    def has_data(self) -> bool:
        return not self.annual.empty or not self.ttm.dropna().empty

    @property
    def last_annual_date(self) -> pd.Timestamp | None:
        return self.annual.index.max() if not self.annual.empty else None

    @property
    def last_quarter_date(self) -> pd.Timestamp | None:
        return self.quarterly.index.max() if not self.quarterly.empty else None

    # -- access ------------------------------------------------------------
    def get(self, concept: str, default: float = NAN) -> float:
        """Latest TTM / point-in-time value for a concept."""
        val = self.ttm.get(concept, default)
        return float(val) if is_num(val) else default

    def annual_series(self, concept: str, years: int | None = None) -> pd.Series:
        if concept not in self.annual:
            return pd.Series(dtype=float)
        s = self.annual[concept].dropna()
        return s.iloc[-years:] if years else s

    def quarterly_series(self, concept: str, quarters: int | None = None) -> pd.Series:
        if concept not in self.quarterly:
            return pd.Series(dtype=float)
        s = self.quarterly[concept].dropna()
        return s.iloc[-quarters:] if quarters else s

    def ttm_series(self, concept: str) -> pd.Series:
        """Rolling 4-quarter sum through history -- lets us chart TTM trends."""
        s = self.quarterly_series(concept)
        if len(s) < 4:
            return pd.Series(dtype=float)
        return s.rolling(4).sum().dropna()


def _assemble(symbol: str, cik: str, facts: dict, profile: dict) -> Fundamentals:
    """Turn a raw companyfacts bag into aligned annual / quarterly frames."""
    annual_raw: dict[str, dict[pd.Timestamp, float]] = {}
    quarterly_raw: dict[str, dict[pd.Timestamp, float]] = {}
    instant_raw: dict[str, dict[pd.Timestamp, float]] = {}
    filed_lookup: dict[pd.Timestamp, str] = {}

    sources: dict[str, list[str]] = {}

    for concept in ALL_CONCEPTS:
        rows_by_tag: dict[str, list[dict]] = {}
        for tag, row in _iter_unit_rows(facts, concept):
            rows_by_tag.setdefault(tag, []).append(row)
        if not rows_by_tag:
            continue

        # Walk candidate tags in priority order and *merge*, rather than taking
        # the first tag with any data.  Companies migrate tags when the FASB
        # deprecates one: NVDA's revenue sits under
        # RevenueFromContractWithCustomerExcludingAssessedTax through FY2022 and
        # under Revenues after it.  First-match would silently return a series
        # that stopped four years ago.  Higher-priority tags win any period they
        # cover; lower-priority tags only fill gaps.
        acc_annual: dict[pd.Timestamp, float] = {}
        acc_quarters: dict[pd.Timestamp, float] = {}
        acc_instant: dict[pd.Timestamp, float] = {}
        used: list[str] = []

        is_instant = concept.kind == "stock" or (
            concept.kind == "shares" and concept.statement == "balance"
        )

        for tag in concept.tags:
            rows = rows_by_tag.get(tag)
            if not rows:
                continue

            if is_instant:
                picked = _dedupe_latest_filed(
                    (r for r in rows if r.get("start") is None and r.get("end")), ("end",)
                )
                if not picked:  # some filers stamp a start date on instants too
                    picked = _dedupe_latest_filed((r for r in rows if r.get("end")), ("end",))
                added = False
                for (end,), row in picked.items():
                    try:
                        val = float(row["val"])
                    except (TypeError, ValueError, KeyError):
                        continue
                    # A listed company cannot have zero shares outstanding, but
                    # filers do tag 0 on some cover pages (multi-class
                    # registrants, certain form types). Zero is finite, so it
                    # would otherwise be accepted as real and make market cap
                    # collapse to nothing.
                    if concept.kind == "shares" and val <= 0:
                        continue
                    ts = pd.Timestamp(end).normalize()
                    if ts not in acc_instant:
                        acc_instant[ts] = val
                        added = True
                    _record_first_filed(filed_lookup, ts, row.get("filed"))
                if added:
                    used.append(tag)
                continue

            picked = _dedupe_latest_filed(
                (r for r in rows if r.get("start") and r.get("end")), ("start", "end")
            )
            if not picked:
                continue

            durations: list[tuple[pd.Timestamp, pd.Timestamp, float]] = []
            annual_series: dict[pd.Timestamp, float] = {}
            for (start, end), row in picked.items():
                try:
                    val = float(row["val"])
                except (TypeError, ValueError, KeyError):
                    continue
                s_ts, e_ts = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
                durations.append((s_ts, e_ts, val))
                _record_first_filed(filed_lookup, e_ts, row.get("filed"))
                if _n_quarters(s_ts, e_ts) == 4:
                    annual_series[e_ts] = val

            if concept.kind in ("per_share", "shares", "ratio"):
                # Neither per-share figures nor weighted-average share counts are
                # additive across periods, so a residual must never be derived:
                # subtracting three quarterly averages from an annual average
                # yields a large negative number, not Q4.  Discrete periods only.
                quarters = {e: v for s, e, v in durations if _n_quarters(s, e) == 1}
            else:
                # Derive per-tag, never across tags: subtracting a discrete
                # quarter under one tag from a YTD total under another would
                # silently mix two different definitions of revenue.
                quarters = _derive_quarters(durations)

            added = False
            for date, val in annual_series.items():
                if date not in acc_annual:
                    acc_annual[date] = val
                    added = True
            for date, val in quarters.items():
                if date not in acc_quarters:
                    acc_quarters[date] = val
                    added = True
            if added:
                used.append(tag)

        if acc_annual:
            annual_raw[concept.name] = acc_annual
        if acc_quarters:
            quarterly_raw[concept.name] = acc_quarters
        if acc_instant:
            instant_raw[concept.name] = acc_instant
        if used:
            sources[concept.name] = used

    # ---- align onto canonical period grids -------------------------------
    def build(flows: dict[str, dict[pd.Timestamp, float]],
              stocks: dict[str, dict[pd.Timestamp, float]]) -> pd.DataFrame:
        """Align flows onto a canonical grid, then snap instants onto it.

        The grid is derived from *flow* periods only.  Deriving it from every
        date -- instants included -- lets a cover-page date 20 days past year end
        hijack the cluster label, so the fiscal year appears to end in October.
        """
        grid_map = _cluster([d for series in flows.values() for d in series])
        if not grid_map:
            return pd.DataFrame()
        grid = sorted(set(grid_map.values()))

        columns: dict[str, dict[pd.Timestamp, float]] = {}
        for name, series in flows.items():
            col = columns.setdefault(name, {})
            for date, val in series.items():
                col[grid_map.get(date, date)] = val
        for name, series in stocks.items():
            col = columns.setdefault(name, {})
            for date, val in series.items():
                label = _snap(date, grid)
                if label is None:
                    continue
                # Several instants can snap to one period (a 10-K cover date and
                # the balance sheet date); the one closest to period end wins.
                prior = col.get(label)
                if prior is None or abs((date - label).days) < col.get(("__d", label), 99):
                    col[label] = val
                    col[("__d", label)] = abs((date - label).days)
        for col in columns.values():
            for k in [k for k in col if isinstance(k, tuple)]:
                del col[k]

        frame = pd.DataFrame(columns).sort_index()
        frame.index.name = "period_end"
        return frame

    annual = build(annual_raw, instant_raw)
    quarterly = build(quarterly_raw, instant_raw)

    # Instants only exist on reporting dates; carry the balance sheet forward
    # within each frame so a row is never half-empty for mechanical reasons.
    stock_cols = [c.name for c in ALL_CONCEPTS
                  if (c.kind == "stock" or (c.kind == "shares" and c.statement == "balance"))]
    for frame in (annual, quarterly):
        present = [c for c in stock_cols if c in frame]
        if present:
            frame[present] = frame[present].ffill()

    # Drop rows with no flow data at all (pure balance-sheet-date artefacts).
    flow_cols = [c.name for c in ALL_CONCEPTS if c.kind in ("flow", "per_share")]
    for name in ("annual", "quarterly"):
        frame = annual if name == "annual" else quarterly
        present = [c for c in flow_cols if c in frame]
        if present and not frame.empty:
            keep = frame[present].notna().any(axis=1)
            trimmed = frame[keep]
            if name == "annual":
                annual = trimmed
            else:
                quarterly = trimmed

    # Second pass for pretax_income -> operating_income, which depends on the
    # first pass having filled pretax_income from net income plus tax.
    for _ in range(2):
        annual = _derive_columns(annual)
        quarterly = _derive_columns(quarterly)

    ttm = _build_ttm(annual, quarterly, instant_raw)
    ttm = _fill_ttm_from_frames(ttm, annual, quarterly)
    # Align onto the quarterly grid where possible; fall back to annual.
    labels = list(quarterly.index) if not quarterly.empty else list(annual.index)
    filed = _align_filed_dates(filed_lookup, labels)
    return Fundamentals(
        symbol=symbol.upper(), cik=cik, profile=profile,
        annual=annual, quarterly=quarterly, ttm=ttm, filed_dates=filed,
        sources=sources, run_rate=_run_rate(quarterly, annual),
    )


def _record_first_filed(lookup: dict, period: pd.Timestamp, filed) -> None:
    """Track the EARLIEST filing that disclosed a period.

    Values use the latest filing (restatements supersede), but availability
    must use the first. The two are routinely a year apart, because a 10-Q
    carries the prior year's comparatives: taking the latest filing date as the
    publication date back-dates a full year of hindsight into every
    point-in-time series built from it.
    """
    if not filed:
        return
    stamp = str(filed)
    current = lookup.get(period)
    if current is None or stamp < current:
        lookup[period] = stamp


def _align_filed_dates(filed_raw: dict, labels: list) -> pd.Series:
    """Map first-filed dates from raw period dates onto the fiscal grid."""
    if not filed_raw or not labels:
        return pd.Series(dtype="object")
    grid = sorted(labels)
    out: dict[pd.Timestamp, str] = {}
    for raw_date, stamp in filed_raw.items():
        label = _snap(raw_date, grid)
        if label is None:
            continue
        current = out.get(label)
        if current is None or stamp < current:
            out[label] = stamp
    return pd.Series(out, dtype="object").sort_index()


def _fill_ttm_from_frames(ttm: pd.Series, annual: pd.DataFrame,
                          quarterly: pd.DataFrame) -> pd.Series:
    """Backfill TTM entries from the derived statement frames.

    :func:`_build_ttm` reads balance-sheet items straight out of the raw instant
    facts, so it never sees the columns :func:`_derive_columns` reconstructed --
    which left total liabilities (and therefore NCAV and reproduction value)
    missing for any filer that reports assets and equity but no liabilities
    subtotal. This closes that gap.
    """
    stock_names = {c.name for c in ALL_CONCEPTS
                   if c.kind == "stock" or (c.kind == "shares" and c.statement == "balance")}
    for frame in (quarterly, annual):  # most recent basis first
        if frame.empty:
            continue
        last = frame.iloc[-1]
        for name in stock_names:
            if name in last and is_num(last.get(name)) and not is_num(ttm.get(name)):
                ttm[name] = float(last[name])
    return ttm


def _derive_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill line items a filer never tagged, using accounting identities.

    Not every company tags every subtotal. Exxon, for instance, reports no
    ``OperatingIncomeLoss`` at all, which leaves operating margin undefined and
    silently disables every EBIT-based model. Where an identity recovers the
    figure exactly, it is better to use it than to report nothing.

    Each derived column is marked in ``derived_<name>`` so no downstream
    consumer mistakes a reconstruction for a filed number.
    """
    if frame.empty:
        return frame

    def col(name: str) -> pd.Series:
        return frame[name] if name in frame else pd.Series(np.nan, index=frame.index)

    # EBIT = pretax income + interest expense. Exact when interest is the only
    # item between the two, which is the common case; it slightly overstates
    # EBIT when large non-operating gains sit above the pretax line.
    derivations: tuple[tuple[str, pd.Series], ...] = (
        ("gross_profit", col("revenue") - col("cost_of_revenue")),
        ("pretax_income", col("net_income") + col("tax_expense")),
        ("operating_income", col("pretax_income") + col("interest_expense")),
        ("total_liabilities", col("total_assets") - col("equity")),
        ("current_assets", col("cash") + col("short_term_investments")
         + col("receivables") + col("inventory")),
    )
    for name, candidate in derivations:
        if candidate.isna().all():
            continue
        existing = col(name)
        fill = candidate.where(existing.isna())
        if fill.notna().any():
            frame[name] = existing.fillna(candidate)
            frame[f"derived_{name}"] = fill.notna().astype(float)
    return frame


def _build_ttm(
    annual: pd.DataFrame,
    quarterly: pd.DataFrame,
    instants: dict[str, dict[pd.Timestamp, float]],
) -> pd.Series:
    """Trailing-twelve-month flows + latest-reported stocks.

    Flow concepts prefer a rolling sum of the four most recent quarters; when
    quarterly coverage is incomplete the latest fiscal year is used instead, and
    ``ttm_basis`` records which happened so callers can judge staleness.
    """
    out: dict[str, float] = {}
    basis = "none"
    as_of: pd.Timestamp | None = None

    flow_names = [c.name for c in ALL_CONCEPTS if c.kind == "flow"]
    per_share_names = [c.name for c in ALL_CONCEPTS if c.kind == "per_share"]
    share_flow_names = [c.name for c in ALL_CONCEPTS if c.kind == "shares" and c.statement == "income"]

    if not quarterly.empty and len(quarterly) >= 4:
        tail = quarterly.iloc[-4:]
        span_ok = (tail.index[-1] - tail.index[0]).days <= 300  # 4 real quarters
        if span_ok:
            basis, as_of = "ttm_quarters", tail.index[-1]
            for name in flow_names + per_share_names:
                if name in tail and tail[name].notna().sum() == 4:
                    out[name] = float(tail[name].sum())
            for name in share_flow_names:
                if name in tail and tail[name].notna().any():
                    # Already period averages: average them, never sum.
                    out[name] = float(tail[name].dropna().mean())

    if not annual.empty:
        last = annual.iloc[-1]
        if basis == "none":
            basis, as_of = "last_fiscal_year", annual.index[-1]
        for name in flow_names + per_share_names + share_flow_names:
            if name not in out and name in last and is_num(last.get(name)):
                out[name] = float(last[name])

    # Balance-sheet items: always the most recent instant, whatever its source.
    for name, series in instants.items():
        if series:
            latest = max(series)
            out[name] = float(series[latest])
            as_of = max(as_of, latest) if as_of is not None else latest

    result = pd.Series(out, dtype=float)
    result["ttm_basis"] = {"none": 0.0, "last_fiscal_year": 1.0, "ttm_quarters": 2.0}[basis]
    if as_of is not None:
        result["as_of_epoch"] = float(pd.Timestamp(as_of).value // 10**9)
    return result


def load_fundamentals(symbol: str, ttl: int | None = None, chain: bool = True) -> Fundamentals:
    """Fetch and assemble fundamentals for ``symbol``.

    Returns an empty :class:`Fundamentals` (``has_data is False``) rather than
    raising when the ticker is not an SEC registrant -- ETFs, ADRs filing 20-F
    without XBRL, and foreign listings all land here legitimately.
    """
    symbol = symbol.upper().strip()
    cik = ticker_to_cik(symbol)
    if not cik:
        log.info("sec: no CIK for %s (not an SEC XBRL filer?)", symbol)
        return Fundamentals(symbol=symbol, cik="")

    profile = fetch_submissions(cik, ttl=ttl)
    facts = fetch_companyfacts(cik, ttl=ttl)
    if not facts:
        return Fundamentals(symbol=symbol, cik=cik, profile=profile)

    result = _assemble(symbol, cik, facts, profile)

    # A listed operating company with under three years of annual filings is
    # usually a reorganised registrant, not a startup.  Try to recover the
    # predecessor's history before giving the caller a two-quarter company.
    if chain and len(result.annual) < MIN_ANNUAL_YEARS:
        merged, used = facts, []
        for other in find_predecessor_ciks(cik, profile):
            older = fetch_companyfacts(other, ttl=ttl)
            if not older:
                continue
            candidate = _merge_facts(merged, older)
            trial = _assemble(symbol, cik, candidate, profile)
            if len(trial.annual) > len(result.annual):
                merged, result, used = candidate, trial, used + [other]
        if used:
            result.chained_ciks = used
            result.notes.append(
                "History stitched across CIKs "
                + ", ".join([cik] + used)
                + " (corporate reorganisation); pre-reorganisation figures are the "
                  "predecessor entity's and may differ in consolidation scope."
            )

    if len(result.annual) < MIN_ANNUAL_YEARS:
        result.notes.append(
            f"Only {len(result.annual)} fiscal year(s) of XBRL data available -- "
            "growth, trend and quality metrics needing history will be unavailable."
        )
    return result


def as_of_filter(facts: dict, as_of: str | pd.Timestamp) -> dict:
    """Strip facts filed after ``as_of``, for point-in-time backtesting.

    Backtesting a fundamental strategy on restated, as-known-today figures is
    the single most common way to manufacture alpha that never existed.  Pass
    the output of this through :func:`_assemble` to see only what a filer had
    actually published by a given date.
    """
    cutoff = str(pd.Timestamp(as_of).date())
    out = {"cik": facts.get("cik"), "entityName": facts.get("entityName"), "facts": {}}
    for taxonomy, tags in (facts.get("facts") or {}).items():
        new_tags = {}
        for tag, entry in tags.items():
            new_units = {}
            for unit, rows in (entry.get("units") or {}).items():
                kept = [r for r in rows if str(r.get("filed", "")) <= cutoff]
                if kept:
                    new_units[unit] = kept
            if new_units:
                new_tags[tag] = {**{k: v for k, v in entry.items() if k != "units"},
                                 "units": new_units}
        if new_tags:
            out["facts"][taxonomy] = new_tags
    return out


def _run_rate(quarterly: pd.DataFrame, annual: pd.DataFrame) -> pd.Series:
    """Annualised run rate from the most recent *contiguous* quarters.

    A deliberate last resort for filers whose full-year figures never appear in
    us-gaap -- Exxon, for one, tags only quarterly and year-to-date revenue, so
    a true TTM cannot be assembled at all.  Scaling 1-3 quarters to a year is
    wrong for any seasonal business (a retailer's Q3 run rate badly understates
    the year), which is exactly why this is kept out of :attr:`Fundamentals.ttm`
    and labelled with the quarter count used.
    """
    if quarterly.empty:
        return pd.Series(dtype=float)
    flow_names = [c.name for c in ALL_CONCEPTS if c.kind == "flow"]
    present = [c for c in flow_names if c in quarterly]
    if not present:
        return pd.Series(dtype=float)

    out: dict[str, float] = {}
    tail = quarterly.iloc[-4:]
    for name in present:
        series = tail[name].dropna()
        if series.empty:
            continue
        # Only trust quarters that are genuinely consecutive.
        consecutive = [series.index[-1]]
        for prev in reversed(series.index[:-1]):
            if 60 <= (consecutive[-1] - prev).days <= 110:
                consecutive.append(prev)
            else:
                break
        n = len(consecutive)
        if n == 0:
            continue
        out[name] = float(series.loc[consecutive].sum()) * 4.0 / n
        out.setdefault("_quarters_used", float(n))
        out["_quarters_used"] = max(out["_quarters_used"], float(n))
    return pd.Series(out, dtype=float)


# --------------------------------------------------------------------------- #
# Corporate-reorganisation chaining
# --------------------------------------------------------------------------- #
BROWSE_URL = "https://www.sec.gov/cgi-bin/browse-edgar"

# Reorganisations that move a listing to a brand-new CIK, leaving decades of
# filings stranded on an entity that no longer carries the ticker.  The ticker
# file only ever points at the live registrant, so without a chain a 140-year-old
# company reads as two quarters old.  Auto-discovery below catches most of these;
# this map is the override for the ones it cannot, newest CIK first.
PREDECESSOR_CIKS: dict[str, tuple[str, ...]] = {
    # ExxonMobil's 2026 holdco reorganisation (new TX holdco over the NJ corp).
    "0002115436": ("0000034088",),
}

_NAME_STOPWORDS = frozenset({
    "inc", "corp", "corporation", "co", "company", "holdings", "holding", "group",
    "the", "ltd", "limited", "plc", "lp", "llc", "sa", "nv", "ag", "new", "class",
})


def _name_tokens(name: str) -> list[str]:
    words = [w.strip(".,&/-").lower() for w in str(name or "").split()]
    return [w for w in words if w and w not in _NAME_STOPWORDS and not w.isdigit()]


def search_companies(query: str, ttl: int | None = None) -> list[dict]:
    """Look up registrants by company name via EDGAR's browse interface.

    Returns ``[{cik, name, sic}]``.  The atom feed has two shapes -- a single
    ``<company-info>`` block for an exact hit, or one ``<entry>`` per match --
    so both are parsed.
    """
    import re

    cfg = settings()
    key = make_key("sec:search", query.lower())
    ttl = cfg.ttl_profile if ttl is None else ttl
    cached = cache().get_json(key, ttl)
    if cached is not None:
        return cached
    try:
        resp = get(
            BROWSE_URL,
            {"company": query, "action": "getcompany", "type": "10-K",
             "count": "40", "output": "atom"},
            allow_status=(404,),
        )
    except (OfflineError, Exception):  # noqa: BLE001 - search is best-effort
        return cache().get_json(key, None) or []
    if resp.status_code == 404:
        return []

    text = resp.text
    ciks = re.findall(r"<cik>(\d+)</cik>", text)
    names = re.findall(r"<conformed-name>([^<]+)</conformed-name>", text)
    sics = re.findall(r"<assigned-sic>(\d+)</assigned-sic>", text)
    out = []
    for idx, cik in enumerate(ciks):
        out.append({
            "cik": cik.zfill(10),
            "name": names[idx] if idx < len(names) else "",
            "sic": sics[idx] if idx < len(sics) else "",
        })
    cache().set_json(key, out)
    return out


def find_predecessor_ciks(cik: str, profile: dict, max_candidates: int = 4) -> list[str]:
    """Discover CIKs holding the earlier filing history of the same business.

    Only triggered when a registrant's own XBRL history is implausibly short.
    A candidate must share the SIC code, overlap on at least one significant
    name token, and actually carry more annual history than the primary -- three
    conditions that together make a false positive unlikely but not impossible,
    which is why :attr:`Fundamentals.chained_ciks` always records what was used.
    """
    cik = str(cik).zfill(10)
    override = PREDECESSOR_CIKS.get(cik)
    if override:
        return list(override)

    name = profile.get("name") or ""
    tokens = _name_tokens(name)
    if not tokens:
        return []
    sic = str(profile.get("sic") or "")

    key = make_key("sec:predecessor", cik)
    cached = cache().get_json(key, settings().ttl_profile)
    if cached is not None:
        return cached

    query = " ".join(tokens[:2])
    found: list[str] = []
    for hit in search_companies(query)[: max_candidates * 3]:
        other = hit["cik"]
        if other == cik:
            continue
        if sic and hit.get("sic") and hit["sic"] != sic:
            continue
        if not set(_name_tokens(hit.get("name", ""))) & set(tokens):
            continue
        found.append(other)
        if len(found) >= max_candidates:
            break

    cache().set_json(key, found)
    return found


def _merge_facts(primary: dict, older: dict) -> dict:
    """Union two companyfacts documents, letting the newest filing win.

    Rows are simply concatenated; :func:`_dedupe_latest_filed` resolves any
    overlapping period in favour of the later filing date, which is the correct
    precedence -- the successor's restatement supersedes the predecessor's
    original.  Only the predecessor's genuinely older periods survive.
    """
    merged = {
        "cik": primary.get("cik"),
        "entityName": primary.get("entityName"),
        "facts": {},
    }
    for source in (older, primary):  # primary second so its metadata wins
        for taxonomy, tags in (source.get("facts") or {}).items():
            tax_block = merged["facts"].setdefault(taxonomy, {})
            for tag, entry in tags.items():
                existing = tax_block.setdefault(
                    tag, {k: v for k, v in entry.items() if k != "units"} | {"units": {}}
                )
                for unit, rows in (entry.get("units") or {}).items():
                    existing["units"].setdefault(unit, []).extend(rows)
    return merged


MIN_ANNUAL_YEARS = 3  # below this, suspect a reorganisation rather than a young company
