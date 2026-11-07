"""Build a factor matrix over a universe of symbols.

Screening is the one place where data cost dominates design. A full
``companyfacts`` document is 2-6 MB, so a thousand-name screen means gigabytes
and, at SEC's rate limit, close to an hour. Two mechanisms make it tractable:

1. **Caching.** Everything lands in the SQLite cache, so the second run of a
   screen is near-instant and only new filings are fetched.
2. **Two-stage screening.** :func:`prefilter_by_frames` uses SEC's frames API --
   one request returns a single concept for *every* filer in the country -- to
   knock out the obvious misses before any per-company work happens.

The matrix itself is a plain DataFrame indexed by symbol, with one column per
metric from :mod:`stocklab.analysis.metrics` plus technical and risk columns.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..analysis.metrics import compute_metrics
from ..analysis.technicals import technical_snapshot
from ..data.providers.sec import fetch_frame, load_fundamentals, ticker_to_cik
from ..data.providers.yahoo import fetch_history, fetch_meta
from ..util.mathx import is_num, pct_rank, winsorize, zscore

log = logging.getLogger(__name__)

# Metrics worth ranking cross-sectionally, and the direction that is "good".
FACTOR_DIRECTIONS: dict[str, bool] = {
    # value: lower is better
    "pe": False, "pe_5y_avg_earnings": False, "pb": False, "ps": False,
    "p_fcf": False, "ev_ebitda": False, "ev_ebit": False, "ev_sales": False,
    "net_debt_to_ebitda": False, "peg": False,
    # quality / profitability: higher is better
    "roic": True, "roe": True, "roa": True, "croic": True, "gross_margin": True,
    "operating_margin": True, "fcf_margin": True, "piotroski_f": True,
    "interest_coverage": True, "current_ratio": True, "cash_conversion": True,
    "fcf_yield": True, "earnings_yield": True, "ebit_ev_yield": True,
    "shareholder_yield": True, "dividend_yield": True, "altman_z": True,
    "roic_5y_avg": True,
    # growth: higher is better
    "revenue_cagr_3y": True, "revenue_cagr_5y": True, "eps_cagr_3y": True,
    "eps_cagr_5y": True, "fcf_cagr_5y": True, "revenue_trend_growth": True,
    "revenue_trend_r2": True,
    # momentum / trend
    "momentum_12_1": True, "return_12m": True, "return_6m": True,
    "trend_template_score": True, "rs_slope_ann": True, "trend_r2": True,
    # risk: lower is better
    "vol63": False, "accruals_ratio": False, "beneish_m": False,
    "share_count_cagr_5y": False, "sbc_intensity": False,
}


@dataclass
class MatrixResult:
    matrix: pd.DataFrame
    failures: dict[str, str]

    def __len__(self) -> int:
        return len(self.matrix)


def load_one(symbol: str, *, with_technicals: bool = True,
             benchmark_prices: pd.Series | None = None) -> dict | None:
    """Every factor for one symbol, or ``None`` if it cannot be valued."""
    symbol = symbol.upper().strip()
    fundamentals = load_fundamentals(symbol)
    if not fundamentals.has_data:
        return None

    meta = fetch_meta(symbol)
    price = meta.get("price")
    if not is_num(price):
        return None

    row: dict = {"symbol": symbol, "name": fundamentals.name}
    row.update(compute_metrics(fundamentals, price))
    row["exchange"] = meta.get("exchange")

    if with_technicals:
        history = fetch_history(symbol, period="3y")
        if not history.empty and len(history) > 60:
            tech = technical_snapshot(history, benchmark_prices)
            # Prefix-free merge: technical keys do not collide with metric keys
            # except `close`, which is dropped in favour of the live price.
            tech.pop("close", None)
            row.update(tech)
    return row


def build_matrix(
    symbols: list[str],
    *,
    with_technicals: bool = True,
    benchmark: str = "SPY",
    max_workers: int = 4,
    progress: bool = False,
) -> MatrixResult:
    """Factor matrix for a list of symbols, indexed by symbol.

    Threaded, but modestly: the HTTP layer rate-limits per host anyway, so more
    workers just queue up behind the limiter. Four is enough to overlap SEC
    and Yahoo requests without tripping either.
    """
    benchmark_prices = None
    if with_technicals:
        bench = fetch_history(benchmark, period="3y")
        if not bench.empty:
            benchmark_prices = bench["adj_close"]

    rows: list[dict] = []
    failures: dict[str, str] = {}
    symbols = [s.upper().strip() for s in dict.fromkeys(symbols) if s and s.strip()]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(load_one, sym, with_technicals=with_technicals,
                        benchmark_prices=benchmark_prices): sym
            for sym in symbols
        }
        for done, future in enumerate(as_completed(futures), start=1):
            sym = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001 - never let one name kill a screen
                failures[sym] = f"{type(exc).__name__}: {exc}"
                continue
            if row is None:
                failures[sym] = "no usable fundamentals or price"
            else:
                rows.append(row)
            if progress and done % 25 == 0:
                log.info("screened %d/%d", done, len(symbols))

    if not rows:
        return MatrixResult(pd.DataFrame(), failures)

    matrix = pd.DataFrame(rows).set_index("symbol")
    # Strings (sector, industry) must survive; everything else becomes numeric.
    for col in matrix.columns:
        if col in ("name", "sector", "industry", "exchange", "altman_zone"):
            continue
        matrix[col] = pd.to_numeric(matrix[col], errors="coerce")
    return MatrixResult(matrix.sort_index(), failures)


def add_ranks(matrix: pd.DataFrame, factors: list[str] | None = None,
              by_sector: bool = False) -> pd.DataFrame:
    """Append ``<factor>_z`` and ``<factor>_pct`` columns.

    Winsorizing before standardising is not optional: one company with a
    one-cent EPS produces a P/E of 9,000, and an un-winsorized z-score then
    compresses every other name into a single bucket.

    ``by_sector=True`` ranks within sector, which is how valuation multiples
    should almost always be compared -- a utility on 18x and a software company
    on 18x are not equally priced.
    """
    out = matrix.copy()
    factors = factors or [f for f in FACTOR_DIRECTIONS if f in out.columns]

    def rank_block(frame: pd.DataFrame) -> pd.DataFrame:
        block = {}
        for factor in factors:
            if factor not in frame.columns:
                continue
            series = pd.to_numeric(frame[factor], errors="coerce")
            if series.notna().sum() < 3:
                continue
            ascending = FACTOR_DIRECTIONS.get(factor, True)
            clipped = winsorize(series)
            block[f"{factor}_z"] = zscore(clipped, robust=True) * (1.0 if ascending else -1.0)
            block[f"{factor}_pct"] = pct_rank(series, ascending=not ascending)
        return pd.DataFrame(block, index=frame.index)

    if by_sector and "sector" in out.columns:
        pieces = [rank_block(group) for _, group in out.groupby("sector", dropna=False)
                  if len(group) >= 3]
        ranked = pd.concat(pieces) if pieces else pd.DataFrame(index=out.index)
    else:
        ranked = rank_block(out)

    return out.join(ranked, how="left")


def composite_score(matrix: pd.DataFrame, weights: dict[str, float],
                    require: int | None = None) -> pd.Series:
    """Weighted blend of ``<factor>_z`` columns into one score.

    Names missing a factor are scored on what they have, with the weights
    re-normalised -- otherwise a company that simply does not report R&D would
    be penalised as though its R&D were zero. ``require`` sets a minimum number
    of present factors before a name is scored at all.
    """
    numerator = pd.Series(0.0, index=matrix.index)
    denominator = pd.Series(0.0, index=matrix.index)
    present = pd.Series(0, index=matrix.index)

    for factor, weight in weights.items():
        col = f"{factor}_z"
        if col not in matrix.columns or weight <= 0:
            continue
        series = pd.to_numeric(matrix[col], errors="coerce")
        mask = series.notna()
        numerator[mask] += series[mask] * weight
        denominator[mask] += weight
        present[mask] += 1

    score = (numerator / denominator.replace(0.0, np.nan))
    if require:
        score = score.where(present >= require)
    return score


def prefilter_by_frames(concept_tag: str, period: str, minimum: float | None = None,
                        maximum: float | None = None, unit: str = "USD") -> set[str]:
    """Cheap first pass: CIKs passing a threshold on one concept, via SEC frames.

    One HTTP request returns the concept for every filer in the country, which
    is how a universe of thousands gets narrowed before any per-company
    download happens. Returns a set of zero-padded CIKs.
    """
    frame = fetch_frame(concept_tag, period, unit=unit)
    if frame.empty or "val" not in frame:
        return set()
    series = pd.to_numeric(frame["val"], errors="coerce").dropna()
    if minimum is not None:
        series = series[series >= minimum]
    if maximum is not None:
        series = series[series <= maximum]
    return set(series.index.astype(str))


def symbols_for_ciks(ciks: set[str], candidate_symbols: list[str]) -> list[str]:
    """Intersect a CIK set with a ticker list, preserving the ticker order."""
    wanted = {str(c).zfill(10) for c in ciks}
    out = []
    for sym in candidate_symbols:
        cik = ticker_to_cik(sym)
        if cik and cik in wanted:
            out.append(sym)
    return out
