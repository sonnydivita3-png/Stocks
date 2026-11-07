"""The tradeable universe: every US-listed symbol, with the junk filtered out.

Source is Nasdaq Trader's nightly symbol directory, which covers all US
exchanges (not just Nasdaq) and flags ETFs, test issues and companies in
non-compliance -- all of which must be dropped before a screen, or the top of
every value ranking fills with delisting candidates and closed-end funds.
"""
from __future__ import annotations

import io
import logging
import re

import pandas as pd

from ..config import settings
from .cache import cache, make_key
from .http import OfflineError, get_text

log = logging.getLogger(__name__)

NASDAQ_LISTED = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

# Suffixes and patterns that mark something other than common stock.
_NON_COMMON = re.compile(
    r"\b(warrant|right|unit|preferred|depositary|debenture|notes? due|"
    r"trust preferred|convertible|when issued|contingent value)\b",
    re.IGNORECASE,
)
# Fifth-letter share-class codes that are not ordinary common equity.
_BAD_FIFTH_LETTER = set("RWPQUZ")


def _parse_pipe_file(text: str) -> pd.DataFrame:
    # The directory files end with a "File Creation Time" trailer line.
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("File Creation")]
    return pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str).fillna("")


def fetch_listed_symbols(ttl: int | None = None) -> pd.DataFrame:
    """All US-listed symbols: ``symbol, name, exchange, is_etf, test_issue, status``."""
    cfg = settings()
    key = make_key("universe:listed")
    ttl = cfg.ttl_universe if ttl is None else ttl
    cached = cache().get_df(key, ttl)
    if cached is not None and not cached.empty:
        return cached

    frames = []
    try:
        nas = _parse_pipe_file(get_text(NASDAQ_LISTED) or "")
        if not nas.empty:
            frames.append(pd.DataFrame({
                "symbol": nas["Symbol"].str.strip(),
                "name": nas["Security Name"].str.strip(),
                "exchange": "NASDAQ",
                "is_etf": nas.get("ETF", "").eq("Y"),
                "test_issue": nas.get("Test Issue", "").eq("Y"),
                "status": nas.get("Financial Status", ""),
            }))
        oth = _parse_pipe_file(get_text(OTHER_LISTED) or "")
        if not oth.empty:
            frames.append(pd.DataFrame({
                "symbol": oth["ACT Symbol"].str.strip(),
                "name": oth["Security Name"].str.strip(),
                "exchange": oth.get("Exchange", "").map(
                    {"A": "NYSE American", "N": "NYSE", "P": "NYSE Arca",
                     "Z": "BATS", "V": "IEX"}
                ).fillna("OTHER"),
                "is_etf": oth.get("ETF", "").eq("Y"),
                "test_issue": oth.get("Test Issue", "").eq("Y"),
                "status": "",
            }))
    except OfflineError:
        stale = cache().get_df(key, None)
        if stale is not None:
            return stale
        return pd.DataFrame(columns=["symbol", "name", "exchange", "is_etf",
                                     "test_issue", "status"])

    if not frames:
        return pd.DataFrame(columns=["symbol", "name", "exchange", "is_etf",
                                     "test_issue", "status"])
    out = pd.concat(frames, ignore_index=True)
    out = out[out["symbol"].str.len() > 0].drop_duplicates("symbol").reset_index(drop=True)
    cache().set_df(key, out)
    return out


def common_stocks(exclude_otc: bool = True) -> pd.DataFrame:
    """Listed common equity only: no ETFs, warrants, units, preferreds or test issues.

    Also drops Nasdaq's "Financial Status" D (deficient) and E (delinquent)
    issues. Those are exactly the names that top a naive value screen, because
    the market has priced in a problem the trailing financials do not show yet.
    """
    frame = fetch_listed_symbols()
    if frame.empty:
        return frame

    mask = ~frame["test_issue"].astype(bool) & ~frame["is_etf"].astype(bool)
    mask &= ~frame["name"].str.contains(_NON_COMMON, na=False)
    # Nasdaq encodes share class in a 5th letter; a dollar sign marks a non-equity.
    mask &= ~frame["symbol"].str.contains(r"[$]", na=False)
    mask &= ~frame["symbol"].str.len().gt(4).__and__(
        frame["symbol"].str[-1].isin(_BAD_FIFTH_LETTER)
    )
    mask &= ~frame["status"].isin(["D", "E"])
    if exclude_otc:
        mask &= frame["exchange"].ne("OTHER")

    out = frame[mask].copy()
    out["symbol"] = out["symbol"].str.replace(".", "-", regex=False)  # EDGAR/Yahoo style
    return out.reset_index(drop=True)


# Index proxies used as benchmarks and for the regime model. ETFs are used
# rather than index symbols (^GSPC) because they carry dividends in adj_close,
# which makes total-return comparisons correct.
BENCHMARKS = {
    "us_total": "VTI",
    "sp500": "SPY",
    "nasdaq100": "QQQ",
    "small_cap": "IWM",
    "mid_cap": "IJH",
    "equal_weight_sp500": "RSP",
    "developed_ex_us": "EFA",
    "emerging": "EEM",
    "aggregate_bonds": "AGG",
    "long_treasuries": "TLT",
    "gold": "GLD",
    "high_yield": "HYG",
}

SECTOR_ETFS = {
    "Information Technology": "XLK",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

# Hand-maintained large-cap list for quick demos and tests without downloading
# the full directory. Not an index and not maintained for accuracy of membership.
MEGA_CAP_SAMPLE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "AVGO", "TSLA",
    "LLY", "JPM", "V", "XOM", "UNH", "MA", "COST", "HD", "PG", "JNJ", "ABBV",
    "WMT", "NFLX", "CRM", "BAC", "ORCL", "MRK", "KO", "CVX", "AMD", "PEP",
    "ADBE", "TMO", "LIN", "ACN", "MCD", "CSCO", "ABT", "WFC", "GE", "DIS",
    "INTU", "QCOM", "IBM", "TXN", "AMGN", "CAT", "VZ", "NOW", "ISRG", "SPGI",
]
