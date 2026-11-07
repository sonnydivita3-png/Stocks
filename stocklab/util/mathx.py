"""Numerical primitives used across valuation, screening and risk code.

Every function here is deliberately total: it returns ``nan`` rather than
raising on degenerate input (empty series, zero denominators, negative bases
for fractional powers).  Screeners run these over thousands of names where a
single bad row must not abort the batch.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

NAN = float("nan")


# --------------------------------------------------------------------------- #
# scalar helpers
# --------------------------------------------------------------------------- #
def safe_div(num: float | None, den: float | None, default: float = NAN) -> float:
    """``num / den`` guarding against ``None``, zero and non-finite inputs."""
    try:
        if num is None or den is None:
            return default
        num_f, den_f = float(num), float(den)
        if not math.isfinite(num_f) or not math.isfinite(den_f) or den_f == 0.0:
            return default
        return num_f / den_f
    except (TypeError, ValueError):
        return default


def is_num(x) -> bool:
    """True when ``x`` is a finite real number."""
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def cagr(begin: float, end: float, years: float) -> float:
    """Compound annual growth rate.

    Undefined when the starting value is non-positive -- a company that went
    from a loss to a profit has no meaningful growth *rate*, and reporting one
    (e.g. by taking absolute values) is a classic way to rank garbage to the
    top of a screen.  We return ``nan`` and let callers handle it explicitly.
    """
    if not (is_num(begin) and is_num(end)) or years <= 0:
        return NAN
    if begin <= 0 or end <= 0:
        return NAN
    return (end / begin) ** (1.0 / years) - 1.0


def xnpv(rate: float, cashflows: Sequence[tuple[pd.Timestamp, float]]) -> float:
    """Net present value of irregularly dated cashflows (365-day basis)."""
    if not cashflows:
        return NAN
    t0 = min(d for d, _ in cashflows)
    total = 0.0
    for when, amount in cashflows:
        years = (when - t0).days / 365.0
        total += amount / (1.0 + rate) ** years
    return total


def xirr(
    cashflows: Sequence[tuple[pd.Timestamp, float]],
    guess: float = 0.1,
    lo: float = -0.9999,
    hi: float = 10.0,
) -> float:
    """Internal rate of return for irregularly dated cashflows.

    Bisection on ``xnpv``.  Robust where Newton's method diverges, which it
    routinely does on real brokerage histories with large late deposits.
    Requires at least one sign change in the flows.
    """
    flows = [(pd.Timestamp(d), float(v)) for d, v in cashflows if is_num(v)]
    if len(flows) < 2:
        return NAN
    signs = {v > 0 for _, v in flows if v != 0}
    if len(signs) < 2:
        return NAN  # all inflows or all outflows -> no root

    f_lo, f_hi = xnpv(lo, flows), xnpv(hi, flows)
    if not (is_num(f_lo) and is_num(f_hi)) or f_lo * f_hi > 0:
        # widen once before giving up
        hi = 100.0
        f_hi = xnpv(hi, flows)
        if not is_num(f_hi) or f_lo * f_hi > 0:
            return NAN

    for _ in range(200):
        mid = (lo + hi) / 2.0
        f_mid = xnpv(mid, flows)
        if not is_num(f_mid):
            return NAN
        if abs(f_mid) < 1e-9:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2.0


def irr_from_values(values: Sequence[float]) -> float:
    """IRR for evenly spaced (periodic) cashflows."""
    vals = [float(v) for v in values if is_num(v)]
    if len(vals) < 2 or len({v > 0 for v in vals if v != 0}) < 2:
        return NAN
    base = pd.Timestamp("2000-01-01")
    flows = [(base + pd.DateOffset(years=i), v) for i, v in enumerate(vals)]
    return xirr(flows)


# --------------------------------------------------------------------------- #
# series helpers
# --------------------------------------------------------------------------- #
def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip a series to its own quantiles.

    Cross-sectional factor work is dominated by outliers (a P/E of 9,000 from a
    one-cent EPS).  Winsorizing before z-scoring is not optional.
    """
    s = pd.to_numeric(s, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return s
    return s.clip(lower=valid.quantile(lower), upper=valid.quantile(upper))


def zscore(s: pd.Series, robust: bool = False, clip: float | None = 3.0) -> pd.Series:
    """Standardise a cross-section.

    ``robust=True`` uses median / MAD, which is the right default when the
    distribution is as fat-tailed as equity valuation multiples.
    """
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().empty:
        return s * NAN
    if robust:
        centre = s.median()
        scale = (s - centre).abs().median() * 1.4826  # MAD -> sigma
    else:
        centre = s.mean()
        scale = s.std(ddof=0)
    if not is_num(scale) or scale == 0:
        return pd.Series(0.0, index=s.index).where(s.notna())
    z = (s - centre) / scale
    return z.clip(-clip, clip) if clip else z


def pct_rank(s: pd.Series, ascending: bool = True) -> pd.Series:
    """Cross-sectional percentile rank in [0, 1]; ties averaged."""
    s = pd.to_numeric(s, errors="coerce")
    return s.rank(pct=True, ascending=ascending, na_option="keep")


def ewma(s: pd.Series, halflife: float) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").ewm(halflife=halflife, adjust=False).mean()


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Running drawdown (<= 0) of an equity curve."""
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if eq.empty:
        return eq
    return eq / eq.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    dd = drawdown_series(equity)
    return float(dd.min()) if not dd.empty else NAN


def ulcer_index(equity: pd.Series) -> float:
    """RMS drawdown -- penalises depth *and* duration, unlike max drawdown."""
    dd = drawdown_series(equity)
    if dd.empty:
        return NAN
    return float(np.sqrt((dd**2).mean()))


def linreg(y: Sequence[float], x: Sequence[float] | None = None) -> tuple[float, float, float]:
    """OLS fit; returns ``(slope, intercept, r_squared)``."""
    y_arr = np.asarray(pd.to_numeric(pd.Series(list(y)), errors="coerce"), dtype=float)
    mask = np.isfinite(y_arr)
    if x is None:
        x_arr = np.arange(len(y_arr), dtype=float)
    else:
        x_arr = np.asarray(pd.to_numeric(pd.Series(list(x)), errors="coerce"), dtype=float)
        mask &= np.isfinite(x_arr)
    if mask.sum() < 3:
        return NAN, NAN, NAN
    y_arr, x_arr = y_arr[mask], x_arr[mask]
    x_var = x_arr.var()
    if x_var == 0:
        return NAN, NAN, NAN
    slope = float(np.cov(x_arr, y_arr, bias=True)[0, 1] / x_var)
    intercept = float(y_arr.mean() - slope * x_arr.mean())
    resid = y_arr - (slope * x_arr + intercept)
    ss_tot = float(((y_arr - y_arr.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else NAN
    return slope, intercept, r2


def log_trend_growth(s: pd.Series) -> tuple[float, float]:
    """Annualised growth from an OLS fit on log values.

    Preferred over endpoint CAGR for noisy fundamentals: endpoint CAGR is
    hostage to whichever two years you happen to pick, a log-trend fit uses
    every observation.  Returns ``(growth_rate, r_squared)``; r² doubles as a
    *durability* measure -- how much of the history the trend actually explains.
    """
    vals = pd.to_numeric(s, errors="coerce").dropna()
    vals = vals[vals > 0]
    if len(vals) < 3:
        return NAN, NAN
    slope, _, r2 = linreg(np.log(vals.to_numpy(dtype=float)))
    if not is_num(slope):
        return NAN, NAN
    return float(np.exp(slope) - 1.0), r2


def annualize_return(total_return: float, years: float) -> float:
    if not is_num(total_return) or years <= 0 or total_return <= -1:
        return NAN
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def blend(values: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted mean over the keys that actually have finite values.

    Re-normalises across available inputs so a missing model does not silently
    drag a blended estimate toward zero.
    """
    num = den = 0.0
    for key, weight in weights.items():
        val = values.get(key)
        if is_num(val) and is_num(weight) and weight > 0:
            num += float(val) * float(weight)
            den += float(weight)
    return num / den if den > 0 else NAN


def clamp(x: float, lo: float, hi: float) -> float:
    if not is_num(x):
        return NAN
    return max(lo, min(hi, float(x)))


def interp_score(x: float, breakpoints: Sequence[tuple[float, float]]) -> float:
    """Piecewise-linear map from a raw metric to a 0-100 score.

    ``breakpoints`` is ``[(metric, score), ...]`` ascending in metric.  Used to
    turn things like ROIC or net-debt/EBITDA into comparable sub-scores without
    needing a cross-section to rank against.
    """
    if not is_num(x) or not breakpoints:
        return NAN
    pts = sorted(breakpoints, key=lambda p: p[0])
    if x <= pts[0][0]:
        return float(pts[0][1])
    if x >= pts[-1][0]:
        return float(pts[-1][1])
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return float(y1)
            frac = (x - x0) / (x1 - x0)
            return float(y0 + frac * (y1 - y0))
    return NAN


def rolling_apply(s: pd.Series, window: int, fn: Callable[[np.ndarray], float]) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").rolling(window).apply(
        lambda arr: fn(np.asarray(arr, dtype=float)), raw=True
    )


def first_finite(*values) -> float:
    """First finite value among the arguments, else ``nan``.

    The workhorse for XBRL tag fallbacks, where one of six possible tags holds
    the number you want.
    """
    for v in values:
        if is_num(v):
            return float(v)
    return NAN


def sum_finite(*values, require_any: bool = True) -> float:
    """Sum of the finite arguments; ``nan`` if none are finite and required."""
    vals = [float(v) for v in values if is_num(v)]
    if not vals:
        return NAN if require_any else 0.0
    return float(sum(vals))


def mean_finite(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if is_num(v)]
    return float(np.mean(vals)) if vals else NAN


def median_finite(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if is_num(v)]
    return float(np.median(vals)) if vals else NAN


def zero_if_nan(x) -> float:
    """Coerce a missing value to zero.

    Exists because ``x or 0.0`` silently fails on ``nan``: NaN is truthy, so the
    fallback never fires and the NaN propagates through the whole expression.
    Use this only where "not reported" genuinely means "none" -- a company with
    no minority interest -- never to paper over a value that should exist.
    """
    return float(x) if is_num(x) else 0.0
