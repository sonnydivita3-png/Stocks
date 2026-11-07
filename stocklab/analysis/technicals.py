"""Technical indicators, implemented to their original definitions.

Two notes on correctness, because both are commonly got wrong in ways that
change the numbers materially:

* **Wilder smoothing is not an EMA with span=n.** RSI, ATR and ADX all use
  Wilder's recursive average, which is an EMA with ``alpha = 1/n`` (equivalent
  to ``span = 2n-1``). Using ``span=n`` makes every one of them twice as
  twitchy as the indicator people think they are reading.
* **Returns come from ``adj_close``, levels come from ``close``.** A stop at
  "the 50-day moving average" must be a tradeable price, not a
  dividend-adjusted one; a 12-month return must include dividends.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..util.mathx import NAN, is_num, linreg, safe_div


# --------------------------------------------------------------------------- #
# moving averages
# --------------------------------------------------------------------------- #
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def wilder(series: pd.Series, period: int) -> pd.Series:
    """Wilder's recursive moving average: an EMA with ``alpha = 1/period``."""
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


# --------------------------------------------------------------------------- #
# oscillators
# --------------------------------------------------------------------------- #
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index (1978), 0-100."""
    delta = close.diff()
    gain = wilder(delta.clip(lower=0.0), period)
    loss = wilder((-delta).clip(lower=0.0), period)
    rs = gain / loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # All gains and no losses is RSI 100 by definition, not a divide-by-zero.
    return out.where(loss.ne(0.0) | gain.isna(), 100.0)


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 14, smooth: int = 3) -> tuple[pd.Series, pd.Series]:
    lowest = low.rolling(period, min_periods=period).min()
    highest = high.rolling(period, min_periods=period).max()
    k = 100.0 * (close - lowest) / (highest - lowest).replace(0.0, np.nan)
    return k, k.rolling(smooth, min_periods=smooth).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
         ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram."""
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


# --------------------------------------------------------------------------- #
# volatility / range
# --------------------------------------------------------------------------- #
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range -- the unit position sizing and stops should use.

    Expressing a stop in ATRs rather than percent makes risk comparable across
    a 60%-vol biotech and a 12%-vol utility.
    """
    return wilder(true_range(high, low, close), period)


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0
              ) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(close, window)
    sd = close.rolling(window, min_periods=window).std(ddof=0)
    return mid - num_std * sd, mid, mid + num_std * sd


def bollinger_bandwidth(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """Band width relative to the mean: the squeeze/expansion signal."""
    lower, mid, upper = bollinger(close, window, num_std)
    return (upper - lower) / mid.replace(0.0, np.nan)


def donchian(high: pd.Series, low: pd.Series, window: int = 20
             ) -> tuple[pd.Series, pd.Series]:
    return (low.rolling(window, min_periods=window).min(),
            high.rolling(window, min_periods=window).max())


def realized_vol(returns: pd.Series, window: int = 21, trading_days: int = 252) -> pd.Series:
    return returns.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(trading_days)


# --------------------------------------------------------------------------- #
# trend strength
# --------------------------------------------------------------------------- #
def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
        ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Wilder's ADX with +DI / -DI. Above ~25 is a trending market.

    ADX measures trend *strength*, never direction: a collapsing stock prints a
    high ADX. Direction comes from the DI spread.
    """
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr_ = wilder(true_range(high, low, close), period)
    plus_di = 100.0 * wilder(plus_dm, period) / atr_.replace(0.0, np.nan)
    minus_di = 100.0 * wilder(minus_dm, period) / atr_.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return wilder(dx, period), plus_di, minus_di


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume: cumulative volume signed by the day's direction."""
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series,
                     volume: pd.Series, period: int = 14) -> pd.Series:
    """Volume-weighted RSI: divergences here tend to precede price."""
    typical = (high + low + close) / 3.0
    flow = typical * volume
    up = flow.where(typical.diff() > 0, 0.0).rolling(period, min_periods=period).sum()
    down = flow.where(typical.diff() < 0, 0.0).rolling(period, min_periods=period).sum()
    ratio = up / down.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + ratio))


def anchored_vwap(close: pd.Series, volume: pd.Series, anchor: pd.Timestamp | None = None,
                  high: pd.Series | None = None, low: pd.Series | None = None) -> pd.Series:
    """Volume-weighted average price from an anchor date forward.

    The level at which the average holder since the anchor is flat, which is why
    it acts as support/resistance after an earnings gap or a 52-week low.
    """
    typical = close if high is None or low is None else (high + low + close) / 3.0
    if anchor is not None:
        mask = typical.index >= pd.Timestamp(anchor)
        typical, volume = typical[mask], volume[mask]
    cum_vol = volume.cumsum().replace(0.0, np.nan)
    return (typical * volume).cumsum() / cum_vol


# --------------------------------------------------------------------------- #
# support / resistance
# --------------------------------------------------------------------------- #
def swing_points(high: pd.Series, low: pd.Series, lookback: int = 5
                 ) -> tuple[pd.Series, pd.Series]:
    """Fractal swing highs/lows: a bar higher (lower) than ``lookback`` either side."""
    sh = high.rolling(2 * lookback + 1, center=True).max().eq(high) & high.notna()
    sl = low.rolling(2 * lookback + 1, center=True).min().eq(low) & low.notna()
    return high.where(sh), low.where(sl)


def support_resistance(frame: pd.DataFrame, lookback: int = 5, bins: int = 24,
                       within: float = 0.35) -> dict[str, list[float]]:
    """Cluster recent swing points into candidate support/resistance levels.

    Price history is bucketed and buckets with repeated swing points are kept:
    a level that has been tested three times matters more than a single spike.
    ``within`` limits attention to the most recent fraction of the window.
    """
    if frame.empty or len(frame) < 3 * lookback:
        return {"support": [], "resistance": []}
    tail = frame.iloc[-int(len(frame) * within):] if 0 < within < 1 else frame
    sh, sl = swing_points(tail["high"], tail["low"], lookback)
    last = float(tail["close"].iloc[-1])

    def cluster(points: pd.Series) -> list[float]:
        vals = points.dropna().to_numpy(dtype=float)
        if vals.size == 0:
            return []
        lo, hi = vals.min(), vals.max()
        if hi <= lo:
            return [float(lo)]
        edges = np.linspace(lo, hi, bins + 1)
        idx = np.clip(np.digitize(vals, edges) - 1, 0, bins - 1)
        out = []
        for b in range(bins):
            members = vals[idx == b]
            if members.size:
                out.append((float(members.mean()), members.size))
        out.sort(key=lambda t: (-t[1], abs(t[0] - last)))
        return [lvl for lvl, _ in out[:5]]

    res = [lvl for lvl in cluster(sh) if lvl > last]
    sup = [lvl for lvl in cluster(sl) if lvl < last]
    return {"support": sorted(sup, reverse=True)[:4], "resistance": sorted(res)[:4]}


# --------------------------------------------------------------------------- #
# the summary everything else consumes
# --------------------------------------------------------------------------- #
def indicator_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the standard indicator set to an OHLCV frame."""
    out = frame.copy()
    close, high, low = out["close"], out["high"], out["low"]
    volume = out["volume"] if "volume" in out else pd.Series(np.nan, index=out.index)
    px = out["adj_close"] if "adj_close" in out else close

    out["ret"] = px.pct_change()
    for w in (10, 20, 50, 100, 150, 200):
        out[f"sma{w}"] = sma(close, w)
    out["ema21"] = ema(close, 21)
    out["rsi14"] = rsi(close, 14)
    out["atr14"] = atr(high, low, close, 14)
    out["atr_pct"] = out["atr14"] / close
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(close)
    out["bb_lower"], out["bb_mid"], out["bb_upper"] = bollinger(close)
    out["bb_width"] = bollinger_bandwidth(close)
    out["adx14"], out["di_plus"], out["di_minus"] = adx(high, low, close, 14)
    out["vol21"] = realized_vol(out["ret"], 21)
    out["vol63"] = realized_vol(out["ret"], 63)
    out["dc_low20"], out["dc_high20"] = donchian(high, low, 20)
    out["dc_low55"], out["dc_high55"] = donchian(high, low, 55)
    out["obv"] = obv(close, volume)
    out["mfi14"] = money_flow_index(high, low, close, volume, 14)
    out["vol_sma50"] = sma(volume, 50)
    out["rel_volume"] = volume / out["vol_sma50"].replace(0.0, np.nan)
    out["dollar_volume"] = close * volume
    return out


def technical_snapshot(frame: pd.DataFrame, benchmark: pd.Series | None = None) -> dict:
    """Current technical state as a flat dict for screens and entry scoring."""
    if frame.empty or len(frame) < 30:
        return {}
    ind = indicator_frame(frame)
    last = ind.iloc[-1]
    close = float(last["close"])
    px = ind["adj_close"] if "adj_close" in ind else ind["close"]

    snap: dict[str, float] = {
        "close": close,
        "rsi14": _f(last.get("rsi14")),
        "atr14": _f(last.get("atr14")),
        "atr_pct": _f(last.get("atr_pct")),
        "adx14": _f(last.get("adx14")),
        "di_spread": _f(last.get("di_plus")) - _f(last.get("di_minus")),
        "macd_hist": _f(last.get("macd_hist")),
        "bb_width": _f(last.get("bb_width")),
        "vol21": _f(last.get("vol21")),
        "vol63": _f(last.get("vol63")),
        "rel_volume": _f(last.get("rel_volume")),
        "dollar_volume_50d": _f(ind["dollar_volume"].tail(50).mean()),
        "mfi14": _f(last.get("mfi14")),
    }

    for w in (20, 50, 100, 150, 200):
        ma = last.get(f"sma{w}")
        snap[f"sma{w}"] = _f(ma)
        snap[f"dist_sma{w}"] = safe_div(close - _f(ma), _f(ma))

    # 52-week position: where in its own range the stock sits.
    window = px.tail(252)
    hi_52 = float(ind["high"].tail(252).max())
    lo_52 = float(ind["low"].tail(252).min())
    snap["high_52w"], snap["low_52w"] = hi_52, lo_52
    snap["pct_from_52w_high"] = safe_div(close - hi_52, hi_52)
    snap["pct_above_52w_low"] = safe_div(close - lo_52, lo_52)
    snap["range_position_52w"] = safe_div(close - lo_52, hi_52 - lo_52)

    # Trailing total returns (dividends included -- that is why px is adj_close).
    for label, days in (("1m", 21), ("3m", 63), ("6m", 126), ("12m", 252)):
        if len(px) > days:
            snap[f"return_{label}"] = safe_div(float(px.iloc[-1]), float(px.iloc[-1 - days])) - 1.0

    # 12-1 momentum: the academic standard, skipping the most recent month
    # because short-term reversal contaminates it.
    if len(px) > 252:
        snap["momentum_12_1"] = safe_div(float(px.iloc[-22]), float(px.iloc[-253])) - 1.0

    # Trend quality: r-squared of a log-price fit. A steady 30% a year scores
    # near 1.0; the same return delivered in two gaps scores far lower.
    tail = px.tail(126).dropna()
    if len(tail) > 30 and (tail > 0).all():
        slope, _, r2 = linreg(np.log(tail.to_numpy(dtype=float)))
        snap["trend_slope_ann"] = float(np.expm1(slope * 252)) if is_num(slope) else NAN
        snap["trend_r2"] = r2

    snap["above_sma50"] = float(close > _f(last.get("sma50")))
    snap["above_sma200"] = float(close > _f(last.get("sma200")))
    sma200 = ind["sma200"].dropna()
    snap["sma200_rising"] = (
        float(sma200.iloc[-1] > sma200.iloc[-22]) if len(sma200) > 22 else NAN
    )
    snap["golden_cross"] = float(_f(last.get("sma50")) > _f(last.get("sma200")))

    # Minervini's stage-2 trend template: a checklist for "is this uptrend
    # mature enough to buy pullbacks in".
    snap["trend_template_score"] = _trend_template(ind, snap)

    if benchmark is not None and len(benchmark.dropna()) > 60:
        snap.update(relative_strength(px, benchmark))

    levels = support_resistance(frame)
    snap["nearest_support"] = levels["support"][0] if levels["support"] else NAN
    snap["nearest_resistance"] = levels["resistance"][0] if levels["resistance"] else NAN
    return snap


def _f(x) -> float:
    return float(x) if is_num(x) else NAN


def _trend_template(ind: pd.DataFrame, snap: dict) -> float:
    """Minervini trend-template score, 0-8.

    Eight conditions from *Trade Like a Stock Market Wizard*. It is a filter for
    institutional-grade uptrends, deliberately strict: most stocks fail it most
    of the time, which is the point.
    """
    close = snap["close"]
    checks = [
        close > snap.get("sma150", NAN),
        close > snap.get("sma200", NAN),
        snap.get("sma150", NAN) > snap.get("sma200", NAN),
        snap.get("sma200_rising", 0.0) == 1.0,
        snap.get("sma50", NAN) > snap.get("sma150", NAN),
        close > snap.get("sma50", NAN),
        # Well off the lows: 30% minimum in the original formulation.
        is_num(snap.get("pct_above_52w_low")) and snap["pct_above_52w_low"] >= 0.30,
        # Within a quarter of the highs: leaders do not lag by more than that.
        is_num(snap.get("pct_from_52w_high")) and snap["pct_from_52w_high"] >= -0.25,
    ]
    scored = [bool(c) for c in checks if c is not None and not (isinstance(c, float) and np.isnan(c))]
    return float(sum(scored)) if scored else NAN


def relative_strength(px: pd.Series, benchmark: pd.Series) -> dict[str, float]:
    """Performance against a benchmark, plus the slope of the ratio line.

    A rising ratio line while both are falling is what "relative strength"
    actually means, and it is the single most useful screen in a drawdown.
    """
    joined = pd.concat([px.rename("px"), benchmark.rename("bm")], axis=1).dropna()
    if len(joined) < 60:
        return {}
    ratio = joined["px"] / joined["bm"]
    out: dict[str, float] = {}
    for label, days in (("1m", 21), ("3m", 63), ("6m", 126), ("12m", 252)):
        if len(joined) > days:
            stock = float(joined["px"].iloc[-1] / joined["px"].iloc[-1 - days]) - 1.0
            bench = float(joined["bm"].iloc[-1] / joined["bm"].iloc[-1 - days]) - 1.0
            out[f"excess_{label}"] = stock - bench
    tail = ratio.tail(126)
    if len(tail) > 30 and (tail > 0).all():
        slope, _, r2 = linreg(np.log(tail.to_numpy(dtype=float)))
        out["rs_slope_ann"] = float(np.expm1(slope * 252)) if is_num(slope) else NAN
        out["rs_r2"] = r2
    out["rs_new_high_63d"] = float(ratio.iloc[-1] >= ratio.tail(63).max() * 0.999)
    return out
