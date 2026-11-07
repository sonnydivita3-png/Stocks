"""Walk-forward evaluation: choose parameters in-sample, measure out-of-sample.

The problem this exists to solve is the one that invalidates most backtests.
Running fifty parameter combinations over one history and reporting the best is
not a test, it is a search, and the winner's performance is mostly the maximum
of fifty noise draws. :func:`stocklab.backtest.metrics.deflated_sharpe`
quantifies that bias after the fact; walk-forward avoids incurring it.

The procedure: fit on a training window, apply the chosen parameters to the
immediately following test window without further adjustment, roll forward, and
chain only the test-window returns. The resulting curve is what the rule would
have produced for someone who had to decide in advance -- which is the only
situation that ever actually obtains.

An **embargo** gap separates train from test. Any signal built from a trailing
window straddles the boundary (a 12-month momentum score on the first test day
is mostly computed from training data), and without a gap that overlap leaks
information across the split.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from ..config import settings
from ..util.mathx import NAN, is_num, safe_div
from .engine import BacktestResult, run
from .metrics import performance_summary


@dataclass
class Fold:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    chosen_params: dict
    in_sample: dict = field(default_factory=dict)
    out_of_sample: dict = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    folds: list[Fold]
    oos_returns: pd.Series
    oos_equity: pd.Series
    oos_summary: dict
    param_stability: pd.DataFrame
    notes: list[str] = field(default_factory=list)

    @property
    def degradation(self) -> float:
        """In-sample CAGR minus out-of-sample CAGR, averaged across folds.

        The honest measure of overfitting. A large positive number means the
        parameter search was fitting noise; near zero means the rule travels.
        """
        gaps = [f.in_sample.get("cagr", NAN) - f.out_of_sample.get("cagr", NAN)
                for f in self.folds]
        gaps = [g for g in gaps if is_num(g)]
        return float(np.mean(gaps)) if gaps else NAN

    def fold_table(self) -> pd.DataFrame:
        rows = []
        for fold in self.folds:
            rows.append({
                "fold": fold.index,
                "train": f"{fold.train_start.date()} to {fold.train_end.date()}",
                "test": f"{fold.test_start.date()} to {fold.test_end.date()}",
                "params": ", ".join(f"{k}={v}" for k, v in fold.chosen_params.items()),
                "in_sample_cagr": fold.in_sample.get("cagr"),
                "oos_cagr": fold.out_of_sample.get("cagr"),
                "oos_sharpe": fold.out_of_sample.get("sharpe"),
                "oos_max_drawdown": fold.out_of_sample.get("max_drawdown"),
            })
        return pd.DataFrame(rows).set_index("fold")


def make_folds(index: pd.DatetimeIndex, train_years: float = 3.0,
               test_years: float = 1.0, embargo_days: int = 30
               ) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Rolling train/test windows with an embargo gap between them."""
    if len(index) == 0:
        return []
    index = pd.DatetimeIndex(sorted(index))
    start, finish = index[0], index[-1]
    train_span = pd.Timedelta(days=int(train_years * 365.25))
    test_span = pd.Timedelta(days=int(test_years * 365.25))
    embargo = pd.Timedelta(days=embargo_days)

    folds = []
    train_start = start
    while True:
        train_end = train_start + train_span
        test_start = train_end + embargo
        test_end = test_start + test_span
        if test_start >= finish:
            break
        folds.append((train_start, train_end, test_start, min(test_end, finish)))
        train_start = train_start + test_span
    return folds


def walk_forward(
    prices: pd.DataFrame,
    weight_builder: Callable[[pd.DataFrame, dict], pd.DataFrame],
    param_grid: Sequence[dict],
    *,
    train_years: float = 3.0,
    test_years: float = 1.0,
    embargo_days: int = 30,
    objective: str = "sharpe",
    risk_free: float = 0.0,
    initial_capital: float = 100_000.0,
) -> WalkForwardResult:
    """Select parameters on each training window; score the next test window.

    ``weight_builder(prices, params)`` must return a target-weight frame using
    only information available at each date -- the engine's one-bar shift
    protects against same-bar look-ahead but not against a signal that is
    itself clairvoyant.

    ``objective`` is the summary key maximised in-sample. Sharpe is the default;
    ``calmar`` is often the better choice, since it selects for the drawdown an
    investor would actually have to sit through.
    """
    notes: list[str] = []
    prices = prices.sort_index()
    windows = make_folds(pd.DatetimeIndex(prices.index), train_years,
                         test_years, embargo_days)
    if not windows:
        return WalkForwardResult([], pd.Series(dtype=float), pd.Series(dtype=float),
                                 {}, pd.DataFrame(),
                                 ["History too short for even one train/test split."])
    if not param_grid:
        return WalkForwardResult([], pd.Series(dtype=float), pd.Series(dtype=float),
                                 {}, pd.DataFrame(), ["Empty parameter grid."])

    folds: list[Fold] = []
    oos_pieces: list[pd.Series] = []

    for number, (train_start, train_end, test_start, test_end) in enumerate(windows, 1):
        train_prices = prices.loc[train_start:train_end]
        test_prices = prices.loc[test_start:test_end]
        if len(train_prices) < 120 or len(test_prices) < 30:
            continue

        best_params, best_score, best_summary = None, -np.inf, {}
        for params in param_grid:
            try:
                weights = weight_builder(train_prices, params)
            except Exception:  # noqa: BLE001 - a bad parameter set is not fatal
                continue
            if weights is None or weights.empty:
                continue
            trial = run(train_prices, weights, risk_free=risk_free,
                        initial_capital=initial_capital)
            score = trial.summary.get(objective, NAN)
            if is_num(score) and score > best_score:
                best_params, best_score, best_summary = params, score, trial.summary

        if best_params is None:
            continue

        # The test window must be built from the full history up to its end, so
        # trailing indicators are available on its first day -- but parameters
        # are frozen at the training window's choice.
        extended = prices.loc[:test_end]
        oos_weights = weight_builder(extended, best_params)
        oos_weights = oos_weights.loc[test_start:test_end]
        oos = run(test_prices, oos_weights, risk_free=risk_free,
                  initial_capital=initial_capital)

        folds.append(Fold(
            index=number, train_start=train_start, train_end=train_end,
            test_start=test_start, test_end=test_end, chosen_params=best_params,
            in_sample=best_summary, out_of_sample=oos.summary,
        ))
        if not oos.returns.empty:
            oos_pieces.append(oos.returns)

    if not folds:
        return WalkForwardResult([], pd.Series(dtype=float), pd.Series(dtype=float),
                                 {}, pd.DataFrame(),
                                 ["No fold had enough data to evaluate."])

    oos_returns = pd.concat(oos_pieces).sort_index() if oos_pieces else pd.Series(dtype=float)
    oos_returns = oos_returns[~oos_returns.index.duplicated(keep="first")]
    oos_equity = (1.0 + oos_returns).cumprod() * initial_capital
    oos_summary = performance_summary(oos_returns, oos_equity, risk_free=risk_free)

    stability = pd.DataFrame([f.chosen_params for f in folds],
                             index=[f.index for f in folds])
    stability.index.name = "fold"

    changes = sum(1 for a, b in zip(folds, folds[1:]) if a.chosen_params != b.chosen_params)
    if changes > len(folds) / 2:
        notes.append(
            f"The selected parameters changed in {changes} of {len(folds) - 1} "
            "transitions. Unstable choices mean the objective surface is mostly "
            "noise, and the out-of-sample result should be treated as the "
            "ceiling rather than the expectation."
        )
    notes.append(
        f"Out-of-sample window totals {len(oos_returns) / settings().trading_days:.1f} "
        f"years across {len(folds)} fold(s). Everything before each test window "
        "was available for fitting; nothing inside it was."
    )
    notes.append(
        "Survivorship bias still applies: the universe is drawn from currently "
        "listed symbols, which walk-forward does nothing to correct."
    )
    return WalkForwardResult(folds=folds, oos_returns=oos_returns,
                             oos_equity=oos_equity, oos_summary=oos_summary,
                             param_stability=stability, notes=notes)


def momentum_builder(top_n_values: Iterable[int] = (2, 3, 5),
                     lookbacks: Iterable[int] = (126, 189, 252),
                     frequency: str = "M"):
    """A ready-made ``(builder, grid)`` pair for cross-sectional momentum."""
    from .engine import weights_from_ranks

    def builder(prices: pd.DataFrame, params: dict) -> pd.DataFrame:
        lookback = int(params["lookback"])
        # 12-1 construction: skip the most recent month, which is contaminated
        # by short-term reversal.
        scores = prices.shift(21) / prices.shift(lookback) - 1.0
        return weights_from_ranks(scores, top_n=int(params["top_n"]),
                                  frequency=frequency)

    grid = [{"top_n": n, "lookback": lb}
            for n in top_n_values for lb in lookbacks]
    return builder, grid
