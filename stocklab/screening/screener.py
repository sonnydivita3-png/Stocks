"""Declarative screening over a factor matrix.

A screen is a list of :class:`Filter` objects plus an optional scoring blend.
The design point worth calling out is :attr:`Filter.required`: a filter on a
metric the company never reported can either reject it or let it through, and
the right answer differs by metric.

Rejecting on missing data silently biases a screen toward companies with
complete disclosure -- which skews large, US-domiciled and mature. Passing on
missing data lets through names that should have been caught. Neither default
is right, so each filter says which it wants, and :attr:`ScreenResult.excluded`
records why every rejected name was rejected.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from .factors import add_ranks, composite_score

_OPS: dict[str, Callable[[Any, Any], Any]] = {
    ">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le,
    "==": operator.eq, "!=": operator.ne,
}


@dataclass
class Filter:
    field: str
    op: str
    value: Any = None
    required: bool = True
    label: str = ""

    def describe(self) -> str:
        if self.label:
            return self.label
        if self.op == "between":
            return f"{self.field} in [{self.value[0]}, {self.value[1]}]"
        if self.op in ("in", "notin"):
            return f"{self.field} {self.op} {self.value}"
        if self.op in ("notnull", "isnull"):
            return f"{self.field} {self.op}"
        return f"{self.field} {self.op} {self.value}"

    def mask(self, frame: pd.DataFrame) -> pd.Series:
        """Boolean mask of rows passing this filter."""
        if self.field not in frame.columns:
            # An absent column cannot discriminate. Required filters therefore
            # reject everything, which is loud and obvious, rather than quietly
            # being ignored and producing a screen that did not do what it said.
            return pd.Series(not self.required, index=frame.index)

        column = frame[self.field]
        if self.op in ("in", "notin"):
            isin = column.isin(list(self.value))
            result = isin if self.op == "in" else ~isin
            return result.fillna(not self.required)
        if self.op == "notnull":
            return column.notna()
        if self.op == "isnull":
            return column.isna()

        numeric = pd.to_numeric(column, errors="coerce")
        present = numeric.notna()
        if self.op == "between":
            lo, hi = self.value
            passes = (numeric >= lo) & (numeric <= hi)
        else:
            func = _OPS.get(self.op)
            if func is None:
                raise ValueError(f"unsupported operator: {self.op}")
            passes = func(numeric, self.value)
        passes = passes.where(present, not self.required)
        return passes.fillna(not self.required).astype(bool)


@dataclass
class ScreenResult:
    name: str
    passed: pd.DataFrame
    excluded: pd.DataFrame        # symbol -> which filters it failed
    filter_attrition: pd.Series   # how many names each filter removed
    universe_size: int
    notes: list[str] = field(default_factory=list)

    def top(self, n: int = 20, by: str = "score") -> pd.DataFrame:
        if self.passed.empty:
            return self.passed
        column = by if by in self.passed.columns else None
        if column is None:
            return self.passed.head(n)
        return self.passed.sort_values(column, ascending=False).head(n)

    def __len__(self) -> int:
        return len(self.passed)


@dataclass
class Screen:
    name: str
    description: str
    filters: list[Filter] = field(default_factory=list)
    score_weights: dict[str, float] = field(default_factory=dict)
    rank_within_sector: bool = False
    min_factors_for_score: int = 2
    # Honest documentation of what the strategy is and is not. Printed with
    # every result, because a screen without its failure modes is a trap.
    evidence: str = ""
    failure_modes: list[str] = field(default_factory=list)

    def run(self, matrix: pd.DataFrame, limit: int | None = None) -> ScreenResult:
        universe_size = len(matrix)
        if matrix.empty:
            return ScreenResult(self.name, matrix, pd.DataFrame(),
                                pd.Series(dtype=int), 0,
                                ["Empty universe."])

        attrition: dict[str, int] = {}
        failures: dict[str, list[str]] = {}
        keep = pd.Series(True, index=matrix.index)

        for filt in self.filters:
            mask = filt.mask(matrix)
            # Attribute each rejection to the first filter that caught it, so
            # attrition reads as a funnel rather than double-counting.
            newly_failed = keep & ~mask
            attrition[filt.describe()] = int(newly_failed.sum())
            for symbol in matrix.index[newly_failed]:
                failures.setdefault(symbol, []).append(filt.describe())
            keep &= mask

        passed = matrix[keep].copy()
        notes: list[str] = []

        if self.score_weights and not passed.empty:
            ranked = add_ranks(passed, list(self.score_weights),
                               by_sector=self.rank_within_sector)
            passed = ranked
            passed["score"] = composite_score(
                ranked, self.score_weights, require=self.min_factors_for_score
            )
            missing = int(passed["score"].isna().sum())
            if missing:
                notes.append(f"{missing} name(s) lacked enough factors to score "
                             f"(minimum {self.min_factors_for_score}).")
            passed = passed.sort_values("score", ascending=False)

        if limit:
            passed = passed.head(limit)

        excluded = pd.DataFrame(
            {"failed_filters": {s: "; ".join(v) for s, v in failures.items()}}
        )
        return ScreenResult(
            name=self.name, passed=passed, excluded=excluded,
            filter_attrition=pd.Series(attrition, dtype=int),
            universe_size=universe_size,
            notes=notes + list(self.failure_modes),
        )


# --------------------------------------------------------------------------- #
# shared hygiene filters
# --------------------------------------------------------------------------- #
def liquidity_filters(min_market_cap: float = 3e8,
                      min_dollar_volume: float = 2e6) -> list[Filter]:
    """Baseline tradeability screen.

    Worth applying to almost every screen. Value and quality factors look
    strongest in exactly the names that cannot be traded in size, and a
    backtest that ignores liquidity harvests returns that do not survive the
    bid-ask spread.
    """
    return [
        Filter("market_cap", ">=", min_market_cap,
               label=f"market cap >= ${min_market_cap/1e6:.0f}m"),
        Filter("dollar_volume_50d", ">=", min_dollar_volume, required=False,
               label=f"50d dollar volume >= ${min_dollar_volume/1e6:.1f}m"),
    ]


def sanity_filters() -> list[Filter]:
    """Reject names whose financials cannot support any analysis."""
    return [
        Filter("revenue", ">", 0, label="positive revenue"),
        Filter("years_of_data", ">=", 3, label="3+ years of filings"),
        # An unreliable share count makes market cap -- and therefore every
        # multiple -- understated, which reads as "cheap". Excluded so a data
        # gap cannot win a value screen.
        Filter("share_count_reliable", "==", 1.0, required=False,
               label="share count corroborated across sources"),
        # A Beneish M-Score above -1.3 is a strong manipulation flag; names
        # without one are allowed through rather than excluded for silence.
        Filter("beneish_m", "<", -1.30, required=False,
               label="Beneish M-Score below -1.30 (manipulation flag)"),
    ]
