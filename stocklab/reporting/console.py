"""Terminal rendering helpers built on `rich`.

Formatting conventions that matter for reading financial output quickly:
large currency figures are abbreviated (4.98T, not 4975400000000), percentages
carry an explicit sign where direction is the point, and anything unknown
prints as a dash rather than 0.00 -- conflating "no data" with "zero" is how a
missing dividend becomes a 0% yield.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..util.mathx import is_num

console = Console()

DASH = "-"


def money(value: Any, decimals: int = 2, abbreviate: bool = True) -> str:
    if not is_num(value):
        return DASH
    value = float(value)
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    if abbreviate:
        for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
            if magnitude >= threshold:
                return f"{sign}${magnitude / threshold:,.2f}{suffix}"
    return f"{sign}${magnitude:,.{decimals}f}"


def pct(value: Any, decimals: int = 1, signed: bool = False) -> str:
    if not is_num(value):
        return DASH
    fmt = f"{{:+.{decimals}f}}%" if signed else f"{{:.{decimals}f}}%"
    return fmt.format(float(value) * 100.0)


def num(value: Any, decimals: int = 2) -> str:
    if not is_num(value):
        return DASH
    return f"{float(value):,.{decimals}f}"


def ratio(value: Any, decimals: int = 1) -> str:
    if not is_num(value):
        return DASH
    return f"{float(value):,.{decimals}f}x"


def colour_for(value: Any, good_high: bool = True,
               thresholds: tuple[float, float] = (0.0, 0.0)) -> str:
    """Green / yellow / red by value, with direction configurable."""
    if not is_num(value):
        return "dim"
    low, high = thresholds
    value = float(value)
    if good_high:
        if value >= high:
            return "green"
        return "red" if value < low else "yellow"
    if value <= low:
        return "green"
    return "red" if value > high else "yellow"


def signed_text(value: Any, formatter=pct) -> Text:
    if not is_num(value):
        return Text(DASH, style="dim")
    style = "green" if float(value) > 0 else ("red" if float(value) < 0 else "white")
    return Text(formatter(value, signed=True) if formatter is pct else formatter(value),
                style=style)


def kv_table(rows: Sequence[tuple[str, Any]], title: str | None = None,
             width: int = 30) -> Table:
    table = Table(show_header=False, box=None, padding=(0, 1), title=title,
                  title_justify="left", title_style="bold")
    table.add_column("field", style="dim", width=width, no_wrap=True)
    table.add_column("value", justify="right")
    for label, value in rows:
        table.add_row(label, value if isinstance(value, Text) else str(value))
    return table


def frame_table(frame: pd.DataFrame, title: str | None = None,
                formatters: dict[str, Any] | None = None,
                max_rows: int | None = None,
                index_label: str = "") -> Table:
    """Render a DataFrame, applying per-column formatters."""
    formatters = formatters or {}
    table = Table(title=title, title_justify="left", title_style="bold",
                  header_style="bold", box=None, padding=(0, 1))
    if frame.empty:
        table.add_column("(no rows)")
        return table

    shown = frame.head(max_rows) if max_rows else frame
    if index_label or shown.index.name:
        table.add_column(index_label or str(shown.index.name), style="bold cyan",
                         no_wrap=True)
    for column in shown.columns:
        numeric = pd.api.types.is_numeric_dtype(shown[column])
        table.add_column(str(column), justify="right" if numeric else "left",
                         overflow="fold")

    for key, row in shown.iterrows():
        cells = []
        if index_label or shown.index.name:
            cells.append(str(key))
        for column in shown.columns:
            value = row[column]
            formatter = formatters.get(column)
            if formatter is not None:
                cells.append(formatter(value))
            elif isinstance(value, float):
                cells.append(DASH if (value != value) else f"{value:,.3f}")
            else:
                cells.append(DASH if value is None else str(value))
        table.add_row(*cells)
    return table


def notes_panel(notes: Iterable[str], title: str = "Caveats",
                style: str = "yellow") -> Panel | None:
    """Render the caveat list. Returns ``None`` when there is nothing to say."""
    items = [n for n in notes if n]
    if not items:
        return None
    body = "\n".join(f"* {note}" for note in items)
    return Panel(body, title=title, title_align="left", border_style=style,
                 padding=(0, 1))


def header(title: str, subtitle: str = "") -> Panel:
    text = Text(title, style="bold white")
    if subtitle:
        text.append(f"\n{subtitle}", style="dim")
    return Panel(text, border_style="blue", padding=(0, 1))


def score_bar(score: float, width: int = 20) -> Text:
    """A 0-100 score as a coloured bar."""
    if not is_num(score):
        return Text(DASH, style="dim")
    filled = int(round(max(0.0, min(100.0, float(score))) / 100.0 * width))
    style = "green" if score >= 65 else ("yellow" if score >= 45 else "red")
    bar = Text("#" * filled, style=style)
    bar.append("." * (width - filled), style="dim")
    bar.append(f" {score:.0f}", style=style)
    return bar


def print_notes(notes: Iterable[str], title: str = "Caveats",
                style: str = "yellow") -> None:
    panel = notes_panel(notes, title, style)
    if panel is not None:
        console.print(panel)
