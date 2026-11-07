"""A local web application over the same engine the CLI uses.

Runs on the user's own machine (``stocklab serve``) rather than a hosted
service, for one reason that matters: the portfolio lives in a local SQLite
file. A hosted page can only ever show a snapshot someone exported to it,
whereas this reads the real database on every request, so a trade recorded a
minute ago is in the next page load.

Design notes:

* Handlers are plain ``def``, not ``async def``. Starlette runs sync endpoints
  in a threadpool, which is what we want -- the data layer is blocking I/O
  against SEC and Yahoo, and running it on the event loop would stall every
  other request.
* Page data is memoised with a short TTL on top of the provider cache. The
  provider cache stops repeat *network* calls; this stops repeat *assembly*,
  which for a full valuation is several seconds of arithmetic.
* Every page exposes the same payload as JSON at ``/api/...``, so the HTML is a
  view over an API rather than the only way to get at the numbers.
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from jinja2 import ChainableUndefined, Undefined
from starlette.templating import Jinja2Templates

from ..util.mathx import NAN, is_num, safe_div

log = logging.getLogger(__name__)

HERE = Path(__file__).parent


# --------------------------------------------------------------------------- #
# template filters
# --------------------------------------------------------------------------- #
DASH = "\u2013"


def _coerce(value: Any) -> float | None:
    """Numeric value, or ``None`` for anything unusable.

    Jinja hands a missing key through as ``Undefined``, which raises on
    ``float()``. Without this, one absent metric takes down the whole page --
    which is exactly how the performance view first 500'd on a statistic that
    simply is not computed for a portfolio equity curve.
    """
    if value is None or isinstance(value, Undefined):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if is_num(number) else None


def _money(value: Any, dp: int = 2) -> str:
    """Abbreviated currency. Unknown prints as a dash, never as 0.00."""
    number = _coerce(value)
    if number is None:
        return DASH
    value = number
    sign = "-" if value < 0 else ""
    size = abs(value)
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if size >= threshold:
            return f"{sign}${size / threshold:,.2f}{suffix}"
    return f"{sign}${size:,.{dp}f}"


def _price(value: Any) -> str:
    number = _coerce(value)
    return f"${number:,.2f}" if number is not None else DASH


def _pct(value: Any, dp: int = 1, signed: bool = False) -> str:
    number = _coerce(value)
    if number is None:
        return DASH
    fmt = f"{{:+.{dp}f}}%" if signed else f"{{:.{dp}f}}%"
    return fmt.format(number * 100.0)


def _ratio(value: Any, dp: int = 1) -> str:
    number = _coerce(value)
    return f"{number:,.{dp}f}x" if number is not None else DASH


def _plain(value: Any, dp: int = 2) -> str:
    number = _coerce(value)
    return f"{number:,.{dp}f}" if number is not None else DASH


def _sign_class(value: Any) -> str:
    """Pair a colour class with the explicit sign the formatters emit.

    Colour never carries the meaning on its own here: every figure wearing one
    of these classes also shows a + or - from the formatter.
    """
    number = _coerce(value)
    if number is None:
        return "muted"
    return "up" if number > 0 else ("down" if number < 0 else "flat")


def _confidence_class(label: Any) -> str:
    return {"reasonable": "good", "moderate": "warning",
            "low": "serious", "none": "critical"}.get(str(label), "")


def _regime_class(label: Any) -> str:
    return {"risk-on": "good", "constructive": "good", "neutral": "warning",
            "defensive": "serious", "risk-off": "critical"}.get(str(label), "")


def _zone_class(zone: Any) -> str:
    text = str(zone or "")
    if "safe" in text:
        return "good"
    if "grey" in text:
        return "warning"
    if "distress" in text:
        return "critical"
    return ""


TEMPLATES = Jinja2Templates(directory=str(HERE / "templates"))
# ChainableUndefined keeps `a.b.c` from raising when `b` is absent; the filters
# above then render the result as a dash. Set on the environment rather than
# passed to Jinja2Templates, which does not accept it.
TEMPLATES.env.undefined = ChainableUndefined
TEMPLATES.env.filters.update({
    "money": _money, "price": _price, "pct": _pct, "ratio": _ratio,
    "plain": _plain, "signclass": _sign_class,
    "confclass": _confidence_class, "regimeclass": _regime_class,
    "zoneclass": _zone_class,
})
TEMPLATES.env.globals.update({"DASH": DASH})


# --------------------------------------------------------------------------- #
# memoisation
# --------------------------------------------------------------------------- #
@dataclass
class _Entry:
    value: Any
    stored_at: float


class ViewCache:
    """Tiny TTL memo for assembled view models.

    Separate from the provider cache on purpose. That one avoids refetching;
    this one avoids recomputing. A company page costs a handful of seconds of
    pure arithmetic even with every byte already local.
    """

    def __init__(self, ttl: float = 900.0):
        self.ttl = ttl
        self._lock = threading.Lock()
        self._entries: dict[str, _Entry] = {}

    def get_or_build(self, key: str, builder: Callable[[], Any],
                     ttl: float | None = None) -> Any:
        ttl = self.ttl if ttl is None else ttl
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and (time.time() - entry.stored_at) < ttl:
                return entry.value
        value = builder()          # built outside the lock: these are slow
        with self._lock:
            self._entries[key] = _Entry(value, time.time())
        return value

    def invalidate(self, prefix: str = "") -> int:
        with self._lock:
            keys = [k for k in self._entries if k.startswith(prefix)]
            for key in keys:
                del self._entries[key]
            return len(keys)

    def info(self) -> list[dict]:
        with self._lock:
            now = time.time()
            return [{"key": k, "age_seconds": round(now - e.stored_at, 1)}
                    for k, e in sorted(self._entries.items())]


VIEWS = ViewCache()


# --------------------------------------------------------------------------- #
# serialisation helpers
# --------------------------------------------------------------------------- #
def num(value: Any) -> float | None:
    """JSON-safe float. ``None`` rather than ``NaN``, which is not valid JSON."""
    return float(value) if is_num(value) else None


def jsonable(obj: Any) -> Any:
    """Recursively convert pandas/numpy/dataclass values into JSON primitives."""
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return num(obj)
    if isinstance(obj, (datetime, date, pd.Timestamp)):
        stamp = pd.Timestamp(obj)
        # Trade and period dates carry no meaningful time, and rendering
        # "2026-09-20T00:00:00" in a table is just noise.
        if stamp.hour == stamp.minute == stamp.second == stamp.microsecond == 0:
            return stamp.strftime("%Y-%m-%d")
        return stamp.isoformat()
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, pd.Series):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return [{"_index": str(i), **{str(c): jsonable(r[c]) for c in obj.columns}}
                for i, r in obj.iterrows()]
    if hasattr(obj, "__dataclass_fields__"):
        return {k: jsonable(getattr(obj, k)) for k in obj.__dataclass_fields__}
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return jsonable(obj.item())
    except ImportError:
        pass
    return str(obj)


def frame_rows(frame: pd.DataFrame, index_name: str = "key") -> list[dict]:
    if frame is None or frame.empty:
        return []
    return [{index_name: str(i), **{str(c): jsonable(r[c]) for c in frame.columns}}
            for i, r in frame.iterrows()]


# --------------------------------------------------------------------------- #
# view-model builders (shared by HTML and JSON routes)
# --------------------------------------------------------------------------- #
def build_regime(benchmark: str = "SPY") -> dict:
    from ..analysis.regime import assess
    state = assess(benchmark)
    return {
        "label": state["label"],
        "score": num(state["composite_score"]),
        "exposure": num(state["exposure_multiplier"]),
        "components": [
            {"name": k, "score": num(v), "weight": num(state["weights"].get(k))}
            for k, v in state["components"].items()
        ],
        "readings": [
            ("Index vs 200-day", num(state["trend"].get("dist_sma200")), "pct"),
            ("Drawdown from high", num(state["trend"].get("drawdown_from_high")), "pct"),
            ("12-month return", num(state["trend"].get("return_12m")), "pct"),
            ("Days above 200-day (1y)",
             num(state["trend"].get("pct_days_above_sma200_1y")), "pct"),
            ("Breadth above 200-day", num(state["breadth"].get("pct_above_sma200")), "pct"),
            ("Equal-weight minus cap-weight (1y)",
             num(state.get("equal_weight_divergence")), "pct"),
            ("High-yield spread", num(state["credit"].get("hy_oas")), "pp"),
            ("  3-month change", num(state["credit"].get("hy_oas_chg_3m")), "pp"),
            ("VIX", num(state["rates"].get("vix")), "plain"),
            ("10-year Treasury", num(state["rates"].get("ust_10y")), "pp"),
            ("10y minus 3m", num(state["rates"].get("curve_10y3m")), "pp"),
            ("Unemployment", num(state["rates"].get("unemployment")), "pp"),
        ],
        "caveats": state["caveats"],
        "benchmark": benchmark,
    }


def build_company(symbol: str, benchmark: str = "SPY",
                  capital: float = 100_000.0) -> dict:
    from ..company import Company
    from ..entry.timing import build_plan
    from ..valuation import composite

    company = Company.load(symbol, benchmark=benchmark)
    if not company.has_data:
        return {"symbol": symbol.upper(), "error":
                "No usable SEC filings or price data for this symbol. "
                "ETFs, ADRs and foreign filers without XBRL legitimately land here."}

    metrics, technicals = company.metrics, company.technicals
    report = composite.value(company.fundamentals, metrics, company.price)
    plan = build_plan(symbol, company.history,
                      margin_of_safety=report.margin_of_safety,
                      technicals=technicals, capital=capital)

    history = company.history
    series: dict = {}
    if not history.empty:
        weekly = history["adj_close"].tail(1260).resample("W").last().dropna()
        series = {"dates": [d.strftime("%Y-%m-%d") for d in weekly.index],
                  "values": [round(float(v), 4) for v in weekly]}

    bands = []
    if report.relative is not None:
        from ..valuation.multiples import MULTIPLE_SPECS
        for name, band in report.relative.bands.items():
            bands.append({
                "label": MULTIPLE_SPECS.get(name, {}).get("label", name),
                "current": num(band.current), "p10": num(band.p10),
                "p25": num(band.p25), "median": num(band.median),
                "p75": num(band.p75), "p90": num(band.p90),
                "percentile": num(band.percentile),
                "implied": num(band.implied_price_at_median),
                "years": num(band.years_covered), "read": band.rich_or_cheap,
            })

    return {
        "symbol": company.symbol, "name": company.name,
        "sector": company.sector, "industry": company.fundamentals.industry,
        "is_financial": company.fundamentals.is_financial,
        "is_reit": company.fundamentals.is_reit,
        "price": num(company.price),
        "years_of_data": num(metrics.get("years_of_data")),
        "share_count_basis": metrics.get("share_count_basis"),
        "metrics": {k: num(v) for k, v in metrics.items() if isinstance(v, float)},
        "altman_zone": metrics.get("altman_zone"),
        "technicals": {k: num(v) for k, v in technicals.items()},
        "valuation": {
            "fair_value": num(report.fair_value), "low": num(report.low),
            "high": num(report.high),
            "margin_of_safety": num(report.margin_of_safety),
            "dispersion": num(report.dispersion), "confidence": report.confidence,
            "implied_return": num(report.implied_return),
            "implied_erp": num(report.implied_equity_premium),
            "risk_free": num(report.risk_free),
            "wacc": num(report.cost_of_capital.wacc),
            "cost_of_equity": num(report.cost_of_capital.cost_of_equity),
            "beta": num(report.cost_of_capital.beta_adjusted),
            "rating": report.cost_of_capital.synthetic_rating,
            "models": [
                {"name": e.name, "value": num(e.value_per_share),
                 "upside": num(safe_div(e.value_per_share, company.price) - 1.0
                               if is_num(e.value_per_share) else NAN),
                 "weight": num(e.weight) if e.applicable else 0.0,
                 "applicable": bool(e.applicable), "rationale": e.rationale}
                for e in report.estimates
            ],
            "scenarios": frame_rows(report.scenarios, "scenario"),
            "bands": bands,
            "notes": report.notes,
        },
        "entry": {
            "score": num(plan.score), "verdict": plan.verdict,
            "best_setup": plan.best_setup,
            "components": {k: num(v) for k, v in plan.components.items()},
            "stop": num(plan.stop.stop), "stop_method": plan.stop.method,
            "risk_per_share": num(plan.stop.risk_per_share),
            "risk_pct": num(plan.stop.risk_pct),
            "targets": {k: num(v) for k, v in plan.stop.targets.items()},
            "size": {k: (num(v) if isinstance(v, float) else v)
                     for k, v in plan.size.items() if k != "notes"},
            "ladder": frame_rows(plan.staged_entries, "tranche"),
            "setups": [
                {"name": s.name, "triggered": bool(s.triggered),
                 "confidence": num(s.confidence), "failed": s.failed,
                 "notes": s.notes}
                for s in plan.setups.values()
            ],
            "invalidation": plan.invalidation,
            "notes": plan.notes,
        },
        "series": series,
        "data_notes": company.notes + list(metrics.get("share_count_notes") or []),
    }


def build_portfolio() -> dict:
    from ..portfolio.ledger import build as build_ledger
    from ..portfolio.store import store as get_store

    store = get_store()
    accounts_list = store.accounts()
    state, accounts = build_ledger(store)
    positions = state.positions()
    symbols = sorted({p.symbol for p in positions})

    prices: dict[str, float] = {}
    if symbols:
        from ..data.providers.yahoo import fetch_meta
        for symbol in symbols:
            price = (fetch_meta(symbol) or {}).get("price")
            if is_num(price):
                prices[symbol] = float(price)

    rows = []
    for position in state.positions(prices=prices, by_account=True):
        account = accounts.get(position.account_id)
        rows.append({
            "symbol": position.symbol,
            "account": account.name if account else "-",
            "account_type": account.type.label if account else "-",
            "sheltered": bool(account.type.sheltered) if account else False,
            "quantity": num(position.quantity),
            "cost_per_share": num(position.cost_per_share),
            "price": num(position.price),
            "cost_basis": num(position.cost_basis),
            "value": num(position.market_value),
            "gain": num(position.unrealized_gain),
            "gain_pct": num(position.unrealized_pct),
            "long_term": num(position.long_term_quantity),
            "short_term": num(position.short_term_quantity),
        })

    combined = []
    for position in state.positions(prices=prices):
        combined.append({
            "symbol": position.symbol, "quantity": num(position.quantity),
            "price": num(position.price), "value": num(position.market_value),
            "cost_basis": num(position.cost_basis),
            "gain": num(position.unrealized_gain),
            "gain_pct": num(position.unrealized_pct),
        })

    invested = sum(r["value"] or 0.0 for r in combined)
    cash = state.total_cash()
    total = invested + max(cash, 0.0)
    for row in combined:
        row["weight"] = num(safe_div(row["value"], total))

    return {
        "accounts": [
            {"name": a.name, "type": a.type.label,
             "tax": ("tax-free" if a.type.tax_free else
                     "tax-deferred" if a.type.tax_deferred else "taxable"),
             "basis_method": a.cost_basis_method.upper(),
             "broker": a.broker, "cash": num(state.total_cash(a.id)),
             "positions": len({l.symbol for l in state.open_lots(a.id)})}
            for a in accounts_list
        ],
        "positions": sorted(rows, key=lambda r: -(r["value"] or 0.0)),
        "combined": sorted(combined, key=lambda r: -(r["value"] or 0.0)),
        "totals": {
            "invested": num(invested), "cash": num(cash), "total": num(total),
            "cost_basis": num(sum(r["cost_basis"] or 0.0 for r in combined)),
            "gain": num(sum(r["gain"] or 0.0 for r in combined)),
        },
        "warnings": state.warnings,
        "empty": not rows,
    }


def build_risk(benchmark: str = "SPY") -> dict:
    from ..data.providers.sec import load_fundamentals
    from ..data.providers.yahoo import fetch_meta
    from ..portfolio.ledger import build as build_ledger
    from ..portfolio.risk_report import build as build_risk_report
    from ..portfolio.store import store as get_store

    state, accounts = build_ledger(get_store())
    positions = state.positions()
    if not positions:
        return {"empty": True}
    symbols = sorted({p.symbol for p in positions})
    prices, sectors = {}, {}
    for symbol in symbols:
        price = (fetch_meta(symbol) or {}).get("price")
        if is_num(price):
            prices[symbol] = float(price)
        try:
            sectors[symbol] = load_fundamentals(symbol).sector
        except Exception:  # noqa: BLE001
            sectors[symbol] = "Unknown"

    report = build_risk_report(state, accounts, prices, sectors, benchmark=benchmark)
    return {
        "empty": False,
        "total_value": num(report.total_value),
        "concentration": {k: num(v) if isinstance(v, float) else v
                          for k, v in report.concentration.items()},
        "beta": num(report.portfolio_beta), "volatility": num(report.portfolio_vol),
        "stats": {k: num(v) for k, v in report.risk_stats.items()
                  if isinstance(v, float)},
        "sectors": [{"sector": k, "weight": num(v)}
                    for k, v in report.sector_exposure.items()],
        "accounts": [{"account": k, "weight": num(v)}
                     for k, v in report.account_exposure.items()],
        "positions": frame_rows(report.positions, "symbol"),
        "stress": frame_rows(report.stress.set_index("scenario"), "scenario")
        if not report.stress.empty else [],
        "notes": report.notes,
    }


def build_tax(year: int | None = None, short_rate: float = 0.24,
              long_rate: float = 0.15) -> dict:
    from ..data.providers.yahoo import fetch_meta
    from ..portfolio.ledger import build as build_ledger
    from ..portfolio.store import store as get_store
    from ..portfolio.tax import annual_report, harvest_candidates, unrealized_frame

    state, accounts = build_ledger(get_store())
    report = annual_report(state, accounts, year=year, short_term_rate=short_rate,
                           long_term_rate=long_rate)
    prices = {}
    for position in state.positions():
        price = (fetch_meta(position.symbol) or {}).get("price")
        if is_num(price):
            prices[position.symbol] = float(price)
    candidates = harvest_candidates(state, accounts, prices,
                                     short_term_rate=short_rate,
                                     long_term_rate=long_rate)
    unrealized = unrealized_frame(state, accounts, prices)
    approaching = []
    if not unrealized.empty:
        soon = unrealized[(unrealized["term"] == "short")
                          & (unrealized["days_to_long_term"] <= 120)
                          & (~unrealized["sheltered"])]
        approaching = frame_rows(soon.set_index("symbol"), "symbol")

    return {
        "year": report.year,
        "short_term": num(report.short_term_gains),
        "long_term": num(report.long_term_gains),
        "net": num(report.net_gain),
        "disallowed": num(report.disallowed_wash_sales),
        "deductible": num(report.deductible_against_income),
        "carryforward": num(report.loss_carryforward),
        "estimated_tax": num(report.estimated_tax),
        "disposals": frame_rows(report.realized_detail.set_index("symbol"), "symbol")
        if not report.realized_detail.empty else [],
        "harvest": [
            {"symbol": c.symbol, "account": c.account_name,
             "quantity": num(c.quantity), "cost_basis": num(c.cost_basis),
             "value": num(c.market_value), "loss": num(c.loss), "term": c.term,
             "saving": num(c.estimated_tax_saving),
             "wash_risk": bool(c.wash_sale_risk),
             "blocked_until": str(c.blocked_until) if c.blocked_until else None,
             "notes": c.notes}
            for c in candidates
        ],
        "approaching_long_term": approaching,
        "notes": report.notes,
        "rates": {"short": short_rate, "long": long_rate},
    }


def build_performance(benchmark: str = "SPY") -> dict:
    from ..data.providers import fred
    from ..data.providers.yahoo import fetch_meta
    from ..portfolio.ledger import build as build_ledger
    from ..portfolio.performance import analyse, contribution_by_position
    from ..portfolio.store import store as get_store

    store = get_store()
    state, accounts = build_ledger(store)
    transactions = store.transactions()
    if not transactions:
        return {"empty": True}

    risk_free = fred.risk_free_rate()
    report = analyse(state, transactions, benchmark=benchmark, risk_free=risk_free)
    prices = {}
    for position in state.positions():
        price = (fetch_meta(position.symbol) or {}).get("price")
        if is_num(price):
            prices[position.symbol] = float(price)
    contribution = contribution_by_position(state, prices, transactions)

    curve = report.equity_curve
    series = {}
    if not curve.empty:
        weekly = curve.resample("W").last().dropna()
        series = {"dates": [d.strftime("%Y-%m-%d") for d in weekly.index],
                  "values": [round(float(v), 5) for v in weekly]}

    return {
        "empty": False,
        "start": str(report.start), "end": str(report.end),
        "benchmark": benchmark,
        "beginning_value": num(report.beginning_value),
        "ending_value": num(report.ending_value),
        "net_contributions": num(report.net_contributions),
        "gain": num(report.gain),
        "twr": num(report.twr), "twr_annualized": num(report.twr_annualized),
        "mwr": num(report.mwr), "behaviour_gap": num(report.behaviour_gap),
        "benchmark_return": num(report.benchmark_return),
        "risk": {k: num(v) for k, v in report.risk.items()},
        "vs_benchmark": {k: num(v) for k, v in report.vs_benchmark.items()},
        "contribution": frame_rows(contribution, "symbol"),
        "series": series,
        "notes": report.notes,
    }


def build_screen(preset: str, symbols: list[str] | None = None,
                 limit: int = 25) -> dict:
    from ..data.universe import MEGA_CAP_SAMPLE
    from ..screening import presets as preset_module
    from ..screening.factors import build_matrix

    screen = preset_module.get(preset)
    candidates = symbols or list(MEGA_CAP_SAMPLE)
    matrix = build_matrix(candidates, with_technicals=True)
    if matrix.matrix.empty:
        return {"preset": preset, "name": screen.name, "error":
                "No usable data for any symbol in the universe."}
    result = screen.run(matrix.matrix, limit=limit)

    columns = [c for c in ("name", "sector", "market_cap", "score", "pe",
                           "ev_ebit", "fcf_yield", "roic", "revenue_cagr_5y",
                           "dividend_yield", "momentum_12_1", "piotroski_f")
               if c in result.passed.columns]
    return {
        "preset": preset, "name": screen.name,
        "description": screen.description, "evidence": screen.evidence,
        "failure_modes": screen.failure_modes,
        "universe_size": result.universe_size, "passed": len(result),
        "rows": frame_rows(result.passed[columns], "symbol"),
        "attrition": [{"filter": k, "rejected": int(v)}
                      for k, v in result.filter_attrition.items() if v > 0],
        "notes": result.notes,
        "failures": len(matrix.failures),
    }


def build_watchlist() -> dict:
    from ..data.providers.yahoo import fetch_meta
    from ..portfolio.store import store as get_store

    frame = get_store().watchlist()
    rows = []
    for _, row in frame.iterrows():
        symbol = str(row["symbol"])
        price = (fetch_meta(symbol) or {}).get("price")
        target = row.get("target_buy")
        rows.append({
            "symbol": symbol, "list_name": row.get("list_name"),
            "added": str(row.get("added")), "thesis": row.get("thesis") or "",
            "target_buy": num(target), "price": num(price),
            "vs_target": num(safe_div(price, target) - 1.0
                             if (is_num(price) and is_num(target)) else NAN),
        })
    return {"rows": rows, "empty": not rows}


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
def _context(request: Request, **extra) -> dict:
    from ..config import settings
    base = {
        "request": request,
        "nav": [
            ("/", "Dashboard"), ("/portfolio", "Portfolio"),
            ("/performance", "Performance"), ("/risk", "Risk"),
            ("/tax", "Tax"), ("/screen", "Screens"), ("/regime", "Regime"),
        ],
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "data_dir": str(settings().root),
    }
    base.update(extra)
    return base


def _render(request: Request, template: str, **extra) -> Response:
    return TEMPLATES.TemplateResponse(request, template, _context(request, **extra))


def _guard(builder: Callable[[], Any], label: str) -> tuple[Any, str | None]:
    """Run a builder, converting a failure into a message rather than a 500."""
    try:
        return builder(), None
    except Exception as exc:  # noqa: BLE001 - a data outage must not blank the app
        log.warning("%s failed: %s", label, exc)
        log.debug("%s", traceback.format_exc())
        return None, f"{type(exc).__name__}: {exc}"


def dashboard(request: Request) -> Response:
    regime, regime_error = _guard(
        lambda: VIEWS.get_or_build("regime", build_regime, ttl=1800), "regime")
    portfolio, portfolio_error = _guard(
        lambda: VIEWS.get_or_build("portfolio", build_portfolio, ttl=120), "portfolio")
    watchlist, _ = _guard(
        lambda: VIEWS.get_or_build("watchlist", build_watchlist, ttl=120), "watchlist")
    return _render(request, "dashboard.html", regime=regime, regime_error=regime_error,
                   portfolio=portfolio, portfolio_error=portfolio_error,
                   watchlist=watchlist)


def company_page(request: Request) -> Response:
    symbol = request.path_params["symbol"].upper()
    capital = float(request.query_params.get("capital", 100_000))
    data, error = _guard(
        lambda: VIEWS.get_or_build(f"company:{symbol}:{capital:.0f}",
                                   lambda: build_company(symbol, capital=capital)),
        f"company {symbol}")
    return _render(request, "company.html", symbol=symbol, company=data, error=error)


def search(request: Request) -> Response:
    symbol = (request.query_params.get("symbol") or "").strip().upper()
    if not symbol:
        return RedirectResponse("/")
    return RedirectResponse(f"/company/{symbol}")


def portfolio_page(request: Request) -> Response:
    data, error = _guard(
        lambda: VIEWS.get_or_build("portfolio", build_portfolio, ttl=120), "portfolio")
    return _render(request, "portfolio.html", portfolio=data, error=error)


def performance_page(request: Request) -> Response:
    data, error = _guard(
        lambda: VIEWS.get_or_build("performance", build_performance, ttl=600),
        "performance")
    return _render(request, "performance.html", performance=data, error=error)


def risk_page(request: Request) -> Response:
    data, error = _guard(
        lambda: VIEWS.get_or_build("risk", build_risk, ttl=600), "risk")
    return _render(request, "risk.html", risk=data, error=error)


def tax_page(request: Request) -> Response:
    year = request.query_params.get("year")
    year_value = int(year) if year and year.isdigit() else None
    short_rate = float(request.query_params.get("short_rate", 0.24))
    long_rate = float(request.query_params.get("long_rate", 0.15))
    key = f"tax:{year_value}:{short_rate}:{long_rate}"
    data, error = _guard(
        lambda: VIEWS.get_or_build(
            key, lambda: build_tax(year_value, short_rate, long_rate), ttl=300),
        "tax")
    return _render(request, "tax.html", tax=data, error=error)


def screen_index(request: Request) -> Response:
    from ..screening import presets as preset_module
    return _render(request, "screen.html", presets=preset_module.describe_all(),
                   result=None, error=None)


def screen_page(request: Request) -> Response:
    preset = request.path_params["preset"]
    raw = request.query_params.get("symbols", "")
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()] or None
    from ..screening import presets as preset_module
    key = f"screen:{preset}:{raw}"
    data, error = _guard(
        lambda: VIEWS.get_or_build(key, lambda: build_screen(preset, symbols), ttl=1800),
        f"screen {preset}")
    return _render(request, "screen.html", presets=preset_module.describe_all(),
                   result=data, error=error, active=preset)


def regime_page(request: Request) -> Response:
    data, error = _guard(
        lambda: VIEWS.get_or_build("regime", build_regime, ttl=1800), "regime")
    return _render(request, "regime.html", regime=data, error=error)


def refresh(request: Request) -> Response:
    """Drop memoised view models so the next request rebuilds them."""
    prefix = request.query_params.get("prefix", "")
    removed = VIEWS.invalidate(prefix)
    target = request.headers.get("referer") or "/"
    if request.query_params.get("json"):
        return JSONResponse({"invalidated": removed})
    return RedirectResponse(target, status_code=303)


# --- JSON API -------------------------------------------------------------- #
def api(builder: Callable[[Request], Any]):
    def handler(request: Request) -> Response:
        try:
            return JSONResponse(jsonable(builder(request)))
        except Exception as exc:  # noqa: BLE001
            log.warning("api error: %s", exc)
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"},
                                status_code=500)
    return handler


def health(request: Request) -> Response:
    from ..config import settings
    from ..data.cache import cache
    return JSONResponse({
        "status": "ok",
        "version": "0.1.0",
        "data_dir": str(settings().root),
        "offline": settings().offline,
        "provider_cache": cache().stats(),
        "view_cache": VIEWS.info(),
    })


ROUTES = [
    Route("/", dashboard),
    Route("/search", search),
    Route("/company/{symbol}", company_page),
    Route("/portfolio", portfolio_page),
    Route("/performance", performance_page),
    Route("/risk", risk_page),
    Route("/tax", tax_page),
    Route("/screen", screen_index),
    Route("/screen/{preset}", screen_page),
    Route("/regime", regime_page),
    Route("/refresh", refresh, methods=["GET", "POST"]),
    Route("/healthz", health),
    Route("/api/regime", api(lambda r: build_regime())),
    Route("/api/company/{symbol}",
          api(lambda r: build_company(r.path_params["symbol"]))),
    Route("/api/portfolio", api(lambda r: build_portfolio())),
    Route("/api/performance", api(lambda r: build_performance())),
    Route("/api/risk", api(lambda r: build_risk())),
    Route("/api/tax", api(lambda r: build_tax(
        int(r.query_params["year"]) if r.query_params.get("year") else None))),
    Route("/api/screen/{preset}",
          api(lambda r: build_screen(r.path_params["preset"]))),
    Route("/api/watchlist", api(lambda r: build_watchlist())),
    Mount("/static", StaticFiles(directory=str(HERE / "static")), name="static"),
]


def create_app(debug: bool = False) -> Starlette:
    return Starlette(debug=debug, routes=ROUTES)


app = create_app()


def serve(host: str = "127.0.0.1", port: int = 8787, reload: bool = False,
          debug: bool = False) -> None:
    """Run the development server.

    Bound to localhost by default. The app has no authentication and reads a
    real portfolio, so exposing it on ``0.0.0.0`` puts your holdings on the
    local network -- fine on a trusted home network to reach it from a phone,
    not fine on shared wifi.
    """
    import uvicorn
    uvicorn.run("stocklab.web.app:app" if reload else create_app(debug),
                host=host, port=port, reload=reload, log_level="info")
