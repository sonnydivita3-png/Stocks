#!/usr/bin/env python3
"""Build the data files behind the Valuation Desk dashboard.

Run directly, or from the weekly refresh job:

    python scripts/build_desk_snapshot.py --out ./desk --limit 515

It writes four CSVs plus a JSON record of everything computed. The CSVs are
what gets attached to the dashboard; the JSON exists so a later run can resume.

**Resumable by design.** A full pass takes well over an hour at the data
providers' rate limits, and a worker restart partway through would otherwise
lose all of it. Results are appended to the JSON every few companies, and a
rerun skips whatever is already there, so an interrupted run costs minutes
rather than the whole pass.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from stocklab.analysis.regime import assess  # noqa: E402
from stocklab.company import Company  # noqa: E402
from stocklab.data.providers.sec import fetch_frame, fetch_ticker_map  # noqa: E402
from stocklab.data.universe import MEGA_CAP_SAMPLE, common_stocks  # noqa: E402
from stocklab.entry.signals import pullback_to_trend  # noqa: E402
from stocklab.scoring import score, size_band  # noqa: E402
# Aliased on purpose: stocklab.analysis.regime also exports `assess`, and
# importing both under one name silently shadowed the regime's. Every run
# after that failed the market backdrop with "'str' object has no attribute
# 'get'" -- the regime's own symbol being called with the shortlist's body.
from stocklab.shortlist import GATES, SCREENS  # noqa: E402
from stocklab.shortlist import assess as assess_company  # noqa: E402
from stocklab.util.mathx import is_num  # noqa: E402
from stocklab.valuation import composite  # noqa: E402

log = logging.getLogger("desk")

REVENUE_TAGS = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
)

# Revenue floor, ceiling, how many to take by size, how many by growth.
BANDS = (
    ("mega_large", 2.0e10, 1e15, 110, 20),
    ("mid", 4.0e9, 2.0e10, 90, 45),
    ("small", 8.0e8, 4.0e9, 70, 75),
    ("emerging", 6.0e7, 8.0e8, 25, 90),
)

COLUMNS = [
    "symbol", "name", "sector", "industry", "size", "grade", "overall",
    "quality", "growth", "safety", "price_score", "momentum", "coverage",
    "say_quality", "say_growth", "say_safety", "say_price", "price",
    "market_cap", "revenue", "profitable", "roic_5y", "roic", "gross_margin",
    "operating_margin", "fcf_margin", "net_debt_ebitda", "piotroski",
    "altman_z", "rev_cagr_3y", "rev_cagr_5y", "rev_trend_r2", "pe", "ps",
    "fcf_yield", "dividend_yield", "fair_value", "margin_of_safety",
    "implied_return", "implied_erp", "implied_method", "implied_is_floor",
    "confidence", "share_count_ok", "profitable_years", "profitable_years_of",
    "earnings_vs_own_average", "category", "qualifies", "weakest_pillar",
    "gates_passed", "gates_total", "gates_failed", "passes",
    "value_qualifies", "value_passed", "value_total", "value_failed",
    "growth_qualifies", "growth_passed", "growth_total", "growth_failed",
    "speculative_qualifies", "speculative_passed", "speculative_total",
    "speculative_failed", "from_52w_high",
    "momentum_12_1", "vol63", "dollar_volume_50d", "years_of_data",
    "sma50", "sma200", "dist_sma50", "dist_sma200", "sma200_rising",
    "above_sma200", "rsi14", "at_sma50", "at_sma200",
    "entry_confidence_50", "entry_confidence_200",
    "entry_low_50", "entry_high_50", "entry_low_200", "entry_high_200",
    "val_dcf", "val_epv", "val_ri", "val_multiples", "val_ddm", "val_graham",
    "history_revenue", "history_operating_income", "data_notes",
]


def num(value):
    return float(value) if is_num(value) else None


def _revenue_by_cik(period: str) -> dict[str, float]:
    """Revenue for every filer in one calendar period.

    One request per tag returns the whole market, which is what makes building
    a 500-name universe cheap enough to do on every refresh.
    """
    out: dict[str, float] = {}
    for tag in REVENUE_TAGS:
        frame = fetch_frame(tag, period)
        if frame.empty or "val" not in frame:
            continue
        for cik, value in pd.to_numeric(frame["val"], errors="coerce").dropna().items():
            out.setdefault(str(cik), float(value))
    return out


def build_universe(limit: int = 515) -> list[str]:
    """Pick a spread of companies across every size band."""
    now = _revenue_by_cik("CY2025")
    before = _revenue_by_cik("CY2022")
    listed = set(common_stocks()["symbol"])

    ticker_for: dict[str, str] = {}
    for symbol, cik in fetch_ticker_map().items():
        if symbol in listed:
            ticker_for.setdefault(cik, symbol)

    rows = []
    for cik, revenue in now.items():
        symbol = ticker_for.get(cik)
        if not symbol or revenue <= 0:
            continue
        prior = before.get(cik)
        growth = ((revenue / prior) ** (1 / 3) - 1) if (prior and prior > 0) else None
        rows.append({"symbol": symbol, "revenue": revenue, "growth": growth})
    if not rows:
        log.warning("no candidates from the frames API; falling back to the sample list")
        return list(MEGA_CAP_SAMPLE)

    frame = pd.DataFrame(rows).drop_duplicates("symbol").set_index("symbol")
    picked: list[str] = []
    for _, low, high, by_size, by_growth in BANDS:
        band = frame[(frame["revenue"] >= low) & (frame["revenue"] < high)]
        biggest = list(band.sort_values("revenue", ascending=False).head(by_size).index)
        growing = band[band["growth"].notna()]
        # Above ~200% a year is almost always an acquisition or a tiny base.
        growing = growing[(growing["growth"] > 0.05) & (growing["growth"] < 2.0)]
        fastest = list(growing.sort_values("growth", ascending=False).head(by_growth).index)
        picked += biggest + fastest

    universe = list(dict.fromkeys(MEGA_CAP_SAMPLE + picked))
    return universe[:limit]


def analyse(symbol: str) -> dict | None:
    """Full work-up and scorecard for one company."""
    company = Company.load(symbol)
    if not company.has_data:
        return None
    metrics, technicals = company.metrics, company.technicals
    report = composite.value(company.fundamentals, metrics, company.price)
    card = score(metrics, technicals, report)

    row = {
        "symbol": symbol, "name": company.name, "sector": company.sector,
        "industry": company.fundamentals.industry,
        "size": size_band(metrics.get("market_cap")),
        "price": num(company.price), "market_cap": num(metrics.get("market_cap")),
        "revenue": num(metrics.get("revenue")),
        "profitable": bool(is_num(metrics.get("net_income")) and metrics["net_income"] > 0),
        "roic_5y": num(metrics.get("roic_5y_avg")), "roic": num(metrics.get("roic")),
        "gross_margin": num(metrics.get("gross_margin")),
        "operating_margin": num(metrics.get("operating_margin")),
        "fcf_margin": num(metrics.get("fcf_margin")),
        "net_debt_ebitda": num(metrics.get("net_debt_to_ebitda")),
        "piotroski": num(metrics.get("piotroski_f")),
        "altman_z": num(metrics.get("altman_z")),
        "rev_cagr_3y": num(metrics.get("revenue_cagr_3y")),
        "rev_cagr_5y": num(metrics.get("revenue_cagr_5y")),
        "rev_trend_r2": num(metrics.get("revenue_trend_r2")),
        "pe": num(metrics.get("pe")), "ps": num(metrics.get("ps")),
        "fcf_yield": num(metrics.get("fcf_yield")),
        "dividend_yield": num(metrics.get("dividend_yield")),
        "fair_value": num(report.fair_value),
        "margin_of_safety": num(report.margin_of_safety),
        "implied_return": num(report.implied_return),
        "implied_erp": num(report.implied_equity_premium),
        # Which model produced it, and whether it is a ceiling rather than an
        # estimate. A reader has to be told the difference.
        "implied_method": report.implied_return_method,
        "implied_is_floor": bool(report.implied_return_is_floor),
        "confidence": report.confidence,
        # 0 means no two filing sources agreed on how many shares exist, so
        # market value and every multiple built on it are withheld.
        "share_count_ok": bool(metrics.get("share_count_reliable")),
        # How dependable the earnings are, as against how large: one good
        # year after four bad ones is valued as the normal state otherwise.
        "profitable_years": num(metrics.get("profitable_years")),
        "profitable_years_of": num(metrics.get("profitable_years_of")),
        "earnings_vs_own_average": num(metrics.get("earnings_vs_own_average")),
        "from_52w_high": num(technicals.get("pct_from_52w_high")),
        "momentum_12_1": num(technicals.get("momentum_12_1")),
        "dollar_volume_50d": num(technicals.get("dollar_volume_50d")),
        # Where the price sits against the averages buyers watch. A company
        # the screens already like, back at a level that has held before, is
        # a moment to start; the same distance below a falling average is the
        # level having given way.
        "sma50": num(technicals.get("sma50")),
        "sma200": num(technicals.get("sma200")),
        "dist_sma50": num(technicals.get("dist_sma50")),
        "dist_sma200": num(technicals.get("dist_sma200")),
        "sma200_rising": bool(technicals.get("sma200_rising")),
        "above_sma200": bool(technicals.get("above_sma200")),
        "rsi14": num(technicals.get("rsi14")),
        "vol63": num(technicals.get("vol63")),
        "years_of_data": num(metrics.get("years_of_data")),
    }
    scored = card.as_dict()
    scored["price_score"] = scored.pop("price")
    row.update({k: (num(v) if isinstance(v, float) else v) for k, v in scored.items()})

    # What kind of business it is, and whether it clears every condition for a
    # long hold. The grade averages its four pillars, so a strength cancels a
    # weakness; this is a set of gates where nothing compensates for anything.
    verdict = assess_company({
        **metrics,
        "size": row["size"],
        "dollar_volume_50d": technicals.get("dollar_volume_50d"),
        "profitable": row["profitable"],
        "coverage": card.coverage,
        "quality": card.quality, "growth": card.growth,
        "safety": card.safety, "price": card.price,
        "fair_value": report.fair_value,
        "margin_of_safety": report.margin_of_safety,
        "implied_equity_premium": report.implied_equity_premium,
        "implied_return_is_floor": report.implied_return_is_floor,
        "confidence": report.confidence,
        # The reverse DCF's view of what the price assumes, against what the
        # company has actually done. The growth screen's only price test.
        "implied_growth": getattr(report.reverse, "implied_growth", None),
        "historical_growth": getattr(report.reverse, "historical_growth", None),
    })
    # The pullback setup, run against both averages. It asks for more than
    # nearness: above a rising 200-day, RSI reset rather than broken, and no
    # volume spike on the way down.
    for window in (50, 200):
        try:
            setup = pullback_to_trend(company.history, ma_window=window)
        except Exception:  # noqa: BLE001 - a thin history must not stop the run
            continue
        row[f"at_sma{window}"] = bool(setup.triggered)
        row[f"entry_confidence_{window}"] = num(setup.confidence)
        low, high = setup.entry_zone
        row[f"entry_low_{window}"] = num(low)
        row[f"entry_high_{window}"] = num(high)

    # Each valuation model's own answer, so a reader can see where a fair
    # value came from and where the models part company, rather than being
    # handed a blended number to take on trust.
    MODEL_KEYS = {
        "DCF (FCFF, 10y fade)": "val_dcf", "Earnings Power Value": "val_epv",
        "Residual income": "val_ri", "Multiples (own history)": "val_multiples",
        "Dividend discount": "val_ddm", "Graham number": "val_graham",
    }
    for est in report.estimates:
        key = MODEL_KEYS.get(est.name)
        if key:
            row[key] = num(est.value_per_share) if est.applicable else None

    # Five years of the two lines everything else is built on, so the
    # consistency the gates test can be seen rather than asserted.
    def series(column: str) -> str:
        if column not in company.fundamentals.annual.columns:
            return ""
        tail = company.fundamentals.annual[column].dropna().tail(5)
        return "|".join(f"{d.year}:{v / 1e6:.0f}" for d, v in tail.items())

    row["history_revenue"] = series("revenue")
    row["history_operating_income"] = series("operating_income")

    # Every repair and refusal this company's figures went through. A reader
    # deciding whether to trust a number deserves to see why one is missing.
    notes = list(metrics.get("share_count_notes") or [])
    notes += list(metrics.get("income_notes") or [])
    notes += [n for n in (report.notes or []) if "withheld" in n or "set aside" in n]
    row["data_notes"] = " \u00b7 ".join(notes)[:900]

    row["category"] = verdict["category"]
    row["weakest_pillar"] = num(verdict["weakest_pillar"])
    # One row carries all three verdicts: a company is offered against every
    # screen, because the category describes the business and the screens
    # describe the case for buying it.
    for name in SCREENS:
        v = verdict[name]
        row[f"{name}_qualifies"] = v["qualifies"]
        row[f"{name}_passed"] = v["passed"]
        row[f"{name}_total"] = v["total"]
        row[f"{name}_failed"] = "|".join(v["failed"])
    row["passes"] = "|".join(verdict["passes"])
    # The bare names mean the value screen, as they did when there was one.
    row["qualifies"] = verdict["qualifies"]
    row["gates_passed"] = verdict["gates_passed"]
    row["gates_total"] = verdict["gates_total"]
    row["gates_failed"] = row["value_failed"]
    return row


def write_csvs(out_dir: Path, record: dict) -> None:
    companies = sorted(record["companies"],
                       key=lambda c: -(c.get("overall") or -1))
    regime = record.get("regime") or {}

    def cell(row, key):
        value = row.get(key)
        return "" if value is None else value

    with (out_dir / "companies.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for row in companies:
            writer.writerow([cell(row, k) for k in COLUMNS])

    scored = [c for c in companies if c.get("overall") is not None]
    top = [c for c in scored if c["grade"] in ("A", "B")]
    smaller = [c for c in scored
               if c.get("size") in ("small", "micro") or (c.get("market_cap") or 0) < 1e10]
    both = [c for c in scored
            if (c.get("quality") or 0) > 60 and (c.get("price_score") or 0) > 50]
    qualifying = [c for c in companies if c.get("qualifies")]
    growth_ok = [c for c in companies if c.get("growth_qualifies")]
    spec_ok = [c for c in companies if c.get("speculative_qualifies")]
    near = [c for c in companies
            if not c.get("qualifies")
            and (c.get("gates_total") or 0) - (c.get("gates_passed") or 0) == 1]

    # The funnel is the point of the screen as much as the survivors are: it
    # says which condition the market is failing today. When almost nothing
    # qualifies, this is the answer, not an empty list.
    with (out_dir / "screen.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["screen", "step", "key", "label", "why_it_fails",
                         "survivors", "eliminated"])
        for screen, gates in SCREENS.items():
            alive = list(companies)
            for step, gate in enumerate(gates, start=1):
                before = len(alive)
                failed_key = f"{screen}_failed"
                alive = [c for c in alive
                         if gate.key not in (c.get(failed_key) or "").split("|")]
                writer.writerow([screen, step, gate.key, gate.label, gate.why,
                                 len(alive), before - len(alive)])

    with (out_dir / "counts.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value", "note"])
        for row in (
            ("total", len(companies), "companies with usable filings and a price"),
            ("scored", len(scored), "enough data to produce an overall score"),
            ("grade_ab", len(top), "scored 68 or better overall"),
            ("smaller_ab", len([c for c in top if c in smaller]), "of those worth under $10bn"),
            ("smaller_total", len(smaller), "companies worth under $10bn"),
            ("quality_and_price", len(both), "quality above 60 and price above 50"),
            ("qualifying", len(qualifying), "clear every value condition"),
            ("growth_qualifying", len(growth_ok), "clear every growth condition"),
            ("speculative_qualifying", len(spec_ok), "clear every speculative condition"),
            ("near_miss", len(near), "fail exactly one condition"),
            ("as_of", record["generated_at"][:10], "snapshot date"),
        ):
            writer.writerow(row)

    if regime:
        with (out_dir / "regime_components.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["component", "score", "weight"])
            for key, value in regime["components"].items():
                writer.writerow([key.title(), round(value, 1), regime["weights"].get(key)])

        readings = [
            ("Index versus its one-year average", regime["trend"].get("dist_sma200"), "pct"),
            ("Index below its recent high", regime["trend"].get("drawdown_from_high"), "pct"),
            ("Index return over 12 months", regime["trend"].get("return_12m"), "pct"),
            ("Share of large companies rising", regime["breadth"].get("pct_above_sma200"), "pct"),
            ("Extra interest risky borrowers pay", regime["credit"].get("hy_oas"), "pp"),
            ("  change over three months", regime["credit"].get("hy_oas_chg_3m"), "pp"),
            ("Expected market swings (VIX)", regime["rates"].get("vix"), "level"),
            ("Government 10-year interest rate", regime["rates"].get("ust_10y"), "pp"),
            ("Unemployment rate", regime["rates"].get("unemployment"), "pp"),
        ]
        with (out_dir / "regime_readings.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["measure", "value", "unit"])
            for measure, value, unit in readings:
                writer.writerow([measure, "" if value is None else round(value, 4), unit])

        with (out_dir / "regime_summary.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["metric", "value", "note"])
            writer.writerow(["label", regime["label"], "overall market backdrop"])
            writer.writerow(["score", round(regime["score"], 1), "out of 100"])
            writer.writerow(["exposure", round(regime["exposure"], 2), "suggested exposure multiplier"])
            writer.writerow(["as_of", record["generated_at"][:10], "snapshot date"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="./desk", help="directory for the CSVs and JSON")
    parser.add_argument("--limit", type=int, default=515, help="how many companies")
    parser.add_argument("--symbols", default="", help="comma-separated override universe")
    parser.add_argument("--fresh", action="store_true", help="ignore any saved progress")
    parser.add_argument("--save-every", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    record_path = out_dir / "snapshot.json"

    record = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "companies": [], "errors": []}
    done: set[str] = set()
    if record_path.exists() and not args.fresh:
        try:
            record = json.loads(record_path.read_text())
            done = {c["symbol"] for c in record.get("companies", [])}
            log.info("resuming: %d companies already analysed", len(done))
        except (ValueError, KeyError):
            log.warning("saved progress unreadable; starting fresh")

    log.info("market regime...")
    # One network hiccup here used to cost the run its whole market-backdrop
    # section: the failure was caught, logged at warning level and never
    # mentioned again, so the build reported success having written three
    # fewer files than it should have.
    state = None
    for attempt in range(1, 4):
        try:
            state = assess("SPY")
            break
        except Exception as exc:  # noqa: BLE001 - an outage must not stop the run
            log.warning("regime attempt %d/3 failed: %s", attempt, exc)
            if attempt < 3:
                time.sleep(2 ** attempt)
    if state is not None:
        record["regime"] = {
            "label": state["label"], "score": num(state["composite_score"]),
            "exposure": num(state["exposure_multiplier"]),
            "components": {k: num(v) for k, v in state["components"].items()},
            "weights": {k: num(v) for k, v in state["weights"].items()},
            "trend": {k: num(v) for k, v in state["trend"].items()},
            "breadth": {k: num(v) for k, v in state["breadth"].items()},
            "credit": {k: num(v) for k, v in state["credit"].items()},
            "rates": {k: num(v) for k, v in state["rates"].items()},
        }
    else:
        record["regime"] = None

    if args.symbols:
        universe = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        log.info("building universe...")
        universe = build_universe(args.limit)
    todo = [s for s in universe if s not in done]
    log.info("universe %d | to analyse %d", len(universe), len(todo))

    for index, symbol in enumerate(todo, 1):
        try:
            row = analyse(symbol)
            if row is None:
                record["errors"].append(f"{symbol}: no usable data")
            else:
                record["companies"].append(row)
        except Exception as exc:  # noqa: BLE001 - one bad name must not stop the run
            record["errors"].append(f"{symbol}: {type(exc).__name__}: {exc}")
        if index % args.save_every == 0 or index == len(todo):
            record["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            record_path.write_text(json.dumps(record, separators=(",", ":")))
            log.info("[%d/%d] %s — saved", index, len(todo), symbol)

    write_csvs(out_dir, record)
    scored = [c for c in record["companies"] if c.get("overall") is not None]
    log.info("\ndone: %d companies (%d scored), %d errors",
             len(record["companies"]), len(scored), len(record["errors"]))
    log.info("wrote %s", ", ".join(sorted(p.name for p in out_dir.glob("*.csv"))))
    if not record.get("regime"):
        # Say it at the end, where it will be read. A run that quietly ships
        # three fewer files than usual is worse than one that fails.
        log.error("INCOMPLETE: the market backdrop could not be fetched, so "
                  "regime_components.csv, regime_readings.csv and "
                  "regime_summary.csv were not written. Re-run before "
                  "publishing, or the page keeps the previous backdrop.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
