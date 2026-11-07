# stocklab

Equity research, valuation, screening and multi-account portfolio management,
built on primary sources.

Fundamentals come from **SEC EDGAR XBRL** — the numbers companies actually
filed, with filing dates attached, so point-in-time analysis is possible and
restatements are visible. Prices and corporate actions come from Yahoo's chart
endpoint, macro inputs from **FRED**, and the symbol universe from Nasdaq
Trader.

> **Not investment advice.** Every model here encodes assumptions, and they are
> stated in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) along with the
> conditions under which each one breaks. Read the relevant section before
> acting on any output.

---

## Install

```bash
pip install -e ".[dev]"
```

SEC's fair-access policy requires automated clients to declare a contact
address. Override the default with:

```bash
export STOCKLAB_USER_AGENT="stocklab/0.1 (you@example.com)"
```

Data lives in `~/.stocklab` by default (`STOCKLAB_HOME` or `--data-dir` to
change it): a SQLite cache for API payloads and a separate SQLite database for
accounts and transactions.

---

## Valuation Desk (published page)

The page most people will actually use is published at
<https://claude.ai/artifact/XFFePLr1ynhKV6wJq8xEA7>. It ranks 515 US-listed
companies — mega cap down to about $60m — on quality, price, growth and safety,
explains each score in plain English, lets anyone look up a ticker or company
name, and shows the market backdrop. Source and rebuild notes are in
[`dashboard/`](dashboard/README.md); the snapshot behind it is built by
`scripts/build_desk_snapshot.py` and rebuilt weekly.

---

## Web app (local)

```bash
pip install -e ".[web]"
stocklab serve
```

Opens at `http://localhost:8787`: dashboard, company analysis, portfolio by
account, performance, risk, tax and harvesting, screens and regime. It reads
your real portfolio database on every request, so a trade recorded a minute ago
is on the next page load. Every page has a matching `/api/...` JSON endpoint.

`--host 0.0.0.0` makes it reachable from a phone on the same network. There is
no authentication and the pages show your actual holdings, so that is fine on a
home network and not on shared wifi.

## Analysing a company

```bash
stocklab quote AAPL          # one-screen snapshot
stocklab analyse AAPL        # fundamentals + valuation range + entry plan
stocklab value AAPL          # DCF projection and sensitivity grid
stocklab reverse AAPL        # what the current price already assumes
stocklab entry AAPL          # setups, stop placement, position size
```

`analyse` runs every applicable valuation model, reports a **range and the
dispersion between models**, and routes by business type — FCFF and EPV are
excluded for banks (debt is raw material, not financing) and for REITs
(non-cash depreciation makes FCFF understate them badly).

Its most useful single number is the **implied equity risk premium**: the IRR
the market price offers on the projected cash flows, minus the risk-free rate.
That figure needs no assumption about what equities *should* return, which
makes it the one output you do not have to agree with the model to use.

```bash
# What growth would the market need at each price?
stocklab reverse MSFT --prices 300,400,500,600
```

## Screening

```bash
stocklab screen --list-presets
stocklab screen quality
stocklab screen magic-formula --universe listed --max-symbols 500
stocklab screen deep-value --symbols AAPL,MSFT,JPM,XOM
```

Ten presets, each printing its published **evidence** and its **failure modes**
with the results, plus an attrition funnel showing where names were filtered
out. The first run over a large universe is slow (rate limits); everything is
cached, so later runs are fast.

Available: `magic-formula`, `deep-value`, `net-net`, `quality`, `garp`,
`dividend-growth`, `piotroski-value`, `momentum`, `fcf-machines`, `cannibals`.

## Market regime

```bash
stocklab regime
```

Scores trend, breadth, credit spreads, volatility and macro into an **exposure
multiplier bounded to 0.60–1.10** — a sizing input, not an in/out signal. A
model that can send exposure to zero eventually will, and whipsaw costs plus
missed recoveries swamp the drawdown it avoided.

---

## Portfolios across multiple accounts

Account type drives behaviour rather than labelling it: whether a gain is
taxable, whether a loss is harvestable at all, whether a wash sale is deferred
or permanently destroyed, and where an asset belongs.

```bash
stocklab account add "Schwab Taxable" --type taxable --basis-method hifo
stocklab account add "Vanguard Roth"  --type roth_ira
stocklab account add "Fidelity 401k"  --type traditional_401k
stocklab account list
```

Record trades directly or import a broker CSV (column names are matched
case-insensitively, with common broker aliases accepted):

```bash
stocklab txn add --account "Schwab Taxable" --date 2024-03-01 \
                 --type buy --symbol AAPL --quantity 100 --price 170.50

stocklab txn import trades.csv --account "Schwab Taxable" --dry-run
```

Transactions are the source of truth. Lots, positions, cash, realised gains and
performance are **recomputed by replaying them**, so the database can never
hold a derived value that disagrees with the trades behind it — and correcting
a mistyped trade from five years ago makes everything downstream follow.

```bash
stocklab portfolio --by-account
stocklab performance                 # TWR and MWR side by side
stocklab risk                        # concentration, correlation, stress
```

`performance` reports **both** time-weighted and money-weighted returns and
names the gap. TWR measures the strategy; MWR measures your own timing.
Reporting one alone is how people convince themselves they are doing better
than they are.

### Rebalancing

```bash
stocklab rebalance --target AAPL=0.15 --target MSFT=0.15 \
                   --target VTI=0.50 --target CASH=0.20 --save

stocklab rebalance --tax-budget 2000
```

Trades only when a position breaches **both** a relative (25%) and an absolute
(5-point) drift band. Sells route to sheltered accounts first, where they cost
nothing, and within a taxable account to the smallest-gain lots. A tax budget
caps the bill and leaves drift uncorrected rather than exceeding it.

### Tax

```bash
stocklab tax --year 2026 --short-rate 0.24 --long-rate 0.15
stocklab harvest --min-loss 1000
```

Covers short/long classification on the *more than* 12 months rule, the $3,000
ordinary-income limit with carry-forward, and wash sales across every account —
including the case most often missed, where a replacement purchase inside an
IRA destroys the loss **permanently** rather than deferring it.

Harvest candidates come from taxable accounts only. Selling at a loss inside an
IRA accomplishes nothing: the loss is not deductible and basis is irrelevant to
how withdrawals are taxed.

Federal estimates only; state tax is not modelled. Not tax advice.

---

## Backtesting

```python
from stocklab.data.providers.yahoo import fetch_closes, fetch_history
from stocklab.backtest import engine, metrics

prices = fetch_closes(["AAPL", "MSFT", "JNJ", "XOM", "JPM"], period="10y")
momentum = prices.shift(21) / prices.shift(252) - 1.0

result = engine.run(
    prices,
    engine.weights_from_ranks(momentum, top_n=3, frequency="M"),
    benchmark=fetch_history("SPY", period="10y")["adj_close"],
)
print(metrics.compare({"momentum": result}))
print(result.warnings)          # survivorship, turnover, sample length
```

The one-bar weight shift is **mandatory and cannot be disabled**: a signal
derived from a close is never traded at that close. Costs are charged on
turnover with gross and net reported side by side, and survivorship bias is
stated in every result because it cannot be corrected without licensed
point-in-time constituents.

`metrics.deflated_sharpe` subtracts the expected maximum from N trials. With 50
variants on five years of data, a Sharpe near 1.0 arises from selection alone.

---

## Library use

```python
from stocklab.company import Company
from stocklab.valuation import composite

company = Company.load("AAPL")
report = composite.value(company.fundamentals, company.metrics, company.price)

print(report.to_frame())            # every model, its value, its weight
print(report.implied_equity_premium)
print(report.notes)                 # the caveats that apply to this name
```

Layout:

```
stocklab/
  web/         local Starlette app: pages, JSON API, templates, static
  data/        caching, rate-limited HTTP, SEC/Yahoo/FRED providers, universe
  analysis/    metrics, technicals, risk, market regime
  valuation/   wacc, dcf, reverse_dcf, epv, residual_income, ddm, graham,
               multiples, composite
  screening/   factor matrix, declarative screener, strategy presets
  portfolio/   models, store, ledger, performance, tax, rebalance, sizing, risk
  entry/       setups, stops, trade planning
  backtest/    engine, metrics
  reporting/   console rendering
```

---

## Tests

```bash
pytest
```

253 tests, all offline against synthetic fixtures with the HTTP session patched
to fail. They concentrate on the invariants that would otherwise fail silently:
derived Q4 figures summing back to reported annual totals, terminal
reinvestment materially changing a DCF, splits preserving total cost basis, the
one-bar shift destroying look-ahead, Wilder smoothing matching its definition,
and gross exposure never exceeding 1.0.

## Limitations

Read [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §8. In short: no analyst
estimates, no point-in-time index constituents (so backtests are
survivorship-biased), SIC rather than GICS sectors, no qualitative judgement,
and no substitute for reading the filing.
