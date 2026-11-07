# stocklab

Equity research, valuation, screening and multi-account portfolio management.

Built on primary sources: **SEC EDGAR XBRL** for fundamentals (the numbers
companies actually filed, with filing dates attached, so point-in-time
backtests are possible), Yahoo's chart endpoint for prices and corporate
actions, and FRED for the macro inputs that drive discount rates.

> Not investment advice. Every model here encodes assumptions that are stated
> in `docs/METHODOLOGY.md`. Read them before trusting a number.

## Install

```bash
pip install -e ".[dev]"
```

SEC requires automated clients to declare a contact address. Override the
default with:

```bash
export STOCKLAB_USER_AGENT="stocklab/0.1 (you@example.com)"
```

## Status

Under construction. Working so far: data layer, fundamentals assembly, metrics.
