# Methodology

Every number this tool produces rests on assumptions. This document states
them, says where each model breaks, and is honest about what the evidence
supports. Read the section for any output you intend to act on.

---

## 1. Data

### Sources and why these ones

| What | Source | Why |
|---|---|---|
| Fundamentals | SEC EDGAR XBRL (`companyfacts`) | The numbers companies actually filed, with filing dates attached. Vendor APIs silently overwrite restatements, which makes point-in-time analysis impossible. |
| Prices, dividends, splits | Yahoo Finance chart endpoint | Free, long history, carries corporate actions. Unofficial and occasionally wrong, so it is used for **prices only**. |
| Macro | FRED | Risk-free rates, credit spreads, yield curve. Authoritative and free. |
| Symbol universe | Nasdaq Trader symbol directory | Covers all US exchanges and flags ETFs, test issues and non-compliant issuers. |

SEC requires automated clients to declare a contact in the `User-Agent`; set
`STOCKLAB_USER_AGENT` to override the default. Requests are rate-limited well
below the published cap, because the penalty for exceeding it is an IP block.

### Known data problems, and what is done about them

These are not hypothetical. Each was found by validating against real filings.

**Tag migration.** XBRL is a vocabulary, not a schema. NVIDIA reported revenue
under `RevenueFromContractWithCustomerExcludingAssessedTax` through FY2022 and
`Revenues` afterwards. Taking the first tag with data returns a series that
stopped four years ago. Tags are therefore *merged* in priority order, with
higher-priority tags winning any period they cover and lower ones filling gaps.
`Fundamentals.sources` records which tags produced each concept.

**Q4 is almost never filed.** US filers report Q1 discrete, then half-year,
nine-month and full-year cumulatives. Quarterly series are rebuilt by interval
arithmetic: where the quarters already known tile a longer period except for
one contiguous gap, that gap is the remainder. Verified to reproduce reported
annual totals exactly.

**Fiscal periods have no grid.** Different tags close the same period on dates
days apart, and 52/53-week calendars drift. End dates are clustered onto a
canonical grid built from *flow* periods only — including instants lets a 10-K
cover-page date three weeks past year end hijack the cluster label, which made
Apple's fiscal year appear to end in October.

**Restated values versus publication dates.** A period's *value* should come
from the latest filing (restatements supersede), but its *availability* must
come from the earliest. A 10-Q carries prior-year comparatives, so reusing the
latest filing date as the publication date back-dates a full year of hindsight
into every point-in-time series. Both are tracked separately.

**Corporate reorganisations split history across CIKs.** ExxonMobil's 2026
holdco reorganisation moved the XOM ticker to a new registrant whose XBRL
history begins mid-2025; the preceding decades sit on a CIK that no longer
carries a ticker. Where a listed company has under three years of filings, the
predecessor is sought and its filings stitched in. `chained_ciks` always
discloses when this happened, because consolidation scope may differ.

**Split-adjusted prices versus as-filed share counts.** Yahoo's close is
adjusted to today; SEC share counts are as-filed. Multiplying them put Apple's
2016 market cap at $107bn instead of about $600bn and dragged the 10th
percentile of its P/E band down to 4x. Historical share counts are restated
onto today's split basis before use.

**Per-share history straddles two split bases.** Filings restate only the
comparative years they reproduce, so Broadcom's diluted share count reads 429m
for FY2021 and 4,232m for FY2022 — a 10-for-1 split, not a 10x issuance — and
produced a 63%-a-year share-count growth rate. The break sits at the
*restatement* boundary, not the split date, so it is located by matching
adjacent-pair ratios against splits the company actually did.

**Multi-class share counts.** `companyfacts` omits dimensional facts, so a
multi-class registrant's cover page returns one class or nothing: Mastercard
reads 122m against 884m diluted shares. Three sources are reconciled — cover
page, diluted weighted average, and net income divided by diluted EPS — and the
cover page is used only when it corroborates one of the others within 15%.
Names where no corroboration is possible carry
`share_count_reliable = 0` and are excluded from valuation screens, so a data
gap cannot win a value ranking by looking cheap.

**Missing subtotals.** Exxon tags no `OperatingIncomeLoss` at all, which leaves
operating margin undefined and silently disables every EBIT-based model. Where
an accounting identity recovers a figure exactly it is used, and the result is
marked in a `derived_<name>` column.

**Dividends.** XBRL dividend tagging has real gaps — Simon Property Group has
tagged no standard dividend concept since 2019, which reads as no dividend — so
the exchange's own dividend record is preferred. Note that Yahoo caps its event
list around 170 entries and drops the middle of a long history, so dividends
are fetched over a bounded ten-year window and splits over the full history.

### What is still wrong

- **No point-in-time index constituents.** The universe is today's listed
  symbols, so every backtest is survivorship-biased. This typically inflates
  returns by 1–4% a year and cannot be fixed without licensed data.
- **No analyst estimates.** Every forward input is derived from history, which
  is a materially weaker basis than consensus for near-term forecasting.
- **Sector classification is SIC, not GICS.** SIC is free; it is coarse and
  occasionally wrong. Managed care (632x) is reclassified to Health Care by
  hand because its SIC range would otherwise make UnitedHealth a bank.
- **Company extension tags are invisible.** A filer using a custom namespace
  for a line item simply has no value for it.
- **Spin-off basis allocation is not modelled.** Correct treatment needs the
  issuer's Form 8937. The ledger warns rather than guessing.

---

## 2. Valuation

### Cost of capital

Cost of equity is CAPM: `risk-free + beta x ERP + size premium`.

- **Risk-free** is the 10-year Treasury, chosen to roughly match the duration
  of the cash flows being discounted. Using the 3-month bill makes every DCF a
  bet on the front end of the curve.
- **ERP defaults to 4.8%**, in the range of Damodaran's implied estimates. This
  is the single most consequential assumption in the entire tool and it is an
  assumption, not a measurement. Override it deliberately.
- **Betas are shrunk toward 1.0** (Blume, 1975: `0.67 x raw + 0.33`). Raw
  regression betas have standard errors frequently above 0.2 and mean-revert;
  a raw beta of 2.4 from a two-year window describes the past, not the future.
  Estimated from five years of weekly returns — daily returns pick up bid-ask
  bounce and non-synchronous trading, monthly leaves too few observations.
- **Cost of debt** comes from a synthetic rating mapped from interest coverage,
  not from the existing coupon. What matters is what the company would pay to
  borrow *now*. The spread table is approximate and drifts with the credit
  cycle.
- **WACC is clamped to 6–20%.** No business funds itself at 3%, and a computed
  WACC outside that band indicates a broken input rather than an insight.

**Where this breaks:** CAPM is a poor empirical description of the
cross-section of returns. Beta does not explain returns well, and the size
premium has been weak since the 1980s once profitability is controlled for.
The size premium here is a modest haircut for illiquidity and estimation risk,
not a factor claim. For a financial company, a debt-inclusive WACC is
meaningless and the cost of equity should be used directly.

### Discounted cash flow

Two-stage FCFF with a linear growth and margin fade over an explicit horizon,
then a Gordon terminal value.

**Terminal reinvestment is enforced at `g / ROIC`.** A business growing at 2.5%
forever must keep funding that growth. Projecting perpetual growth with no
perpetual reinvestment is free money and inflates terminal value — and
therefore fair value — by 20–40% for a typical company. This is the single most
common error in spreadsheet DCFs.

**Terminal growth is capped at the risk-free rate.** The ten-year yield is a
market-implied estimate of long-run nominal growth, and nothing can outgrow the
economy in perpetuity without becoming it.

**Terminal ROIC fades halfway from today's ROIC to the cost of capital.**
Competition erodes excess returns; assuming a 90% ROIC persists forever is the
other common way to produce a silly valuation.

**Starting assumptions are shrunk toward the mean.** Growth blends the 3-year
CAGR with a log-trend fit and is then pulled 35% toward terminal growth, because
the empirical half-life of excess revenue growth is roughly five years. Margins
blend current with a five-year average. Both are capped.

**Where this breaks:** anything with a terminal share above about 75% is not a
valuation of a forecast, it is a claim about year 11 onward, and the tool says
so. DCF does not apply to financials (debt is raw material) or REITs (non-cash
depreciation and acquisition-led capex make FCFF understate value badly — it
valued a healthy triple-net trust at a quarter of its price). Both are routed
elsewhere.

### Reverse DCF — the most useful output here

A forward DCF asks "what is this worth?", which requires a forecast and is
therefore easy to dismiss. A reverse DCF asks "what would have to be true for
today's price to be correct?", and the price is observable.

**The headline is the implied IRR**, because it needs no equity risk premium
assumption: it is simply the internal rate of return of the projected cash
flows at the market price. *"You are being offered 5.2% a year, against 4.8% on
a Treasury"* is a complete argument that requires no agreement about what
equities should return.

**Implied near-term growth is far more fragile** and is reported with that
caveat. It solves against a fixed terminal rate, and since terminal value is
usually half the total, the near-term knob must move a long way to close any
gap. A 35% reading means "the forecast period alone cannot explain the price",
not that anyone expects 35% growth.

### Earnings Power Value (Greenwald)

Normalised, no-growth earnings capitalised at the cost of capital. Harder to
fool than a DCF because there is no growth forecast and no terminal value.

Its real value is the comparison with **asset reproduction value** — what a
competitor would spend to rebuild the business. EPV above reproduction value
means the business earns more than its assets should permit, which in a
competitive market requires a barrier to entry; the gap *is* the franchise
value. A DCF cannot perform this test because it assumes the moat.

**Where this breaks:** PP&E sits at historical cost, so reproduction value is
understated for anything capital-intensive and the franchise ratio overstates
any advantage. The tool flags balance sheets where PP&E exceeds 35% of assets.
Capitalised R&D (three years) is a crude stand-in for the intangible base.

### Residual income

Book value plus the present value of excess returns on equity. Structurally
more stable than a DCF because most of the value sits in an *observed* number —
today's book value — rather than a terminal value a decade out.

This is the primary model for **financials**, where FCFF has no meaning but
book value and ROE are exactly the terms in which banks are analysed. Excess
returns fade to the cost of equity over the forecast and then decay at a
persistence factor (default 0.6, roughly a two-year half-life), consistent with
the observed speed of ROE mean-reversion.

### Dividend discount

Gated by applicability: rejected for non-payers, for companies returning
capital mainly through buybacks (which a dividend model cannot see), and where
the payout is unsustainable. For REITs the payout test uses **FFO, not net
income** — a REIT distributing 90% of FFO looks like a 180% payout on GAAP
earnings because depreciation on appreciating property sits in between, and
judging it on net income rejects the one sector the model fits best.

### Graham's screens

Included because they still work as filters and are transparent. Two warnings:

- The **Graham formula** `V = EPS x (8.5 + 2g) x 4.4/Y` is the most misused
  equation in value investing. Graham published it as a *description* of how
  the market had historically priced growth and explicitly warned against using
  it to value growth stocks. Growth is capped at 15% here.
- **Net-net investing** showed large excess returns historically (Oppenheimer
  1986; Xiao & Arnold 2008) and still does in backtests, but the modern
  survivor set is dominated by shells and companies consuming the very current
  assets being counted. It needs a basket of 20–30 names; the returns are a
  property of the group, not of any member.

### The composite

Reports a **range and model dispersion**, not a point estimate. When DCF, EPV
and residual income disagree threefold, the disagreement *is* the finding and
the blended midpoint is noise. Models are weighted by whether they fit the
business, not by preference.

---

## 3. Screening

Factors are winsorized before z-scoring — equity multiples are fat-tailed
enough that one company with a one-cent EPS and a P/E of 9,000 will otherwise
compress every other name into a single bucket. Robust (median/MAD) z-scores
are the default. Ranking within sector is available and is almost always right
for multiples: a utility on 18x and a software company on 18x are not equally
priced.

**Missing data is a per-filter decision.** Rejecting on absence biases a screen
toward complete disclosure, which skews large, US-domiciled and mature; passing
on absence lets through names that should have been caught. Neither default is
right, so each filter declares which it wants and every rejection is attributed.

Each preset ships with its published evidence and its failure modes, which are
printed with the results. Summary of the honest position on each:

- **Magic Formula** — Greenblatt reported ~30% annualised 1988–2004, largely
  from small and micro caps where it is hard to trade. Ranks on trailing EBIT,
  so it systematically buys cyclical peaks.
- **Value (deep value, Piotroski)** — the value premium is among the best
  documented effects in asset pricing, and has been weak or negative in US
  large caps for much of the period since 2007. P/B is increasingly
  uninformative where value is intangible.
- **Quality** — gross profitability (Novy-Marx 2013) and accruals (Sloan 1996)
  carry robust premia and, unlike deep value, have held up in large caps. The
  screen has no valuation constraint and will happily return excellent
  businesses at indefensible prices.
- **Momentum** — the most persistent anomaly on record (Jegadeesh & Titman
  1993), across asset classes and more than a century. It also crashes: roughly
  −60% in 2009, clustered precisely when risk appetite returns. High turnover
  makes it tax-inefficient enough in a taxable account to consume the premium.
- **Dividend growth** — growers have beaten both non-payers and the
  highest-yielding quintile, which is dominated by distressed payers about to
  cut. Carries unintended rate duration.

---

## 4. Market regime

Scored from index trend, breadth, credit spreads, volatility and macro, weighted
by the strength of the evidence behind each. Credit and trend carry the most;
the yield curve carries 8% despite its reputation, because its 12–24 month lead
time is useless for sizing a position today.

**The output is an exposure multiplier bounded to 0.60–1.10, not an in/out
signal.** This is deliberate. Nobody times markets reliably, and a model that
can send exposure to zero eventually will — whipsaw costs and missed recoveries
swamp the drawdown avoided. What regime signals *do* have documented predictive
content for is drawdown risk, which belongs in a sizing decision.

Trend components are graded rather than binary: a binary "above the 200-day"
test scores identically whether price is 0.4% or 15% above, producing a cliff
exactly where the signal is least reliable.

---

## 5. Portfolio and tax

### Lot accounting

Transactions are the source of truth; lots, positions, cash, realised gains and
performance are recomputed by replaying them. The database cannot hold a
derived value that disagrees with the trades behind it, and a mistyped trade
from five years ago can be corrected with everything downstream following.

Cost basis supports FIFO, LIFO, HIFO, average and specific-lot. **FIFO is the
broker default and usually the worst choice** in an appreciated position,
because the oldest lot has the lowest basis; HIFO minimises the realised gain
but consumes long-term lots last, so it can convert long-term gains into
short-term ones. There is no universally correct method, only the one you chose
deliberately.

### Wash sales

Detected across **every account**, including retirement accounts. The case most
often missed: when the replacement purchase lands in an IRA or 401(k) the loss
is **permanently** disallowed rather than deferred, because a sheltered account
has no taxable basis for the adjustment to attach to (Rev. Rul. 2008-5).

Not modelled: "substantially identical" across different securities (two S&P
500 funds from different issuers is a grey area the IRS has never resolved),
options, convertibles, constructive sales, straddles.

### Tax

Short versus long on the *more than* 12 months rule — exactly 365 days is still
short-term, and that off-by-one is the difference between ordinary rates and
0/15/20%. Models the $3,000 annual limit on net capital losses against ordinary
income with indefinite carry-forward.

**Harvesting inside a retirement account accomplishes nothing**: the loss is not
deductible and basis is irrelevant to how withdrawals are taxed. Candidates are
drawn from taxable accounts only.

Harvesting resets basis lower, so it **defers** tax rather than eliminating it.
The benefit is the time value of the deferral plus any rate difference, not the
headline saving.

**Federal estimates only. State tax is not modelled** — Wisconsin, for example,
taxes capital gains as ordinary income with a 30% exclusion for assets held
over a year, which no federal-only figure captures. Not tax advice; reconcile
against broker records before filing.

### Asset location

Put heavily taxed assets where they are shielded and tax-efficient assets in
the taxable account. Two corrections the naive version gets wrong: **municipal
bonds in an IRA** waste their exemption entirely, and **a taxable account is
what gives you loss harvesting and the step-up in basis at death** — neither
exists in a sheltered account, and equities benefit most from both.

### Rebalancing

Trades only when a position breaches **both** a relative (25%) and an absolute
(5-point) drift band. Trading on every deviation generates turnover for noise.
Sells route to sheltered accounts first, where a sale costs nothing, and within
a taxable account to the smallest-gain lots. A tax budget caps the bill and
leaves drift uncorrected rather than exceeding it. A plan that sells at a loss
and buys the same name back is flagged as the wash sale it is.

### Performance

Both time-weighted and money-weighted returns, with the gap called out.
TWR measures the strategy; MWR measures the investor's own timing. Reporting
one alone is how people convince themselves they are doing better than they
are — someone who added heavily at a bottom earns an MWR well above their TWR.

---

## 6. Entry timing

Valuation says what to own; entry setups say when a price offers favourable
risk-to-reward, which is a much weaker claim. The honest framing is that entry
timing mostly controls **risk per trade** rather than expected return: a defined
stop lets you size a position properly, which matters more than the entry price.

The composite weights valuation above the technical setup because the evidence
for buying a business cheaply is far better than for any chart pattern. Of the
four setups, pullback-to-trend and breakout-on-volume describe real supply
conditions; the volatility-contraction pattern is a pattern-recognition
heuristic with no published excess-return evidence of its own.

**Place the stop where the thesis is wrong, then size to it.** Choosing a size
first and putting the stop where the loss feels tolerable guarantees being
stopped out by noise, because the level has nothing to do with the stock's
volatility. Everything is measured in **R** (entry-to-stop distance): a system
with a 40% hit rate and 3R winners is excellent; one with an 80% hit rate and
0.2R winners is a slow loss.

---

## 7. Backtesting

Four biases are addressed explicitly rather than assumed away.

1. **Look-ahead.** The one-bar weight shift is mandatory and cannot be
   disabled. Verified: a rule ranking on the day's realised return shows a
   2,236% CAGR applied unshifted and −2.2% through the engine.
2. **Costs.** Commission and slippage charged on turnover, with gross and net
   returns reported side by side. A daily-rebalanced strategy with 5bp of
   slippage pays roughly 12% a year in friction.
3. **Survivorship.** Stated in every result. Cannot be corrected here.
4. **Point-in-time fundamentals.** `as_of_filter` exists for this; a fundamental
   backtest that does not use it is measuring hindsight.

Metrics lead with drawdown depth, duration and time underwater rather than
CAGR, because the drawdown decides whether anyone held the strategy long enough
to collect the return. **Deflated Sharpe** subtracts the expected maximum from
N trials: with 50 variants on five years of data, a Sharpe near 1.0 arises from
selection alone, so any backtest Sharpe quoted without disclosing the number of
trials behind it is close to meaningless.

What this is good for: comparing two rules on the same data with the same
costs. What it is not good for: producing a number to show someone.

---

## 8. What this tool cannot do

- It cannot tell you what a business will earn. Every forward input is
  extrapolated from filed history.
- It has no qualitative judgement: management quality, competitive dynamics,
  regulatory risk, technological disruption, accounting aggressiveness beyond
  what the Beneish inputs capture.
- It cannot value a pre-revenue company, a biotech whose value is one trial
  readout, or anything whose worth rests on an option rather than cash flows.
- It does not know about pending litigation, an acquisition announced
  yesterday, or a going-concern qualification in the filing's footnotes.
- It is not a substitute for reading the 10-K.

## References

Altman (1968) on the Z-Score; Ball, Gerakos, Linnainmaa & Nikolaev (2016) on
cash-based profitability; Bailey & López de Prado (2014) on the deflated Sharpe
ratio; Beneish (1999) on earnings manipulation; Bessembinder (2018) on the
skewness of long-run stock returns; Blume (1975) on beta mean-reversion;
Damodaran on cost of capital, implied ERP and terminal-value discipline;
Fama & French (1992) on value and size; Greenblatt (2005) on the Magic Formula;
Greenwald et al. (2001) on earnings power and asset reproduction value;
Jegadeesh & Titman (1993) on momentum; Novy-Marx (2013) on gross profitability;
Piotroski (2000) on the F-Score; Sloan (1996) on accruals.
