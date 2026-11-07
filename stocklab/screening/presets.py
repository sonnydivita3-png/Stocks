"""Named screening strategies, each shipped with its evidence and its failure modes.

Every preset carries an ``evidence`` string and a ``failure_modes`` list that
are printed with its results. This is not decoration. A screen presented
without the conditions under which it fails is a device for manufacturing
confidence, and most of these strategies have well-documented decades of
underperformance inside their long-run records.

Thresholds are the published ones where published (Greenblatt's ROC and
earnings yield, Graham's two-thirds of NCAV, Piotroski's F >= 8, Minervini's
trend template). Where a strategy has no canonical cut-off, the threshold is
chosen to produce a workable shortlist and is noted as arbitrary.
"""
from __future__ import annotations

from .screener import Filter, Screen, liquidity_filters, sanity_filters


def magic_formula(min_market_cap: float = 5e8) -> Screen:
    return Screen(
        name="Magic Formula (Greenblatt)",
        description="High return on capital bought at a high earnings yield.",
        filters=[
            *sanity_filters(),
            *liquidity_filters(min_market_cap),
            Filter("ebit_ev_yield", ">=", 0.08, label="EBIT/EV >= 8%"),
            Filter("roic", ">=", 0.15, label="ROIC >= 15%"),
            Filter("ebit", ">", 0, label="positive EBIT"),
            # Greenblatt excluded financials and utilities explicitly: ROC is
            # not comparable where the balance sheet is the business, and
            # regulated returns make the ranking meaningless.
            Filter("sector", "notin", ["Financials", "Utilities", "Real Estate"],
                   label="exclude financials, utilities, REITs"),
        ],
        score_weights={"ebit_ev_yield": 1.0, "roic": 1.0},
        evidence=(
            "Greenblatt's 'The Little Book That Beats the Market' reported ~30% "
            "annualised 1988-2004 on a US universe above $50m market cap."
        ),
        failure_modes=[
            "Published results came largely from small and micro caps; the "
            "effect is materially weaker in large caps where it is tradeable.",
            "Greenblatt himself documented three-year stretches of "
            "underperformance within the backtest. The strategy requires "
            "holding through them, which is where most implementations fail.",
            "Ranks on trailing EBIT, so it systematically buys cyclical peaks: "
            "a homebuilder at the top of the cycle screens beautifully.",
        ],
    )


def deep_value(min_market_cap: float = 3e8) -> Screen:
    return Screen(
        name="Deep value",
        description="Cheap on assets and earnings, with enough balance sheet to survive.",
        filters=[
            *sanity_filters(),
            *liquidity_filters(min_market_cap),
            Filter("ev_ebit", "between", (0.5, 10.0), label="EV/EBIT between 0.5 and 10"),
            Filter("pb", "<=", 1.5, label="P/B <= 1.5"),
            Filter("fcf", ">", 0, label="positive free cash flow"),
            Filter("net_debt_to_ebitda", "<=", 3.0, required=False,
                   label="net debt / EBITDA <= 3"),
            # Piotroski's central finding: within cheap stocks, the financially
            # weak half is where the losses are.
            Filter("piotroski_f", ">=", 5.0, required=False, label="F-Score >= 5"),
        ],
        score_weights={"ev_ebit": 1.0, "pb": 0.8, "fcf_yield": 1.0, "piotroski_f": 0.6},
        rank_within_sector=True,
        evidence=(
            "The value premium is one of the most robustly documented effects in "
            "asset pricing (Fama-French 1992 onward), though it has been weak or "
            "negative in US large caps for much of the period since 2007."
        ),
        failure_modes=[
            "Cheapness on trailing figures often means earnings are about to "
            "fall; the screen cannot distinguish a mispricing from a melting ice cube.",
            "Structurally over-weights capital-intensive and cyclical sectors, "
            "so it is a sector bet as much as a factor bet.",
            "P/B is increasingly uninformative where value is intangible: "
            "R&D and brand spend are expensed, so book value understates "
            "software and pharma while flattering banks and heavy industry.",
        ],
    )


def net_net() -> Screen:
    return Screen(
        name="Graham net-net",
        description="Trading below two-thirds of net current asset value.",
        filters=[
            Filter("revenue", ">", 0, label="positive revenue"),
            Filter("market_cap", ">=", 2e7, label="market cap >= $20m"),
            Filter("dollar_volume_50d", ">=", 2e5, required=False,
                   label="50d dollar volume >= $200k"),
            Filter("ncav_discount", ">", 0.0, required=True,
                   label="price below 2/3 of NCAV"),
            Filter("cfo", ">", 0, required=False,
                   label="positive operating cash flow (not burning the assets)"),
        ],
        score_weights={"ncav_discount": 1.0, "piotroski_f": 0.5},
        evidence=(
            "Graham's own records and later replications (Oppenheimer 1986, "
            "Xiao & Arnold 2008) found large excess returns, around 20-30% "
            "annualised, on diversified baskets of net-nets."
        ),
        failure_modes=[
            "Requires a basket of 20-30 names. Individual net-nets fail often; "
            "the strategy's returns are a statistical property of the group.",
            "The modern survivor set is dominated by shells, delisting "
            "candidates and companies consuming the very current assets the "
            "screen is counting. The positive-CFO filter helps but does not fix it.",
            "Almost always microcap and frequently untradeable in size.",
        ],
    )


def quality_compounders(min_market_cap: float = 2e9) -> Screen:
    return Screen(
        name="Quality compounders",
        description="Durable high returns on capital, reinvesting, lightly levered.",
        filters=[
            *sanity_filters(),
            *liquidity_filters(min_market_cap, 1e7),
            Filter("roic_5y_avg", ">=", 0.15, label="5y average ROIC >= 15%"),
            Filter("roic_min_5y", ">=", 0.10, required=False,
                   label="worst-year ROIC >= 10% (durability, not one good year)"),
            Filter("revenue_cagr_5y", ">=", 0.05, label="5y revenue CAGR >= 5%"),
            Filter("revenue_trend_r2", ">=", 0.60, required=False,
                   label="revenue trend r-squared >= 0.60 (steady, not lumpy)"),
            Filter("gross_margin", ">=", 0.30, required=False, label="gross margin >= 30%"),
            Filter("net_debt_to_ebitda", "<=", 3.0, required=False,
                   label="net debt / EBITDA <= 3"),
            Filter("cash_conversion", ">=", 0.80, required=False,
                   label="operating cash flow >= 80% of net income"),
            Filter("accruals_ratio", "<=", 0.10, required=False, label="low accruals"),
        ],
        score_weights={
            "roic_5y_avg": 1.0, "revenue_cagr_5y": 0.8, "fcf_margin": 0.8,
            "revenue_trend_r2": 0.5, "roic_stability": 0.4, "fcf_yield": 0.5,
        },
        evidence=(
            "Quality factors -- profitability and low accruals in particular -- "
            "carry robust premia (Novy-Marx 2013 on gross profitability; Sloan "
            "1996 on accruals), and unlike deep value they have held up in "
            "large caps."
        ),
        failure_modes=[
            "The screen has no valuation constraint, so it will happily return "
            "excellent businesses at indefensible prices. Pair it with the "
            "valuation module before acting.",
            "High trailing ROIC mean-reverts: competition is the base rate. "
            "The worst-year filter tests durability but cannot see the future.",
            "ROIC is flattered by buyback-shrunken equity and by acquisitive "
            "companies that have written off goodwill.",
        ],
    )


def garp(min_market_cap: float = 1e9) -> Screen:
    return Screen(
        name="GARP (growth at a reasonable price)",
        description="Real growth, real profitability, a multiple that is not absurd.",
        filters=[
            *sanity_filters(),
            *liquidity_filters(min_market_cap, 5e6),
            Filter("revenue_cagr_3y", ">=", 0.10, label="3y revenue CAGR >= 10%"),
            Filter("eps_cagr_3y", ">=", 0.10, required=False, label="3y EPS CAGR >= 10%"),
            Filter("peg", "between", (0.0, 2.0), required=False, label="PEG between 0 and 2"),
            Filter("roe", ">=", 0.12, required=False, label="ROE >= 12%"),
            Filter("operating_margin", ">", 0.05, label="operating margin > 5%"),
            Filter("net_debt_to_ebitda", "<=", 3.5, required=False,
                   label="net debt / EBITDA <= 3.5"),
        ],
        score_weights={"revenue_cagr_3y": 1.0, "roic": 0.8, "peg": 0.8,
                       "fcf_margin": 0.5, "eps_cagr_3y": 0.7},
        evidence=(
            "No single canonical study; GARP is a practitioner heuristic "
            "associated with Peter Lynch's record at Magellan (1977-1990)."
        ),
        failure_modes=[
            "PEG has no theoretical basis. It implicitly assumes a fair P/E "
            "equal to the growth rate, which is arbitrary and breaks entirely "
            "at low growth or low interest rates.",
            "Trailing 3-year growth is a weak predictor of forward growth; "
            "revenue growth mean-reverts quickly at scale.",
        ],
    )


def dividend_growth(min_market_cap: float = 1e9) -> Screen:
    return Screen(
        name="Dividend growth",
        description="Growing, covered dividends -- not merely high yields.",
        filters=[
            *sanity_filters(),
            *liquidity_filters(min_market_cap, 3e6),
            Filter("dividend_yield", ">=", 0.015, label="dividend yield >= 1.5%"),
            Filter("dividend_cagr_5y", ">=", 0.03, required=False,
                   label="5y dividend CAGR >= 3%"),
            # Cash cover, not earnings cover: dividends are paid from cash.
            Filter("fcf_payout_ratio", "between", (0.0, 0.80), required=False,
                   label="dividend <= 80% of free cash flow"),
            Filter("payout_ratio", "between", (0.0, 0.85), required=False,
                   label="dividend <= 85% of earnings"),
            Filter("net_debt_to_ebitda", "<=", 3.5, required=False,
                   label="net debt / EBITDA <= 3.5"),
            Filter("fcf", ">", 0, label="positive free cash flow"),
        ],
        score_weights={"dividend_cagr_5y": 1.0, "dividend_yield": 0.7,
                       "fcf_margin": 0.6, "roic": 0.6, "shareholder_yield": 0.5},
        evidence=(
            "Dividend growers and initiators have outperformed non-payers and, "
            "notably, the highest-yielding quintile, which is dominated by "
            "distressed payers about to cut."
        ),
        failure_modes=[
            "The highest yields are usually a market forecast of a cut. The "
            "payout and leverage filters exist to catch that and will not "
            "always succeed.",
            "Misses the many quality companies that return capital through "
            "buybacks instead, which is tax-advantaged for most holders.",
            "Tilts toward rate-sensitive sectors, so it carries unintended "
            "duration risk.",
        ],
    )


def piotroski_value(min_market_cap: float = 2e8) -> Screen:
    return Screen(
        name="Piotroski F-Score value",
        description="High book-to-market names that pass the fundamental health tests.",
        filters=[
            *sanity_filters(),
            *liquidity_filters(min_market_cap, 1e6),
            Filter("pb", "between", (0.0, 1.2), label="P/B <= 1.2 (high book-to-market)"),
            Filter("piotroski_f", ">=", 8.0, label="F-Score >= 8 of 9"),
        ],
        score_weights={"piotroski_f": 1.0, "pb": 0.8, "fcf_yield": 0.6, "roa": 0.5},
        evidence=(
            "Piotroski (2000): within the highest book-to-market quintile, "
            "high-F-Score firms beat low-F-Score firms by roughly 7.5% a year "
            "1976-1996, and the result has broadly replicated out of sample."
        ),
        failure_modes=[
            "The F-Score rewards year-over-year *improvement*, so an excellent "
            "but already-optimised business scores mediocre. It is a test for "
            "distressed value, not for quality.",
            "Requires two consecutive years of eight line items, so it is "
            "unavailable for recent listings and often for foreign filers.",
            "At an F-Score of 8+ with P/B under 1.2, the shortlist in a "
            "late-cycle US market is frequently empty. That is information.",
        ],
    )


def momentum(min_market_cap: float = 1e9) -> Screen:
    return Screen(
        name="Trend momentum",
        description="Established uptrends with relative strength and low drawdown.",
        filters=[
            Filter("revenue", ">", 0, label="positive revenue"),
            *liquidity_filters(min_market_cap, 1e7),
            Filter("momentum_12_1", ">=", 0.10, label="12-1 month momentum >= 10%"),
            Filter("above_sma200", "==", 1.0, label="above the 200-day average"),
            Filter("trend_template_score", ">=", 6.0, required=False,
                   label="Minervini trend template >= 6 of 8"),
            Filter("pct_from_52w_high", ">=", -0.25, required=False,
                   label="within 25% of the 52-week high"),
            Filter("vol63", "<=", 0.80, required=False,
                   label="3-month realised volatility <= 80%"),
        ],
        score_weights={"momentum_12_1": 1.0, "trend_r2": 0.7, "rs_slope_ann": 0.8,
                       "trend_template_score": 0.5, "vol63": 0.4},
        evidence=(
            "Cross-sectional momentum is among the most persistent anomalies on "
            "record (Jegadeesh & Titman 1993), documented across asset classes "
            "and more than a century of equity data. The 12-1 construction skips "
            "the most recent month because short-term reversal contaminates it."
        ),
        failure_modes=[
            "Momentum crashes. It loses violently at inflection points -- "
            "roughly -60% in 2009 as the prior year's losers rebounded -- and "
            "those crashes cluster precisely when risk appetite returns.",
            "High turnover makes it tax-inefficient in a taxable account, which "
            "can consume the entire premium for an individual investor.",
            "The trend-template filter is a discretionary overlay with no "
            "published excess-return evidence of its own.",
        ],
    )


def fcf_machines(min_market_cap: float = 1e9) -> Screen:
    return Screen(
        name="Free cash flow machines",
        description="High, consistent, well-converted free cash flow.",
        filters=[
            *sanity_filters(),
            *liquidity_filters(min_market_cap, 5e6),
            Filter("fcf_yield", ">=", 0.06, label="FCF yield >= 6%"),
            Filter("fcf_positive_years", ">=", 5.0, required=False,
                   label="positive FCF in 5+ of the last 7 years"),
            Filter("fcf_conversion", ">=", 0.75, required=False,
                   label="FCF >= 75% of net income"),
            Filter("net_debt_to_ebitda", "<=", 3.0, required=False,
                   label="net debt / EBITDA <= 3"),
            Filter("sbc_intensity", "<=", 0.08, required=False,
                   label="stock comp <= 8% of revenue"),
        ],
        score_weights={"fcf_yield": 1.0, "fcf_conversion": 0.7, "roic": 0.7,
                       "shareholder_yield": 0.6, "fcf_margin": 0.6},
        evidence=(
            "Cash-based profitability measures subsume accrual-based ones in "
            "predicting returns (Ball, Gerakos, Linnainmaa & Nikolaev 2016)."
        ),
        failure_modes=[
            "A high FCF yield can mean under-investment: capex cut to flatter "
            "near-term cash flow eventually shows up as lost revenue.",
            "The stock-comp filter matters because SBC is a real cost excluded "
            "from cash flow statements; without it, software companies screen "
            "as cash machines while diluting holders every year.",
        ],
    )


def cannibals(min_market_cap: float = 1e9) -> Screen:
    return Screen(
        name="Share cannibals",
        description="Shrinking the share count meaningfully, funded by real cash flow.",
        filters=[
            *sanity_filters(),
            *liquidity_filters(min_market_cap, 5e6),
            Filter("share_count_cagr_5y", "<=", -0.02,
                   label="share count shrinking >= 2% a year over 5 years"),
            Filter("fcf", ">", 0, label="positive free cash flow"),
            Filter("roic", ">=", 0.10, required=False, label="ROIC >= 10%"),
            Filter("net_debt_to_ebitda", "<=", 3.0, required=False,
                   label="net debt / EBITDA <= 3"),
        ],
        score_weights={"share_count_cagr_5y": 1.0, "roic": 0.8, "fcf_yield": 0.8,
                       "shareholder_yield": 0.6},
        evidence=(
            "Buyback-intensity screens have historically outperformed, and the "
            "effect is strongest where repurchases are funded from cash flow "
            "rather than new borrowing."
        ),
        failure_modes=[
            "Buybacks executed at high valuations destroy value as surely as a "
            "bad acquisition. The screen rewards the activity, not the price paid.",
            "Debt-funded buybacks manufacture EPS growth while raising "
            "financial risk; the leverage filter is the only guard here.",
        ],
    )


ALL_PRESETS = {
    "magic-formula": magic_formula,
    "deep-value": deep_value,
    "net-net": net_net,
    "quality": quality_compounders,
    "garp": garp,
    "dividend-growth": dividend_growth,
    "piotroski-value": piotroski_value,
    "momentum": momentum,
    "fcf-machines": fcf_machines,
    "cannibals": cannibals,
}


def get(name: str) -> Screen:
    """Look up a preset by name, building it with default thresholds."""
    key = name.lower().strip()
    if key not in ALL_PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(ALL_PRESETS)}")
    return ALL_PRESETS[key]()


def describe_all() -> list[dict]:
    rows = []
    for key, builder in ALL_PRESETS.items():
        screen = builder()
        rows.append({
            "key": key, "name": screen.name, "description": screen.description,
            "filters": len(screen.filters), "evidence": screen.evidence,
        })
    return rows
