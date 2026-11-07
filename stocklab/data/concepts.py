"""Mapping from financial concepts to US-GAAP XBRL tags.

XBRL is a vocabulary, not a schema.  Two companies in the same industry will
report revenue under different tags, and the same company changes tags between
years when the FASB deprecates one.  There is no way around a hand-curated
fallback list; this module is that list.

Each :class:`Concept` names candidate tags in priority order.  The first tag
with data for a period wins.  ``statement`` and ``kind`` drive how the series is
assembled (flows get summed across quarters, stocks are point-in-time), and
``positive`` normalises sign so that e.g. capex is always a positive outflow
regardless of how the filer signed it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Statement = Literal["income", "balance", "cashflow", "meta"]
Kind = Literal["flow", "stock", "per_share", "shares", "ratio"]


@dataclass(frozen=True)
class Concept:
    name: str
    statement: Statement
    kind: Kind
    tags: tuple[str, ...]
    unit: str = "USD"
    positive: bool = False          # take abs(): the filer's sign is a convention
    taxonomy: str = "us-gaap"
    label: str = ""
    sum_tags: tuple[tuple[str, ...], ...] = field(default=())  # fallback: sum a group

    @property
    def title(self) -> str:
        return self.label or self.name.replace("_", " ").title()


# --------------------------------------------------------------------------- #
# Income statement
# --------------------------------------------------------------------------- #
INCOME = [
    Concept(
        "revenue", "income", "flow",
        (
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
            "SalesRevenueGoodsNet",
            "SalesRevenueServicesNet",
            "RevenuesNetOfInterestExpense",          # banks
            "TotalRevenuesAndOtherIncome",           # energy
            "RegulatedAndUnregulatedOperatingRevenue",
        ),
        label="Revenue",
    ),
    Concept(
        "cost_of_revenue", "income", "flow",
        (
            "CostOfRevenue",
            "CostOfGoodsAndServicesSold",
            "CostOfGoodsSold",
            "CostOfServices",
            "CostOfSales",
        ),
        positive=True, label="Cost of Revenue",
    ),
    Concept("gross_profit", "income", "flow", ("GrossProfit",), label="Gross Profit"),
    Concept(
        "rnd", "income", "flow",
        ("ResearchAndDevelopmentExpense",
         "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"),
        positive=True, label="R&D",
    ),
    Concept(
        "sga", "income", "flow",
        ("SellingGeneralAndAdministrativeExpense",
         "GeneralAndAdministrativeExpense",
         "SellingAndMarketingExpense"),
        positive=True, label="SG&A",
    ),
    Concept(
        "operating_expenses", "income", "flow",
        ("OperatingExpenses", "CostsAndExpenses", "OperatingCostsAndExpenses"),
        positive=True, label="Operating Expenses",
    ),
    Concept(
        "operating_income", "income", "flow",
        ("OperatingIncomeLoss",
         "IncomeLossFromContinuingOperationsBeforeInterestExpenseInterestIncomeIncomeTaxesExtraordinaryItemsNoncontrollingInterestsNet"),
        label="Operating Income",
    ),
    Concept(
        "interest_expense", "income", "flow",
        ("InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt",
         "InterestAndDebtExpense", "InterestExpenseBorrowings"),
        positive=True, label="Interest Expense",
    ),
    Concept(
        "pretax_income", "income", "flow",
        ("IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
         "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
         "IncomeLossFromContinuingOperationsBeforeIncomeTaxesForeign"),
        label="Pretax Income",
    ),
    Concept(
        "tax_expense", "income", "flow",
        ("IncomeTaxExpenseBenefit", "IncomeTaxesPaidNet"),
        label="Income Tax",
    ),
    Concept(
        "net_income", "income", "flow",
        ("NetIncomeLoss",
         "NetIncomeLossAvailableToCommonStockholdersBasic",
         "ProfitLoss",
         "IncomeLossFromContinuingOperations"),
        label="Net Income",
    ),
    Concept(
        "eps_diluted", "income", "per_share",
        ("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"),
        unit="USD/shares", label="Diluted EPS",
    ),
    Concept(
        "eps_basic", "income", "per_share",
        ("EarningsPerShareBasic", "EarningsPerShareBasicAndDiluted"),
        unit="USD/shares", label="Basic EPS",
    ),
    Concept(
        "shares_diluted", "income", "shares",
        ("WeightedAverageNumberOfDilutedSharesOutstanding",
         "WeightedAverageNumberOfSharesOutstandingBasicAndDiluted"),
        unit="shares", label="Diluted Shares (wtd avg)",
    ),
    Concept(
        "shares_basic", "income", "shares",
        ("WeightedAverageNumberOfSharesOutstandingBasic",
         "WeightedAverageNumberOfSharesOutstanding"),
        unit="shares", label="Basic Shares (wtd avg)",
    ),
]

# --------------------------------------------------------------------------- #
# Balance sheet  (instants)
# --------------------------------------------------------------------------- #
BALANCE = [
    Concept(
        "cash", "balance", "stock",
        ("CashAndCashEquivalentsAtCarryingValue",
         "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
         "CashAndDueFromBanks", "Cash"),
        label="Cash & Equivalents",
    ),
    Concept(
        "short_term_investments", "balance", "stock",
        ("ShortTermInvestments", "MarketableSecuritiesCurrent",
         "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
         "OtherShortTermInvestments"),
        label="Short-Term Investments",
    ),
    Concept(
        "long_term_investments", "balance", "stock",
        ("LongTermInvestments", "MarketableSecuritiesNoncurrent",
         "AvailableForSaleSecuritiesDebtSecuritiesNoncurrent"),
        label="Long-Term Investments",
    ),
    Concept(
        "receivables", "balance", "stock",
        ("AccountsReceivableNetCurrent", "ReceivablesNetCurrent",
         "AccountsNotesAndLoansReceivableNetCurrent"),
        label="Receivables",
    ),
    Concept("inventory", "balance", "stock", ("InventoryNet", "InventoryGross"), label="Inventory"),
    Concept("current_assets", "balance", "stock", ("AssetsCurrent",), label="Current Assets"),
    Concept("total_assets", "balance", "stock", ("Assets",), label="Total Assets"),
    Concept(
        "ppe_net", "balance", "stock",
        ("PropertyPlantAndEquipmentNet", "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization"),
        label="PP&E (net)",
    ),
    Concept("goodwill", "balance", "stock", ("Goodwill",), label="Goodwill"),
    Concept(
        "intangibles", "balance", "stock",
        ("IntangibleAssetsNetExcludingGoodwill", "FiniteLivedIntangibleAssetsNet"),
        label="Intangibles",
    ),
    Concept(
        "payables", "balance", "stock",
        ("AccountsPayableCurrent", "AccountsPayableAndAccruedLiabilitiesCurrent"),
        label="Payables",
    ),
    Concept("current_liabilities", "balance", "stock", ("LiabilitiesCurrent",), label="Current Liabilities"),
    Concept("total_liabilities", "balance", "stock", ("Liabilities",), label="Total Liabilities"),
    Concept(
        "short_term_debt", "balance", "stock",
        ("DebtCurrent", "ShortTermBorrowings", "LongTermDebtCurrent",
         "CommercialPaper", "OtherShortTermBorrowings",
         "NotesPayableCurrent"),
        positive=True, label="Short-Term Debt",
    ),
    Concept(
        "long_term_debt", "balance", "stock",
        ("LongTermDebtNoncurrent", "LongTermDebt",
         "LongTermNotesPayable", "SeniorNotes"),
        positive=True, label="Long-Term Debt",
    ),
    Concept(
        "operating_lease_liability", "balance", "stock",
        ("OperatingLeaseLiabilityNoncurrent", "OperatingLeaseLiability"),
        positive=True, label="Operating Lease Liability",
    ),
    Concept(
        "equity", "balance", "stock",
        ("StockholdersEquity",
         "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
        label="Shareholders' Equity",
    ),
    Concept(
        "retained_earnings", "balance", "stock",
        ("RetainedEarningsAccumulatedDeficit",), label="Retained Earnings",
    ),
    Concept(
        "preferred_equity", "balance", "stock",
        ("PreferredStockValue", "PreferredStockValueOutstanding",
         "TemporaryEquityCarryingAmountAttributableToParent"),
        positive=True, label="Preferred Equity",
    ),
    Concept(
        "minority_interest", "balance", "stock",
        ("MinorityInterest", "StockholdersEquityAttributableToNoncontrollingInterest"),
        label="Minority Interest",
    ),
    Concept(
        "shares_outstanding", "balance", "shares",
        ("EntityCommonStockSharesOutstanding",),
        unit="shares", taxonomy="dei", label="Shares Outstanding",
    ),
    Concept(
        "shares_issued", "balance", "shares",
        ("CommonStockSharesOutstanding", "CommonStockSharesIssued"),
        unit="shares", label="Common Shares Outstanding",
    ),
]

# --------------------------------------------------------------------------- #
# Cash flow statement
# --------------------------------------------------------------------------- #
CASHFLOW = [
    Concept(
        "cfo", "cashflow", "flow",
        ("NetCashProvidedByUsedInOperatingActivities",
         "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
        label="Operating Cash Flow",
    ),
    Concept(
        "capex", "cashflow", "flow",
        ("PaymentsToAcquirePropertyPlantAndEquipment",
         "PaymentsToAcquireProductiveAssets",
         "PaymentsForCapitalImprovements",
         "PaymentsToAcquireOtherPropertyPlantAndEquipment"),
        positive=True, label="Capital Expenditures",
    ),
    Concept(
        "cfi", "cashflow", "flow",
        ("NetCashProvidedByUsedInInvestingActivities",
         "NetCashProvidedByUsedInInvestingActivitiesContinuingOperations"),
        label="Investing Cash Flow",
    ),
    Concept(
        "cff", "cashflow", "flow",
        ("NetCashProvidedByUsedInFinancingActivities",
         "NetCashProvidedByUsedInFinancingActivitiesContinuingOperations"),
        label="Financing Cash Flow",
    ),
    Concept(
        "depreciation_amortization", "cashflow", "flow",
        ("DepreciationDepletionAndAmortization",
         "DepreciationAmortizationAndAccretionNet",
         "DepreciationAndAmortization",
         "Depreciation"),
        positive=True, label="D&A",
    ),
    Concept(
        "stock_comp", "cashflow", "flow",
        ("ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"),
        positive=True, label="Stock-Based Comp",
    ),
    Concept(
        "dividends_paid", "cashflow", "flow",
        ("PaymentsOfDividendsCommonStock", "PaymentsOfDividends",
         "PaymentsOfOrdinaryDividends", "PaymentsOfDividendsAndDividendEquivalents",
         "PaymentsOfCapitalDistribution", "PaymentsOfDistributionsToAffiliates"),
        positive=True, label="Dividends Paid",
    ),
    Concept(
        "buybacks", "cashflow", "flow",
        ("PaymentsForRepurchaseOfCommonStock",
         "PaymentsForRepurchaseOfEquity"),
        positive=True, label="Share Repurchases",
    ),
    Concept(
        "acquisitions", "cashflow", "flow",
        ("PaymentsToAcquireBusinessesNetOfCashAcquired",
         "PaymentsToAcquireBusinessesAndInterestInAffiliates"),
        positive=True, label="Acquisitions",
    ),
    Concept(
        "debt_issued", "cashflow", "flow",
        ("ProceedsFromIssuanceOfLongTermDebt", "ProceedsFromNotesPayable"),
        positive=True, label="Debt Issued",
    ),
    Concept(
        "debt_repaid", "cashflow", "flow",
        ("RepaymentsOfLongTermDebt", "RepaymentsOfDebt", "RepaymentsOfNotesPayable"),
        positive=True, label="Debt Repaid",
    ),
]

ALL_CONCEPTS: tuple[Concept, ...] = tuple(INCOME + BALANCE + CASHFLOW)
BY_NAME: dict[str, Concept] = {c.name: c for c in ALL_CONCEPTS}

FLOW_CONCEPTS = tuple(c.name for c in ALL_CONCEPTS if c.kind in ("flow", "per_share"))
STOCK_CONCEPTS = tuple(c.name for c in ALL_CONCEPTS if c.kind == "stock")


# --------------------------------------------------------------------------- #
# SIC -> sector.  Coarse on purpose: the point is peer grouping and knowing
# when a valuation model does not apply (banks, REITs, insurers), not GICS
# fidelity.  SEC publishes SIC, not GICS, and SIC is free.
# --------------------------------------------------------------------------- #
_SIC_RANGES: tuple[tuple[int, int, str], ...] = (
    (100, 999, "Agriculture"),
    (1000, 1099, "Materials"),
    (1200, 1299, "Energy"),
    (1300, 1399, "Energy"),
    (1400, 1499, "Materials"),
    (1500, 1799, "Industrials"),
    (2000, 2199, "Consumer Staples"),
    (2200, 2399, "Consumer Discretionary"),
    (2400, 2599, "Industrials"),
    (2600, 2699, "Materials"),
    (2700, 2799, "Communication Services"),
    (2800, 2824, "Materials"),
    (2830, 2836, "Health Care"),
    (2840, 2899, "Consumer Staples"),
    (2900, 2999, "Energy"),
    (3000, 3299, "Industrials"),
    (3300, 3399, "Materials"),
    (3400, 3569, "Industrials"),
    (3570, 3579, "Information Technology"),
    (3580, 3599, "Industrials"),
    (3600, 3629, "Industrials"),
    (3630, 3669, "Consumer Discretionary"),
    (3670, 3679, "Information Technology"),
    (3680, 3689, "Information Technology"),
    (3690, 3699, "Information Technology"),
    (3700, 3799, "Consumer Discretionary"),
    (3800, 3829, "Information Technology"),
    (3830, 3851, "Health Care"),
    (3860, 3999, "Consumer Discretionary"),
    (4000, 4299, "Industrials"),
    (4400, 4799, "Industrials"),
    (4800, 4899, "Communication Services"),
    (4900, 4999, "Utilities"),
    (5000, 5199, "Industrials"),
    (5200, 5499, "Consumer Discretionary"),
    (5500, 5599, "Consumer Discretionary"),
    (5600, 5799, "Consumer Discretionary"),
    (5800, 5899, "Consumer Discretionary"),
    (5900, 5999, "Consumer Discretionary"),
    (6000, 6199, "Financials"),
    (6200, 6299, "Financials"),
    (6320, 6329, "Health Care"),    # managed care: UNH, ELV, CI, HUM
    (6300, 6411, "Financials"),
    (6500, 6599, "Real Estate"),
    (6798, 6798, "Real Estate"),
    (6700, 6799, "Financials"),
    (7000, 7099, "Consumer Discretionary"),
    (7200, 7299, "Consumer Discretionary"),
    (7300, 7349, "Industrials"),
    (7350, 7359, "Industrials"),
    (7370, 7379, "Information Technology"),
    (7380, 7399, "Industrials"),
    (7800, 7999, "Communication Services"),
    (8000, 8099, "Health Care"),
    (8200, 8299, "Consumer Discretionary"),
    (8300, 8399, "Health Care"),
    (8700, 8799, "Industrials"),
)

# Industries where the standard enterprise-value / FCFF toolkit is invalid.
# Managed-care organisations carry 632x SIC codes but are operating companies,
# not balance-sheet lenders: EV/EBITDA and net debt are meaningful for them, so
# they are excluded from the financial-sector carve-out below.
MANAGED_CARE_SIC = frozenset(range(6320, 6330))
FINANCIAL_SIC = frozenset(range(6000, 6500)) - MANAGED_CARE_SIC
REIT_SIC = frozenset({6798})
INSURANCE_SIC = frozenset(range(6311, 6412))
UTILITY_SIC = frozenset(range(4900, 5000))


def sector_for_sic(sic: int | str | None) -> str:
    try:
        code = int(sic)
    except (TypeError, ValueError):
        return "Unknown"
    for lo, hi, sector in _SIC_RANGES:
        if lo <= code <= hi:
            return sector
    return "Unknown"


def is_financial(sic: int | str | None) -> bool:
    """Banks, brokers, insurers.

    EV/EBITDA, FCFF and net-debt metrics are meaningless here -- debt is raw
    material, not financing -- so valuation code must branch on this.
    """
    try:
        return int(sic) in FINANCIAL_SIC
    except (TypeError, ValueError):
        return False


def is_reit(sic: int | str | None) -> bool:
    try:
        return int(sic) in REIT_SIC
    except (TypeError, ValueError):
        return False
