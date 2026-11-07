"""stocklab: equity research, valuation, screening and portfolio management.

Built on primary sources -- SEC EDGAR XBRL for fundamentals, so figures are
traceable to a filed tag and a filing date, which is what makes point-in-time
analysis possible.

Not investment advice. Every model here encodes assumptions; they are stated in
``docs/METHODOLOGY.md`` along with the conditions under which each one breaks.

Quick start::

    from stocklab.company import Company
    from stocklab.valuation import composite

    company = Company.load("AAPL")
    report = composite.value(company.fundamentals, company.metrics, company.price)
    print(report.to_frame())
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Company",
    "settings",
    "configure",
]


def __getattr__(name: str):
    """Lazy re-exports.

    Importing the package must not pull in pandas, scipy and the whole provider
    stack, because the CLI's ``--help`` and the test collector both import it.
    """
    if name == "Company":
        from .company import Company
        return Company
    if name in ("settings", "configure"):
        from . import config
        return getattr(config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
