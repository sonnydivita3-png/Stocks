"""Runtime configuration: filesystem layout, network manners, model defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ENV_PREFIX = "STOCKLAB_"


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(ENV_PREFIX + name, default)


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    try:
        return float(raw) if raw is not None else default
    except ValueError:
        return default


def _default_root() -> Path:
    override = _env("HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".stocklab"


@dataclass
class Settings:
    """Everything tunable in one place.

    Defaults for the market-wide assumptions (ERP, terminal growth) are set to
    defensible long-run values, but they are *assumptions* and they drive
    valuation output more than any other input.  Override them deliberately
    rather than inheriting them by accident -- see ``docs/METHODOLOGY.md``.
    """

    root: Path = field(default_factory=_default_root)

    # --- network manners ---------------------------------------------------
    # SEC's fair-access policy caps clients at 10 requests/second and requires a
    # declared User-Agent. Exceeding it gets your IP blocked, not throttled.
    # SEC's fair-access policy (sec.gov/os/webmaster-faq#developers) requires
    # automated clients to declare an identifying name plus a contact address;
    # www.sec.gov returns 403 without one.  Sent only to sec.gov hosts.
    # Override with STOCKLAB_USER_AGENT.
    user_agent: str = field(
        default_factory=lambda: _env("USER_AGENT", "stocklab/0.1 (sonnydivita3@gmail.com)")
    )
    sec_rate_limit_per_sec: float = field(default_factory=lambda: _env_float("SEC_RPS", 6.0))
    request_timeout: float = field(default_factory=lambda: _env_float("TIMEOUT", 30.0))
    max_retries: int = 3
    offline: bool = field(default_factory=lambda: _env("OFFLINE", "0") not in ("0", "", "false"))

    # --- cache TTLs in seconds --------------------------------------------
    ttl_prices: int = 60 * 60 * 6          # intraday staleness is fine for analysis
    ttl_fundamentals: int = 60 * 60 * 24 * 7  # 10-Qs land quarterly; a week is safe
    ttl_profile: int = 60 * 60 * 24 * 30
    ttl_macro: int = 60 * 60 * 12
    ttl_universe: int = 60 * 60 * 24 * 7

    # --- market assumptions ------------------------------------------------
    risk_free_fallback: float = 0.042      # used only when FRED is unreachable
    equity_risk_premium: float = 0.048     # Damodaran-style implied ERP, long run
    terminal_growth: float = 0.025         # <= long-run nominal GDP; see docs
    small_cap_premium: float = 0.015       # applied below $2bn market cap
    micro_cap_premium: float = 0.030       # applied below $300m market cap
    default_tax_rate: float = 0.23         # marginal, incl. state, for FCFF
    min_wacc: float = 0.06                 # floor: no business funds itself at 3%
    max_wacc: float = 0.20

    # --- portfolio defaults ------------------------------------------------
    default_cost_basis_method: str = "fifo"  # fifo | lifo | average | hifo
    rebalance_band: float = 0.05           # relative drift tolerance
    min_trade_value: float = 100.0         # don't generate dust trades
    commission_per_trade: float = 0.0
    slippage_bps: float = 5.0

    # --- risk --------------------------------------------------------------
    trading_days: int = 252
    var_confidence: float = 0.95

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser()

    # --- derived paths -----------------------------------------------------
    @property
    def cache_db(self) -> Path:
        return self.ensure(self.root) / "cache.sqlite"

    @property
    def portfolio_db(self) -> Path:
        return self.ensure(self.root) / "portfolio.sqlite"

    @property
    def reports_dir(self) -> Path:
        return self.ensure(self.root / "reports")

    @staticmethod
    def ensure(path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path

    def size_premium(self, market_cap: float | None) -> float:
        """Extra cost of equity for small and micro caps.

        Empirically the raw 'size premium' has been weak since the 1980s once
        you control for profitability, but illiquidity and estimation risk are
        real for a $200m company.  This is a modest haircut, not a Fama-French
        factor claim.
        """
        if market_cap is None or market_cap <= 0:
            return 0.0
        if market_cap < 300e6:
            return self.micro_cap_premium
        if market_cap < 2e9:
            return self.small_cap_premium
        return 0.0


SETTINGS = Settings()


def settings() -> Settings:
    return SETTINGS


def configure(**kwargs) -> Settings:
    """Mutate the process-wide settings (used by the CLI and tests)."""
    for key, value in kwargs.items():
        if not hasattr(SETTINGS, key):
            raise KeyError(f"unknown setting: {key}")
        setattr(SETTINGS, key, value)
    SETTINGS.__post_init__()
    return SETTINGS
