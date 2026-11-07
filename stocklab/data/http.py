"""Shared HTTP client: one session, polite rate limiting, bounded retries."""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

import requests

from ..config import settings

log = logging.getLogger(__name__)


class OfflineError(RuntimeError):
    """Raised when a network fetch is needed but the tool is in offline mode."""


class RateLimiter:
    """Fixed-interval limiter with a host-wide cooldown after a throttle.

    A lock plus a monotonic clock. SEC's fair-access policy is a hard 10 req/s
    and the penalty for breaching it is a block on your IP, so the conservative
    default is worth the lost throughput.

    The cooldown matters more than the interval. Per-request backoff does not
    help when a server has already decided to throttle you: every other
    in-flight thread keeps hammering it and keeps the throttle alive. On a 429
    the whole host is parked for a fixed interval, so a screen slows down once
    instead of failing repeatedly.
    """

    def __init__(self, per_second: float):
        self.min_interval = 1.0 / per_second if per_second > 0 else 0.0
        self._lock = threading.Lock()
        self._last = 0.0
        self._cooldown_until = 0.0

    def penalise(self, seconds: float) -> None:
        """Park every request to this host for ``seconds``."""
        with self._lock:
            self._cooldown_until = max(self._cooldown_until, time.monotonic() + seconds)

    def wait(self) -> None:
        # Two phases, deliberately separate: sleep out any host cooldown first,
        # then enforce the inter-request interval exactly once. Folding them
        # into one loop (and advancing `_last` each pass) never terminates,
        # because the interval check is then measured against a reservation
        # that keeps moving into the future.
        sleep_for = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                remaining = self._cooldown_until - now
                if remaining <= 0:
                    sleep_for = max(0.0, self.min_interval - (now - self._last))
                    self._last = now + sleep_for
                    break
            time.sleep(min(remaining, 5.0))
        if sleep_for > 0:
            time.sleep(sleep_for)


_LIMITERS: dict[str, RateLimiter] = {}
_LIMITER_LOCK = threading.Lock()
_SESSION: requests.Session | None = None
_SESSION_LOCK = threading.Lock()

# Hosts needing their own budget. Anything unlisted gets the default.
_HOST_RPS = {
    "data.sec.gov": None,       # filled from settings at first use
    "www.sec.gov": None,
    # Yahoo's chart endpoint is undocumented and starts returning 429 above
    # roughly 2 req/s sustained from one IP. The limit is not published, so this
    # is set conservatively: a screen is slower but it finishes.
    "query1.finance.yahoo.com": 1.5,
    "query2.finance.yahoo.com": 1.5,
    "stooq.com": 4.0,
    "fred.stlouisfed.org": 8.0,
    "www.nasdaqtrader.com": 4.0,
}
_DEFAULT_RPS = 5.0

# Status codes worth retrying: transient server faults and explicit throttles.
_RETRY_STATUS = {429, 500, 502, 503, 504}


def _limiter(host: str) -> RateLimiter:
    with _LIMITER_LOCK:
        if host not in _LIMITERS:
            rps = _HOST_RPS.get(host, _DEFAULT_RPS)
            if rps is None:
                rps = settings().sec_rate_limit_per_sec
            _LIMITERS[host] = RateLimiter(rps)
        return _LIMITERS[host]


def session() -> requests.Session:
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is None:
            sess = requests.Session()
            sess.headers.update(
                {
                    # SEC rejects requests without a declared, identifying UA.
                    "User-Agent": settings().user_agent,
                    "Accept-Encoding": "gzip, deflate",
                    "Accept": "application/json,text/plain,*/*",
                }
            )
            _SESSION = sess
        return _SESSION


def get(
    url: str,
    params: dict[str, Any] | None = None,
    *,
    timeout: float | None = None,
    allow_status: tuple[int, ...] = (),
    headers: dict[str, str] | None = None,
) -> requests.Response:
    """Rate-limited GET with exponential backoff.

    ``allow_status`` lists codes returned to the caller instead of raising --
    useful for 404s, which for these APIs mean "no such ticker", not "broken".
    """
    cfg = settings()
    if cfg.offline:
        raise OfflineError(f"offline mode: refusing to fetch {url}")

    host = requests.utils.urlparse(url).netloc
    timeout = timeout or cfg.request_timeout
    last_err: Exception | None = None

    for attempt in range(cfg.max_retries + 1):
        _limiter(host).wait()
        try:
            resp = session().get(url, params=params, timeout=timeout, headers=headers)
        except requests.RequestException as exc:
            last_err = exc
            if attempt >= cfg.max_retries:
                break
            time.sleep(min(2.0**attempt, 8.0))
            continue

        if resp.status_code in allow_status or resp.ok:
            return resp
        if resp.status_code in _RETRY_STATUS and attempt < cfg.max_retries:
            # Honour Retry-After when the server bothers to send one. Yahoo does
            # not, and recovers slowly once throttled, so 429 gets a much longer
            # backoff than a transient 503.
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                delay = float(retry_after)
            elif resp.status_code == 429:
                delay = min(5.0 * (2.0**attempt), 45.0)
                # Park the whole host, not just this request.
                _limiter(host).penalise(delay)
            else:
                delay = min(2.0**attempt, 8.0)
            log.debug("retrying %s after %s (status %s)", url, delay, resp.status_code)
            time.sleep(delay)
            continue
        last_err = requests.HTTPError(f"{resp.status_code} for {url}", response=resp)
        break

    raise last_err or requests.RequestException(f"failed: {url}")


def get_json(url: str, params: dict[str, Any] | None = None, **kwargs):
    """GET returning parsed JSON, or ``None`` for a 404."""
    resp = get(url, params, allow_status=(404,), **kwargs)
    if resp.status_code == 404:
        return None
    try:
        return resp.json()
    except ValueError:
        return None


def get_text(url: str, params: dict[str, Any] | None = None, **kwargs) -> str | None:
    resp = get(url, params, allow_status=(404,), **kwargs)
    return None if resp.status_code == 404 else resp.text
