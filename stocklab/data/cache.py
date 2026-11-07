"""SQLite-backed cache for API payloads and dataframes.

Rationale: every data source here is rate-limited, slow, or both.  SEC
``companyfacts`` is a 4 MB JSON document per company; refetching it inside a
screen over 500 names would take half an hour and probably earn an IP block.
A single-file SQLite cache is used instead of a directory of pickles because it
gives atomic writes, survives concurrent readers, and is trivially inspectable.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import sqlite3
import threading
import time
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    key        TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    payload    BLOB NOT NULL,
    stored_at  REAL NOT NULL,
    bytes      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cache_stored ON cache(stored_at);
"""


def make_key(*parts: Any) -> str:
    """Stable cache key from arbitrary parts.

    Long keys are hashed so URLs with query strings don't blow past index
    limits, but a readable prefix is kept so ``sqlite3 cache.sqlite`` is useful
    for debugging.
    """
    raw = "|".join("" if p is None else str(p) for p in parts)
    if len(raw) <= 160:
        return raw
    digest = hashlib.sha256(raw.encode()).hexdigest()[:24]
    return f"{raw[:120]}#{digest}"


class Cache:
    """Thread-safe TTL cache.  ``ttl=0`` forces a refresh; ``ttl=None`` never expires."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else settings().cache_db
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._local = threading.local()
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    # -- raw ---------------------------------------------------------------
    def _get_raw(self, key: str, ttl: int | None) -> tuple[str, bytes] | None:
        with self._lock:
            row = self._conn().execute(
                "SELECT kind, payload, stored_at FROM cache WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        kind, payload, stored_at = row
        if ttl is not None and ttl >= 0 and (time.time() - stored_at) > ttl:
            return None
        return kind, payload

    def _put_raw(self, key: str, kind: str, payload: bytes) -> None:
        with self._lock:
            self._conn().execute(
                "INSERT OR REPLACE INTO cache(key, kind, payload, stored_at, bytes)"
                " VALUES (?,?,?,?,?)",
                (key, kind, payload, time.time(), len(payload)),
            )

    # -- json --------------------------------------------------------------
    def get_json(self, key: str, ttl: int | None = None):
        hit = self._get_raw(key, ttl)
        if hit is None:
            return None
        kind, payload = hit
        if kind != "json":
            return None
        return json.loads(gzip.decompress(payload).decode())

    def set_json(self, key: str, obj) -> None:
        blob = gzip.compress(json.dumps(obj, separators=(",", ":")).encode(), 6)
        self._put_raw(key, "json", blob)

    # -- dataframe ---------------------------------------------------------
    def get_df(self, key: str, ttl: int | None = None) -> pd.DataFrame | None:
        hit = self._get_raw(key, ttl)
        if hit is None:
            return None
        kind, payload = hit
        if kind != "df":
            return None
        try:
            buf = io.BytesIO(gzip.decompress(payload))
            frame = pd.read_csv(buf, index_col=0)
        except Exception:
            return None
        # pandas >= 3 reads string indices as the `str` dtype rather than
        # `object`, so test for "not numeric" instead of naming a dtype.
        if not pd.api.types.is_numeric_dtype(frame.index):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")  # a ticker index must not "parse"
                    frame.index = pd.to_datetime(frame.index, format="ISO8601")
            except Exception:
                pass
        return frame

    def set_df(self, key: str, frame: pd.DataFrame) -> None:
        buf = io.StringIO()
        frame.to_csv(buf)
        self._put_raw(key, "df", gzip.compress(buf.getvalue().encode(), 6))

    # -- maintenance -------------------------------------------------------
    def age(self, key: str) -> float | None:
        """Seconds since ``key`` was stored, or ``None`` if absent."""
        with self._lock:
            row = self._conn().execute(
                "SELECT stored_at FROM cache WHERE key = ?", (key,)
            ).fetchone()
        return None if row is None else time.time() - row[0]

    def delete(self, key: str) -> None:
        with self._lock:
            self._conn().execute("DELETE FROM cache WHERE key = ?", (key,))

    def purge(self, older_than: float | None = None, prefix: str | None = None) -> int:
        clauses, params = [], []
        if older_than is not None:
            clauses.append("stored_at < ?")
            params.append(time.time() - older_than)
        if prefix:
            clauses.append("key LIKE ?")
            params.append(prefix + "%")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._lock:
            cur = self._conn().execute(f"DELETE FROM cache{where}", params)
            return cur.rowcount or 0

    def stats(self) -> dict:
        with self._lock:
            rows = self._conn().execute(
                "SELECT kind, COUNT(*), SUM(bytes), MIN(stored_at) FROM cache GROUP BY kind"
            ).fetchall()
        return {
            kind: {
                "entries": count,
                "mb": round((total or 0) / 1e6, 2),
                "oldest_age_hours": round((time.time() - (oldest or time.time())) / 3600, 1),
            }
            for kind, count, total, oldest in rows
        }


_CACHE: Cache | None = None


def cache() -> Cache:
    """Process-wide cache, re-created when the configured path changes.

    Comparing against the resolved path keeps a later `--data-dir` from being
    silently ignored once an instance exists.
    """
    global _CACHE
    wanted = settings().cache_db
    if _CACHE is None or _CACHE.path != wanted:
        _CACHE = Cache(wanted)
    return _CACHE


def reset_cache(path: Path | str | None = None) -> Cache:
    global _CACHE
    _CACHE = Cache(path)
    return _CACHE
