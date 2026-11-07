"""SQLite persistence for accounts, transactions and target allocations.

Only *inputs* are stored. Lots, positions, realised gains and performance are
recomputed from the transaction log on every read (see :mod:`.ledger`), so the
database can never hold a derived value that disagrees with the trades that
produced it.
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..config import settings
from .models import Account, AccountType, Transaction, TxnType

_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS accounts (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    name     TEXT NOT NULL UNIQUE,
    type     TEXT NOT NULL,
    broker   TEXT DEFAULT '',
    owner    TEXT DEFAULT '',
    currency TEXT DEFAULT 'USD',
    cost_basis_method TEXT DEFAULT 'fifo',
    notes    TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS transactions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id INTEGER NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    date       TEXT NOT NULL,
    type       TEXT NOT NULL,
    symbol     TEXT DEFAULT '',
    quantity   REAL DEFAULT 0,
    price      REAL DEFAULT 0,
    amount     REAL DEFAULT 0,
    fees       REAL DEFAULT 0,
    lot_ids    TEXT DEFAULT '',
    notes      TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_txn_account ON transactions(account_id, date);
CREATE INDEX IF NOT EXISTS idx_txn_symbol ON transactions(symbol, date);

CREATE TABLE IF NOT EXISTS targets (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    model      TEXT NOT NULL,
    symbol     TEXT NOT NULL,
    weight     REAL NOT NULL,
    account_id INTEGER REFERENCES accounts(id) ON DELETE CASCADE,
    notes      TEXT DEFAULT '',
    UNIQUE(model, symbol, account_id)
);

CREATE TABLE IF NOT EXISTS watchlist (
    symbol     TEXT NOT NULL,
    list_name  TEXT NOT NULL DEFAULT 'default',
    added      TEXT NOT NULL,
    thesis     TEXT DEFAULT '',
    target_buy REAL,
    notes      TEXT DEFAULT '',
    PRIMARY KEY (symbol, list_name)
);
"""


class PortfolioStore:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else settings().portfolio_db
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self.connection() as conn:
            conn.executescript(_SCHEMA)

    def connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    # ------------------------------------------------------------------ #
    # accounts
    # ------------------------------------------------------------------ #
    def add_account(self, account: Account) -> int:
        cur = self.connection().execute(
            "INSERT INTO accounts(name, type, broker, owner, currency,"
            " cost_basis_method, notes) VALUES (?,?,?,?,?,?,?)",
            (account.name, account.type.value, account.broker, account.owner,
             account.currency, account.cost_basis_method, account.notes),
        )
        account.id = int(cur.lastrowid)
        return account.id

    def get_account(self, identifier: int | str) -> Account | None:
        if isinstance(identifier, int):
            row = self.connection().execute(
                "SELECT * FROM accounts WHERE id = ?", (identifier,)).fetchone()
        else:
            row = self.connection().execute(
                "SELECT * FROM accounts WHERE name = ? COLLATE NOCASE",
                (identifier,)).fetchone()
        return self._account_from_row(row) if row else None

    def accounts(self) -> list[Account]:
        rows = self.connection().execute("SELECT * FROM accounts ORDER BY id").fetchall()
        return [self._account_from_row(r) for r in rows]

    @staticmethod
    def _account_from_row(row: sqlite3.Row) -> Account:
        return Account(
            id=row["id"], name=row["name"], type=AccountType(row["type"]),
            broker=row["broker"] or "", owner=row["owner"] or "",
            currency=row["currency"] or "USD",
            cost_basis_method=row["cost_basis_method"] or "fifo",
            notes=row["notes"] or "",
        )

    def delete_account(self, account_id: int) -> None:
        self.connection().execute("DELETE FROM accounts WHERE id = ?", (account_id,))

    def rename_account(self, account_id: int, new_name: str) -> None:
        self.connection().execute(
            "UPDATE accounts SET name = ? WHERE id = ?", (new_name, account_id))

    # ------------------------------------------------------------------ #
    # transactions
    # ------------------------------------------------------------------ #
    def add_transaction(self, txn: Transaction) -> int:
        cur = self.connection().execute(
            "INSERT INTO transactions(account_id, date, type, symbol, quantity,"
            " price, amount, fees, lot_ids, notes) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (txn.account_id, txn.date.isoformat(), txn.type.value, txn.symbol,
             txn.quantity, txn.price, txn.amount, txn.fees,
             ",".join(str(i) for i in txn.lot_ids), txn.notes),
        )
        txn.id = int(cur.lastrowid)
        return txn.id

    def add_transactions(self, txns: Iterable[Transaction]) -> int:
        count = 0
        conn = self.connection()
        conn.execute("BEGIN")
        try:
            for txn in txns:
                self.add_transaction(txn)
                count += 1
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return count

    def transactions(self, account_id: int | None = None, symbol: str | None = None,
                     start: date | None = None, end: date | None = None
                     ) -> list[Transaction]:
        """Transactions in chronological order, ties broken by insertion id.

        Deterministic ordering matters: two trades on the same day must replay
        in a fixed order or the cost basis depends on query order.
        """
        clauses, params = [], []
        if account_id is not None:
            clauses.append("account_id = ?")
            params.append(account_id)
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol.upper())
        if start:
            clauses.append("date >= ?")
            params.append(start.isoformat())
        if end:
            clauses.append("date <= ?")
            params.append(end.isoformat())
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self.connection().execute(
            f"SELECT * FROM transactions{where} ORDER BY date, id", params).fetchall()
        return [self._txn_from_row(r) for r in rows]

    @staticmethod
    def _txn_from_row(row: sqlite3.Row) -> Transaction:
        raw_lots = row["lot_ids"] or ""
        lot_ids = tuple(int(x) for x in raw_lots.split(",") if x.strip().isdigit())
        return Transaction(
            id=row["id"], account_id=row["account_id"],
            date=pd.Timestamp(row["date"]).date(), type=TxnType(row["type"]),
            symbol=row["symbol"] or "", quantity=row["quantity"] or 0.0,
            price=row["price"] or 0.0, amount=row["amount"] or 0.0,
            fees=row["fees"] or 0.0, lot_ids=lot_ids, notes=row["notes"] or "",
        )

    def delete_transaction(self, txn_id: int) -> None:
        self.connection().execute("DELETE FROM transactions WHERE id = ?", (txn_id,))

    def symbols(self, account_id: int | None = None) -> list[str]:
        clause = " AND account_id = ?" if account_id is not None else ""
        params = (account_id,) if account_id is not None else ()
        rows = self.connection().execute(
            f"SELECT DISTINCT symbol FROM transactions WHERE symbol != ''{clause}"
            " ORDER BY symbol", params).fetchall()
        return [r["symbol"] for r in rows]

    # ------------------------------------------------------------------ #
    # target allocations
    # ------------------------------------------------------------------ #
    def set_target(self, model: str, symbol: str, weight: float,
                   account_id: int | None = None, notes: str = "") -> None:
        self.connection().execute(
            "INSERT INTO targets(model, symbol, weight, account_id, notes)"
            " VALUES (?,?,?,?,?)"
            " ON CONFLICT(model, symbol, account_id) DO UPDATE SET"
            " weight = excluded.weight, notes = excluded.notes",
            (model, symbol.upper(), float(weight), account_id, notes),
        )

    def targets(self, model: str = "default",
                account_id: int | None = None) -> pd.Series:
        if account_id is None:
            rows = self.connection().execute(
                "SELECT symbol, weight FROM targets WHERE model = ?"
                " AND account_id IS NULL", (model,)).fetchall()
        else:
            rows = self.connection().execute(
                "SELECT symbol, weight FROM targets WHERE model = ?"
                " AND account_id = ?", (model, account_id)).fetchall()
        if not rows:
            return pd.Series(dtype=float)
        return pd.Series({r["symbol"]: r["weight"] for r in rows}, dtype=float)

    def models(self) -> list[str]:
        rows = self.connection().execute(
            "SELECT DISTINCT model FROM targets ORDER BY model").fetchall()
        return [r["model"] for r in rows]

    def clear_targets(self, model: str) -> None:
        self.connection().execute("DELETE FROM targets WHERE model = ?", (model,))

    # ------------------------------------------------------------------ #
    # watchlist
    # ------------------------------------------------------------------ #
    def watch(self, symbol: str, list_name: str = "default", thesis: str = "",
              target_buy: float | None = None, notes: str = "") -> None:
        self.connection().execute(
            "INSERT INTO watchlist(symbol, list_name, added, thesis, target_buy, notes)"
            " VALUES (?,?,?,?,?,?)"
            " ON CONFLICT(symbol, list_name) DO UPDATE SET"
            " thesis = excluded.thesis, target_buy = excluded.target_buy,"
            " notes = excluded.notes",
            (symbol.upper(), list_name, date.today().isoformat(), thesis,
             target_buy, notes),
        )

    def watchlist(self, list_name: str | None = None) -> pd.DataFrame:
        if list_name:
            rows = self.connection().execute(
                "SELECT * FROM watchlist WHERE list_name = ? ORDER BY symbol",
                (list_name,)).fetchall()
        else:
            rows = self.connection().execute(
                "SELECT * FROM watchlist ORDER BY list_name, symbol").fetchall()
        if not rows:
            return pd.DataFrame(columns=["symbol", "list_name", "added", "thesis",
                                         "target_buy", "notes"])
        return pd.DataFrame([dict(r) for r in rows])

    def unwatch(self, symbol: str, list_name: str = "default") -> None:
        self.connection().execute(
            "DELETE FROM watchlist WHERE symbol = ? AND list_name = ?",
            (symbol.upper(), list_name))


_STORE: PortfolioStore | None = None


def store(path: Path | str | None = None) -> PortfolioStore:
    global _STORE
    if _STORE is None or path is not None:
        _STORE = PortfolioStore(path)
    return _STORE
