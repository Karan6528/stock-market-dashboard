"""Holdings stored in SQLite (data/portfolio.db) so they survive app restarts."""
from __future__ import annotations

import sqlite3
from contextlib import closing

import pandas as pd

from .config import DATA_DIR

DB_PATH = DATA_DIR / "portfolio.db"


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS stocks (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               ticker TEXT NOT NULL,
               shares REAL NOT NULL,
               buy_price REAL NOT NULL
           )"""
    )
    cols = {r[1] for r in conn.execute("PRAGMA table_info(stocks)")}
    if "buy_date" not in cols:  # migrate the original schema
        conn.execute("ALTER TABLE stocks ADD COLUMN buy_date TEXT")
    return conn


def load() -> pd.DataFrame:
    with closing(_connect()) as conn:
        return pd.read_sql_query("SELECT id, ticker, shares, buy_price, buy_date FROM stocks ORDER BY id", conn)


def add(ticker: str, shares: float, buy_price: float, buy_date: str | None) -> None:
    with closing(_connect()) as conn, conn:
        conn.execute("INSERT INTO stocks (ticker, shares, buy_price, buy_date) VALUES (?, ?, ?, ?)",
                     (ticker.upper().strip(), float(shares), float(buy_price), buy_date))


def delete(row_id: int) -> None:
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM stocks WHERE id = ?", (int(row_id),))


def replace_all(df: pd.DataFrame) -> None:
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM stocks")
        for r in df.itertuples(index=False):
            conn.execute("INSERT INTO stocks (ticker, shares, buy_price, buy_date) VALUES (?, ?, ?, ?)",
                         (str(r.ticker).upper().strip(), float(r.shares), float(r.buy_price),
                          None if pd.isna(r.buy_date) else str(r.buy_date)))
