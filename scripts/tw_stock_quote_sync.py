#!/usr/bin/env python3
"""Sync today's stock price change (漲跌幅) and trading volume (交易量) into tw_stock_list.

Official sources (OpenAPI):
- TWSE listed:    https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL
- TPEx OTC:       https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes
- TPEx emerging:  https://www.tpex.org.tw/openapi/v1/tpex_esb_daily_close_quotes

Columns added / updated in tw_stock_list:
  close_price  REAL   -- 收盤價
  change_val   TEXT   -- 漲跌 (e.g. "+1.60", "-0.28")
  change_pct   TEXT   -- 漲跌幅 (e.g. "+2.02%", "-0.74%")
  volume       INTEGER -- 交易量 (股)
  quote_date   TEXT   -- 資料日期 (ISO format, e.g. "2026-04-13")

Usage:
  python scripts/tw_stock_quote_sync.py
  python scripts/tw_stock_quote_sync.py --db data/tw_stock_list.sqlite3
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import requests

TWSE_DAY_ALL_URL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TPEX_OTC_QUOTE_URL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
TPEX_ESB_QUOTE_URL = "https://www.tpex.org.tw/openapi/v1/tpex_esb_daily_close_quotes"

REQUEST_TIMEOUT = 60


@dataclass(frozen=True)
class Quote:
    stock_code: str
    close_price: float | None
    change_val: str   # "+1.60" / "-0.28" / "" if unavailable
    change_pct: str   # "+2.02%" / "-0.74%" / ""
    volume: int | None
    quote_date: str   # ISO "2026-04-13"


def fetch_json(url: str) -> list[dict]:
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def roc_to_iso(roc_date: str) -> str:
    """Convert ROC date string to ISO format.

    TWSE/TPEx use 7-digit ROC dates: YYYMMDD (e.g. '1150410' → '2026-04-10').
    """
    s = roc_date.strip()
    if len(s) == 7:
        ad_year = int(s[:3]) + 1911
        return f"{ad_year}-{s[3:5]}-{s[5:7]}"
    return s  # unexpected format — return as-is


def to_float(val: str) -> float | None:
    """Parse a numeric string; return None if blank or non-numeric."""
    s = val.strip().replace(",", "")
    if not s or s in {"----", "--", "N/A"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def compute_pct(close: float, change: float) -> str:
    prev = close - change
    if prev == 0:
        return ""
    pct = change / prev * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"


def fmt_change(change: float) -> str:
    sign = "+" if change >= 0 else ""
    return f"{sign}{change:.2f}"


def load_twse_quotes() -> list[Quote]:
    data = fetch_json(TWSE_DAY_ALL_URL)
    out: list[Quote] = []
    for row in data:
        code = (row.get("Code") or "").strip()
        if not code:
            continue
        close = to_float(row.get("ClosingPrice") or "")
        change = to_float(row.get("Change") or "")
        vol_raw = to_float(row.get("TradeVolume") or "")
        date = roc_to_iso(row.get("Date") or "")

        if close is not None and change is not None:
            chg_str = fmt_change(change)
            pct_str = compute_pct(close, change)
        else:
            chg_str = ""
            pct_str = ""

        out.append(
            Quote(
                stock_code=code,
                close_price=close,
                change_val=chg_str,
                change_pct=pct_str,
                volume=int(vol_raw) if vol_raw is not None else None,
                quote_date=date,
            )
        )
    return out


def load_tpex_quotes(url: str) -> list[Quote]:
    try:
        data = fetch_json(url)
    except Exception as e:
        print(f"  warning: failed to fetch {url}: {e}")
        return []

    out: list[Quote] = []
    for row in data:
        code = (row.get("SecuritiesCompanyCode") or "").strip()
        if not code:
            continue

        close = to_float(row.get("Close") or "")
        # TPEx Change already has +/- sign as a string, e.g. "+0.28"
        change_raw = (row.get("Change") or "").strip()
        change = to_float(change_raw)
        vol_raw = to_float(row.get("TradingShares") or "")
        date = roc_to_iso(row.get("Date") or "")

        if close is not None and change is not None:
            # Preserve the original sign string from the API; re-format for consistency
            chg_str = fmt_change(change)
            pct_str = compute_pct(close, change)
        else:
            chg_str = ""
            pct_str = ""

        out.append(
            Quote(
                stock_code=code,
                close_price=close,
                change_val=chg_str,
                change_pct=pct_str,
                volume=int(vol_raw) if vol_raw is not None else None,
                quote_date=date,
            )
        )
    return out


def ensure_quote_columns(conn: sqlite3.Connection) -> None:
    """Add quote columns to tw_stock_list if they don't exist yet."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(tw_stock_list)")}
    new_cols = [
        ("close_price", "REAL"),
        ("change_val",  "TEXT"),
        ("change_pct",  "TEXT"),
        ("volume",      "INTEGER"),
        ("quote_date",  "TEXT"),
    ]
    for col_name, col_type in new_cols:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE tw_stock_list ADD COLUMN {col_name} {col_type}")
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync today's TW stock quotes into tw_stock_list")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/tw_stock_list.sqlite3"),
    )
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(f"DB not found: {args.db}. Run tw_stock_list_sync.py first.")

    print("Fetching TWSE listed quotes ...")
    twse_quotes = load_twse_quotes()
    print(f"  {len(twse_quotes)} records")

    print("Fetching TPEx OTC quotes ...")
    otc_quotes = load_tpex_quotes(TPEX_OTC_QUOTE_URL)
    print(f"  {len(otc_quotes)} records")

    print("Fetching TPEx emerging quotes ...")
    esb_quotes = load_tpex_quotes(TPEX_ESB_QUOTE_URL)
    print(f"  {len(esb_quotes)} records")

    # Merge; first seen wins (TWSE → OTC → ESB)
    all_quotes: dict[str, Quote] = {}
    for q in (*twse_quotes, *otc_quotes, *esb_quotes):
        all_quotes.setdefault(q.stock_code, q)

    conn = sqlite3.connect(args.db)
    try:
        ensure_quote_columns(conn)

        updated = 0
        for q in all_quotes.values():
            cur = conn.execute(
                """
                UPDATE tw_stock_list
                SET close_price = ?,
                    change_val  = ?,
                    change_pct  = ?,
                    volume      = ?,
                    quote_date  = ?
                WHERE stock_code = ?
                """,
                (q.close_price, q.change_val, q.change_pct, q.volume, q.quote_date, q.stock_code),
            )
            updated += cur.rowcount

        conn.commit()
    finally:
        conn.close()

    print(f"Updated {updated} rows in {args.db.resolve()}")


if __name__ == "__main__":
    main()
