#!/usr/bin/env python3
"""Sync Taiwan stock code list (with industry classification) into SQLite.

Official sources (OpenAPI):
- TWSE listed companies:        https://openapi.twse.com.tw/v1/opendata/t187ap03_L
- TPEx OTC companies:           https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O
- TPEx emerging companies (興櫃): https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_R

Industry code mapping source:
  TWSE Listed Company Industry Classification and Adjustment Guidelines
  https://twse-regulation.twse.com.tw/m/LawContent.aspx?FID=FL007104

This script builds a single table `tw_stock_list` in the target SQLite DB.
The DB is intentionally separate from tpex_industry_chain.sqlite3 because
the data sources span both TWSE and TPEx (not TPEx-only).

Usage:
  python scripts/tw_stock_list_sync.py
  python scripts/tw_stock_list_sync.py --db data/tw_stock_list.sqlite3
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass(frozen=True)
class Stock:
    stock_code: str
    stock_name: str
    market: str          # TWSE_LISTED / TPEX_OTC / TPEX_EMERGING
    industry_code: str   # raw code from API (e.g. "24")
    industry_name: str   # resolved Chinese name (e.g. "半導體業")
    source_url: str
    source_as_of: str    # yyyyMMdd-ish string from API if available


TWSE_LISTED_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
TPEX_OTC_URL = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"
TPEX_EMERGING_URL = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_R"

REQUEST_TIMEOUT = 60

# Official TWSE/TPEx unified industry classification codes.
# Source: TWSE Listed Company Industry Classification and Adjustment Guidelines
#   https://twse-regulation.twse.com.tw/m/LawContent.aspx?FID=FL007104
# Both TWSE and TPEx (OTC/Emerging) share the same code system.
# Codes 07 and 13 were retired in the latest reclassification (Jul 2023):
#   former 07 (化學工業) → 21; former 13 (電子工業) → split into 24–31.
INDUSTRY_CODE_MAP: dict[str, str] = {
    "01": "水泥工業",
    "02": "食品工業",
    "03": "塑膠工業",
    "04": "紡織纖維",
    "05": "電機機械",
    "06": "電器電纜",
    "08": "玻璃陶瓷",
    "09": "造紙工業",
    "10": "鋼鐵工業",
    "11": "橡膠工業",
    "12": "汽車工業",
    "14": "建材營造",
    "15": "航運業",
    "16": "觀光餐旅",
    "17": "金融保險",
    "18": "貿易百貨",
    "19": "綜合",
    "20": "其他",
    "21": "化學工業",
    "22": "生技醫療業",
    "23": "油電燃氣業",
    "24": "半導體業",
    "25": "電腦及週邊設備業",
    "26": "光電業",
    "27": "通信網路業",
    "28": "電子零組件業",
    "29": "電子通路業",
    "30": "資訊服務業",
    "31": "其他電子業",
    "32": "文化創意業",
    "33": "農業科技業",
    "34": "電子商務",
    "35": "綠能環保",
    "36": "數位雲端",
    "37": "運動休閒",
    "38": "居家生活",
}


def resolve_industry(code: str) -> str:
    """Return the official Chinese industry name for a given code.

    Falls back to the raw code string if the code is not in the mapping,
    so new categories added by regulators are still stored rather than lost.
    """
    return INDUSTRY_CODE_MAP.get(code.strip(), code.strip())


def fetch_json(url: str) -> list[dict]:
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def load_twse_listed() -> list[Stock]:
    data = fetch_json(TWSE_LISTED_URL)
    out: list[Stock] = []
    for row in data:
        code = (row.get("公司代號") or "").strip()
        name = (row.get("公司簡稱") or row.get("公司名稱") or "").strip()
        as_of = (row.get("出表日期") or "").strip()
        ind_code = (row.get("產業別") or "").strip()
        if not code or not name:
            continue
        out.append(
            Stock(
                stock_code=code,
                stock_name=name,
                market="TWSE_LISTED",
                industry_code=ind_code,
                industry_name=resolve_industry(ind_code),
                source_url=TWSE_LISTED_URL,
                source_as_of=as_of,
            )
        )
    return out


def load_tpex(url: str, market: str) -> list[Stock]:
    data = fetch_json(url)
    out: list[Stock] = []
    for row in data:
        code = (row.get("SecuritiesCompanyCode") or "").strip()
        name = (row.get("CompanyAbbreviation") or row.get("CompanyName") or "").strip()
        as_of = (row.get("Date") or "").strip()
        ind_code = (row.get("SecuritiesIndustryCode") or "").strip()
        if not code or not name:
            continue
        out.append(
            Stock(
                stock_code=code,
                stock_name=name,
                market=market,
                industry_code=ind_code,
                industry_name=resolve_industry(ind_code),
                source_url=url,
                source_as_of=as_of,
            )
        )
    return out


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tw_stock_list (
            stock_code    TEXT PRIMARY KEY,
            stock_name    TEXT NOT NULL,
            market        TEXT NOT NULL,
            industry_code TEXT NOT NULL DEFAULT '',
            industry_name TEXT NOT NULL DEFAULT '',
            source_url    TEXT NOT NULL,
            source_as_of  TEXT NOT NULL DEFAULT '',
            synced_at     TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tw_stock_market   ON tw_stock_list (market)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tw_stock_industry ON tw_stock_list (industry_code)")
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync TW stock list (with industry) into SQLite")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/tw_stock_list.sqlite3"),
        help="Target SQLite database path (default: data/tw_stock_list.sqlite3)",
    )
    args = parser.parse_args()

    args.db.parent.mkdir(parents=True, exist_ok=True)
    synced_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    stocks: list[Stock] = []
    stocks.extend(load_twse_listed())
    stocks.extend(load_tpex(TPEX_OTC_URL, "TPEX_OTC"))
    stocks.extend(load_tpex(TPEX_EMERGING_URL, "TPEX_EMERGING"))

    # If a code appears in multiple markets (should be rare), keep the first seen.
    dedup: dict[str, Stock] = {}
    for s in stocks:
        dedup.setdefault(s.stock_code, s)

    conn = sqlite3.connect(args.db)
    try:
        init_db(conn)
        conn.execute("DELETE FROM tw_stock_list")
        conn.executemany(
            """
            INSERT INTO tw_stock_list
              (stock_code, stock_name, market, industry_code, industry_name,
               source_url, source_as_of, synced_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    s.stock_code,
                    s.stock_name,
                    s.market,
                    s.industry_code,
                    s.industry_name,
                    s.source_url,
                    s.source_as_of,
                    synced_at,
                )
                for s in dedup.values()
            ],
        )
        conn.commit()
    finally:
        conn.close()

    print(f"rows: {len(dedup)}  db: {args.db.resolve()}")


if __name__ == "__main__":
    main()
