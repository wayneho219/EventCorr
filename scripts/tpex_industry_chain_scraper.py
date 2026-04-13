#!/usr/bin/env python3
"""Scrape TPEx Industry Chain site (ic.tpex.org.tw) into SQLite/JSON.

Columns: major category (from index) -> chain topic (introduce?ic=) ->
segment (companyList_*) -> Taiwan listed stock code & name.

Usage:
  python scripts/tpex_industry_chain_scraper.py
  python scripts/tpex_industry_chain_scraper.py --out data/tpex_industry_chain.sqlite3 \
      --json data/tpex_industry_chain.json
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://ic.tpex.org.tw/"
INDEX_PATH = "index.php"
INTRO_PATH = "introduce.php"
CHAIN_TITLE_SUFFIX = "\u7522\u696d\u93c8\u7c21\u4ecb"
REQUEST_TIMEOUT = 30
SLEEP_SEC = 0.35


@dataclass(frozen=True)
class Row:
    major_industry: str
    chain_ic: str
    chain_topic: str
    segment_code: str
    segment_name: str
    listing_bucket: str
    stock_code: str
    stock_name: str


def fetch_text(session: requests.Session, path: str, params: dict | None = None) -> str:
    url = urljoin(BASE, path)
    r = session.get(url, params=params or {}, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    return r.text


def parse_major_by_ic_from_index(html: str) -> dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    mapping: dict[str, str] = {}

    def ic_from_href(href: str | None) -> str | None:
        if not href:
            return None
        m = re.search(r"introduce\.php\?ic=([A-Z0-9]+)", href)
        return m.group(1) if m else None

    def major_for_element(el) -> str | None:
        item = el.find_parent("div", class_="item")
        if not item:
            return None
        link = item.find("a", class_="link")
        if not link:
            return None
        txt = link.find("span", class_="txt")
        return txt.get_text(strip=True) if txt else None

    for a in soup.select('a[href*="introduce.php?ic="]'):
        ic = ic_from_href(a.get("href"))
        if ic:
            mapping[ic] = major_for_element(a) or ""

    for span in soup.find_all("span", class_="itemLink"):
        oc = span.get("onclick") or ""
        m = re.search(r"introduce\.php\?ic=([A-Z0-9]+)", oc)
        if not m:
            continue
        ic = m.group(1)
        mapping[ic] = major_for_element(span) or ""

    return mapping


def normalize_chain_topic(h3_text: str) -> str:
    t = h3_text.strip()
    if t.endswith(CHAIN_TITLE_SUFFIX):
        t = t[: -len(CHAIN_TITLE_SUFFIX)].strip()
    return t


def listing_bucket_for_link(a_tag) -> str:
    tr = a_tag.find_parent("tr")
    if not tr:
        return ""
    first_td = tr.find("td")
    if not first_td:
        return ""
    b = first_td.find("b")
    if not b:
        return ""
    return (b.get_text(strip=True) or "").strip()


def parse_intro_page(html: str, chain_ic: str, major_industry: str) -> list[Row]:
    soup = BeautifulSoup(html, "html.parser")
    h3 = soup.find("h3")
    chain_topic = normalize_chain_topic(h3.get_text(strip=True)) if h3 else ""

    rows: list[Row] = []
    for div in soup.select("div[id^='companyList_']"):
        if div.find_parent("noscript"):
            continue
        seg_id = div.get("id") or ""
        m = re.match(r"companyList_([0-9A-Z]+)", seg_id)
        segment_code = m.group(1) if m else seg_id.replace("companyList_", "")
        segment_name = (div.get("title") or "").strip()

        for a in div.select('a[href*="company_basic.php?stk_code="]'):
            href = a.get("href") or ""
            sm = re.search(r"stk_code=(\d+)", href)
            if not sm:
                continue
            code = sm.group(1)
            name = (a.get("title") or a.get_text(strip=True) or "").strip()
            bucket = listing_bucket_for_link(a)
            rows.append(
                Row(
                    major_industry=major_industry,
                    chain_ic=chain_ic,
                    chain_topic=chain_topic,
                    segment_code=segment_code,
                    segment_name=segment_name,
                    listing_bucket=bucket,
                    stock_code=code,
                    stock_name=name,
                )
            )
    return rows


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tpex_industry_chain (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            major_industry TEXT NOT NULL,
            chain_ic TEXT NOT NULL,
            chain_topic TEXT NOT NULL,
            segment_code TEXT NOT NULL,
            segment_name TEXT NOT NULL,
            listing_bucket TEXT NOT NULL DEFAULT '',
            stock_code TEXT NOT NULL,
            stock_name TEXT NOT NULL,
            scraped_at TEXT NOT NULL,
            UNIQUE (chain_ic, segment_code, stock_code, listing_bucket)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chain_ic ON tpex_industry_chain (chain_ic)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_stock_code ON tpex_industry_chain (stock_code)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_segment ON tpex_industry_chain (segment_name)"
    )
    conn.commit()


def upsert_rows(conn: sqlite3.Connection, rows: Iterable[Row], scraped_at: str) -> int:
    n = 0
    for r in rows:
        conn.execute(
            """
            INSERT OR REPLACE INTO tpex_industry_chain (
                major_industry, chain_ic, chain_topic, segment_code, segment_name,
                listing_bucket, stock_code, stock_name, scraped_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r.major_industry,
                r.chain_ic,
                r.chain_topic,
                r.segment_code,
                r.segment_name,
                r.listing_bucket,
                r.stock_code,
                r.stock_name,
                scraped_at,
            ),
        )
        n += 1
    conn.commit()
    return n


def export_json(rows: list[Row], path: Path, scraped_at: str) -> None:
    payload = [
        {
            "major_industry": r.major_industry,
            "chain_ic": r.chain_ic,
            "chain_topic": r.chain_topic,
            "segment_code": r.segment_code,
            "segment_name": r.segment_name,
            "listing_bucket": r.listing_bucket,
            "stock_code": r.stock_code,
            "stock_name": r.stock_name,
            "scraped_at": scraped_at,
        }
        for r in rows
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="TPEx industry chain -> SQLite / JSON")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/tpex_industry_chain.sqlite3"),
        help="Output SQLite path",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional JSON export path",
    )
    parser.add_argument(
        "--sleep", type=float, default=SLEEP_SEC, help="Seconds between HTTP requests"
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "EventCorr/1.0 research scraper",
            "Accept-Language": "zh-TW,zh;q=0.9",
        }
    )

    index_html = fetch_text(session, INDEX_PATH)
    major_by_ic = parse_major_by_ic_from_index(index_html)
    chain_codes = sorted(major_by_ic.keys(), key=lambda x: (len(x), x))

    scraped_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    all_rows: list[Row] = []

    for ic in chain_codes:
        html = fetch_text(session, INTRO_PATH, params={"ic": ic})
        major = major_by_ic.get(ic, "")
        rows = parse_intro_page(html, chain_ic=ic, major_industry=major)
        all_rows.extend(rows)
        time.sleep(max(0.0, args.sleep))

    conn = sqlite3.connect(args.out)
    try:
        init_db(conn)
        conn.execute("DELETE FROM tpex_industry_chain")
        conn.commit()
        n = upsert_rows(conn, all_rows, scraped_at)
    finally:
        conn.close()

    print(f"chains: {len(chain_codes)}  rows: {n}  db: {args.out.resolve()}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        export_json(all_rows, args.json, scraped_at)
        print(f"json: {args.json.resolve()}")


if __name__ == "__main__":
    main()
