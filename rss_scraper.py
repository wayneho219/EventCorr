import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time

# ── 設定區 ────────────────────────────────────────────────────────────────────
PTT_BASE      = "https://www.ptt.cc"
PTT_BOARDS    = ["Stock"]  # 要爬的看板，可加 "StockForum" 等
MAX_PER_STOCK = 10          # 每支股票最多保留幾篇

# PTT 網頁版需帶 over18 cookie，不需登入帳號
_SESSION = requests.Session()
_SESSION.cookies.set("over18", "1")
_SESSION.headers.update({
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0.0.0 Safari/537.36",
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
    "Referer":         "https://www.ptt.cc/bbs/Stock/index.html",
})


def _get(url: str, retries: int = 3) -> requests.Response | None:
    """帶重試的 GET，遇到連線錯誤最多重試 retries 次。"""
    for attempt in range(1, retries + 1):
        try:
            resp = _SESSION.get(url, timeout=15)
            resp.raise_for_status()
            return resp
        except Exception as e:
            print(f"  [WARN] 第 {attempt} 次請求失敗 {url}：{e}")
            if attempt < retries:
                time.sleep(2 * attempt)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# PTT 搜尋 + 全文抓取
# ──────────────────────────────────────────────────────────────────────────────

def _search_articles(board: str, query: str, max_count: int) -> list[dict]:
    """
    用 PTT 內建搜尋取得指定關鍵字的文章清單。
    URL 格式：https://www.ptt.cc/bbs/{board}/search?q={query}

    回傳：[{"title": ..., "url": ..., "date": ...}, ...]，最多 max_count 筆。
    """
    results = []
    url = f"{PTT_BASE}/bbs/{board}/search?q={requests.utils.quote(query)}"

    while len(results) < max_count:
        resp = _get(url)
        if resp is None:
            break

        soup = BeautifulSoup(resp.text, "html.parser")

        for row in soup.select("div.r-ent"):
            title_tag = row.select_one("div.title a")
            date_tag  = row.select_one("div.date")
            if not title_tag:
                continue  # 已刪除文章

            results.append({
                "title": title_tag.get_text(strip=True),
                "url":   PTT_BASE + title_tag["href"],
                "date":  date_tag.get_text(strip=True) if date_tag else "",
            })

            if len(results) >= max_count:
                break

        # 翻下一頁搜尋結果
        prev = soup.select_one("a.btn.wide:-soup-contains('上頁')")
        if not prev or len(results) >= max_count:
            break
        url = PTT_BASE + prev["href"]
        time.sleep(0.3)

    return results[:max_count]


def _get_article_content(url: str) -> str:
    """取得單篇文章內文，移除 header 與推文。"""
    resp = _get(url)
    if resp is None:
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")
    main = soup.select_one("div#main-content")
    if not main:
        return ""

    for tag in main.select("div.article-metaline, div.article-metaline-right"):
        tag.decompose()
    for tag in main.select("div.push"):
        tag.decompose()

    content = main.get_text(separator="\n")
    content = re.split(r"\n--\s*\n", content)[0]
    content = re.sub(r"\n{3,}", "\n\n", content).strip()
    return content


# ──────────────────────────────────────────────────────────────────────────────
# 主函式
# ──────────────────────────────────────────────────────────────────────────────

def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "stock_id", "ArticleTitle", "ArticleText", "ArticleCreateTime"
    ])


def fetch_and_process_ptt(
    stock_dict: dict,
    boards: list = PTT_BOARDS,
    max_per_stock: int = MAX_PER_STOCK,
) -> pd.DataFrame:
    """
    對每支股票用代號直接搜尋 PTT，取前 max_per_stock 篇，不需帳號。

    :param stock_dict:    {'2330': ['台積電', ...], ...}
    :param boards:        要搜尋的看板 list
    :param max_per_stock: 每支股票最多取幾篇
    :return: DataFrame 欄位：stock_id, ArticleTitle, ArticleText, ArticleCreateTime
    """
    parsed_data: list[dict] = []
    seen_urls:   set[str]   = set()

    for stock_id in stock_dict:
        for board in boards:
            print(f"  搜尋 [{board}] 股票代號：{stock_id}")
            articles = _search_articles(board, stock_id, max_per_stock)
            print(f"    找到 {len(articles)} 篇，抓取全文...")

            for art in articles:
                url = art["url"]
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                content = _get_article_content(url)
                time.sleep(0.3)

                parsed_data.append({
                    "stock_id":          stock_id,
                    "ArticleTitle":      art["title"],
                    "ArticleText":       content,
                    "ArticleCreateTime": art["date"],
                    "_source_url":       url,
                })

    # ── 轉 DataFrame ──────────────────────────────────────────────────
    df = pd.DataFrame(parsed_data)

    if df.empty:
        print("\n未找到任何文章。")
        return _empty_df()

    # PTT 列表日期格式為 "4/13"，補上今年年份
    current_year = pd.Timestamp.now().year
    df["ArticleCreateTime"] = pd.to_datetime(
        df["ArticleCreateTime"].apply(lambda d: f"{current_year}/{d.strip()}"),
        format="%Y/%m/%d", errors="coerce"
    )

    df = df.drop_duplicates(subset=["stock_id", "_source_url"])
    df = df.sort_values(["stock_id", "ArticleCreateTime"], ascending=[True, False])
    df = df.drop(columns=["_source_url"]).reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 全市場清單對接
# ──────────────────────────────────────────────────────────────────────────────

def load_stock_dict_from_csv(csv_path: str) -> dict:
    """
    從全市場股票清單 CSV 建立 stock_dict。

    預期欄位：
        stock_id   — 股票代號，例如 2330
        stock_name — 公司簡稱，例如 台積電
    可選欄位：
        stock_alias — 其他常用名稱（逗號分隔），例如 TSMC,神山
    """
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    stock_dict: dict = {}

    for _, row in df.iterrows():
        sid      = str(row["stock_id"]).strip()
        name     = str(row["stock_name"]).strip()
        keywords = [name, sid]

        if "stock_alias" in df.columns and row["stock_alias"]:
            aliases = [a.strip() for a in row["stock_alias"].split(",") if a.strip()]
            keywords.extend(aliases)

        stock_dict[sid] = list(dict.fromkeys(k for k in keywords if k))

    return stock_dict


# ──────────────────────────────────────────────────────────────────────────────
# 使用範例
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── 方式 A：手動指定測試股票 ──────────────────────────────────────
    my_stock_dict = {
        "2330": ["台積電", "TSMC", "神山", "2330"],
        "2317": ["鴻海", "Foxconn", "2317"],
        "3324": ["雙鴻", "3324"],
        "3017": ["奇鋐", "3017"],
        "2354": ["鴻準", "2354"],
    }

    # ── 方式 B：從 CSV 載入全市場清單 ────────────────────────────────
    # my_stock_dict = load_stock_dict_from_csv("tw_stocks.csv")

    print("開始從 PTT 搜尋各股票文章...")
    result_df = fetch_and_process_ptt(my_stock_dict)

    print(f"\n完成！共解析出 {len(result_df)} 筆資料。")
    print(result_df.to_string())

    output_path = "articles.csv"
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n已存檔 → {output_path}")
