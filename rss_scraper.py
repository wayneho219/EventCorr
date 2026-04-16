import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import json
import os

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

        # 翻下一頁搜尋結果（按鈕存在但無 href 表示已是第一頁）
        prev = soup.select_one("a.btn.wide:-soup-contains('上頁')")
        if not prev or not prev.get("href") or len(results) >= max_count:
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


CHECKPOINT_FILE = "_ptt_checkpoint.json"   # 斷點紀錄檔


def _load_checkpoint() -> set[str]:
    """讀取斷點，回傳已完成的 stock_id 集合。"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, encoding="utf-8") as f:
            return set(json.load(f).get("done", []))
    return set()


def _save_checkpoint(done: set[str]) -> None:
    """將已完成的 stock_id 寫入斷點檔。"""
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({"done": list(done)}, f, ensure_ascii=False)


def _append_to_csv(rows: list[dict], output_path: str, write_header: bool) -> None:
    """將一批資料 append 寫入 CSV。"""
    if not rows:
        return
    current_year = pd.Timestamp.now().year
    df = pd.DataFrame(rows)
    df["ArticleCreateTime"] = pd.to_datetime(
        df["ArticleCreateTime"].apply(lambda d: f"{current_year}/{d.strip()}"),
        format="%Y/%m/%d", errors="coerce"
    )
    df = df.drop_duplicates(subset=["stock_id", "_source_url"])
    df = df.drop(columns=["_source_url"])
    df.to_csv(
        output_path,
        mode="w" if write_header else "a",
        header=write_header,
        index=False,
        encoding="utf-8-sig",
    )


def fetch_and_process_ptt(
    stock_dict: dict,
    boards: list = PTT_BOARDS,
    max_per_stock: int = MAX_PER_STOCK,
    output_path: str = "articles.csv",
) -> pd.DataFrame:
    """
    對每支股票用代號直接搜尋 PTT，取前 max_per_stock 篇，不需帳號。
    支援斷點續跑：中斷後重新執行會跳過已完成的股票。
    每處理完一支股票即批次寫入 CSV，不需等全部完成。

    :param stock_dict:    {'2330': ['台積電', ...], ...}
    :param boards:        要搜尋的看板 list
    :param max_per_stock: 每支股票最多取幾篇
    :param output_path:   輸出 CSV 路徑
    :return: DataFrame 欄位：stock_id, ArticleTitle, ArticleText, ArticleCreateTime
    """
    done_ids    = _load_checkpoint()
    seen_urls:  set[str] = set()
    first_write = not os.path.exists(output_path)  # 檔案不存在時才寫 header

    total   = len(stock_dict)
    skipped = len(done_ids & stock_dict.keys())
    print(f"共 {total} 支股票，已完成 {skipped} 支，剩餘 {total - skipped} 支。")

    for i, stock_id in enumerate(stock_dict, 1):
        if stock_id in done_ids:
            print(f"  [{i}/{total}] {stock_id} 已完成，跳過。")
            continue

        batch: list[dict] = []

        for board in boards:
            print(f"  [{i}/{total}] 搜尋 [{board}] {stock_id}...")
            articles = _search_articles(board, stock_id, max_per_stock)
            print(f"    找到 {len(articles)} 篇，抓取全文...")

            for art in articles:
                url = art["url"]
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                content = _get_article_content(url)
                time.sleep(0.3)

                batch.append({
                    "stock_id":          stock_id,
                    "ArticleTitle":      art["title"],
                    "ArticleText":       content,
                    "ArticleCreateTime": art["date"],
                    "_source_url":       url,
                })

        # ── 批次寫入 CSV ──────────────────────────────────────────────
        _append_to_csv(batch, output_path, write_header=first_write)
        first_write = False
        print(f"    寫入 {len(batch)} 筆 → {output_path}")

        # ── 更新斷點 ──────────────────────────────────────────────────
        done_ids.add(stock_id)
        _save_checkpoint(done_ids)

    # ── 清除斷點（全部完成）─────────────────────────────────────────
    if done_ids >= stock_dict.keys():
        os.remove(CHECKPOINT_FILE)
        print(f"\n所有股票處理完畢，已清除斷點檔。")

    # ── 讀回完整結果回傳 ─────────────────────────────────────────────
    if os.path.exists(output_path):
        return pd.read_csv(output_path, encoding="utf-8-sig")
    return _empty_df()


# ──────────────────────────────────────────────────────────────────────────────
# 全市場清單對接
# ──────────────────────────────────────────────────────────────────────────────

def load_stock_dict_from_csv(csv_path: str) -> dict:
    """
    從全市場股票清單 CSV 建立 stock_dict。
    代號欄位自動偵測：優先用 stock_code，其次 stock_id。

    預期欄位：
        stock_code 或 stock_id — 股票代號，例如 2330
    可選欄位：
        stock_name  — 公司簡稱，例如 台積電
        stock_alias — 其他常用名稱（逗號分隔），例如 TSMC,神山
    """
    df = pd.read_csv(csv_path, dtype=str).fillna("")

    if "stock_code" in df.columns:
        id_col = "stock_code"
    elif "stock_id" in df.columns:
        id_col = "stock_id"
    else:
        raise ValueError(f"{csv_path} 找不到 stock_code 或 stock_id 欄位，請確認欄位名稱。")

    stock_dict: dict = {}

    for _, row in df.iterrows():
        sid      = str(row[id_col]).strip()
        keywords = [sid]

        if "stock_name" in df.columns and row["stock_name"]:
            keywords.insert(0, str(row["stock_name"]).strip())

        if "stock_alias" in df.columns and row["stock_alias"]:
            aliases = [a.strip() for a in row["stock_alias"].split(",") if a.strip()]
            keywords.extend(aliases)

        stock_dict[sid] = list(dict.fromkeys(k for k in keywords if k))

    return stock_dict


# ──────────────────────────────────────────────────────────────────────────────
# 使用範例
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":



    # ── 方式 B：從 CSV 載入全市場清單 ────────────────────────────────
    my_stock_dict = load_stock_dict_from_csv("tw_stocks.csv")

    output_path = "articles.csv"

    print("開始從 PTT 搜尋各股票文章...")
    result_df = fetch_and_process_ptt(my_stock_dict, output_path=output_path)

    print(f"\n完成！共 {len(result_df)} 筆資料已存至 {output_path}")
