"""
Stock Comment Hierarchical Topic Modeling Pipeline
===================================================
第一階層固定為 tw_stocks.csv 產業分類（industry_name）
PHASE = "cluster"  → 執行 Step 1-5，輸出 label_input.txt 供人工命名
PHASE = "label"    → 讀取 topic_labels.json，套用標籤並輸出視覺化結果
"""

# ┌─────────────────────────────────────────┐
# │  設定執行階段：                          │
# │    "cluster" ── 分群 + 輸出命名素材      │
# │    "label"   ── 套用標籤 + 輸出結果      │
# └─────────────────────────────────────────┘
PHASE = "label"

# AUTO_LEVELS = True  → 自動偵測各層最佳群數（CH 分數偏好少群，通常不建議）
# AUTO_LEVELS = False → 使用下方 LEVELS 手動指定（建議）
AUTO_LEVELS = False

LEVELS: dict[str, int] = {   # AUTO_LEVELS=False 時才使用
    "medium": 25,   # 建議範圍：15–30（中層主題數）
    "fine":   60,   # 建議範圍：40–80（細層主題數）
}

# 自動偵測的搜尋範圍（可視需求調整）
AUTO_RANGES = {
    "medium": range(8, 30),
    "fine":   range(20, 60),
}

import json
import os
import re
import numpy as np
import pandas as pd
import jieba
import plotly.graph_objects as go

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from umap import UMAP
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 ── DOCUMENTS + 產業分類 JOIN
# ══════════════════════════════════════════════════════════════════════════════
print("STEP 1 | 載入資料 + 產業分類 JOIN")
df = pd.read_csv("articles.csv", encoding="utf-8-sig", parse_dates=["ArticleCreateTime"])
df = df.dropna(subset=["ArticleText"])
df["ArticleText"] = df["ArticleText"].astype(str).str.strip()
df["stock_id"] = df["stock_id"].astype(str)

# 從 tw_stocks.csv 取得產業分類（第一階層）
stocks_df = pd.read_csv("tw_stocks.csv", encoding="utf-8-sig", dtype={"stock_code": str})
industry_map = stocks_df.set_index("stock_code")["industry_name"].to_dict()
df["industry_name"] = df["stock_id"].map(industry_map).fillna("其他")

# 文字清理：去 URL、去非中英數符號、壓縮重複字元
def clean_text(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^\u4e00-\u9fff\w\s]", " ", text)
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)
    return text.strip()

df["ArticleText"] = df["ArticleText"].apply(clean_text)
df = df.reset_index(drop=True)
docs = df["ArticleText"].tolist()

print(f"  總筆數: {len(df)}")
print(f"  股票數: {df['stock_id'].nunique()}")
print(f"  產業數: {df['industry_name'].nunique()}")
print(f"  產業列表: {sorted(df['industry_name'].unique())}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 ── EMBEDDING  (BGE-large-zh, 1024d)
# ══════════════════════════════════════════════════════════════════════════════
EMBED_CACHE = "_embeddings.npy"
if os.path.exists(EMBED_CACHE):
    print("STEP 2 | 載入快取 embedding（跳過重算）")
    embeddings = np.load(EMBED_CACHE)
else:
    print("STEP 2 | BGE-large-zh Embedding (1024d)")
    embed_model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
    embeddings = embed_model.encode(
        docs, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    )
    np.save(EMBED_CACHE, embeddings)
    print(f"  embedding 已存檔 → {EMBED_CACHE}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 ── DIMENSION REDUCTION  (UMAP + T-SNE)
# ══════════════════════════════════════════════════════════════════════════════
print("STEP 3 | UMAP 15d + T-SNE 2d")
umap_embeddings = UMAP(
    n_components=15, n_neighbors=30, min_dist=0.0,
    metric="cosine", random_state=42
).fit_transform(embeddings)

tsne_2d = TSNE(
    n_components=2, perplexity=50, learning_rate="auto",
    init="pca", random_state=42, n_jobs=-1
).fit_transform(umap_embeddings)

df["tsne_x"], df["tsne_y"] = tsne_2d[:, 0], tsne_2d[:, 1]

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 ── HIERARCHICAL CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
print("STEP 4 | Ward Hierarchical Clustering")
Z = linkage(pdist(umap_embeddings, metric="euclidean"), method="ward")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4.5 ── AUTO LEVEL DETECTION（Calinski-Harabasz）
# ══════════════════════════════════════════════════════════════════════════════
if AUTO_LEVELS:
    from sklearn.metrics import calinski_harabasz_score

    def _best_k(k_range: range) -> int:
        best_k, best_score = k_range.start, -1.0
        for k in k_range:
            labels = fcluster(Z, k, criterion="maxclust")
            score = calinski_harabasz_score(umap_embeddings, labels)
            if score > best_score:
                best_score, best_k = score, k
        return best_k

    print("STEP 4.5 | 自動偵測最佳群數（Calinski-Harabasz）")
    medium_k = _best_k(AUTO_RANGES["medium"])
    fine_k   = _best_k(range(max(AUTO_RANGES["fine"].start, medium_k + 3),
                             AUTO_RANGES["fine"].stop))
    LEVELS = {"medium": medium_k, "fine": fine_k}
    print(f"  → medium={medium_k}, fine={fine_k}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 ── MULTI-LEVEL CUTTING
# ══════════════════════════════════════════════════════════════════════════════
print("STEP 5 | Multi-level Cutting")
cluster_assignments: dict[str, np.ndarray] = {}
for level, n in LEVELS.items():
    arr = fcluster(Z, n, criterion="maxclust")
    cluster_assignments[level] = arr
    df[f"cluster_{level}"] = arr

# 中途存檔（供 Phase 2 使用）
df.to_parquet("_checkpoint.parquet", index=False)
np.save("_cluster_fine.npy",   cluster_assignments["fine"])
np.save("_cluster_medium.npy", cluster_assignments["medium"])
np.save("_linkage.npy", Z)
print("  分群結果已暫存")


# ══════════════════════════════════════════════════════════════════════════════
# 關鍵詞萃取工具
# ══════════════════════════════════════════════════════════════════════════════
def extract_keywords(indices: list[int], n: int = 15) -> list[str]:
    subset = [docs[i] for i in indices]
    tokenized = [" ".join(jieba.cut(d)) for d in subset]
    vect = TfidfVectorizer(
        max_features=n,
        token_pattern=r"(?u)\b[\u4e00-\u9fff]{2,}\b",
    )
    try:
        vect.fit(tokenized)
        scores = np.asarray(vect.transform(tokenized).mean(axis=0)).flatten()
        top_idx = scores.argsort()[::-1][:n]
        vocab = {v: k for k, v in vect.vocabulary_.items()}
        return [vocab[i] for i in top_idx if i in vocab]
    except Exception:
        return []


def sample_sentences(indices: list[int], n: int = 3) -> list[str]:
    """從群中取樣最短的幾句（較乾淨）。"""
    picked = sorted(indices, key=lambda i: len(docs[i]))[:n]
    return [docs[i][:80].replace("\n", " ") for i in picked]


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 ── 輸出命名素材
# ══════════════════════════════════════════════════════════════════════════════
if PHASE == "cluster":
    print("\n輸出命名素材 → label_input.txt / label_template.json")

    lines = []
    template: dict[str, dict[str, str]] = {}

    # 產業統計（第一階層，無需命名）
    lines.append(f"\n{'═'*60}")
    lines.append("【第一階層】產業分類（來自 tw_stocks.csv，無需命名）")
    lines.append(f"{'═'*60}")
    ind_counts = df["industry_name"].value_counts()
    for ind, cnt in ind_counts.items():
        lines.append(f"  {ind}: {cnt} 篇")

    # NLP 分群命名素材（medium / fine）
    for level in ["fine", "medium"]:
        n_clusters = LEVELS[level]
        assignments = cluster_assignments[level]
        lines.append(f"\n{'═'*60}")
        lines.append(f"【{level.upper()} 層 NLP 主題】{n_clusters} 個主題")
        lines.append(f"{'═'*60}")
        template[level] = {}

        for cid in range(1, n_clusters + 1):
            idx = list(np.where(assignments == cid)[0])
            if not idx:
                continue
            kws = extract_keywords(idx)
            samples = sample_sentences(idx)
            count = len(idx)

            # 顯示該群主要來自哪些產業
            ind_dist = df.iloc[idx]["industry_name"].value_counts().head(3)
            ind_str = ", ".join(f"{k}({v})" for k, v in ind_dist.items())

            lines.append(f"\n[{level} 主題 {cid:02d}]  ({count} 篇)")
            lines.append(f"  主要產業：{ind_str}")
            lines.append(f"  關鍵詞：{', '.join(kws)}")
            lines.append("  範例句：")
            for s in samples:
                lines.append(f"    • {s}")
            lines.append(f"  → 請命名：＿＿＿＿＿＿")

            template[level][str(cid)] = ""

    with open("label_input.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    with open("label_template.json", "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

    print("\n  label_input.txt   ← 貼給 LLM 或自行命名")
    print("  label_template.json ← 填入名稱後將檔名改為 topic_labels.json")
    print("\n完成 Phase 1。命名完畢後：")
    print("  1. 將名稱填入 label_template.json")
    print("  2. 存為 topic_labels.json")
    print('  3. 將程式頂端 PHASE = "cluster" 改為 PHASE = "label"')
    print("  4. 重新執行")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 ── 套用標籤並輸出結果
# ══════════════════════════════════════════════════════════════════════════════
elif PHASE == "label":
    print("\nPHASE 2 | 套用標籤 + 輸出視覺化")

    if not os.path.exists("topic_labels.json"):
        raise FileNotFoundError("找不到 topic_labels.json，請先完成 Phase 1 命名步驟。")

    with open("topic_labels.json", encoding="utf-8") as f:
        topic_labels: dict[str, dict[str, str]] = json.load(f)

    # 從暫存還原分群（跳過耗時步驟）
    df = pd.read_parquet("_checkpoint.parquet")
    for level in ["medium", "fine"]:
        arr = np.load(f"_cluster_{level}.npy")
        int_labels = {int(k): v for k, v in topic_labels[level].items()}
        df[f"label_{level}"] = pd.Series(arr).map(int_labels).values

    # ── T-SNE 散點圖（以產業為顏色）────────────────────────────────────────
    import plotly.express as px
    fig_scatter = px.scatter(
        df, x="tsne_x", y="tsne_y",
        color="industry_name",
        hover_data=["stock_id", "ArticleCreateTime", "label_medium", "label_fine"],
        title="T-SNE ── 產業分布（顏色=產業，可切換 medium/fine 標籤）",
        width=1400, height=900,
    )
    fig_scatter.write_html("tsne_industry.html", include_mathjax=False)
    print("  tsne_industry.html")

    for level in ["medium", "fine"]:
        fig = px.scatter(
            df, x="tsne_x", y="tsne_y",
            color=f"label_{level}",
            hover_data=["stock_id", "industry_name", "ArticleCreateTime", f"label_{level}"],
            title=f"T-SNE ── {level} 層 NLP 主題",
            width=1400, height=900,
        )
        fig.write_html(f"tsne_{level}.html", include_mathjax=False)
        print(f"  tsne_{level}.html")

    # ── 各層主題統計 CSV ─────────────────────────────────────────────────────
    for level in ["medium", "fine"]:
        stat = (
            df.groupby([f"label_{level}", "industry_name"]).size()
            .unstack(fill_value=0)
            .rename_axis(f"主題({level})")
        )
        stat["總計"] = stat.sum(axis=1)
        stat.sort_values("總計", ascending=False).to_csv(
            f"topics_{level}.csv", encoding="utf-8-sig"
        )
        print(f"  topics_{level}.csv")

    # ── 完整結果 ─────────────────────────────────────────────────────────────
    out_cols = ["stock_id", "industry_name", "ArticleCreateTime",
                "label_medium", "label_fine"]
    df[out_cols].to_csv("result_all.csv", encoding="utf-8-sig", index=False)
    print("  result_all.csv")

    # ══════════════════════════════════════════════════════════════════════════
    # 樹狀圓餅圖
    # 層次：Root → 產業(industry) → 中主題(medium) → 細主題(fine)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n建立樹狀圓餅圖資料...")

    _ids     = ["All"]
    _labels  = ["全部文章"]
    _parents = [""]
    _values  = [len(df)]
    _colors  = ["#ffffff"]

    # 產業顏色（第一階層）
    INDUSTRY_COLORS = [
        "#4C78A8","#F58518","#E45756","#72B7B2","#54A24B",
        "#EECA3B","#B279A2","#FF9DA6","#9D755D","#BAB0AC",
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
        "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5",
        "#c49c94","#f7b6d2","#c7c7c7","#dbdb8d","#9edae5",
        "#6baed6","#fd8d3c","#74c476","#9e9ac8","#e7969c",
    ]
    industry_color_map: dict[str, str] = {}
    industries_sorted = sorted(df["industry_name"].dropna().unique())

    for i, industry in enumerate(industries_sorted):
        color = INDUSTRY_COLORS[i % len(INDUSTRY_COLORS)]
        industry_color_map[industry] = color
        node_id = f"ind|{industry}"
        count = int((df["industry_name"] == industry).sum())
        _ids.append(node_id)
        _labels.append(industry)
        _parents.append("All")
        _values.append(count)
        _colors.append(color)

    # Medium 主題（第二階層）：parent = 最多文章所屬的產業
    medium_color_map: dict[str, str] = {}
    for medium in sorted(df["label_medium"].dropna().unique()):
        mask = df["label_medium"] == medium
        dominant_industry = df.loc[mask, "industry_name"].mode()[0]
        parent_id = f"ind|{dominant_industry}"
        color = industry_color_map.get(dominant_industry, "#cccccc")
        medium_color_map[medium] = color
        node_id = f"med|{medium}"
        _ids.append(node_id)
        _labels.append(str(medium))
        _parents.append(parent_id)
        _values.append(int(mask.sum()))
        _colors.append(color)

    # Fine 主題（第三階層）：parent = 最多文章所屬的 medium 主題
    for fine in sorted(df["label_fine"].dropna().unique()):
        mask = df["label_fine"] == fine
        dominant_medium = df.loc[mask, "label_medium"].mode()[0]
        parent_id = f"med|{dominant_medium}"
        color = medium_color_map.get(dominant_medium, "#cccccc")
        node_id = f"fin|{fine}"
        _ids.append(node_id)
        _labels.append(str(fine))
        _parents.append(parent_id)
        _values.append(int(mask.sum()))
        _colors.append(color)

    # ── 由下往上重新計算父節點值，確保 parent >= sum(children)（branchvalues="total" 必要條件）
    # 因為 dominant_industry/medium 分配可能造成跨層不一致
    from collections import defaultdict
    _values = list(_values)
    id_to_idx = {nid: i for i, nid in enumerate(_ids)}
    for _pass in range(3):  # 4層樹需3次傳遞：fine→medium→industry→root
        child_sums: dict = defaultdict(int)
        for pid, val in zip(_parents, _values):
            if pid:
                child_sums[pid] += val
        for pid_id, csum in child_sums.items():
            idx = id_to_idx[pid_id]
            if _values[idx] < csum:
                _values[idx] = csum

    # ── Sunburst（樹狀圓餅圖） ───────────────────────────────────────────────
    fig_sb = go.Figure(go.Sunburst(
        ids=_ids,
        labels=_labels,
        parents=_parents,
        values=_values,
        marker=dict(colors=_colors),
        branchvalues="total",
        insidetextorientation="radial",
        hovertemplate="<b>%{label}</b><br>文章數: %{value}<br>占比: %{percentParent:.1%}<extra></extra>",
    ))
    fig_sb.update_layout(
        title="台股新聞主題樹狀圓餅圖（產業 → 中主題 → 細主題）",
        width=1000, height=1000,
        margin=dict(t=60, l=10, r=10, b=10),
    )
    fig_sb.write_html("tree_sunburst.html", include_mathjax=False)
    print("  tree_sunburst.html")

    # ── Treemap（矩形樹圖） ──────────────────────────────────────────────────
    fig_tm = go.Figure(go.Treemap(
        ids=_ids,
        labels=_labels,
        parents=_parents,
        values=_values,
        marker=dict(colors=_colors),
        branchvalues="total",
        textinfo="label+value",
        hovertemplate="<b>%{label}</b><br>文章數: %{value}<br>占比: %{percentParent:.1%}<extra></extra>",
    ))
    fig_tm.update_layout(
        title="台股新聞主題 Treemap（產業 → 中主題 → 細主題）",
        width=1400, height=900,
        margin=dict(t=50, l=10, r=10, b=10),
    )
    fig_tm.write_html("tree_treemap.html", include_mathjax=False)
    print("  tree_treemap.html")

    # ── Icicle（橫向層次圖） ─────────────────────────────────────────────────
    fig_ic = go.Figure(go.Icicle(
        ids=_ids,
        labels=_labels,
        parents=_parents,
        values=_values,
        marker=dict(colors=_colors),
        branchvalues="total",
        tiling=dict(orientation="v", pad=3),
        textfont=dict(size=13),
        hovertemplate="<b>%{label}</b><br>文章數: %{value}<br>占比: %{percentParent:.1%}<extra></extra>",
    ))
    fig_ic.update_layout(
        title="台股新聞主題層次圖（產業 → 中主題 → 細主題）",
        width=1400, height=900,
        margin=dict(t=50, l=10, r=10, b=10),
    )
    fig_ic.write_html("tree_icicle.html", include_mathjax=False)
    print("  tree_icicle.html")

    # ── Dendrogram ───────────────────────────────────────────────────────────
    if os.path.exists("_linkage.npy"):
        from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

        Z_loaded = np.load("_linkage.npy")
        ddata = scipy_dendrogram(Z_loaded, truncate_mode="lastp", p=60, no_plot=True)

        shapes_x, shapes_y = [], []
        for xs, ys in zip(ddata["icoord"], ddata["dcoord"]):
            shapes_x += xs + [None]
            shapes_y += ys + [None]

        leaf_positions = sorted(set(
            v for xs, ys in zip(ddata["icoord"], ddata["dcoord"])
            for v, y in zip(xs, ys) if y == 0.0
        ))

        fig_dend = go.Figure()
        fig_dend.add_trace(go.Scatter(
            x=shapes_x, y=shapes_y,
            mode="lines",
            line=dict(color="#4C78A8", width=1.2),
            hoverinfo="skip",
        ))
        fig_dend.update_layout(
            title="Ward Hierarchical Clustering Dendrogram（top 60 nodes）",
            xaxis=dict(
                tickmode="array",
                tickvals=leaf_positions,
                ticktext=ddata["ivl"],
                tickangle=-60,
                tickfont=dict(size=10),
            ),
            yaxis=dict(title="Distance"),
            width=1400, height=650,
            plot_bgcolor="white",
            showlegend=False,
        )
        fig_dend.write_html("dendrogram.html", include_mathjax=False)
        print("  dendrogram.html")

    print("\n完成 Phase 2！")
    print("\n輸出檔案：")
    print("  tree_sunburst.html  ← 樹狀圓餅圖（產業 → 中主題 → 細主題）")
    print("  tree_treemap.html   ← 矩形樹圖")
    print("  tree_icicle.html    ← 層次圖")
    print("  tsne_industry.html  ← T-SNE 散點（產業顏色）")
    print("  tsne_medium.html    ← T-SNE 散點（medium 主題顏色）")
    print("  tsne_fine.html      ← T-SNE 散點（fine 主題顏色）")
    print("  result_all.csv      ← 完整分群結果")
