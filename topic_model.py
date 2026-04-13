"""
Stock Comment Hierarchical Topic Modeling Pipeline
===================================================
PHASE = "cluster"  → 執行 Step 1-5，輸出 label_input.txt 供人工命名
PHASE = "label"    → 讀取 topic_labels.json，套用標籤並輸出視覺化結果
"""

# ┌─────────────────────────────────────────┐
# │  設定執行階段：                          │
# │    "cluster" ── 分群 + 輸出命名素材      │
# │    "label"   ── 套用標籤 + 輸出結果      │
# └─────────────────────────────────────────┘
PHASE = "label"

# AUTO_LEVELS = True  → 自動偵測各層最佳群數（執行較慢，約多 1–2 分鐘）
# AUTO_LEVELS = False → 使用下方 LEVELS 手動指定
AUTO_LEVELS = True

LEVELS: dict[str, int] = {   # AUTO_LEVELS=False 時才使用
    "coarse": 5,
    "medium": 15,
    "fine":   50,
}

# 自動偵測的搜尋範圍（可視需求調整）
AUTO_RANGES = {
    "coarse": range(3, 10),
    "medium": range(8, 30),
    "fine":   range(25, 75),
}

import json
import re
import numpy as np
import pandas as pd
import jieba
import plotly.express as px

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from umap import UMAP
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 ── DOCUMENTS
# ══════════════════════════════════════════════════════════════════════════════
print("STEP 1 | 載入資料")
df = pd.read_csv("articles.csv", encoding="utf-8-sig", parse_dates=["ArticleCreateTime"])
df = df.dropna(subset=["ArticleText"])
df["ArticleText"] = df["ArticleText"].astype(str).str.strip()

# 文字清理：去 URL、去非中英數符號、壓縮重複字元
def clean_text(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)                  # 去 URL
    text = re.sub(r"[^\u4e00-\u9fff\w\s]", " ", text)        # 去特殊符號
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)                # 壓縮重複字（哈哈哈哈→哈哈）
    return text.strip()

df["ArticleText"] = df["ArticleText"].apply(clean_text)

df = df.reset_index(drop=True)
docs = df["ArticleText"].tolist()
print(f"  總筆數: {len(df)}")
print(f"  2317: {(df.StockId==2317).sum()}, 2354: {(df.StockId==2354).sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 ── EMBEDDING  (BGE-large-zh, 1024d)
# ══════════════════════════════════════════════════════════════════════════════
import os
EMBED_CACHE = "_embeddings.npy"
if os.path.exists(EMBED_CACHE):
    print("STEP 2 | 載入快取 embedding（跳過重算）")
    embeddings = np.load(EMBED_CACHE)
else:
    print("STEP 2 | BGE-large-zh Embedding (1024d)")
    embed_model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
    embeddings = embed_model.encode(docs, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    np.save(EMBED_CACHE, embeddings)
    print(f"  embedding 已存檔 → {EMBED_CACHE}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 ── DIMENSION REDUCTION  (UMAP + T-SNE)
# ══════════════════════════════════════════════════════════════════════════════
print("STEP 3 | UMAP 15d + T-SNE 2d")
umap_embeddings = UMAP(n_components=15, n_neighbors=30, min_dist=0.0, metric="cosine", random_state=42).fit_transform(embeddings)
tsne_2d = TSNE(n_components=2, perplexity=50, learning_rate="auto", init="pca", random_state=42, n_jobs=-1).fit_transform(umap_embeddings)
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
    coarse_k = _best_k(AUTO_RANGES["coarse"])
    medium_k = _best_k(range(max(AUTO_RANGES["medium"].start, coarse_k + 2),
                             AUTO_RANGES["medium"].stop))
    fine_k   = _best_k(range(max(AUTO_RANGES["fine"].start, medium_k + 3),
                             AUTO_RANGES["fine"].stop))
    LEVELS = {"coarse": coarse_k, "medium": medium_k, "fine": fine_k}
    print(f"  → coarse={coarse_k}, medium={medium_k}, fine={fine_k}")

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
np.save("_cluster_coarse.npy", cluster_assignments["coarse"])
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

    for level in ["fine", "medium", "coarse"]:
        n_clusters = LEVELS[level]
        assignments = cluster_assignments[level]
        lines.append(f"\n{'═'*60}")
        lines.append(f"【{level.upper()} 層】{n_clusters} 個主題")
        lines.append(f"{'═'*60}")
        template[level] = {}

        for cid in range(1, n_clusters + 1):
            idx = list(np.where(assignments == cid)[0])
            if not idx:
                continue
            kws = extract_keywords(idx)
            samples = sample_sentences(idx)
            count = len(idx)

            lines.append(f"\n[{level} 主題 {cid:02d}]  ({count} 筆)")
            lines.append(f"  關鍵詞：{', '.join(kws)}")
            lines.append("  範例句：")
            for s in samples:
                lines.append(f"    • {s}")
            lines.append(f"  → 請命名：＿＿＿＿＿＿")

            template[level][str(cid)] = ""   # 空白待填

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

    if not __import__("os").path.exists("topic_labels.json"):
        raise FileNotFoundError("找不到 topic_labels.json，請先完成 Phase 1 命名步驟。")

    with open("topic_labels.json", encoding="utf-8") as f:
        topic_labels: dict[str, dict[str, str]] = json.load(f)

    # 從暫存還原分群（跳過耗時的 embedding/clustering 步驟）
    df = pd.read_parquet("_checkpoint.parquet")
    for level in LEVELS:
        arr = np.load(f"_cluster_{level}.npy")
        int_labels = {int(k): v for k, v in topic_labels[level].items()}
        df[f"label_{level}"] = pd.Series(arr).map(int_labels).values

    # T-SNE 散點圖
    for level in LEVELS:
        fig = px.scatter(
            df, x="tsne_x", y="tsne_y",
            color=f"label_{level}",
            symbol=df["StockId"].astype(str),
            hover_data=["StockId", "ArticleCreateTime", f"label_{level}"],
            title=f"T-SNE ── {level} 層主題（△=2317 注意股  ○=2354 非注意股）",
            width=1200, height=800,
        )
        fig.write_html(f"tsne_{level}.html")
        print(f"  tsne_{level}.html")

    # 各層主題統計
    for level in LEVELS:
        stat = (
            df.groupby([f"label_{level}", "StockId"]).size()
            .unstack(fill_value=0)
            .rename_axis(f"主題({level})")
        )
        stat["總計"] = stat.sum(axis=1)
        stat.sort_values("總計", ascending=False).to_csv(f"topics_{level}.csv", encoding="utf-8-sig")
        print(f"  topics_{level}.csv")

    # 完整結果
    out_cols = ["ArticleId", "StockId", "ArticleCreateTime",
                "label_coarse", "label_medium", "label_fine"]
    df[out_cols].to_csv("result_all.csv", encoding="utf-8-sig", index=False)
    print("  result_all.csv")

    # 樹狀圖 ── 建立共用層次資料（coarse → medium → fine）
    import plotly.graph_objects as go

    _ids      = ["All"]
    _labels   = ["全部主題"]
    _parents  = [""]
    _values   = [len(df)]
    _colors   = ["#ffffff"]

    COARSE_COLORS = [
        "#4C78A8","#F58518","#E45756","#72B7B2","#54A24B",
        "#EECA3B","#B279A2","#FF9DA6","#9D755D","#BAB0AC",
    ]
    color_map: dict[str, str] = {}

    for i, coarse in enumerate(sorted(df["label_coarse"].dropna().unique())):
        c = COARSE_COLORS[i % len(COARSE_COLORS)]
        color_map[f"c|{coarse}"] = c
        _ids.append(f"c|{coarse}")
        _labels.append(str(coarse))
        _parents.append("All")
        _values.append(int((df["label_coarse"] == coarse).sum()))
        _colors.append(c)

    for medium in sorted(df["label_medium"].dropna().unique()):
        dominant_coarse = df.loc[df["label_medium"] == medium, "label_coarse"].mode()[0]
        parent_key = f"c|{dominant_coarse}"
        color_map[f"m|{medium}"] = color_map.get(parent_key, "#cccccc")
        _ids.append(f"m|{medium}")
        _labels.append(str(medium))
        _parents.append(parent_key)
        _values.append(int((df["label_medium"] == medium).sum()))
        _colors.append(color_map[f"m|{medium}"])

    for fine in sorted(df["label_fine"].dropna().unique()):
        dominant_medium = df.loc[df["label_fine"] == fine, "label_medium"].mode()[0]
        parent_key = f"m|{dominant_medium}"
        _ids.append(f"f|{fine}")
        _labels.append(str(fine))
        _parents.append(parent_key)
        _values.append(int((df["label_fine"] == fine).sum()))
        _colors.append(color_map.get(parent_key, "#cccccc"))

    # Icicle（橫向層次圖，最易閱讀各層主題）
    fig_ic = go.Figure(go.Icicle(
        ids=_ids, labels=_labels, parents=_parents, values=_values,
        marker=dict(colors=_colors),
        branchvalues="total",
        tiling=dict(orientation="v", pad=3),
        textfont=dict(size=13),
    ))
    fig_ic.update_layout(
        title="主題層次圖 Icicle（Coarse → Medium → Fine）",
        width=1400, height=900,
        margin=dict(t=50, l=10, r=10, b=10),
    )
    fig_ic.write_html("tree_icicle.html")
    print("  tree_icicle.html")

    # Treemap（矩形樹圖，面積代表文章數）
    fig_tm = go.Figure(go.Treemap(
        ids=_ids, labels=_labels, parents=_parents, values=_values,
        marker=dict(colors=_colors),
        branchvalues="total",
        textfont=dict(size=13),
        textinfo="label+value",
    ))
    fig_tm.update_layout(
        title="主題層次圖 Treemap（Coarse → Medium → Fine）",
        width=1400, height=900,
        margin=dict(t=50, l=10, r=10, b=10),
    )
    fig_tm.write_html("tree_treemap.html")
    print("  tree_treemap.html")

    # Sunburst（圓形層次圖）
    fig_sb = go.Figure(go.Sunburst(
        ids=_ids, labels=_labels, parents=_parents, values=_values,
        marker=dict(colors=_colors),
        branchvalues="total",
        insidetextorientation="radial",
    ))
    fig_sb.update_layout(
        title="主題層次圖 Sunburst（Coarse → Medium → Fine）",
        width=950, height=950,
    )
    fig_sb.write_html("tree_sunburst.html")
    print("  tree_sunburst.html")

    # 樹狀圖 ── Dendrogram 互動式 HTML（若 Phase 1 的 linkage matrix 存在）
    if os.path.exists("_linkage.npy"):
        from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
        import plotly.graph_objects as go_d

        Z_loaded = np.load("_linkage.npy")

        # 用 scipy 計算 dendrogram 座標，取最頂端 60 個節點
        ddata = scipy_dendrogram(Z_loaded, truncate_mode="lastp", p=60,
                                 no_plot=True)

        # 將每條連線轉成 plotly trace
        shapes_x, shapes_y = [], []
        for xs, ys in zip(ddata["icoord"], ddata["dcoord"]):
            shapes_x += xs + [None]
            shapes_y += ys + [None]

        # 葉節點標籤：顯示該節點包含幾篇文章
        leaf_x = [(xs[0] + xs[-1]) / 2 for xs in ddata["icoord"]
                  if ddata["dcoord"][ddata["icoord"].index(xs)][0] == 0
                  or ddata["dcoord"][ddata["icoord"].index(xs)][3] == 0]
        tick_labels = ddata["ivl"]
        tick_pos = [(min(ddata["icoord"][i]) + max(ddata["icoord"][i])) / 2
                    for i in range(len(ddata["icoord"]))]
        # 葉節點 x 位置
        leaf_positions = sorted(set(
            v for xs, ys in zip(ddata["icoord"], ddata["dcoord"])
            for v, y in zip(xs, ys) if y == 0.0
        ))

        fig_dend = go_d.Figure()
        fig_dend.add_trace(go_d.Scatter(
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
                ticktext=tick_labels,
                tickangle=-60,
                tickfont=dict(size=10),
            ),
            yaxis=dict(title="Distance"),
            width=1400, height=650,
            plot_bgcolor="white",
            showlegend=False,
        )
        fig_dend.write_html("dendrogram.html")
        print("  dendrogram.html")

    print("\n完成 Phase 2！")
