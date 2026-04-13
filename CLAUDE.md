# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**EventCorr** is an AI-driven data pipeline for Taiwan stock market analysis. It correlates news events with individual stock performance by:
1. Scraping RSS financial news feeds and tagging articles to matching stocks
2. Running hierarchical topic modeling (NLP clustering) on article text to identify concept stock groups and supply chain themes

There is a secondary pipeline described in [SA.md](SA.md) — an "AI Investment Decision Room" focused on dynamic stock labeling and sentiment analysis for Taiwan stocks (台股).

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Additional packages required by `topic_model.py` that are not in `requirements.txt`:
```bash
pip install feedparser sentence-transformers umap-learn scikit-learn jieba
```

## Running the Pipeline

### Step 1: RSS Scraping (`rss_scraper.py`)

Run directly to test scraping with built-in sample RSS feeds and stock dictionary:
```bash
python rss_scraper.py
```

The core function is `fetch_and_process_rss(rss_urls, stock_dict)` — import and call it with your own RSS URL list and stock-to-keyword mapping dict. Output is a pandas DataFrame with columns: `stock_id`, `ArticleTitle`, `Tags`, `ArticleText`, `ArticleCreateTime`.

To use the full market stock list, load from CSV per [transform.py](transform.py):
```bash
# transform.py shows how to convert tw_stocks.csv → stock dict for the scraper
python transform.py
```

### Step 2: Topic Modeling (`topic_model.py`)

The script runs in two phases controlled by the `PHASE` variable at the top of the file:

**Phase 1 — Cluster** (generate naming material):
```bash
# Set PHASE = "cluster" at top of topic_model.py, then:
python topic_model.py
# Outputs: label_input.txt (for human/LLM labeling), label_template.json
```

**Phase 2 — Label** (apply labels and visualize):
```bash
# 1. Fill in label_template.json and save as topic_labels.json
# 2. Set PHASE = "label" at top of topic_model.py, then:
python topic_model.py
# Outputs: tsne_*.html, topics_*.csv, result_all.csv, tree_*.html, dendrogram.html
```

**Input required:** `articles.csv` with columns `ArticleText`, `ArticleCreateTime`, `StockId`.

**Caching:** Embeddings are cached to `_embeddings.npy` to avoid recomputation. Delete this file to force re-embedding.

## Architecture

### Data Flow

```
RSS Feeds → rss_scraper.py → articles.csv
                                  ↓
                          topic_model.py (Phase 1: cluster)
                                  ↓
                          label_input.txt / label_template.json
                                  ↓
                          [Human/LLM labels → topic_labels.json]
                                  ↓
                          topic_model.py (Phase 2: label)
                                  ↓
                    result_all.csv + HTML visualizations
```

### Topic Modeling Pipeline (topic_model.py)

1. **Embedding** — BGE-large-zh-v1.5 (1024d) via `sentence-transformers`
2. **Dim reduction** — UMAP (15d) → T-SNE (2d for visualization)
3. **Clustering** — Ward hierarchical clustering on UMAP embeddings
4. **Multi-level cutting** — Three granularity levels: coarse / medium / fine
5. **Auto-detection** — Calinski-Harabasz score selects optimal cluster counts (when `AUTO_LEVELS=True`)
6. **Keyword extraction** — TF-IDF + jieba tokenization per cluster
7. **Visualization** — Interactive HTML: scatter plots, icicle/treemap/sunburst hierarchy charts, dendrogram

### Intermediate Files (generated, not committed)

- `_embeddings.npy` — cached sentence embeddings
- `_checkpoint.parquet` — DataFrame with cluster assignments for Phase 2 resume
- `_cluster_{level}.npy` — cluster assignment arrays (coarse/medium/fine)
- `_linkage.npy` — Ward linkage matrix for dendrogram
