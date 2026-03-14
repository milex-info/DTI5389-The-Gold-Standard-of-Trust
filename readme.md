# The Gold Standard of Trust

Data processing, network analysis, and machine learning pipeline for the 2026 DTI5389 group project. This project infers buyer-seller transactions from comment threads on [r/Pmsforsale](https://www.reddit.com/r/Pmsforsale/) (a peer-to-peer precious metals marketplace), visualizes the resulting trust network, and applies graph embeddings to study power users, reputation dynamics, and fraud detection.

---

## Prerequisites

- **Python 3.10+**
- **Environment variable:** `GOOGLE_API_KEY` (required for Step 2 — Gemini feature extraction)

Install all dependencies:

```bash
pip install pandas networkx matplotlib requests numpy torch node2vec scikit-learn umap-learn python-louvain google-generativeai
```

---

## Quick Start

The full pipeline is run with two orchestrator scripts:

```bash
# Phase 1: Data processing (merge, extract features, scrape flair, infer transactions)
python process_data.py

# Phase 2: Analysis (graph embeddings, link prediction, research questions)
python run_analysis.py --all
```

To skip slow or API-dependent steps (pre-built data is included in the repo):

```bash
python process_data.py --skip-extract --skip-flair
```

---

## Pipeline Steps

Scripts run in the following order. Each step's output feeds into the next.

### Step 1 — Merge Raw Data

**Script:** `data_scripts/merge.py`

Concatenates all CSV files in `raw_data/` (9 scraping exports) into a single deduplicated dataset. Deduplicates by the Reddit `id` column, keeping the most recent version of each record. If LLM-extracted feature columns already exist in the output file, they are preserved across re-merges.

- **Input:** `raw_data/*.csv`
- **Output:** `processed_data/merged_reddit_data.csv`

### Step 2 — Extract Features (LLM)

**Script:** `data_scripts/extract_features.py`

Uses the Google Gemini API to extract three structured fields from each post's title and body:

- `payment_method` — zelle, venmo, paypal, crypto, etc.
- `transaction_value` — estimated dollar value of listed items
- `item_type` — gold, silver, platinum, palladium, mixed, etc.

Processes posts in batches (default 15) with exponential-backoff retry on rate limits. Skips already-processed rows unless `--force` is given. Requires the `GOOGLE_API_KEY` environment variable.

- **Input:** `processed_data/merged_reddit_data.csv`
- **Output:** Same file, with three new columns added
- **Flags:** `--force`, `--batch-size N`, `--dry-run`
- **Skippable:** `python process_data.py --skip-extract`

### Step 3 — Scrape Seller Flair

**Script:** `data_scripts/generate_sellerlist.py`

Fetches each seller's Reddit reputation flair (e.g., `S: 846 | B: 148`) from the Reddit JSON API. Applies a 3-second rate-limit delay between requests. This step is very slow (~3 seconds per seller) and the pre-built output is included in the repository.

- **Input:** `processed_data/merged_reddit_data.csv`
- **Output:** `processed_data/seller_list.csv`
- **Skippable:** `python process_data.py --skip-flair`

### Step 4 — Infer Transactions & Build Network Graph

**Script:** `data_scripts/market_graph.py`

The core transaction inference engine. Uses a two-rule heuristic to identify buyer-seller transactions from comment threads:

- **Rule 1 (Reply-Based):** A comment contains a buyer expression ("BIN", "I'll take", "dibs") and the post author replies with a confirmation ("yours", "sold", "trade pending"). Confidence: 1.0 (strong) or 0.6 (weak signals).
- **Rule 2 (SOLD Fallback):** A post is marked SOLD and exactly one unmatched buyer expressed interest. Confidence: 0.4 (single buyer) or 0.3 (earliest BIN among multiple).

Builds a directed NetworkX graph and renders it with a dark theme — gold-stroke seller nodes, white semi-transparent buyer nodes. Outputs both PNG and SVG. Optionally produces a top-N sellers subgraph.

- **Input:** `processed_data/merged_reddit_data.csv`
- **Output:** `processed_data/transactions.csv`, `processed_data/transactions_top5.csv`, `visualizations/transaction_network.{png,svg}`, `visualizations/transaction_network_top5.{png,svg}`

### Step 5 — Graph Embeddings & Analysis

**Script:** `analysis_scripts/embeddings_analysis.py`

Performs graph analytics and addresses three research questions:

1. **Graph Construction** — Builds a feature-rich MultiDiGraph with edge features (transaction value, payment method, item type, timestamp) and node features (flair score, transaction counts, volume, role ratio, unique partners). Note: `account_age` is randomly simulated, not sourced from real data.
2. **Node2Vec Grid Search** — 25-configuration grid search over `p` and `q` parameters. Evaluates link prediction AUC-ROC on a 20% held-out edge set.
3. **GraphSAGE Training** — Custom 2-layer GraphSAGE model (PyTorch) trained for 50 epochs with link prediction loss.
4. **UMAP Visualization** — Projects 64D embeddings to 2D. Three panels: log flair score, Louvain community, and seller/buyer role ratio.
5. **RQ1: Power Users** — Logistic regression and K-Means clustering on embeddings.
6. **RQ2: Reputation vs. Volume** — Flair-only vs. embedding-based regression; cosine similarity correlation.
7. **RQ3: Fraud Detection** — Isolation Forest anomaly detection, GraphSAGE reconstruction error, betweenness centrality for middleman identification.

- **Input:** `processed_data/transactions.csv`, `processed_data/seller_list.csv`, `processed_data/merged_reddit_data.csv`
- **Output:** `visualizations/embedding_umap.png`, `analysis_scripts/research_questions_results.txt`

Run via: `python run_analysis.py --embeddings` or `python run_analysis.py --all`

---

## CLI Reference

### `process_data.py`

| Flag | Description |
|---|---|
| `--skip-extract` | Skip Gemini feature extraction (Step 2) |
| `--skip-flair` | Skip Reddit flair scraping (Step 3) |
| `--top-sellers N` | Number of top sellers for the filtered subgraph (default: 5) |

### `run_analysis.py`

| Flag | Description |
|---|---|
| `--embeddings` | Run the embeddings analysis |
| `--all` | Run all registered analyses |

### `data_scripts/market_graph.py`

| Flag | Default | Description |
|---|---|---|
| `--input <path>` | `processed_data/merged_reddit_data.csv` | Input CSV path |
| `--edges <filename>` | `transactions.csv` | Output edge list filename |
| `--image <filename>` | `transaction_network.png` | Output graph image filename |
| `--include-low-confidence` | off | Include weak-signal matches (confidence 0.6) |
| `--allow-earliest-fallback` | off | Pick earliest BIN when multiple unmatched buyers exist |
| `--top-sellers N` | 0 (disabled) | Generate a filtered top-N sellers subgraph |

---

## Outputs

| File | Description |
|---|---|
| `processed_data/merged_reddit_data.csv` | Deduplicated union of all raw CSVs with LLM-extracted features |
| `processed_data/seller_list.csv` | Seller reputation flair scores |
| `processed_data/transactions.csv` | Full inferred transaction edge list |
| `processed_data/transactions_top5.csv` | Transactions involving top 5 sellers only |
| `visualizations/transaction_network.png` | Full marketplace network graph |
| `visualizations/transaction_network.svg` | Full marketplace network graph (vector) |
| `visualizations/transaction_network_top5.png` | Top-5-sellers subgraph |
| `visualizations/transaction_network_top5.svg` | Top-5-sellers subgraph (vector) |
| `visualizations/embedding_umap.png` | 3-panel UMAP projection of Node2Vec embeddings |
| `analysis_scripts/research_questions_results.txt` | Numerical results for all three research questions |

---

## Data Source

All raw data was collected from [r/Pmsforsale](https://www.reddit.com/r/Pmsforsale/) using [Apify's Reddit Scraper Pro](https://apify.com/). Nine scraping runs span from February 10 to March 13, 2026. The dataset includes both posts (sale listings) and comments (buyer interest, seller confirmations, general discussion).
