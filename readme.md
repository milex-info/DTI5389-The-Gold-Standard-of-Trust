# The Gold Standard of Trust

Data processing and network analysis pipeline for the 2026 DTI5389 group project. This project infers buyer-seller transactions from comment threads on [r/Pmsforsale](https://www.reddit.com/r/Pmsforsale/) (a peer-to-peer precious metals marketplace) and visualizes the resulting trust network.

---

## Prerequisites

- **Python 3.10+**
- **Environment variable:** `GOOGLE_API_KEY` (required for Step 2 — Gemini feature extraction)

Install all dependencies:

```bash
pip install pandas networkx matplotlib requests numpy google-generativeai
```

---

## Quick Start

```bash
python process_data.py
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

---

## CLI Reference

### `process_data.py`

| Flag | Description |
|---|---|
| `--skip-extract` | Skip Gemini feature extraction (Step 2) |
| `--skip-flair` | Skip Reddit flair scraping (Step 3) |
| `--top-sellers N` | Number of top sellers for the filtered subgraph (default: 5) |

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

---

## Data Source

All raw data was collected from [r/Pmsforsale](https://www.reddit.com/r/Pmsforsale/) using [Apify's Reddit Scraper Pro](https://apify.com/). Nine scraping runs span from February 10 to March 13, 2026. The dataset includes both posts (sale listings) and comments (buyer interest, seller confirmations, general discussion).
