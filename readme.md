# The Gold Standard of Trust

Data processing and network-graph analysis scripts for the 2026 DTI5389 group project, **"The Gold Standard of Trust."**

This project scrapes, merges, and analyzes buyer-seller transaction data from the Reddit subreddit [r/Pmsforsale](https://www.reddit.com/r/Pmsforsale/) — a peer-to-peer marketplace where users buy and sell precious metals (gold, silver, platinum, etc.). The goal is to infer transactions from comment threads and visualize the resulting trust network as a directed graph.

---

## Repository Contents

```
DTI5389-The-Gold-Standard-of-Trust/
│
├── merge.py                        # Merges raw scraped CSVs into one deduplicated file
├── market_graph.py                 # Extracts transactions and builds network visualizations
│
├── raw_data/                       # Raw CSV exports from Apify's Reddit Scraper Pro
│   ├── dataset_reddit-scraper-pro_2026-02-10_01-45-22-831.csv
│   ├── dataset_reddit-scraper-pro_2026-02-17_20-34-29-659.csv
│   ├── dataset_reddit-scraper-pro_2026-02-26_03-57-29-102.csv
│   ├── dataset_reddit-scraper-pro_2026-02-26_04-51-38-009.csv
│   └── dataset_reddit-scraper-pro_2026-03-01_21-08-49-886.csv
│
├── processed_data/                 # Outputs from the scripts
│   ├── merged_reddit_data.csv      # Deduplicated union of all raw CSVs
│   ├── transactions.csv            # Full inferred transaction edge list
│   └── transactions_top5.csv       # Transactions involving the top 5 sellers only
│
└── visualizations/                 # Generated network graph images
    ├── transaction_network.png     # Full marketplace network graph
    └── transaction_network_top5.png# Top-5-sellers subgraph
```

---

## How It Works

### Data Collection

Reddit post and comment data from r/Pmsforsale is collected using [Apify's Reddit Scraper Pro](https://apify.com/). Five scraping runs were performed between February 10 and March 1, 2026, producing the CSV files in `raw_data/`.

### Data Merging (`merge.py`)

Because data was scraped in multiple batches over time, the raw CSVs contain overlapping records. `merge.py` concatenates all files and deduplicates rows by their Reddit `id` column, keeping the most recent version of each record. The result is a single unified dataset written to `processed_data/merged_reddit_data.csv`.

### Transaction Inference (`market_graph.py`)

On r/Pmsforsale, sellers create posts listing items for sale and buyers express interest through comments. The script uses a two-rule heuristic engine to infer buyer-seller transactions:

**Rule 1 — Reply-Based Detection (High Confidence)**

1. A non-author comment contains a **buyer expression** (e.g., "BIN", "I'll take", "dibs", "claim").
2. The post author (seller) directly replies to that comment with a **seller confirmation** (e.g., "yours", "sold", "trade pending", "paid").
3. If both signals are strong, the transaction is recorded with confidence score **1.0**. Weaker signal pairs (e.g., "chat" + "replied") score **0.6** and are only included with the `--include-low-confidence` flag.

**Rule 2 — SOLD Post Fallback (Lower Confidence)**

1. If a post is marked as SOLD (in the title, body, or flair) and exactly one buyer expressed interest without being matched by Rule 1, a transaction is inferred with confidence score **0.4**.
2. When multiple unmatched buyers exist and `--allow-earliest-fallback` is enabled, the earliest "BIN" comment wins (confidence score **0.3**).

### Network Visualization

Inferred transactions are assembled into a directed graph using NetworkX, where:
- **Nodes** represent Reddit users (sellers and buyers).
- **Directed edges** point from seller to buyer, representing completed transactions.
- **Edge labels** show the number of transactions between each pair.
- Node sizes scale with the user's degree (number of connections).

An optional `--top-sellers N` mode produces a filtered subgraph highlighting only the top N sellers (colored orange) and their buyers (colored blue).

---

## Prerequisites

- **Python 3.10** or later
- **pip** (Python package manager)

### Required Python Packages

| Package | Purpose |
|---|---|
| `pandas` | CSV reading, DataFrame operations, deduplication |
| `networkx` | Directed graph construction and layout algorithms |
| `matplotlib` | Network graph rendering and image export |

---

## Step-by-Step Usage Guide

### 1. Clone the Repository

```bash
git clone https://github.com/milex-info/DTI5389-The-Gold-Standard-of-Trust.git
cd DTI5389-The-Gold-Standard-of-Trust
```

### 2. Install Dependencies

```bash
pip install pandas networkx matplotlib
```

### 3. Merge the Raw Data

Run `merge.py` to combine all scraped CSV files in `raw_data/` into a single deduplicated file:

```bash
python merge.py
```

**Output:** `processed_data/merged_reddit_data.csv`

You should see output similar to:

```
Found 5 CSV files:
  dataset_reddit-scraper-pro_2026-02-10_01-45-22-831.csv: 108493 rows, 45 columns
  dataset_reddit-scraper-pro_2026-02-17_20-34-29-659.csv: 116470 rows, 55 columns
  ...
Total rows before dedup: 481350
Total rows after dedup:  419829
Duplicates removed:      61521

Merged file written to: .../processed_data/merged_reddit_data.csv
```

### 4. Generate Transactions and Network Graph

Run `market_graph.py` to extract transactions and produce the visualizations:

```bash
python market_graph.py --top-sellers 5
```

This will generate four output files:

| Output | Description |
|---|---|
| `processed_data/transactions.csv` | Full edge list of all inferred transactions |
| `visualizations/transaction_network.png` | Network graph of the entire marketplace |
| `processed_data/transactions_top5.csv` | Edge list filtered to top 5 sellers |
| `visualizations/transaction_network_top5.png` | Subgraph of top 5 sellers and their buyers |

You should see output similar to:

```
Reading processed_data/merged_reddit_data.csv ...
  419829 rows loaded  (12345 posts, 407484 comments)
  Rule 1 (reply-based): 1650 edges
  Rule 2 (SOLD fallback): 160 edges
  Total: 1810 transactions -> transactions.csv
  Network visualisation -> transaction_network.png
  Nodes: 1523  Edges: 1701

-- Top 5 Sellers --
  #1  rooneyskywalker  (58 transactions)
  #2  xxSpeedysxx  (44 transactions)
  #3  zenpathfinder  (37 transactions)
  ...
  156 transactions -> transactions_top5.csv
  Network visualisation -> transaction_network_top5.png
```

### Command-Line Options for `market_graph.py`

| Flag | Default | Description |
|---|---|---|
| `--input <path>` | `processed_data/merged_reddit_data.csv` | Path to the input CSV file |
| `--edges <filename>` | `transactions.csv` | Filename for the output transaction edge list |
| `--image <filename>` | `transaction_network.png` | Filename for the output network graph image |
| `--include-low-confidence` | off | Include lower-confidence matches (e.g., "chat"/"pm" + "replied") |
| `--allow-earliest-fallback` | off | For SOLD posts with multiple buyers, pick the earliest BIN comment |
| `--top-sellers N` | 0 (disabled) | Also generate a filtered view of the top N sellers by transaction count |

#### Examples

Generate only the full graph (no top-sellers view):

```bash
python market_graph.py
```

Include low-confidence transactions and use the earliest-BIN fallback:

```bash
python market_graph.py --include-low-confidence --allow-earliest-fallback --top-sellers 10
```

Use a custom input file and output names:

```bash
python market_graph.py --input my_data.csv --edges my_edges.csv --image my_graph.png
```

---

## Output Descriptions

### Transaction CSV (`transactions.csv`)

Each row represents one inferred buyer-seller transaction:

| Column | Description |
|---|---|
| `seller` | Reddit username of the post author (seller) |
| `buyer` | Reddit username of the commenter (buyer) |
| `post_id` | Reddit post ID where the transaction occurred |
| `buyer_comment_id` | ID of the buyer's comment expressing interest |
| `seller_comment_id` | ID of the seller's confirming reply (empty for Rule 2 matches) |
| `buyer_comment_time` | Timestamp of the buyer's comment |
| `seller_comment_time` | Timestamp of the seller's reply |
| `rule` | Which inference rule matched (e.g., `seller_confirmed`, `sold_post_single_buyer`) |
| `confidence` | Confidence tier (`high`, `low`, `sold_post_single_buyer`, `sold_post_earliest_bin`) |
| `confidence_score` | Numeric score: 1.0 (high), 0.6 (low), 0.4 (single buyer), 0.3 (earliest BIN) |

### Network Graph Images

- **`transaction_network.png`** — Full directed graph of the entire marketplace. Every user who participated in at least one inferred transaction appears as a node.
- **`transaction_network_top5.png`** — Filtered subgraph showing only the top 5 sellers (orange nodes) and their buyers (blue nodes), with a legend distinguishing the two groups.

---

## Data Source

All raw data was collected from the [r/Pmsforsale](https://www.reddit.com/r/Pmsforsale/) subreddit using [Apify's Reddit Scraper Pro](https://apify.com/). The scraping runs span from February 10 to March 1, 2026. The dataset includes both posts (sale listings) and comments (buyer interest, seller confirmations, general discussion).
