# The Gold Standard of Trust

Data processing, network-graph analysis, and machine learning scripts for the 2026 DTI5389 group project, **"The Gold Standard of Trust."**

This project scrapes, merges, and analyzes buyer-seller transaction data from the Reddit subreddit [r/Pmsforsale](https://www.reddit.com/r/Pmsforsale/) — a peer-to-peer marketplace where users buy and sell precious metals (gold, silver, platinum, etc.). The goal is to infer transactions from comment threads, visualize the resulting trust network as a directed graph, and answer research questions about power users, reputation dynamics, and fraud detection using graph neural network embeddings.

---

## Repository Contents

```
DTI5389-The-Gold-Standard-of-Trust/
│
├── merge.py                        # Merges raw scraped CSVs into one deduplicated file
├── generate_sellerlist.py          # Scrapes Reddit flair (reputation scores) for each seller
├── market_graph.py                 # Extracts transactions and builds network visualizations
├── analysis.py                     # Graph embeddings, link prediction, and research question analysis
│
├── raw_data/                       # Raw CSV exports from Apify's Reddit Scraper Pro
│   ├── dataset_reddit-scraper-pro_2026-02-10_01-45-22-831.csv
│   ├── dataset_reddit-scraper-pro_2026-02-17_20-34-29-659.csv
│   ├── dataset_reddit-scraper-pro_2026-02-26_03-57-29-102.csv
│   ├── dataset_reddit-scraper-pro_2026-02-26_04-51-38-009.csv
│   ├── dataset_reddit-scraper-pro_2026-03-01_21-08-49-886.csv
│   └── dataset_reddit-scraper-pro_2026-03-04_17-13-23-132.csv
│
├── processed_data/                 # Outputs from the scripts
│   ├── merged_reddit_data.csv      # Deduplicated union of all raw CSVs (~419,830 rows)
│   ├── seller_list.csv             # Seller flair/reputation scores (~3,187 rows)
│   ├── transactions.csv            # Full inferred transaction edge list (~1,811 rows)
│   └── transactions_top5.csv       # Transactions involving the top 5 sellers only (~157 rows)
│
├── visualizations/                 # Generated images
│   ├── transaction_network.png     # Full marketplace network graph
│   ├── transaction_network_top5.png# Top-5-sellers subgraph
│   └── embedding_umap.png         # UMAP projection of Node2Vec embeddings (3-panel)
│
└── research_questions_results.txt  # Machine-generated results from analysis.py
```

---

## How It Works

The project follows a four-stage pipeline:

### Stage 1: Data Collection

Reddit post and comment data from r/Pmsforsale is collected using [Apify's Reddit Scraper Pro](https://apify.com/). Six scraping runs were performed between February 10 and March 4, 2026, producing the CSV files in `raw_data/`. Each file contains 45-55 columns of Reddit metadata including post titles, comment bodies, author names, timestamps, flair text, and parent-child comment relationships.

### Stage 2: Data Merging (`merge.py`)

Because data was scraped in multiple batches over time, the raw CSVs contain overlapping records. `merge.py` concatenates all files and deduplicates rows by their Reddit `id` column, keeping the most recent version of each record. The result is a single unified dataset written to `processed_data/merged_reddit_data.csv`.

### Stage 3: Seller Flair Scraping (`generate_sellerlist.py`)

On r/Pmsforsale, users earn reputation flair in the format `S: <count> | B: <count>` reflecting the number of completed sales and buys. `generate_sellerlist.py` extracts all post authors (sellers) from the merged dataset and fetches each user's current flair text from the Reddit JSON API. A 3-second rate-limit delay is applied between requests to respect Reddit's terms of service.

**Output:** `processed_data/seller_list.csv` with columns `dataType`, `authorName`, `postUrl`, and `sellerFlair`.

### Stage 4: Transaction Inference (`market_graph.py`)

On r/Pmsforsale, sellers create posts listing items for sale and buyers express interest through comments. The script uses a two-rule heuristic engine to infer buyer-seller transactions:

**Rule 1 — Reply-Based Detection (High Confidence)**

1. A non-author comment contains a **buyer expression** (e.g., "BIN", "I'll take", "dibs", "claim").
2. The post author (seller) directly replies to that comment with a **seller confirmation** (e.g., "yours", "sold", "trade pending", "paid").
3. If both signals are strong, the transaction is recorded with confidence score **1.0**. Weaker signal pairs (e.g., "chat" + "replied") score **0.6** and are only included with the `--include-low-confidence` flag.

**Rule 2 — SOLD Post Fallback (Lower Confidence)**

1. If a post is marked as SOLD (in the title, body, or flair) and exactly one buyer expressed interest without being matched by Rule 1, a transaction is inferred with confidence score **0.4**.
2. When multiple unmatched buyers exist and `--allow-earliest-fallback` is enabled, the earliest "BIN" comment wins (confidence score **0.3**).

### Stage 5: Network Visualization

Inferred transactions are assembled into a directed graph using NetworkX, where:
- **Nodes** represent Reddit users (sellers and buyers).
- **Directed edges** point from seller to buyer, representing completed transactions.
- **Edge labels** show the number of transactions between each pair.
- Node sizes scale with the user's degree (number of connections).

An optional `--top-sellers N` mode produces a filtered subgraph highlighting only the top N sellers (colored orange) and their buyers (colored blue).

### Stage 6: Graph Analysis and Machine Learning (`analysis.py`)

The analysis script performs advanced graph analytics and addresses three research questions. It operates in seven steps:

**Step 1 — Graph Construction.** Builds a MultiDiGraph from the transaction edge list with edge features (transaction value, payment method, item type as one-hot vectors, timestamps) and node features (flair score, account age, transaction counts, volume, seller/buyer role ratio, number of unique trading partners). All continuous features are normalized with StandardScaler.

**Step 2-3 — Embeddings and Link Prediction.** Holds out 20% of edges for validation. Runs a Node2Vec grid search over `p` and `q` in `{0.25, 0.5, 1, 2, 4}` (25 configurations), evaluating AUC-ROC on the held-out edges. Also trains a custom 2-layer GraphSAGE model (PyTorch) with a link prediction loss for 50 epochs and evaluates its AUC-ROC and Average Precision.

**Step 4 — Embedding Visualization.** Projects Node2Vec embeddings to 2D using UMAP. Runs Louvain community detection on the graph. Produces a 3-panel visualization (`embedding_umap.png`) colored by: (1) log flair score, (2) community membership, and (3) seller/buyer role ratio.

**Step 5 — RQ1: Power Users.** Identifies power users (top 10% by transaction count). Trains a logistic regression classifier on embeddings to predict power-user status. Reports AUC, K-Means cluster assignments, and centroid distances for named target users.

**Step 6 — RQ2: Reputation and Volume.** Compares a flair-only linear regression baseline against an embedding-based model for predicting trading volume. Computes the Pearson correlation between user embedding similarity and transaction value.

**Step 7 — RQ3: Fraud Detection.** Runs Isolation Forest anomaly detection on embeddings (5% contamination). Computes GraphSAGE reconstruction error to identify anomalous nodes. Calculates betweenness centrality to find potential middlemen.

All numerical results are written to `research_questions_results.txt`.

---

## Research Questions and Results

Results were generated on 2026-03-08 from the full transaction dataset (1,811 inferred transactions, 1,523 unique users).

### RQ1: Do "Power Users" Emerge as Market-Makers?

| Metric | Value |
|---|---|
| Power User Classification AUC | 0.5481 |

The AUC near 0.5 suggests that power-user status is not strongly separable from structural position alone in the embedding space. Three named target users (GregHutch1964, rooneyskywalker, xxSpeedysxx) all fell in the same K-Means cluster (Cluster 3), with centroid distances between 1.92 and 2.17, indicating they occupy a similar region of the network.

### RQ2: Does Reputation Correlate with Trading Volume?

| Model | R-squared |
|---|---|
| Baseline (Flair only) | 0.0984 |
| Embeddings | 0.0815 |

| Metric | Value |
|---|---|
| Correlation (user similarity vs. transaction value) | -0.0093 |

Both models explain less than 10% of the variance in trading volume. The near-zero correlation between user embedding similarity and transaction value suggests that in this marketplace, flair reputation has a weak relationship with actual trading volume and pricing outcomes.

### RQ3: How Does the Community Self-Regulate Against Fraud?

| Metric | Value |
|---|---|
| Anomalous nodes detected (Isolation Forest) | 56 |

**Top anomalous nodes by GraphSAGE reconstruction error:**

| User | Reconstruction Error |
|---|---|
| Objective-Cap-7697 | 0.5469 |
| xxSpeedysxx | 0.5186 |
| rooneyskywalker | 0.5116 |
| stackinggold | 0.5036 |
| zenpathfinder | 0.4961 |

**Potential middlemen (top betweenness centrality):**

| User | Betweenness Centrality |
|---|---|
| Objective-Cap-7697 | 0.0097 |
| OminiousMetal7443 | 0.0045 |
| YouGetWhatYouPayFour | 0.0041 |
| stillwaters23 | 0.0037 |
| Strange_grass23 | 0.0035 |

Notably, Objective-Cap-7697 appears at the top of both the reconstruction error and betweenness centrality lists, making them the strongest candidate for further investigation.

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
| `requests` | HTTP requests to Reddit JSON API for flair scraping |
| `numpy` | Numerical operations and array manipulation |
| `torch` (PyTorch) | GraphSAGE neural network model |
| `node2vec` | Node2Vec graph embedding algorithm |
| `scikit-learn` | Logistic/linear regression, K-Means, Isolation Forest, metrics |
| `umap-learn` | UMAP dimensionality reduction for embedding visualization |
| `python-louvain` | Louvain community detection algorithm |

---

## Step-by-Step Usage Guide

### 1. Clone the Repository

```bash
git clone https://github.com/milex-info/DTI5389-The-Gold-Standard-of-Trust.git
cd DTI5389-The-Gold-Standard-of-Trust
```

### 2. Install Dependencies

```bash
pip install pandas networkx matplotlib requests numpy torch node2vec scikit-learn umap-learn python-louvain
```

### 3. Merge the Raw Data

Run `merge.py` to combine all scraped CSV files in `raw_data/` into a single deduplicated file:

```bash
python merge.py
```

**Output:** `processed_data/merged_reddit_data.csv`

You should see output similar to:

```
Found 6 CSV files:
  dataset_reddit-scraper-pro_2026-02-10_01-45-22-831.csv: 108493 rows, 45 columns
  dataset_reddit-scraper-pro_2026-02-17_20-34-29-659.csv: 116470 rows, 55 columns
  ...
Total rows before dedup: 481350
Total rows after dedup:  419830
Duplicates removed:      61520
```

### 4. Scrape Seller Flair (Optional — Pre-built Data Included)

Run `generate_sellerlist.py` to fetch each seller's reputation flair from Reddit. This step is slow (~3 seconds per seller due to rate limiting) and the output is already included in the repository.

```bash
python generate_sellerlist.py
```

**Output:** `processed_data/seller_list.csv`

> **Note:** This script makes live HTTP requests to Reddit. It may take several hours to complete for all ~3,187 sellers. The pre-built `seller_list.csv` is provided so you can skip this step.

### 5. Generate Transactions and Network Graph

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
  419830 rows loaded  (12345 posts, 407485 comments)
  Rule 1 (reply-based): 1650 edges
  Rule 2 (SOLD fallback): 161 edges
  Total: 1811 transactions -> transactions.csv
  Network visualisation -> transaction_network.png
  Nodes: 1523  Edges: 1701

-- Top 5 Sellers --
  #1  rooneyskywalker  (58 transactions)
  #2  xxSpeedysxx  (44 transactions)
  #3  zenpathfinder  (37 transactions)
  ...
  157 transactions -> transactions_top5.csv
  Network visualisation -> transaction_network_top5.png
```

### 6. Run the Analysis

Run `analysis.py` to train graph embeddings and answer the research questions:

```bash
python analysis.py
```

This will:
1. Build a feature-enriched MultiDiGraph from the transaction data
2. Run a Node2Vec grid search (25 p/q configurations) and train a GraphSAGE model
3. Generate a 3-panel UMAP visualization of the embeddings
4. Evaluate three research questions (power users, reputation, fraud detection)

**Outputs:**
- `visualizations/embedding_umap.png` — UMAP embedding visualization
- `research_questions_results.txt` — Numerical results for all three research questions

> **Note:** The Node2Vec grid search and GraphSAGE training may take several minutes depending on hardware. A GPU is not required but will speed up the PyTorch operations.

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

### Seller List CSV (`seller_list.csv`)

Each row represents one post author with their Reddit flair:

| Column | Description |
|---|---|
| `dataType` | Always `post` |
| `authorName` | Reddit username of the seller |
| `postUrl` | URL of the sale post |
| `sellerFlair` | Flair text (e.g., `S: 846 \| B: 148`) or empty if unavailable |

### Network Graph Images

- **`transaction_network.png`** — Full directed graph of the entire marketplace. Every user who participated in at least one inferred transaction appears as a node (~1,523 nodes, ~1,701 edges).
- **`transaction_network_top5.png`** — Filtered subgraph showing only the top 5 sellers (orange nodes) and their buyers (blue nodes), with a legend distinguishing the two groups.

### Embedding Visualization (`embedding_umap.png`)

A 3-panel UMAP projection of the 64-dimensional Node2Vec embeddings:

1. **Left panel** — Nodes colored by log(flair score), showing how reputation maps onto network structure.
2. **Center panel** — Nodes colored by Louvain community membership, revealing trading clusters.
3. **Right panel** — Nodes colored by role ratio (red = predominantly seller, blue = predominantly buyer), showing the marketplace's structural division.

### Research Results (`research_questions_results.txt`)

Machine-generated text file containing the numerical outputs of all three research question analyses (RQ1: power users, RQ2: reputation vs. volume, RQ3: fraud detection). Generated automatically by `analysis.py`.

---

## Data Source

All raw data was collected from the [r/Pmsforsale](https://www.reddit.com/r/Pmsforsale/) subreddit using [Apify's Reddit Scraper Pro](https://apify.com/). Six scraping runs span from February 10 to March 4, 2026. The dataset includes both posts (sale listings) and comments (buyer interest, seller confirmations, general discussion).
