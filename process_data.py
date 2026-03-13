#!/usr/bin/env python3
"""
Orchestrator script that runs the four data-processing steps in order:

  1. merge.py              -- Merge raw Reddit scraper CSVs into a single
                              deduplicated file (processed_data/merged_reddit_data.csv).
  2. extract_features.py   -- Use Google Gemini to extract payment_method,
                              transaction_value, and item_type from post text.
  3. generate_sellerlist.py -- Fetch seller flair/reputation scores from Reddit
                              and write processed_data/seller_list.csv.
  4. market_graph.py       -- Infer buyer-seller transactions and produce
                              processed_data/transactions.csv plus network
                              visualisations in visualizations/.

Usage:
    python process_data.py                     # run all four steps
    python process_data.py --skip-flair        # skip the slow Reddit API scrape
    python process_data.py --skip-extract      # skip the Gemini feature extraction
    python process_data.py --top-sellers 5     # also generate top-N seller output
"""

import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PROCESSING_DIR = os.path.join(PROJECT_ROOT, "data_scripts")


def run_step(description, script, extra_args=None):
    """Run a Python script inside data_processing/ and abort on failure."""
    path = os.path.join(DATA_PROCESSING_DIR, script)
    cmd = [sys.executable, path] + (extra_args or [])

    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nERROR: {script} exited with code {result.returncode}. Aborting.")
        sys.exit(result.returncode)


def main():
    ap = argparse.ArgumentParser(description="Run the full data-processing pipeline")
    ap.add_argument(
        "--skip-flair", action="store_true",
        help="Skip generate_sellerlist.py (the Reddit API flair scrape is slow)",
    )
    ap.add_argument(
        "--skip-extract", action="store_true",
        help="Skip extract_features.py (the Gemini feature extraction)",
    )
    ap.add_argument(
        "--top-sellers", type=int, default=5, metavar="N",
        help="Pass --top-sellers N to market_graph.py (default: 5)",
    )
    args = ap.parse_args()

    # Step 1: Merge raw CSVs
    run_step("Step 1/4: Merging raw Reddit scraper CSVs", "merge.py")

    # Step 2: Extract transaction features using Gemini
    if args.skip_extract:
        print("\n-- Skipping extract_features.py (--skip-extract) --")
    else:
        run_step("Step 2/4: Extracting transaction features (Gemini)", "extract_features.py")

    # Step 3: Scrape seller flair from Reddit
    if args.skip_flair:
        print("\n-- Skipping generate_sellerlist.py (--skip-flair) --")
    else:
        run_step("Step 3/4: Fetching seller flair from Reddit", "generate_sellerlist.py")

    # Step 4: Infer transactions and build network graph
    graph_args = ["--top-sellers", str(args.top_sellers)]
    run_step("Step 4/4: Inferring transactions and building network graph", "market_graph.py", graph_args)

    print(f"\n{'=' * 60}")
    print("  Pipeline complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
