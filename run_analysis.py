#!/usr/bin/env python3
"""
Orchestrator for analysis scripts in analysis_scripts/.

Usage:
    python run_analysis.py --embeddings      # run the graph-embeddings analysis
    python run_analysis.py --all             # run every available analysis
"""

import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "analysis_scripts")

SCRIPTS = {
    "embeddings": {
        "file": "embeddings_analysis.py",
        "description": "Graph embeddings analysis (Node2Vec, GraphSAGE, RQ1-RQ3)",
    },
    # Add future analysis scripts here, e.g.:
    # "sentiment": {
    #     "file": "sentiment_analysis.py",
    #     "description": "Sentiment analysis of post text",
    # },
}


def run_script(name, info):
    path = os.path.join(ANALYSIS_DIR, info["file"])
    if not os.path.isfile(path):
        print(f"ERROR: {path} not found. Skipping '{name}'.")
        return False

    print(f"\n{'=' * 60}")
    print(f"  Running: {name} — {info['description']}")
    print(f"  Script:  {path}")
    print(f"{'=' * 60}\n")

    result = subprocess.run([sys.executable, path])
    if result.returncode != 0:
        print(f"\nERROR: {info['file']} exited with code {result.returncode}.")
        return False
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Run analysis scripts from analysis_scripts/",
    )
    ap.add_argument(
        "--embeddings", action="store_true",
        help="Run the graph-embeddings analysis (Node2Vec, GraphSAGE, RQ1-RQ3)",
    )
    ap.add_argument(
        "--all", action="store_true",
        help="Run every available analysis script",
    )
    args = ap.parse_args()

    # Collect which analyses to run
    selected = []
    if args.all:
        selected = list(SCRIPTS.keys())
    else:
        if args.embeddings:
            selected.append("embeddings")

    if not selected:
        ap.print_help()
        print("\nNo analysis selected. Use --embeddings or --all.")
        sys.exit(1)

    failed = []
    for name in selected:
        ok = run_script(name, SCRIPTS[name])
        if not ok:
            failed.append(name)

    print(f"\n{'=' * 60}")
    if failed:
        print(f"  Done. {len(failed)} analysis script(s) failed: {', '.join(failed)}")
    else:
        print(f"  All {len(selected)} analysis script(s) completed successfully.")
    print(f"{'=' * 60}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
