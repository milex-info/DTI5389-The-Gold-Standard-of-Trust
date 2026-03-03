"""
Merge all Reddit scraper CSV files in this directory into a single
deduplicated CSV file (merged_reddit_data.csv).

- Columns are the union of all input files (missing columns filled with empty values).
- Duplicates are removed by the 'id' column, keeping the last occurrence
  (files are loaded oldest-first so newer scrapes take priority).
"""

import glob
import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "processed_data/merged_reddit_data.csv")


def main():
    # Discover CSV files (exclude the output file) and sort by name (oldest first)
    csv_files = sorted(
        f
        for f in glob.glob(os.path.join(SCRIPT_DIR, "raw_data/*.csv"))
        if os.path.basename(f) != os.path.basename(OUTPUT_FILE)
    )

    if not csv_files:
        print("No CSV files found to merge.")
        return

    print(f"Found {len(csv_files)} CSV files:")
    frames = []
    for path in csv_files:
        df = pd.read_csv(path, dtype=str)  # read everything as strings to avoid type issues
        print(f"  {os.path.basename(path)}: {len(df)} rows, {len(df.columns)} columns")
        frames.append(df)

    # Concatenate all frames; pandas aligns columns automatically
    combined = pd.concat(frames, ignore_index=True)
    print(f"\nTotal rows before dedup: {len(combined)}")

    # Deduplicate on 'id', keeping last occurrence (newest scrape wins)
    merged = combined.drop_duplicates(subset="id", keep="last")
    print(f"Total rows after dedup:  {len(merged)}")
    print(f"Duplicates removed:      {len(combined) - len(merged)}")

    # Write output
    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"\nMerged file written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
