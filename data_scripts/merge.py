"""
Merge all Reddit scraper CSV files in this directory into a single
deduplicated CSV file (merged_reddit_data.csv).

- Columns are the union of all input files (missing columns filled with empty values).
- Duplicates are removed by the 'id' column, keeping the last occurrence
  (files are loaded oldest-first so newer scrapes take priority).
- Any enrichment columns previously added to the output file (e.g. by
  extract_features.py) are automatically preserved across re-merges.
"""

import glob
import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "processed_data", "merged_reddit_data.csv")


def main():
    # Discover CSV files (exclude the output file) and sort by name (oldest first)
    csv_files = sorted(
        f
        for f in glob.glob(os.path.join(PROJECT_ROOT, "raw_data", "*.csv"))
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

    # Collect the union of all column names present in the raw data
    raw_columns = set()
    for df in frames:
        raw_columns.update(df.columns)

    # Concatenate all frames; pandas aligns columns automatically
    combined = pd.concat(frames, ignore_index=True)
    print(f"\nTotal rows before dedup: {len(combined)}")

    # Deduplicate on 'id', keeping last occurrence (newest scrape wins)
    merged = combined.drop_duplicates(subset="id", keep="last")
    print(f"Total rows after dedup:  {len(merged)}")
    print(f"Duplicates removed:      {len(combined) - len(merged)}")

    # Preserve enrichment columns from the existing output file (if any).
    # Any column in the old output that does not appear in any raw CSV is
    # considered an enrichment column (e.g. payment_method, transaction_value,
    # item_type added by extract_features.py).
    if os.path.isfile(OUTPUT_FILE):
        existing = pd.read_csv(OUTPUT_FILE, dtype=str, keep_default_na=False)
        enriched_cols = [c for c in existing.columns if c not in raw_columns]

        if enriched_cols:
            print(f"\nPreserving {len(enriched_cols)} enrichment column(s) "
                  f"from existing file: {', '.join(enriched_cols)}")

            # Build a lookup keyed by 'id' with only the enrichment columns
            enrichment_src = existing[["id"] + enriched_cols]
            enrichment = enrichment_src.drop_duplicates(
                subset="id", keep="last"
            )

            # Left-join enrichment data onto the fresh merge
            merged = merged.merge(enrichment, on="id", how="left")

            # Fill NaN in enrichment columns with empty string to stay
            # consistent with the dtype=str convention used throughout
            merged[enriched_cols] = merged[enriched_cols].fillna("")
        else:
            print("\nNo enrichment columns found in existing file.")
    else:
        print("\nNo existing output file found; skipping enrichment preservation.")

    # Write output
    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"\nMerged file written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
