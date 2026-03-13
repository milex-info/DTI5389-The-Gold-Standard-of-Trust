#!/usr/bin/env python3
"""
Extract transaction features (payment_method, transaction_value, item_type)
from r/Pmsforsale post text using Google Gemini.

Reads  : processed_data/merged_reddit_data.csv
Writes : same file, with three new columns added and populated for post rows.

Usage:
    python extract_features.py                  # process all unprocessed posts
    python extract_features.py --force          # reprocess every post
    python extract_features.py --batch-size 10  # adjust batch size
    python extract_features.py --dry-run        # preview without writing
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import google.generativeai as genai

# ── Paths ───────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CSV_PATH = os.path.join(PROJECT_ROOT, "processed_data", "merged_reddit_data.csv")

# ── Gemini setup ────────────────────────────────────────────────────────────

SYSTEM_INSTRUCTION = """\
You are a structured-data extraction assistant for the Reddit marketplace
r/Pmsforsale, where users buy, sell, and trade precious metals.

For each post you receive you must extract exactly three fields:

1. **payment_method** – The accepted payment method(s) mentioned in the post.
   Return a comma-separated lowercase list chosen from:
   zelle, venmo, paypal, cashapp, crypto, applepay, googlepay, money order, check, cash, unknown
   If the post mentions "PPFF" or "PayPal FF" or "PayPal Friends and Family", return "paypal".
   If the post mentions "PPGS" or "PayPal G&S" or "PayPal Goods and Services", return "paypal".
   If no payment method is mentioned, return "unknown".

2. **transaction_value** – The approximate total dollar value of all items
   listed for sale/trade in the post.  Return a single number (integer).
   - If the post lists multiple items with prices, SUM them all.
   - If a price says "spot" or "at spot", estimate using these approximate
     spot prices: gold oz ≈ $2900, silver oz ≈ $33, platinum oz ≈ $1000.
     Scale by fractional weights (e.g. 1/10 oz gold ≈ $290).
   - If the post is a WTB (Want To Buy) post with a budget, use that budget.
   - If you truly cannot determine any value, return 0.

3. **item_type** – The type of precious metal(s) in the post.
   Return one of: gold, silver, platinum, palladium, copper, mixed, unknown
   - If multiple metal types are listed, return "mixed".
   - Coins and bars count as the metal they are made of.
   - "Junk silver" or "90%" refers to silver.

Return your answer as a JSON array.  Each element must have these keys:
  "id"                – the post ID you were given
  "payment_method"    – string
  "transaction_value" – integer
  "item_type"         – string

Return ONLY the JSON array, no markdown fences, no commentary.
"""

# Maximum characters of body text to send per post (to stay within token limits)
MAX_BODY_CHARS = 2000

# ── Helpers ─────────────────────────────────────────────────────────────────


def configure_gemini() -> genai.GenerativeModel:
    """Configure and return a Gemini model instance."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY environment variable is not set.")
        sys.exit(1)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-3-flash-preview",
        system_instruction=SYSTEM_INSTRUCTION,
    )
    return model


def build_prompt(posts: List[Dict[str, str]]) -> str:
    """Build a prompt for a batch of posts."""
    parts = []
    for p in posts:
        title = (p.get("title") or "").strip()
        body = (p.get("body") or "").strip()
        # Truncate very long bodies
        if len(body) > MAX_BODY_CHARS:
            body = body[:MAX_BODY_CHARS] + "..."
        parts.append(f'--- POST ID: {p["id"]} ---\nTitle: {title}\nBody: {body}\n')

    return (
        f"Extract payment_method, transaction_value, and item_type from "
        f"each of the following {len(posts)} r/Pmsforsale posts.\n\n"
        + "\n".join(parts)
    )


def parse_response(text: str, expected_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Parse the JSON array response from Gemini.

    Returns a dict mapping post ID -> {payment_method, transaction_value, item_type}.
    """
    # Strip markdown fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (```json or ```)
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find a JSON array in the text
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return {}
        else:
            return {}

    if not isinstance(data, list):
        return {}

    result = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        pid = str(item.get("id", "")).strip()
        if not pid:
            continue
        result[pid] = {
            "payment_method": str(item.get("payment_method", "unknown")).strip().lower(),
            "transaction_value": _parse_numeric(item.get("transaction_value", 0)),
            "item_type": str(item.get("item_type", "unknown")).strip().lower(),
        }

    return result


def _parse_numeric(val: Any) -> str:
    """Convert a value to a numeric string, or empty string if not parseable."""
    if val is None:
        return ""
    try:
        n = float(val)
        if n == 0:
            return ""
        return str(int(n))
    except (ValueError, TypeError):
        # Try to extract a number from a string like "$300"
        m = re.search(r"[\d,]+\.?\d*", str(val))
        if m:
            try:
                n = float(m.group().replace(",", ""))
                return str(int(n)) if n > 0 else ""
            except ValueError:
                pass
        return ""


def call_gemini_with_retry(
    model: genai.GenerativeModel,
    prompt: str,
    max_retries: int = 5,
    base_delay: float = 2.0,
) -> Optional[str]:
    """Call Gemini with exponential backoff on retryable errors."""
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            err_str = str(e).lower()
            # Retry on rate limit, server errors, or transient issues
            if any(kw in err_str for kw in ["429", "rate", "quota", "500", "503", "overloaded", "resource"]):
                delay = base_delay * (2 ** attempt)
                print(f"    Retryable error (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"    Waiting {delay:.1f}s before retry...")
                time.sleep(delay)
            else:
                print(f"    Non-retryable error: {e}")
                return None
    print(f"    Exhausted {max_retries} retries.")
    return None


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract transaction features from post text using Gemini"
    )
    ap.add_argument("--force", action="store_true",
                    help="Reprocess all posts even if columns already exist")
    ap.add_argument("--batch-size", type=int, default=15,
                    help="Number of posts per Gemini API call (default: 15)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview extraction on first batch without writing")
    args = ap.parse_args()

    # ── 1. Load CSV ─────────────────────────────────────────────────────────
    print(f"Reading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
    print(f"  {len(df)} rows loaded "
          f"({(df['dataType'] == 'post').sum()} posts, "
          f"{(df['dataType'] == 'comment').sum()} comments)")

    # ── 2. Add columns if missing ───────────────────────────────────────────
    for col in ("payment_method", "transaction_value", "item_type"):
        if col not in df.columns:
            df[col] = ""
            print(f"  Added column: {col}")
        else:
            print(f"  Column already exists: {col}")

    # ── 3. Identify posts to process ────────────────────────────────────────
    post_mask = df["dataType"] == "post"
    posts_idx = df.index[post_mask].tolist()

    if not args.force:
        # Skip posts that already have all three fields populated
        already_done = (
            (df.loc[post_mask, "payment_method"].str.strip() != "")
            & (df.loc[post_mask, "transaction_value"].str.strip() != "")
            & (df.loc[post_mask, "item_type"].str.strip() != "")
        )
        posts_to_process = [i for i in posts_idx if not already_done.get(i, False)]
    else:
        posts_to_process = posts_idx

    total = len(posts_to_process)
    print(f"\n  Posts to process: {total} (of {len(posts_idx)} total posts)")

    if total == 0:
        print("  Nothing to do. Use --force to reprocess all posts.")
        return

    # ── 4. Configure Gemini ─────────────────────────────────────────────────
    model = configure_gemini()
    print("  Gemini model configured (gemini-2.0-flash)")

    # ── 5. Process in batches ───────────────────────────────────────────────
    batch_size = args.batch_size
    num_batches = (total + batch_size - 1) // batch_size
    processed = 0
    failed = 0

    print(f"\n  Processing {total} posts in {num_batches} batches of up to {batch_size}...\n")

    for batch_num in range(num_batches):
        start = batch_num * batch_size
        end = min(start + batch_size, total)
        batch_indices = posts_to_process[start:end]

        # Build batch data
        batch_posts = []
        for idx in batch_indices:
            row = df.loc[idx]
            pid = str(row.get("parsedId", row.get("id", ""))).strip()
            batch_posts.append({
                "id": pid,
                "title": str(row.get("title", "")),
                "body": str(row.get("body", "")),
                "df_index": idx,
            })

        prompt = build_prompt(batch_posts)

        # Call Gemini
        response_text = call_gemini_with_retry(model, prompt)

        if response_text is None:
            print(f"  Batch {batch_num + 1}/{num_batches}: FAILED (no response)")
            failed += len(batch_posts)
            continue

        # Parse response
        expected_ids = [p["id"] for p in batch_posts]
        results = parse_response(response_text, expected_ids)

        # Map results back to DataFrame
        batch_ok = 0
        for post in batch_posts:
            pid = post["id"]
            idx = post["df_index"]
            if pid in results:
                r = results[pid]
                df.at[idx, "payment_method"] = r["payment_method"]
                df.at[idx, "transaction_value"] = r["transaction_value"]
                df.at[idx, "item_type"] = r["item_type"]
                batch_ok += 1
            else:
                # Mark as unknown so we don't reprocess on next run
                df.at[idx, "payment_method"] = "unknown"
                df.at[idx, "transaction_value"] = ""
                df.at[idx, "item_type"] = "unknown"
                failed += 1

        processed += batch_ok
        pct = (start + len(batch_posts)) / total * 100

        print(f"  Batch {batch_num + 1}/{num_batches}: "
              f"{batch_ok}/{len(batch_posts)} extracted  "
              f"[{start + len(batch_posts)}/{total} = {pct:.1f}%]")

        if args.dry_run:
            print("\n  --dry-run: showing first batch results and stopping.\n")
            for post in batch_posts:
                pid = post["id"]
                if pid in results:
                    r = results[pid]
                    print(f"    {pid}: payment={r['payment_method']}, "
                          f"value={r['transaction_value']}, type={r['item_type']}")
                else:
                    print(f"    {pid}: (no result)")
            return

        # Small delay between batches to be respectful of rate limits
        if batch_num < num_batches - 1:
            time.sleep(0.5)

    # ── 6. Write updated CSV ────────────────────────────────────────────────
    print(f"\n  Extraction complete: {processed} succeeded, {failed} failed/missing")
    print(f"  Writing updated CSV to {CSV_PATH} ...")
    df.to_csv(CSV_PATH, index=False)
    print("  Done.")


if __name__ == "__main__":
    main()
