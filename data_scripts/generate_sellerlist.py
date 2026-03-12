import os
import pandas as pd
import requests
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

df = pd.read_csv(os.path.join(PROJECT_ROOT, "processed_data", "merged_reddit_data.csv"))
seller_list = df.loc[df["dataType"] == "post", ["dataType", "authorName", "postUrl"]].copy()

seller_list["sellerFlair"] = None

session = requests.Session()
session.headers.update(
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FlairFetcher/1.0"}
)

total = len(seller_list)

for i, (idx, row) in enumerate(seller_list.iterrows()):
    url = row["postUrl"] + ".json"
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        flair = data[0]["data"]["children"][0]["data"]["author_flair_text"]
        seller_list.at[idx, "sellerFlair"] = flair
    except Exception:
        seller_list.at[idx, "sellerFlair"] = None

    print(f"Row {i + 1}/{total} - {row['authorName']}: {seller_list.at[idx, 'sellerFlair']}")

    if i < total - 1:
        time.sleep(3)

output_path = os.path.join(PROJECT_ROOT, "processed_data", "seller_list.csv")
seller_list.to_csv(output_path, index=False)

print(f"Wrote {total} rows to {output_path}")
