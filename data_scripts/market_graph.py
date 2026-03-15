#!/usr/bin/env python3
"""
Extract buyer-seller transactions from r/Pmsforsale Reddit data and build
a directed network graph of the marketplace.

Input : CSV file with posts and comments (Apify Reddit scraper format).
Output: transactions.csv  – edge list describing inferred transactions
        transaction_network.png – NetworkX / Matplotlib visualisation

Rules for inferring transactions
---------------------------------
HIGH-CONFIDENCE
  A comment B by a non-author contains a buyer expression (BIN, I'll take, …).
  A comment C by the post author is a *direct reply* to B (parentId == id of B).
  C contains a seller confirmation phrase (yours, sold, trade pending, …).

LOW-CONFIDENCE  (opt-in via --include-low-confidence)
  Same structure but the buyer expression is only "chat" / "pm" and/or the
  seller reply is only "replied" / "chat".

SOLD-POST FALLBACK
  Post body/title/flair contains "SOLD" and exactly one buyer expressed
  interest (via a comment that was not already matched above).  Optionally,
  when multiple buyers exist, the earliest BIN wins (--allow-earliest-fallback).
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # non-interactive backend – works without a display
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ── Graph palette ───────────────────────────────────────────────────────────
_BG      = "#16161D"
_GOLD    = "#FDE4A3"
_WHITE50 = "#FFFFFF"  # used with alpha=0.5 where needed

# Try to use IBM Plex Sans; fall back to a generic sans-serif.
_LABEL_FONT = "IBM Plex Sans"
if not any(_LABEL_FONT.lower() in f.name.lower() for f in fm.fontManager.ttflist):
    _LABEL_FONT = "sans-serif"


# ── Defaults ────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_INPUT = os.path.join(
    PROJECT_ROOT,
    "processed_data", "merged_reddit_data.csv",
)

# ── Pattern lists ───────────────────────────────────────────────────────────

BUYER_PATTERNS_HIGH: List[str] = [
    r"\bbin\b",
    r"\bb\.i\.n\.",
    r"\bbuy it now\b",
    r"\bi will take\b",
    r"\bi['']ll take\b",
    r"\bi will buy\b",
    r"\bi['']ll buy\b",
    r"\btake it\b",
    r"\bdibs\b",
    r"\bclaim\b",
]

BUYER_PATTERNS_LOW: List[str] = [
    r"\bchat\b",
    r"\bpm\b",
    r"\bmessage\b",
    r"\binterested\b",
]

SELLER_PATTERNS_HIGH: List[str] = [
    r"\byours\b",
    r"\bsold to you\b",
    r"\bsold\b",
    r"\btrade pending\b",
    r"\bpending\b",
    r"\bpaid\b",
    r"\binvoice sent\b",
]

SELLER_PATTERNS_LOW: List[str] = [
    r"\breplied\b",
    r"\bpm sent\b",
]

SOLD_PATTERN = re.compile(r"\bsold\b", re.IGNORECASE)

# Confidence scores assigned to each rule type
CONFIDENCE_SCORES: Dict[str, float] = {
    "high":                   1.0,
    "low":                    0.6,
    "sold_post_single_buyer": 0.4,
    "sold_post_earliest_bin": 0.3,
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _compile(patterns: List[str]) -> List[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


_BUYER_HIGH  = _compile(BUYER_PATTERNS_HIGH)
_BUYER_LOW   = _compile(BUYER_PATTERNS_LOW)
_SELLER_HIGH = _compile(SELLER_PATTERNS_HIGH)
_SELLER_LOW  = _compile(SELLER_PATTERNS_LOW)


def _matches(text: str, compiled: List[re.Pattern]) -> bool:
    return any(p.search(text) for p in compiled)


def detect_buyer(text: str) -> Optional[str]:
    """Return 'BIN' | 'CHAT' | None."""
    if not isinstance(text, str) or not text.strip():
        return None
    if _matches(text, _BUYER_HIGH):
        return "BIN"
    if _matches(text, _BUYER_LOW):
        return "CHAT"
    return None


def detect_seller(text: str) -> Optional[str]:
    """Return 'HIGH' | 'LOW' | None."""
    if not isinstance(text, str) or not text.strip():
        return None
    if _matches(text, _SELLER_HIGH):
        return "HIGH"
    if _matches(text, _SELLER_LOW):
        return "LOW"
    return None


def normalize_id(raw) -> str:
    """Strip Reddit type prefixes (t1_, t3_) and whitespace."""
    if raw is None:
        return ""
    s = str(raw).strip()
    if s.startswith(("t1_", "t3_")):
        return s[3:]
    return s


def _first(row: pd.Series, cols: List[str]) -> str:
    """Return the first non-empty string value from *cols*."""
    for c in cols:
        v = str(row.get(c, "")).strip()
        if v:
            return v
    return ""


def _timestamp(row: pd.Series) -> pd.Timestamp:
    raw = _first(row, ["commentCreatedAt", "createdAt", "postCreatedAt"])
    return pd.to_datetime(raw, errors="coerce", utc=True)


def _is_post_sold(row: pd.Series) -> bool:
    blob = " ".join(
        str(row.get(c, ""))
        for c in ("body", "title", "postTitle", "flair")
    )
    return bool(SOLD_PATTERN.search(blob))


# ── Main pipeline ───────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build buyer-seller network graph from r/Pmsforsale CSV data"
    )
    ap.add_argument("--input",  default=DEFAULT_INPUT,
                    help="Input CSV file path")
    ap.add_argument("--edges",  default="transactions.csv",
                    help="Output edge-list CSV path")
    ap.add_argument("--image",  default="transaction_network.png",
                    help="Output network PNG path")
    ap.add_argument("--include-low-confidence", action="store_true",
                    help="Include chat/replied pairs (flagged as low-confidence)")
    ap.add_argument("--allow-earliest-fallback", action="store_true",
                    help="For SOLD posts with >1 buyer, pick earliest BIN")
    ap.add_argument("--top-sellers", type=int, default=0, metavar="N",
                    help="Also output a filtered view with only the top N sellers "
                         "(by transaction count) and their buyers.  "
                         "Produces <edges>_top<N>.csv and <image>_top<N>.png")
    args = ap.parse_args()

    # ── 1. Load ─────────────────────────────────────────────────────────────
    print(f"Reading {args.input} ...")
    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    df["body"] = df["body"].fillna("")
    print(f"  {len(df)} rows loaded  "
          f"({(df['dataType']=='post').sum()} posts, "
          f"{(df['dataType']=='comment').sum()} comments)")

    # ── 2. Index posts ──────────────────────────────────────────────────────
    posts_df   = df[df["dataType"] == "post"]
    comment_df = df[df["dataType"] == "comment"]

    post_author: Dict[str, str] = {}
    post_sold:   Dict[str, bool] = {}
    for _, row in posts_df.iterrows():
        pid = normalize_id(_first(row, ["parsedId", "id", "parsedPostId"]))
        if not pid:
            continue
        post_author[pid] = str(row.get("authorName", "")).strip()
        post_sold[pid]   = _is_post_sold(row)

    # ── 3. Index comments ───────────────────────────────────────────────────
    Comment = Dict[str, object]
    comments: List[Comment] = []
    for _, row in comment_df.iterrows():
        cid  = normalize_id(_first(row, ["parsedId", "id"]))
        par  = normalize_id(_first(row, ["parsedParentId", "parentId"]))
        pid  = normalize_id(_first(row, ["parsedPostId", "postId"]))
        if not cid or not pid:
            continue
        author = str(row.get("authorName", "")).strip()
        # skip AutoModerator – it's a bot, not a buyer
        if author == "AutoModerator":
            continue
        comments.append({
            "id":        cid,
            "author":    author,
            "body":      str(row.get("body", "")),
            "parent_id": par,
            "post_id":   pid,
            "ts":        _timestamp(row),
        })

    parent_children: Dict[str, List[Comment]] = {}
    post_comments:   Dict[str, List[Comment]] = {}
    for c in comments:
        parent_children.setdefault(c["parent_id"], []).append(c)
        post_comments.setdefault(c["post_id"], []).append(c)

    # sort children by timestamp so earliest comes first
    for kids in parent_children.values():
        kids.sort(key=lambda c: (c["ts"] if pd.notna(c["ts"]) else pd.Timestamp.max, c["id"]))

    # ── 4. Rule 1 – buyer comment + seller reply ────────────────────────────
    transactions: List[Dict[str, object]] = []
    seen: set = set()

    for c in comments:
        pid    = c["post_id"]
        seller = post_author.get(pid)
        if not seller:
            continue
        if not c["author"] or c["author"] == seller:
            continue

        btype = detect_buyer(c["body"])
        if btype is None:
            continue
        if btype == "CHAT" and not args.include_low_confidence:
            continue

        # look for seller's direct reply
        for child in parent_children.get(c["id"], []):
            if child["author"] != seller:
                continue
            sconf = detect_seller(child["body"])
            if sconf is None:
                continue
            if sconf == "LOW" and not args.include_low_confidence:
                continue

            conf  = "high" if sconf == "HIGH" else "low"
            rule  = "seller_confirmed" if sconf == "HIGH" else (
                "chat_replied" if btype == "CHAT" else "seller_acknowledged"
            )
            key = (pid, c["author"], c["id"], child["id"])
            if key in seen:
                continue
            seen.add(key)

            transactions.append({
                "seller":              seller,
                "buyer":               c["author"],
                "post_id":             pid,
                "buyer_comment_id":    c["id"],
                "seller_comment_id":   child["id"],
                "buyer_comment_time":  c["ts"],
                "seller_comment_time": child["ts"],
                "rule":                rule,
                "confidence":          conf,
                "confidence_score":    CONFIDENCE_SCORES[conf],
            })
            break  # first confirming reply wins

    print(f"  Rule 1 (reply-based): {len(transactions)} edges")

    # ── 5. Rule 2 – SOLD post fallback ──────────────────────────────────────
    already_matched = {(t["post_id"], t["buyer"]) for t in transactions}
    fallback_count = 0

    for pid, sold in post_sold.items():
        if not sold:
            continue
        seller = post_author.get(pid, "")
        if not seller:
            continue

        candidates: List[Tuple[str, Comment]] = []
        for c in post_comments.get(pid, []):
            if not c["author"] or c["author"] == seller:
                continue
            bt = detect_buyer(c["body"])
            if bt is None:
                continue
            if bt == "CHAT" and not args.include_low_confidence:
                continue
            if (pid, c["author"]) in already_matched:
                continue
            candidates.append((bt, c))

        if len(candidates) == 1:
            bt, c = candidates[0]
            transactions.append({
                "seller":              seller,
                "buyer":               c["author"],
                "post_id":             pid,
                "buyer_comment_id":    c["id"],
                "seller_comment_id":   "",
                "buyer_comment_time":  c["ts"],
                "seller_comment_time": pd.NaT,
                "rule":                "sold_post_single_buyer",
                "confidence":          "sold_post_single_buyer",
                "confidence_score":    CONFIDENCE_SCORES["sold_post_single_buyer"],
            })
            fallback_count += 1

        elif len(candidates) > 1 and args.allow_earliest_fallback:
            bins = [x for x in candidates if x[0] == "BIN"]
            pool = bins if bins else candidates
            pool.sort(key=lambda x: (
                x[1]["ts"] if pd.notna(x[1]["ts"]) else pd.Timestamp.max,
                x[1]["id"],
            ))
            _, c = pool[0]
            transactions.append({
                "seller":              seller,
                "buyer":               c["author"],
                "post_id":             pid,
                "buyer_comment_id":    c["id"],
                "seller_comment_id":   "",
                "buyer_comment_time":  c["ts"],
                "seller_comment_time": pd.NaT,
                "rule":                "sold_post_earliest_bin",
                "confidence":          "sold_post_earliest_bin",
                "confidence_score":    CONFIDENCE_SCORES["sold_post_earliest_bin"],
            })
            fallback_count += 1

    print(f"  Rule 2 (SOLD fallback): {fallback_count} edges")

    # ── 6. Output CSV ───────────────────────────────────────────────────────
    if not transactions:
        print("No transactions found.")
        return

    out = pd.DataFrame(transactions)
    out.to_csv(os.path.join(PROJECT_ROOT, 'processed_data', args.edges), index=False)
    print(f"  Total: {len(transactions)} transactions -> {args.edges}")

    # ── 7. Build NetworkX graph ─────────────────────────────────────────────
    G = nx.DiGraph()
    weights: Dict[Tuple[str, str], float] = {}
    counts:  Dict[Tuple[str, str], int]   = {}
    for t in transactions:
        pair = (t["seller"], t["buyer"])
        weights[pair] = weights.get(pair, 0.0) + float(t["confidence_score"])
        counts[pair]  = counts.get(pair, 0) + 1

    for pair, w in weights.items():
        G.add_edge(pair[0], pair[1], weight=w, count=counts[pair])

    # ── 8. Draw ─────────────────────────────────────────────────────────────
    pos = nx.spring_layout(G, seed=42, k=1.8)
    deg = dict(G.degree())

    # Identify seller vs buyer nodes
    seller_set = {t["seller"] for t in transactions}
    seller_nodes = [n for n in G.nodes() if n in seller_set]
    buyer_nodes  = [n for n in G.nodes() if n not in seller_set]

    seller_sizes = [300 + 120 * deg[n] for n in seller_nodes]
    buyer_sizes  = [300 + 120 * deg[n] for n in buyer_nodes]

    edge_widths = [0.5 + 0.7 * G[u][v]["weight"] for u, v in G.edges()]

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    # Seller nodes: hollow with #FDE4A3 2px stroke
    nx.draw_networkx_nodes(
        G, pos, nodelist=seller_nodes, ax=ax,
        node_size=seller_sizes, node_color=_BG, alpha=1.0,
        edgecolors=_GOLD, linewidths=2,
    )
    # Buyer nodes: white at 50% opacity
    nx.draw_networkx_nodes(
        G, pos, nodelist=buyer_nodes, ax=ax,
        node_size=buyer_sizes, node_color=_WHITE50, alpha=0.5,
        edgecolors="none", linewidths=0,
    )

    # Edges: white at 50% opacity, no labels
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths, edge_color=_WHITE50,
        arrows=True, arrowsize=14, alpha=0.5,
        connectionstyle="arc3,rad=0.1",
    )

    # Only seller labels: IBM Plex Sans Regular, #FDE4A3
    seller_labels = {nd: nd for nd in seller_nodes}
    nx.draw_networkx_labels(
        G, pos, labels=seller_labels, ax=ax,
        font_size=7, font_weight="normal",
        font_color=_GOLD, font_family=_LABEL_FONT,
    )

    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(PROJECT_ROOT, 'visualizations', args.image),
                dpi=150, facecolor=_BG)
    svg_name = _split_ext(args.image)[0] + ".svg"
    fig.savefig(os.path.join(PROJECT_ROOT, 'visualizations', svg_name),
                facecolor=_BG)
    print(f"  Network visualisation -> {args.image}, {svg_name}")
    print(f"  Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")

    # ── 9. Top-sellers mode ─────────────────────────────────────────────────
    if args.top_sellers and args.top_sellers > 0:
        _top_sellers(transactions, args.top_sellers, args.edges, args.image)


# ── Top-sellers helper ──────────────────────────────────────────────────────

def _top_sellers(
    transactions: List[Dict[str, object]],
    n: int,
    base_edges: str,
    base_image: str,
) -> None:
    """Filter to the top *n* sellers by transaction count, output a separate
    CSV edge list and a cleaner network visualisation."""

    print(f"\n-- Top {n} Sellers --")

    # Count transactions per seller
    seller_counts: Dict[str, int] = {}
    for t in transactions:
        s = str(t["seller"])
        seller_counts[s] = seller_counts.get(s, 0) + 1

    ranked = sorted(seller_counts.items(), key=lambda x: x[1], reverse=True)
    top_set = {name for name, _ in ranked[:n]}

    for rank, (name, count) in enumerate(ranked[:n], 1):
        print(f"  #{rank}  {name}  ({count} transactions)")

    # Filter transactions
    filtered = [t for t in transactions if str(t["seller"]) in top_set]
    if not filtered:
        print("  No transactions for top sellers (unexpected).")
        return

    # ── Output CSV ──────────────────────────────────────────────────────────
    stem_e, ext_e = _split_ext(base_edges)
    out_path = f"{stem_e}_top{n}{ext_e}"
    pd.DataFrame(filtered).to_csv(os.path.join(PROJECT_ROOT, 'processed_data', out_path), index=False)
    print(f"  {len(filtered)} transactions -> {out_path}")

    # ── Build subgraph ──────────────────────────────────────────────────────
    G = nx.DiGraph()
    weights: Dict[Tuple[str, str], float] = {}
    counts:  Dict[Tuple[str, str], int]   = {}
    for t in filtered:
        pair = (str(t["seller"]), str(t["buyer"]))
        weights[pair] = weights.get(pair, 0.0) + float(t["confidence_score"])
        counts[pair]  = counts.get(pair, 0) + 1

    for pair, w in weights.items():
        G.add_edge(pair[0], pair[1], weight=w, count=counts[pair])

    # ── Clustered layout: buyers orbit their primary seller ──────────────
    rng = random.Random(42)

    seller_nodes = [nd for nd in G.nodes() if nd in top_set]
    buyer_nodes  = [nd for nd in G.nodes() if nd not in top_set]

    # 1. Place sellers evenly on a large circle
    seller_pos: Dict[str, Tuple[float, float]] = {}
    n_sellers = len(seller_nodes)
    for i, s in enumerate(seller_nodes):
        angle = 2 * math.pi * i / n_sellers
        seller_pos[s] = (math.cos(angle), math.sin(angle))

    # 2. Map each buyer to their primary seller (most transactions)
    primary_seller: Dict[str, str] = {}
    for buyer in buyer_nodes:
        best_seller, best_count = seller_nodes[0], 0
        for seller in seller_nodes:
            if G.has_edge(seller, buyer):
                c = G[seller][buyer]["count"]
                if c > best_count:
                    best_count = c
                    best_seller = seller
        primary_seller[buyer] = best_seller

    # 3. Group buyers by primary seller
    clusters: Dict[str, List[str]] = {s: [] for s in seller_nodes}
    for buyer, seller in primary_seller.items():
        clusters[seller].append(buyer)

    # 4. Position buyers in a ring around their primary seller
    pos: Dict[str, Tuple[float, float]] = {}
    pos.update(seller_pos)
    orbit_radius = 0.35
    for seller, buyers in clusters.items():
        sx, sy = seller_pos[seller]
        n_buyers = len(buyers)
        for j, buyer in enumerate(buyers):
            angle = 2 * math.pi * j / max(n_buyers, 1)
            jitter_r = rng.uniform(-0.04, 0.04)
            jitter_a = rng.uniform(-0.15, 0.15)
            r = orbit_radius + jitter_r
            a = angle + jitter_a
            pos[buyer] = (sx + r * math.cos(a), sy + r * math.sin(a))

    deg = dict(G.degree())

    seller_sizes = [600 + 160 * deg[nd] for nd in seller_nodes]
    buyer_sizes  = [250 + 80 * deg[nd] for nd in buyer_nodes]

    edge_widths = [0.6 + 0.8 * G[u][v]["weight"] for u, v in G.edges()]

    fig, ax = plt.subplots(figsize=(16, 11))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    # Seller nodes: hollow with #FDE4A3 2px stroke
    nx.draw_networkx_nodes(
        G, pos, nodelist=seller_nodes, ax=ax,
        node_size=seller_sizes, node_color=_BG, alpha=1.0,
        edgecolors=_GOLD, linewidths=2,
    )
    # Buyer nodes: white at 50% opacity
    nx.draw_networkx_nodes(
        G, pos, nodelist=buyer_nodes, ax=ax,
        node_size=buyer_sizes, node_color=_WHITE50, alpha=0.5,
        edgecolors="none", linewidths=0,
    )

    # Edges: white at 50% opacity, no labels
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths, edge_color=_WHITE50,
        arrows=True, arrowsize=14, alpha=0.5,
        connectionstyle="arc3,rad=0.12",
    )

    # Only seller labels: IBM Plex Sans Regular, #FDE4A3
    seller_labels = {nd: nd for nd in seller_nodes}
    nx.draw_networkx_labels(
        G, pos, labels=seller_labels, ax=ax,
        font_size=9, font_weight="normal",
        font_color=_GOLD, font_family=_LABEL_FONT,
    )

    ax.axis("off")

    fig.tight_layout()
    stem_i, ext_i = _split_ext(base_image)
    img_path = f"{stem_i}_top{n}{ext_i}"
    fig.savefig(os.path.join(PROJECT_ROOT, 'visualizations', img_path),
                dpi=150, facecolor=_BG)
    svg_path = _split_ext(img_path)[0] + ".svg"
    fig.savefig(os.path.join(PROJECT_ROOT, 'visualizations', svg_path),
                facecolor=_BG)
    print(f"  Network visualisation -> {img_path}, {svg_path}")
    print(f"  Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")


def _split_ext(path: str) -> Tuple[str, str]:
    """Split 'foo.csv' into ('foo', '.csv')."""
    root, ext = os.path.splitext(path)
    return root, ext


if __name__ == "__main__":
    main()
