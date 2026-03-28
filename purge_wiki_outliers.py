#!/usr/bin/env python3
"""
purge_wiki_outliers.py — Audit and purge off-topic wiki chunks from ChromaDB.

Pulls all wiki chunks, groups by source URL, flags non-Axe-Fx-III pages,
shows a full report, then purges flagged chunks from both collections.

Usage:
    python purge_wiki_outliers.py           # dry run — report only
    python purge_wiki_outliers.py --purge   # actually delete
"""

import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction

# ── Config ─────────────────────────────────────────────────────────────────────
DB_PATH        = Path.home() / "rag" / "db"
OLD_COLLECTION = "axefx3"
NEW_COLLECTION = "axefx3_bge"

# URL patterns that indicate off-topic wiki pages
BAD_PATTERNS = [
    r'FM3', r'FM9', r'AM4', r'VP4', r'AX8', r'ICON',
    r'Category:FM', r'Category:AM', r'Category:AX',
    r'File:.*(?:FM3|FM9|AM4|VP4|AX8|ICON)',
]
BAD_RE = re.compile('|'.join(BAD_PATTERNS), re.IGNORECASE)

# Also flag wiki "Category:", "File:", "Special:" pages that are clearly nav/meta
META_RE = re.compile(r'title=(File:|Category:|Special:|Talk:|Template:)', re.IGNORECASE)


class PassthroughEF(EmbeddingFunction):
    def __init__(self): pass
    def __call__(self, input): raise RuntimeError("PassthroughEF called")


def get_all_wiki_chunks(col):
    """Read all wiki chunks from a collection."""
    chunks = {"ids": [], "metadatas": [], "documents": []}
    offset = 0
    batch  = 1000

    while True:
        r = col.get(
            where={"source_type": "wiki"},
            limit=batch,
            offset=offset,
            include=["metadatas", "documents"],
        )
        if not r["ids"]:
            break
        chunks["ids"].extend(r["ids"])
        chunks["metadatas"].extend(r["metadatas"])
        chunks["documents"].extend(r["documents"])
        offset += len(r["ids"])
        print(f"  Read {offset:,} wiki chunks...", end="\r")

    print()
    return chunks


def audit(chunks):
    """Group chunks by source URL and flag bad ones."""
    by_source = defaultdict(list)  # source_url -> list of (id, meta, doc)

    for id_, meta, doc in zip(chunks["ids"], chunks["metadatas"], chunks["documents"]):
        src = meta.get("source", meta.get("filename", "unknown"))
        by_source[src].append((id_, meta, doc))

    flagged   = {}   # source_url -> list of ids
    kept      = {}   # source_url -> count

    for src, items in by_source.items():
        if BAD_RE.search(src) or META_RE.search(src):
            flagged[src] = [i for i, m, d in items]
        else:
            kept[src] = len(items)

    return flagged, kept


def main():
    parser = argparse.ArgumentParser(description="Audit and purge off-topic wiki chunks")
    parser.add_argument("--purge", action="store_true", help="Actually delete flagged chunks")
    args = parser.parse_args()

    print("\nWiki Outlier Audit")
    print("=" * 60)

    client  = chromadb.PersistentClient(path=str(DB_PATH))
    new_col = client.get_collection(name=NEW_COLLECTION, embedding_function=PassthroughEF())
    old_col = client.get_collection(name=OLD_COLLECTION)

    print(f"Reading wiki chunks from '{NEW_COLLECTION}'...")
    chunks  = get_all_wiki_chunks(new_col)
    total   = len(chunks["ids"])
    print(f"  {total:,} wiki chunks across both collections.\n")

    flagged, kept = audit(chunks)

    flagged_ids = [id_ for ids in flagged.values() for id_ in ids]
    kept_count  = sum(kept.values())

    # ── Report ─────────────────────────────────────────────────────────────────
    print(f"FLAGGED — {len(flagged):,} sources, {len(flagged_ids):,} chunks to purge:")
    print("-" * 60)
    for src, ids in sorted(flagged.items(), key=lambda x: -len(x[1])):
        # Extract the page title for readability
        m = re.search(r'title=([^&]+)', src)
        title = m.group(1).replace('+', ' ') if m else src[-60:]
        print(f"  [{len(ids):3d} chunks]  {title}")

    print(f"\nKEPT — {len(kept):,} sources, {kept_count:,} chunks:")
    print("-" * 60)
    for src, count in sorted(kept.items(), key=lambda x: -x[1])[:20]:
        m = re.search(r'title=([^&]+)', src)
        title = m.group(1).replace('+', ' ') if m else src[-60:]
        print(f"  [{count:3d} chunks]  {title}")
    if len(kept) > 20:
        print(f"  ... and {len(kept)-20} more sources")

    print(f"\nSummary:")
    print(f"  Total wiki chunks : {total:,}")
    print(f"  To purge          : {len(flagged_ids):,} ({len(flagged_ids)/total*100:.1f}%)")
    print(f"  To keep           : {kept_count:,}")

    if not args.purge:
        print(f"\nDry run — re-run with --purge to delete flagged chunks from both collections.")
        return

    # ── Purge ──────────────────────────────────────────────────────────────────
    print(f"\nPurging {len(flagged_ids):,} chunks from both collections...")

    batch_size = 500
    for i in range(0, len(flagged_ids), batch_size):
        batch = flagged_ids[i:i+batch_size]
        new_col.delete(ids=batch)
        old_col.delete(ids=batch)
        print(f"  Deleted {min(i+batch_size, len(flagged_ids)):,}/{len(flagged_ids):,}")

    print(f"\nDone.")
    print(f"  '{NEW_COLLECTION}' now has {new_col.count():,} chunks")
    print(f"  '{OLD_COLLECTION}' now has {old_col.count():,} chunks")


if __name__ == "__main__":
    main()
