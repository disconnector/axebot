#!/usr/bin/env python3
"""
Corpus noise cleanup — removes low-quality chunks from ChromaDB.
Targets wiki category index entries, VTT metadata artifacts, and
any chunk too short to carry semantic meaning.
Run as a one-time or scheduled cleanup job.
"""

import sys
import logging
from pathlib import Path

import chromadb

DB_PATH    = Path.home() / "rag" / "db"
COLLECTION = "axefx3"
LOG_PATH   = Path.home() / "rag" / "cleanup_noise.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_PATH)),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("cleanup")

# Minimum word count — anything shorter is almost certainly noise
MIN_WORDS = 30

# Boilerplate patterns — if 2+ match, chunk is noise
# 2+ of these = noise
NOISE_PATTERNS = [
    "login", "log in", "register", "create account",
    "navigation menu", "jump to navigation", "jump to search",
    "retrieved from", "last edited", "privacy policy",
    "terms of use", "edit this page", "view history",
    "this page was last", "permanent link", "page information",
    "what links here", "related changes", "special pages",
    "printable version", "in other languages",
]

# 1 of these = instant noise (highly specific boilerplate)
NOISE_PATTERNS_STRONG = [
    "maintained by members of the community",
    "wiki for products made by fractal audio",
    "url: https://forum.fractalaudio.com",
    "revision history of \"",
    "video history: legend",
]

def is_noise(doc: str) -> bool:
    if doc is None:
        return True
    words = doc.split()
    if len(words) < MIN_WORDS:
        return True
    doc_lower = doc.lower()
    if any(p in doc_lower for p in NOISE_PATTERNS_STRONG):
        return True
    hits = sum(1 for p in NOISE_PATTERNS if p in doc_lower)
    return hits >= 2

def main():
    log.info("Starting noise cleanup...")
    client = chromadb.PersistentClient(path=str(DB_PATH))
    col    = client.get_collection(COLLECTION)

    total_before = col.count()
    log.info(f"Collection size before: {total_before}")

    ids_to_delete = []
    offset = 0
    batch  = 2000
    scanned = 0

    while True:
        r = col.get(include=['documents', 'metadatas'], limit=batch, offset=offset)
        if not r['ids']:
            break
        for eid, doc, meta in zip(r['ids'], r['documents'], r['metadatas']):
            # Only clean wiki and youtube — PDFs and forum were curated more carefully
            if meta.get('source_type') not in ('wiki', 'youtube'):
                continue
            if is_noise(doc):
                ids_to_delete.append(eid)
        scanned += len(r['ids'])
        log.info(f"  Scanned {scanned}/{total_before}, noise found so far: {len(ids_to_delete)}")
        offset += batch

    if ids_to_delete:
        # Delete in batches to avoid SQLite variable limit
        batch_size = 500
        for i in range(0, len(ids_to_delete), batch_size):
            col.delete(ids=ids_to_delete[i:i+batch_size])
        log.info(f"Deleted {len(ids_to_delete)} noisy chunks")
    else:
        log.info("No noisy chunks found")

    total_after = col.count()
    log.info(f"Collection size after: {total_after} (removed {total_before - total_after})")
    log.info("Cleanup complete.")

if __name__ == "__main__":
    main()
