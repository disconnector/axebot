#!/usr/bin/env python3
"""
reembed.py — Re-embed the entire AxeBot corpus using the Mac Studio remote server.

Reads all documents from the old collection, embeds them with the new model
(bge-large-en-v1.5 on the Mac Studio M1 Max), and writes them to a new collection.

The old collection is left untouched — easy to roll back.

Usage:
    EMBED_SERVER_URL=http://your-embed-server:8600 python reembed.py
    EMBED_SERVER_URL=http://your-embed-server:8600 python reembed.py --dry-run
    EMBED_SERVER_URL=http://your-embed-server:8600 python reembed.py --verify-only
"""

import os
import sys
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import chromadb
import numpy as np
import requests

# ── Config ─────────────────────────────────────────────────────────────────────
DB_PATH          = Path.home() / "rag" / "db"
OLD_COLLECTION   = "axefx3"
NEW_COLLECTION   = "axefx3_bge"
EMBED_SERVER_URL = os.environ.get("EMBED_SERVER_URL", "http://your-embed-server:8600")
READ_BATCH       = 1000   # chunks to read from ChromaDB at a time
EMBED_BATCH      = 256    # texts to send to Mac Studio per request
WRITE_BATCH      = 256    # chunks to upsert into new collection at a time
EMBED_TIMEOUT    = 300    # seconds — generous for large batches over LAN


# ── Helpers ────────────────────────────────────────────────────────────────────
def check_server():
    """Verify the embed server is up and print model info."""
    try:
        r = requests.get(f"{EMBED_SERVER_URL}/health", timeout=5)
        info = r.json()
        print(f"  Embed server : {info['model']} on {info['device']}")
        print(f"  Server URL   : {EMBED_SERVER_URL}")
        return info
    except Exception as e:
        print(f"ERROR: Embed server unreachable at {EMBED_SERVER_URL}: {e}")
        print("  Start the server on the Mac Studio and set EMBED_SERVER_URL.")
        sys.exit(1)


def embed_texts(texts: list[str], batch_size: int = EMBED_BATCH) -> list[list[float]]:
    """Send texts to the Mac Studio and return embedding vectors."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        r = requests.post(
            f"{EMBED_SERVER_URL}/embed",
            json={"texts": batch, "batch_size": batch_size},
            timeout=EMBED_TIMEOUT,
        )
        r.raise_for_status()
        all_embeddings.extend(r.json()["embeddings"])
    return all_embeddings


def read_all_chunks(col) -> tuple[list, list, list]:
    """Read ALL documents, ids, and metadatas from a collection in batches."""
    all_ids, all_docs, all_metas = [], [], []
    offset = 0

    while True:
        batch = col.get(
            limit=READ_BATCH,
            offset=offset,
            include=["documents", "metadatas"],
        )
        if not batch["ids"]:
            break
        all_ids.extend(batch["ids"])
        all_docs.extend(batch["documents"])
        all_metas.extend(batch["metadatas"])
        offset += len(batch["ids"])
        print(f"  Read {offset:,} / {col.count():,} chunks...", end="\r")

    print()
    return all_ids, all_docs, all_metas


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Re-embed AxeBot corpus with bge-large-en-v1.5")
    parser.add_argument("--dry-run",     action="store_true", help="Read and embed a sample, don't write")
    parser.add_argument("--verify-only", action="store_true", help="Check both collections exist and compare counts")
    args = parser.parse_args()

    print("\nAxeBot Corpus Re-Embedder")
    print("=" * 50)
    print(f"  Old collection : {OLD_COLLECTION}")
    print(f"  New collection : {NEW_COLLECTION}")
    print(f"  DB path        : {DB_PATH}")

    # ── Connect ────────────────────────────────────────────────────────────────
    server_info = check_server()

    client   = chromadb.PersistentClient(path=str(DB_PATH))
    old_col  = client.get_collection(name=OLD_COLLECTION)
    old_count = old_col.count()
    print(f"  Old corpus     : {old_count:,} chunks\n")

    # ── Verify-only mode ───────────────────────────────────────────────────────
    if args.verify_only:
        try:
            new_col   = client.get_collection(name=NEW_COLLECTION)
            new_count = new_col.count()
            print(f"  New corpus     : {new_count:,} chunks")
            pct = new_count / old_count * 100 if old_count else 0
            print(f"  Progress       : {pct:.1f}% complete")
            if new_count == old_count:
                print("\n  Re-embedding is COMPLETE. Ready to cut over.")
            else:
                print(f"\n  {old_count - new_count:,} chunks still to go.")
        except Exception:
            print("  New collection does not exist yet.")
        return

    # ── Dry run ────────────────────────────────────────────────────────────────
    if args.dry_run:
        print("DRY RUN — reading 100 chunks and embedding them as a test...")
        sample = old_col.get(limit=100, include=["documents", "metadatas"])
        t0 = time.perf_counter()
        vecs = embed_texts(sample["documents"])
        elapsed = time.perf_counter() - t0
        tps = len(sample["documents"]) / elapsed
        dim = len(vecs[0])
        print(f"\n  Embedded {len(vecs)} chunks in {elapsed:.2f}s ({tps:.0f} chunks/sec)")
        print(f"  Vector dim     : {dim}")
        print(f"  Sample magnitude: {float(np.linalg.norm(vecs[0])):.4f}  (bge-large always ~1.0)")
        projected_time = old_count / tps
        print(f"\n  Projected time for full corpus ({old_count:,} chunks): "
              f"{projected_time:.0f}s ({projected_time/60:.1f} min)")
        print("\nDry run complete. Run without --dry-run to proceed.")
        return

    # ── Full re-embed ──────────────────────────────────────────────────────────
    # Create new collection with a passthrough embedding function
    # (embeddings are pre-computed and passed directly — ChromaDB won't re-embed)
    from chromadb.utils.embedding_functions import EmbeddingFunction

    class PassthroughEF(EmbeddingFunction):
        """Dummy EF — we supply pre-computed embeddings directly."""
        def __call__(self, input):
            raise RuntimeError("PassthroughEF should never be called — pass embeddings directly")

    new_col = client.get_or_create_collection(
        name=NEW_COLLECTION,
        embedding_function=PassthroughEF(),
        metadata={"model": server_info["model"], "source": OLD_COLLECTION},
    )

    already_done = new_col.count()
    if already_done > 0:
        print(f"Resuming — {already_done:,} chunks already in new collection.")

    # ── Read all chunks ────────────────────────────────────────────────────────
    print("Reading all chunks from old collection...")
    all_ids, all_docs, all_metas = read_all_chunks(old_col)
    print(f"  {len(all_ids):,} chunks loaded into memory.\n")

    # Skip already-done chunks (for resuming interrupted runs)
    if already_done > 0:
        existing = set(new_col.get(limit=already_done, include=[])["ids"])
        skip = [(i, d, m) for i, d, m in zip(all_ids, all_docs, all_metas) if i in existing]
        todo = [(i, d, m) for i, d, m in zip(all_ids, all_docs, all_metas) if i not in existing]
        print(f"  Skipping {len(skip):,} already-embedded chunks.")
        print(f"  Embedding {len(todo):,} remaining chunks.\n")
        all_ids  = [x[0] for x in todo]
        all_docs = [x[1] for x in todo]
        all_metas= [x[2] for x in todo]

    if not all_ids:
        print("Nothing to do — all chunks already re-embedded.")
        return

    # ── Embed + write in batches ───────────────────────────────────────────────
    total   = len(all_ids)
    done    = 0
    t_start = time.perf_counter()

    print(f"Embedding {total:,} chunks via Mac Studio ({EMBED_SERVER_URL})...")
    print(f"Batch size: {EMBED_BATCH} for embedding, {WRITE_BATCH} for writes.\n")

    for write_start in range(0, total, WRITE_BATCH):
        write_end = min(write_start + WRITE_BATCH, total)

        batch_ids   = all_ids[write_start:write_end]
        batch_docs  = all_docs[write_start:write_end]
        batch_metas = all_metas[write_start:write_end]

        # Embed this write-batch (may be split further for the embed server)
        batch_vecs = embed_texts(batch_docs, batch_size=EMBED_BATCH)

        # Write to new collection with pre-computed embeddings
        new_col.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=batch_vecs,
        )

        done += len(batch_ids)
        elapsed  = time.perf_counter() - t_start
        rate     = done / elapsed
        eta      = (total - done) / rate if rate > 0 else 0

        print(f"  {done:>6,}/{total:,}  "
              f"({done/total*100:5.1f}%)  "
              f"{rate:6.0f} chunks/sec  "
              f"ETA: {eta/60:.1f} min")

    total_elapsed = time.perf_counter() - t_start
    print(f"\nDone! {total:,} chunks re-embedded in {total_elapsed:.0f}s "
          f"({total_elapsed/60:.1f} min)")
    print(f"New collection '{NEW_COLLECTION}': {new_col.count():,} chunks")
    print(f"\nNext step: update server.py to use collection '{NEW_COLLECTION}'")
    print(f"  Change: COLLECTION = \"{NEW_COLLECTION}\"")
    print(f"  And:    EMBED_SERVER_URL = \"{EMBED_SERVER_URL}\"")
    print(f"  Then:   systemctl --user restart rag-api")


if __name__ == "__main__":
    main()
