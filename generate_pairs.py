#!/usr/bin/env python3
"""
Training pair generator for fine-tuning the axebot embedding model.

Generates (query, passage) pairs from the corpus two ways:
  1. Synthetic queries — Haiku reads each chunk and writes questions a user might ask
  2. Forum mining — consecutive [POST N] pairs are natural Q&A

Output: ~/rag/training_pairs.jsonl
Format: {"anchor": "user question", "positive": "relevant passage"}
"""

import re
import json
import time
import random
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import chromadb
import anthropic

DB_PATH     = Path.home() / "rag" / "db"
CORPUS_BASE = Path.home() / "fractal"
FORUM_DIR   = CORPUS_BASE / "forum" / "threads"
OUTPUT      = Path.home() / "rag" / "training_pairs.jsonl"
HAIKU_MODEL = "claude-haiku-4-5-20251001"
WORKERS     = 15  # concurrent Haiku requests

QUERY_GEN_PROMPT = """You are helping build a training dataset for a search system about the Axe-Fx III guitar processor.

Read the passage below and write {n} short questions that a guitarist or audio engineer might type when searching for this information. 

Rules:
- Write natural, conversational questions (how a real user would phrase them)
- Include some informal phrasing ("how do I make it less fizzy") alongside technical phrasing ("High Cut parameter range")
- Each question should be answerable using ONLY the information in this passage
- One question per line, no numbering or bullets
- Do not repeat the same question with different words

Passage:
{passage}

Questions:"""


def get_collection():
    client = chromadb.PersistentClient(path=str(DB_PATH))
    from chromadb.utils import embedding_functions
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_collection(name="axefx3", embedding_function=ef)


def sample_chunks(collection, source_types, n, min_words=40):
    results = collection.get(
        where={"source_type": {"$in": source_types}},
        limit=min(n * 4, 50000),
        include=["documents", "metadatas"]
    )
    pairs = list(zip(results["documents"], results["metadatas"]))
    pairs = [(doc, meta) for doc, meta in pairs if len(doc.split()) >= min_words]
    random.shuffle(pairs)
    return pairs[:n]


def generate_queries_for_chunk(passage, n_queries, client):
    """Ask Haiku to write n_queries questions for this passage. Thread-safe."""
    try:
        response = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": QUERY_GEN_PROMPT.format(n=n_queries, passage=passage[:1500])
            }]
        )
        text = response.content[0].text.strip()
        questions = [q.strip() for q in text.splitlines() if q.strip()]
        return questions[:n_queries]
    except Exception as e:
        print(f"  WARNING: query gen failed: {e}")
        return []


def mine_forum_pairs(max_pairs=500):
    pairs = []
    files = list(FORUM_DIR.glob("*.txt")) + list(FORUM_DIR.glob("**/*.txt"))
    random.shuffle(files)

    for fpath in files:
        if len(pairs) >= max_pairs:
            break
        try:
            text = fpath.read_text(errors="replace")
            posts = re.split(r'\[POST\s+\d+\]', text)
            posts = [p.strip() for p in posts if len(p.strip().split()) >= 20]
            if len(posts) < 2:
                continue
            anchor = posts[0][:800]
            for reply in posts[1:4]:
                if len(reply.split()) >= 20:
                    pairs.append({"anchor": anchor, "positive": reply[:800]})
        except Exception:
            continue

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate training pairs for embedding fine-tuning")
    parser.add_argument("--wiki-samples",  type=int, default=800,  help="Wiki chunks to generate queries for")
    parser.add_argument("--pdf-samples",   type=int, default=600,  help="PDF chunks to generate queries for")
    parser.add_argument("--forum-pairs",   type=int, default=500,  help="Forum Q&A pairs to mine")
    parser.add_argument("--queries-each",  type=int, default=3,    help="Queries to generate per chunk")
    parser.add_argument("--workers",       type=int, default=WORKERS, help="Concurrent Haiku requests")
    parser.add_argument("--dry-run",       action="store_true",    help="Show stats without calling API")
    args = parser.parse_args()

    print("Loading collection...")
    collection = get_collection()
    # Each thread gets its own client instance (not thread-safe to share)
    all_pairs  = []

    # ── Forum mining (free, no API calls) ────────────────────────────────────
    print(f"\nMining forum threads (target: {args.forum_pairs} pairs)...")
    forum_pairs = mine_forum_pairs(max_pairs=args.forum_pairs)
    all_pairs.extend(forum_pairs)
    print(f"  Mined {len(forum_pairs)} forum pairs")

    # ── Sample chunks ─────────────────────────────────────────────────────────
    print(f"\nSampling {args.wiki_samples} wiki chunks...")
    wiki_chunks = sample_chunks(collection, ["wiki"], args.wiki_samples)
    print(f"  Got {len(wiki_chunks)} chunks after filtering")

    print(f"\nSampling {args.pdf_samples} PDF chunks...")
    pdf_chunks = sample_chunks(collection, ["pdf"], args.pdf_samples)
    print(f"  Got {len(pdf_chunks)} chunks after filtering")

    all_chunks = wiki_chunks + pdf_chunks
    total_api_calls = len(all_chunks)
    est_cost = total_api_calls * ((200 * 0.80 + 100 * 4.00) / 1_000_000)

    print(f"\nPlan:")
    print(f"  Forum pairs:        {len(forum_pairs)}")
    print(f"  Chunks to process:  {len(all_chunks)}")
    print(f"  Queries per chunk:  {args.queries_each}")
    print(f"  Expected pairs:     ~{len(forum_pairs) + len(all_chunks) * args.queries_each}")
    print(f"  Estimated cost:     ${est_cost:.2f}")
    print(f"  Concurrent workers: {args.workers}")

    if args.dry_run:
        print("\n--dry-run: stopping here")
        return

    # ── Generate synthetic queries (concurrent) ───────────────────────────────
    print(f"\nGenerating queries via Haiku ({args.workers} workers)...")

    synthetic_pairs = []
    lock = threading.Lock()
    completed = 0
    start_time = time.time()

    def process_chunk(item):
        nonlocal completed
        doc, meta = item
        # Each thread creates its own client
        client = anthropic.Anthropic()
        questions = generate_queries_for_chunk(doc, args.queries_each, client)
        pairs = [{"anchor": q, "positive": doc} for q in questions]

        with lock:
            synthetic_pairs.extend(pairs)
            completed += 1
            if completed % 50 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                remaining = (total_api_calls - completed) / rate if rate > 0 else 0
                print(f"  {completed}/{total_api_calls} chunks — "
                      f"{len(synthetic_pairs)} queries — "
                      f"{rate:.1f}/s — "
                      f"~{remaining:.0f}s remaining")

        return pairs

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_chunk, item) for item in all_chunks]
        # Wait for all to complete (exceptions surface here)
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"  ERROR in worker: {e}")

    all_pairs.extend(synthetic_pairs)

    elapsed = time.time() - start_time
    print(f"\nGeneration complete in {elapsed:.0f}s")

    # ── Write output ──────────────────────────────────────────────────────────
    random.shuffle(all_pairs)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\nDone!")
    print(f"  Total pairs written: {len(all_pairs)}")
    print(f"  Output: {OUTPUT}")
    print(f"\nNext step: python train_embeddings.py")


if __name__ == "__main__":
    main()
