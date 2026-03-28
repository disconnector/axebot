#!/usr/bin/env python3
"""
Retrieval quality benchmark for axebot corpus.
Run before and after cleanup to compare chunk quality.
Saves results to a JSON file for diff comparison.
"""

import json
import sys
import time
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH    = Path.home() / "rag" / "db"
COLLECTION = "axefx3"
EMBED_MODEL = "/home/rich/rag/models/axefx-embed/final"
TOP_K = 8

# Representative benchmark queries — mix of factual, experiential, vague
QUERIES = [
    # Factual — should hit PDFs/wiki
    "What does the Input Trim parameter do?",
    "How do I configure scene switching on the FC-12?",
    "What is the difference between Triode Plate Frequency and Triode Plate Capacitance?",
    "What amp models are based on Marshall circuits?",
    "How does the Cab block work with IRs?",

    # Experiential — should hit forum/youtube
    "How do I get a Van Halen brown sound?",
    "My amp sounds fizzy, how do I fix it?",
    "Tips for dialing in a clean Fender tone",

    # Vague — most likely to surface noise chunks
    "Tell me about the Axe-Fx III",
    "What are the best amp models?",
    "How do I get good tone?",
    "Axe-Fx III settings",
]

NOISE_PATTERNS = [
    "login", "log in", "register", "create account",
    "navigation menu", "jump to navigation", "jump to search",
    "retrieved from", "last edited", "privacy policy",
    "terms of use", "edit this page", "view history",
    "this page was last", "permanent link", "page information",
    "what links here", "related changes", "special pages",
]

def is_noise(doc):
    if not doc:
        return True
    if len(doc.split()) < 20:
        return True
    doc_lower = doc.lower()
    return sum(1 for p in NOISE_PATTERNS if p in doc_lower) >= 2

def score_chunk(doc, meta):
    """Return a quality score for a retrieved chunk."""
    if is_noise(doc):
        return "NOISE"
    words = len(doc.split())
    source = meta.get('source_type', '?')
    filename = (meta.get('filename') or meta.get('model_name') or '')[:60]
    return f"OK ({source}, {words}w) — {filename}"

def run_benchmark(label):
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {label}")
    print(f"{'='*60}")

    client = chromadb.PersistentClient(path=str(DB_PATH))
    col    = client.get_collection(COLLECTION)
    model  = SentenceTransformer(EMBED_MODEL)

    results = {
        "label": label,
        "corpus_size": col.count(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "queries": []
    }

    total_chunks = 0
    noise_chunks = 0

    for query in QUERIES:
        embedding = model.encode([query])[0].tolist()
        r = col.query(query_embeddings=[embedding], n_results=TOP_K,
                      include=['documents', 'metadatas'])

        docs  = r['documents'][0]
        metas = r['metadatas'][0]

        query_result = {
            "query": query,
            "chunks": []
        }

        query_noise = 0
        for doc, meta in zip(docs, metas):
            score = score_chunk(doc, meta)
            is_n  = score == "NOISE"
            if is_n:
                query_noise += 1
                noise_chunks += 1
            total_chunks += 1

            query_result["chunks"].append({
                "source_type": meta.get('source_type'),
                "filename":    (meta.get('filename') or '')[:60],
                "words":       len((doc or '').split()),
                "noise":       is_n,
                "preview":     (doc or '')[:120],
            })

        noise_pct = round(query_noise / TOP_K * 100)
        print(f"\n[{noise_pct:3d}% noise] {query}")
        for c in query_result["chunks"]:
            marker = "🗑 " if c["noise"] else "✓ "
            print(f"  {marker}[{c['source_type']}] {c['words']}w — {c['preview'][:80]}")

        results["queries"].append(query_result)

    overall_noise_pct = round(noise_chunks / total_chunks * 100)
    results["summary"] = {
        "total_chunks_evaluated": total_chunks,
        "noise_chunks": noise_chunks,
        "noise_pct": overall_noise_pct,
    }

    print(f"\n{'='*60}")
    print(f"SUMMARY: {noise_chunks}/{total_chunks} chunks were noise ({overall_noise_pct}%)")
    print(f"Corpus size: {col.count()}")

    return results

if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "benchmark"
    out_path = Path.home() / "rag" / f"benchmark_{label}.json"
    results = run_benchmark(label)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")
