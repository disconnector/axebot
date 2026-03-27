#!/usr/bin/env python3
"""
RAG Demo — Training Edition
Shows every layer of the pipeline explicitly for educational purposes:
  1. Raw question
  2. Query type detection + source filter
  3. HyDE hypothetical answer (what gets embedded)
  4. Raw chunks returned from vector DB (before reranking)
  5. Reranker scores — watch the order change
  6. Prompt construction
  7. LLM generation

Usage: python demo.py "how do I make my amp sound less fizzy"
       python demo.py "what are the types of chorus available" --no-hyde --no-rerank
"""

import argparse
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
import anthropic

DB_PATH       = Path.home() / "rag" / "db"
EMBED_MODEL   = "/home/rich/rag/models/axefx-embed/final"
RERANK_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION    = "axefx3"
CLAUDE_MODEL  = "claude-sonnet-4-6"
HAIKU_MODEL   = "claude-haiku-4-5-20251001"

RERANK_FETCH_MULTIPLIER = 4

# ── Colors ─────────────────────────────────────────────────────────────────────
ORANGE  = "\033[38;5;208m"
TEAL    = "\033[38;5;45m"
GREEN   = "\033[38;5;82m"
YELLOW  = "\033[38;5;226m"
CYAN    = "\033[38;5;51m"
RED     = "\033[38;5;196m"
DIM     = "\033[2m"
BOLD    = "\033[1m"
RESET   = "\033[0m"
DIVIDER = f"{DIM}{'─' * 72}{RESET}"

def banner(title, color=ORANGE):
    width = 72
    bar   = "═" * width
    pad   = (width - len(title) - 2) // 2
    print(f"\n{color}{BOLD}╔{bar}╗")
    print(f"║{' ' * pad} {title} {' ' * (width - pad - len(title) - 1)}║")
    print(f"╚{bar}╝{RESET}\n")

def section(step, title, color=TEAL):
    print(f"\n{color}{BOLD}▶  STEP {step} — {title}{RESET}")
    print(DIVIDER)

# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Axe-Fx III technician and tone engineer with deep knowledge of:
- Fractal Audio hardware (Axe-Fx III, FC-6, FC-12, FX8, AX8)
- All amp models, cab IRs, and effect blocks
- MIDI configuration and external controller setup
- Expression pedal and FC footswitch programming
- Firmware features and release notes
- Signal routing, I/O configuration, and performance rigs

Answer questions accurately based on the provided context from manuals, forum discussions, wiki pages, and video transcripts.
If the context doesn't fully answer the question, say so — don't hallucinate Axe-Fx-specific details.
Be concise but thorough. Use technical terminology correctly. When making a specific claim, cite the source number in brackets, e.g. [3]."""

HYDE_PROMPT = """You are an Axe-Fx III expert. A user has asked the following question.

Write a concise hypothetical answer (3-5 sentences) using precise Axe-Fx III technical terminology — 
parameter names, block names, amp model names, menu paths — exactly as they appear in Fractal Audio 
documentation. Do not hedge or say "it depends." Write as if you are certain of the answer.

This hypothetical answer will be used as a search query against a technical corpus, not shown to the user.
Return only the hypothetical answer, nothing else.

Question: {question}"""

# ── Query type detection ───────────────────────────────────────────────────────

SOURCE_FILTERS = {
    "factual":      {"source_type": {"$in": ["pdf", "wiki"]}},
    "experiential": {"source_type": {"$in": ["forum", "youtube"]}},
    "general":      None,
}
TOP_K_MAP = {"factual": 6, "experiential": 12, "general": 8}

def detect_query_type(question):
    q = question.lower()
    factual_signals      = ["what is","what does","what are","how does","how do",
                            "define","explain","parameter","setting","value","range",
                            "what's the","what block","which block","what type"]
    experiential_signals = ["how do i","how can i","tips","recommend","best way",
                            "sound like","dial in","settings for","tone","preset",
                            "get the","achieve","improve","tweak","advice"]
    fs = sum(1 for w in factual_signals      if w in q)
    es = sum(1 for w in experiential_signals if w in q)
    if fs > es:   return "factual"
    elif es > fs: return "experiential"
    else:         return "general"

# ── Source colors ──────────────────────────────────────────────────────────────
SRC_COLOR = {"pdf": ORANGE, "wiki": TEAL, "forum": GREEN, "youtube": YELLOW}

def src_color(src_type):
    return SRC_COLOR.get(src_type, DIM)

# ── Main demo ──────────────────────────────────────────────────────────────────

def run_demo(question, use_hyde=True, use_rerank=True, pause=False):
    client = anthropic.Anthropic()

    def maybe_pause():
        if pause:
            input(f"\n{DIM}  [ press ENTER to continue ]{RESET}\n")

    banner("RAG PIPELINE DEMO — AXEBOT", ORANGE)

    # ── STEP 1: Question ───────────────────────────────────────────────────────
    section(1, "USER QUESTION", ORANGE)
    print(f"  {BOLD}{question}{RESET}")
    maybe_pause()

    # ── STEP 2: Query type ─────────────────────────────────────────────────────
    section(2, "QUERY TYPE DETECTION", TEAL)
    query_type  = detect_query_type(question)
    top_k       = TOP_K_MAP[query_type]
    src_filter  = SOURCE_FILTERS[query_type]
    filter_desc = str(src_filter) if src_filter else "all sources (no filter)"
    fetch_k     = top_k * RERANK_FETCH_MULTIPLIER if use_rerank else top_k

    print(f"  Query type   : {YELLOW}{BOLD}{query_type.upper()}{RESET}")
    print(f"  Source filter: {YELLOW}{filter_desc}{RESET}")
    print(f"  Final top-k  : {YELLOW}{top_k} chunks{RESET}")
    if use_rerank:
        print(f"  Fetch from DB: {YELLOW}{fetch_k} candidates{RESET}  {DIM}(reranker will cull to {top_k}){RESET}")
    print(f"\n  {DIM}factual → PDFs + wiki | experiential → forum + youtube | general → everything{RESET}")
    maybe_pause()

    # ── STEP 3: HyDE ──────────────────────────────────────────────────────────
    section(3, "HyDE QUERY EXPANSION (Hypothetical Document Embedding)", TEAL)
    if use_hyde:
        print(f"  {DIM}Asking Haiku to write a technical hypothetical answer...{RESET}\n")
        try:
            hyde_response = client.messages.create(
                model=HAIKU_MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": HYDE_PROMPT.format(question=question)}]
            )
            search_text = hyde_response.content[0].text.strip()
        except Exception as e:
            print(f"  {DIM}HyDE failed ({e}), using raw question{RESET}")
            search_text = question

        print(f"  {DIM}Original question:{RESET}")
        print(f"  {YELLOW}  \"{question}\"{RESET}\n")
        print(f"  {DIM}HyDE expansion (this vector gets sent to ChromaDB):{RESET}")
        print(f"  {GREEN}  \"{search_text}\"{RESET}")
        print(f"\n  {DIM}The hypothetical answer uses technical vocabulary that matches the corpus —")
        print(f"  so its vector lands much closer to real relevant chunks.{RESET}")
    else:
        search_text = question
        print(f"  {DIM}HyDE disabled — embedding raw question directly{RESET}")
        print(f"  {YELLOW}  \"{search_text}\"{RESET}")
    maybe_pause()

    # ── STEP 4: Vector search ──────────────────────────────────────────────────
    section(4, "VECTOR DATABASE SEARCH (bi-encoder, broad recall)", TEAL)
    print(f"  {DIM}Embedding search text → 384-dimensional vector...{RESET}")
    print(f"  {DIM}HNSW approximate nearest-neighbor search across 98k+ chunks...{RESET}\n")

    chroma_client = chromadb.PersistentClient(path=str(DB_PATH))
    ef            = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    collection    = chroma_client.get_collection(name=COLLECTION, embedding_function=ef)

    query_kwargs = dict(query_texts=[search_text], n_results=fetch_k)
    if src_filter:
        query_kwargs["where"] = src_filter

    results   = collection.query(**query_kwargs, include=["documents", "metadatas", "distances"])
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]
    raw_chunks = list(zip(docs, metas, distances))

    label = f"Retrieved {len(raw_chunks)} candidates" + (" (before reranking)" if use_rerank else "")
    print(f"  {BOLD}{GREEN}{label}:{RESET}\n")

    for i, (doc, meta, dist) in enumerate(raw_chunks, 1):
        src_type = meta.get("source_type", "?")
        filename = meta.get("filename", "?")
        section_ = meta.get("section", "")
        sc       = src_color(src_type)
        print(f"  {BOLD}[{i:2d}]{RESET} {sc}{src_type:8s}{RESET}  "
              f"{DIM}{filename[:45]:<45}{RESET}  dist={DIM}{dist:.4f}{RESET}")
        if section_:
            print(f"        {DIM}section: {section_[:65]}{RESET}")
        preview = doc.replace("\n", " ").strip()[:140]
        print(f"        {DIM}\"{preview}...\"{RESET}\n")

    maybe_pause()

    # ── STEP 5: Reranker ───────────────────────────────────────────────────────
    if use_rerank:
        section(5, "RERANKER (cross-encoder, precision filtering)", CYAN)
        print(f"  {DIM}Model: {RERANK_MODEL}{RESET}")
        print(f"  {DIM}Scoring {len(raw_chunks)} (question, chunk) pairs together...{RESET}")
        print(f"  {DIM}Unlike the bi-encoder, the cross-encoder reads both at once —")
        print(f"  every query token can attend to every chunk token.{RESET}\n")

        reranker = CrossEncoder(RERANK_MODEL)
        pairs    = [(question, doc) for doc, _, _ in raw_chunks]
        scores   = reranker.predict(pairs)

        # Build list of (score, doc, meta, original_rank)
        scored = sorted(
            zip(scores, docs, metas, range(1, len(raw_chunks)+1)),
            key=lambda x: x[0],
            reverse=True
        )

        print(f"  {BOLD}Reranked — top {top_k} kept, rest discarded:{RESET}\n")
        kept_chunks = []
        for new_rank, (score, doc, meta, old_rank) in enumerate(scored, 1):
            src_type = meta.get("source_type", "?")
            filename = meta.get("filename", "?")
            sc       = src_color(src_type)
            kept     = new_rank <= top_k

            rank_change = old_rank - new_rank
            if rank_change > 0:
                move = f"{GREEN}▲{rank_change:+d}{RESET}"
            elif rank_change < 0:
                move = f"{RED}▼{rank_change:+d}{RESET}"
            else:
                move = f"{DIM}  ={RESET}"

            status = f"{BOLD}{GREEN}KEEP{RESET}" if kept else f"{DIM}drop{RESET}"
            print(f"  [{new_rank:2d}] {status}  score={YELLOW}{score:+.4f}{RESET}  {move}  "
                  f"{sc}{src_type:8s}{RESET}  {DIM}{filename[:40]}{RESET}")
            preview = doc.replace("\n", " ").strip()[:100]
            print(f"        {DIM}\"{preview}...\"{RESET}\n")

            if kept:
                kept_chunks.append((doc, meta))

        print(f"\n  {DIM}The reranker score is not a similarity percentage —")
        print(f"  it's a relevance logit. Higher = more directly answers your question.{RESET}")
        maybe_pause()
    else:
        kept_chunks = [(doc, meta) for doc, meta, _ in raw_chunks[:top_k]]
        section(5, "RERANKER", DIM)
        print(f"  {DIM}Reranker disabled — using top {top_k} vector search results as-is{RESET}")
        maybe_pause()

    # ── STEP 6: Prompt construction ────────────────────────────────────────────
    section(6, "PROMPT CONSTRUCTION", TEAL)
    context_parts = []
    for i, (doc, meta) in enumerate(kept_chunks, 1):
        src_type = meta.get("source_type", "unknown")
        fname    = meta.get("filename", "unknown")
        context_parts.append(f"[{i}] ({src_type}: {fname})\n{doc}")
    context = "\n\n---\n\n".join(context_parts)

    full_prompt = f"""Here is relevant context from the Axe-Fx III documentation, forum, wiki, and video transcripts:

{context}

---

Question: {question}

Answer:"""

    print(f"  {DIM}System prompt: {len(SYSTEM_PROMPT)} chars{RESET}")
    print(f"  {YELLOW}  [{len(kept_chunks)} reranked chunks — {len(context)} chars of context]{RESET}")
    print(f"  {YELLOW}  [question: \"{question}\"]{RESET}")
    print(f"\n  {DIM}Total prompt: ~{(len(SYSTEM_PROMPT) + len(full_prompt)) // 4} tokens{RESET}")
    print(f"  {DIM}The model has never seen the Axe-Fx III manual —")
    print(f"  it only knows what's in those {len(kept_chunks)} chunks.{RESET}")
    maybe_pause()

    # ── STEP 7: LLM generation ─────────────────────────────────────────────────
    section(7, f"LLM GENERATION ({CLAUDE_MODEL})", ORANGE)
    print(f"  {DIM}Streaming response...{RESET}\n")
    print(f"  {BOLD}", end="", flush=True)

    answer_parts = []
    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": full_prompt}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            answer_parts.append(text)
    answer = "".join(answer_parts)
    print(f"{RESET}")

    # ── Summary ────────────────────────────────────────────────────────────────
    banner("PIPELINE COMPLETE", GREEN)
    print(f"  Question    : {YELLOW}{question}{RESET}")
    print(f"  Query type  : {YELLOW}{query_type}{RESET}")
    print(f"  HyDE        : {YELLOW}{'enabled' if use_hyde else 'disabled'}{RESET}")
    print(f"  Reranker    : {YELLOW}{'enabled — fetched ' + str(fetch_k) + ', kept ' + str(top_k) if use_rerank else 'disabled'}{RESET}")
    print(f"  Chunks used : {YELLOW}{len(kept_chunks)}{RESET}")
    print(f"  Sources     :", end="")
    seen   = set()
    labels = []
    for _, meta in kept_chunks:
        key = meta.get("source_type", "?")
        if key not in seen:
            labels.append(key);  seen.add(key)
    print(f" {YELLOW}{', '.join(labels)}{RESET}")
    print(f"  Model       : {YELLOW}{CLAUDE_MODEL}{RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="RAG demo with full pipeline visibility")
    parser.add_argument("question",          nargs="+",          help="Question to ask")
    parser.add_argument("--no-hyde",         action="store_true", help="Disable HyDE expansion")
    parser.add_argument("--no-rerank",       action="store_true", help="Disable reranker")
    parser.add_argument("--pause",           action="store_true", help="Pause between steps (presenter mode)")
    args = parser.parse_args()

    run_demo(
        question=" ".join(args.question),
        use_hyde=not args.no_hyde,
        use_rerank=not args.no_rerank,
        pause=args.pause,
    )

if __name__ == "__main__":
    main()
