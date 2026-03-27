#!/usr/bin/env python3
"""
Axe-Fx III RAG Ingestion Script
Ingests PDFs, forum threads, wiki HTML, and YouTube VTT transcripts into ChromaDB.
Each source type uses a chunking strategy matched to its structure.
Supports incremental updates — skips files already in the database.
"""

import os
import re
import sys
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import anthropic
import chromadb
from chromadb.utils import embedding_functions

# ── Configuration ─────────────────────────────────────────────────────────────
DB_PATH       = Path.home() / "rag" / "db"
CORPUS_BASE   = Path.home() / "fractal"
GUIDES_DIR    = CORPUS_BASE / "guides"
FORUM_DIR     = CORPUS_BASE / "forum" / "threads"
WIKI_DIR      = CORPUS_BASE / "wiki"
VTT_DIR       = CORPUS_BASE / "youtube-transcripts"

CHUNK_SIZE    = 512   # target size in words
CHUNK_OVERLAP = 64    # overlap in words
EMBED_MODEL   = "/home/rich/rag/models/axefx-embed/final"
COLLECTION    = "axefx3"
ENRICH_MODEL  = "claude-haiku-4-5-20251001"  # fast + cheap for PDF preprocessing

ENRICHMENT_PROMPT = """You are processing text extracted from an Axe-Fx III technical manual page for a RAG system.

The text may contain parameter lists, tables, or bare enumerations that lack context and embed poorly in a vector database.

Your job:
1. Identify parameter names, option lists, feature lists, and tables
2. For each one, insert a natural language sentence immediately after it that describes what it contains and what it means
3. Resolve ambiguous pronouns or orphaned references (e.g. "Set it to -18dB" -> "Set the Input Gain to -18dB")
4. Do NOT remove or alter any original text — only insert additions
5. Do NOT add headers, commentary, or any text outside the document itself

Return only the enriched text, nothing else.

TEXT:
{text}"""


def get_client():
    DB_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(DB_PATH))


def get_collection(client):
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_or_create_collection(name=COLLECTION, embedding_function=ef)


# ── Chunking strategies ────────────────────────────────────────────────────────

def split_sentences(text):
    """Split text into sentences at punctuation boundaries."""
    sentences = re.split(r'(?<=[.?!])\s+(?=[A-Z"\']|$)', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_fixed(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Fixed word-count chunking. Used as a last-resort fallback only."""
    ws = text.split()
    chunks = []
    i = 0
    while i < len(ws):
        chunk = " ".join(ws[i:i + size])
        if chunk.strip():
            chunks.append(chunk)
        i += size - overlap
    return chunks


def chunk_by_paragraph(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Paragraph-aware chunking for structured text (PDFs, forum posts).

    Splits on double newlines first, accumulates paragraphs up to `size` words,
    then carries an overlap window into the next chunk. Oversized single
    paragraphs fall back to sentence chunking so we never hard-cut mid-sentence.
    """
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]

    chunks = []
    current_words = []

    for para in paragraphs:
        para_words = para.split()

        # Paragraph alone exceeds limit — split it by sentence instead
        if len(para_words) > size:
            if current_words:
                chunks.append(" ".join(current_words))
                current_words = current_words[-overlap:]
            for sc in chunk_by_sentence(para, size, overlap):
                chunks.append(sc)
            last = chunks[-1].split() if chunks else []
            current_words = last[-overlap:] if last else []
            continue

        # Adding this paragraph would overflow — flush and start fresh
        if len(current_words) + len(para_words) > size and current_words:
            chunks.append(" ".join(current_words))
            current_words = current_words[-overlap:] + para_words
        else:
            current_words.extend(para_words)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def chunk_by_sentence(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Sentence-boundary chunking for unstructured text (transcripts, dense prose).

    Accumulates complete sentences up to `size` words, then carries an overlap
    window forward. Sentences that alone exceed `size` fall back to fixed chunking.
    """
    sentences = split_sentences(text)

    chunks = []
    current_words = []

    for sentence in sentences:
        sent_words = sentence.split()

        # Single sentence exceeds limit — hard-cut it
        if len(sent_words) > size:
            if current_words:
                chunks.append(" ".join(current_words))
                current_words = current_words[-overlap:]
            for fc in chunk_fixed(sentence, size, overlap):
                chunks.append(fc)
            last = chunks[-1].split() if chunks else []
            current_words = last[-overlap:] if last else []
            continue

        if len(current_words) + len(sent_words) > size and current_words:
            chunks.append(" ".join(current_words))
            current_words = current_words[-overlap:] + sent_words
        else:
            current_words.extend(sent_words)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def already_ingested(collection, filename):
    results = collection.get(where={"filename": filename}, limit=1)
    return len(results["ids"]) > 0


def make_id(filename, chunk_id):
    raw = f"{filename}::{chunk_id}"
    return hashlib.md5(raw.encode()).hexdigest()


# ── PDF ingestion ──────────────────────────────────────────────────────────────
def needs_enrichment(text):
    """
    Detect pages that are list-heavy and would embed poorly without enrichment.

    Heuristic: if more than half the lines are short (< 40 chars) and the
    average line length is under 35 chars, the page is likely parameter lists,
    tables, or bare enumerations rather than prose.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 4:
        return False
    avg_len = sum(len(l) for l in lines) / len(lines)
    short_ratio = sum(1 for l in lines if len(l) < 40) / len(lines)
    return short_ratio > 0.5 and avg_len < 35


def enrich_page(text, claude_client):
    """
    Send a list-heavy page to Haiku for semantic enrichment.
    Haiku adds natural language descriptions alongside bare parameter lists
    so the resulting chunks embed with full semantic richness.
    """
    try:
        response = claude_client.messages.create(
            model=ENRICH_MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": ENRICHMENT_PROMPT.format(text=text)}]
        )
        return response.content[0].text
    except Exception as e:
        print(f"    [ENRICH] WARNING: enrichment failed, using raw text: {e}")
        return text


def ingest_pdfs(collection):
    """
    Strategy: paragraph-based with sentence fallback.
    List-heavy pages are enriched via Haiku before chunking.

    Manuals have deliberate paragraph structure — each paragraph is one complete
    thought. Paragraph chunking respects that structure. Pages detected as
    list-heavy (parameter tables, option enumerations) are sent to Claude Haiku
    first, which inserts natural language descriptions alongside bare lists so
    they embed with full semantic richness. Page number is preserved in metadata
    for citation.
    """
    import PyPDF2
    claude_client = anthropic.Anthropic()
    added = 0
    files = list(GUIDES_DIR.glob("*.pdf")) + list(GUIDES_DIR.glob("**/*.pdf"))
    for pdf_path in files:
        fname = pdf_path.name
        if already_ingested(collection, fname):
            print(f"  [PDF] {fname}: already ingested, skipping")
            continue
        try:
            reader = PyPDF2.PdfReader(str(pdf_path))
            all_chunks = []
            enriched_pages = 0
            for page_num, page in enumerate(reader.pages):
                text = (page.extract_text() or "").strip()
                if not text:
                    continue
                if needs_enrichment(text):
                    text = enrich_page(text, claude_client)
                    enriched_pages += 1
                for chunk in chunk_by_paragraph(text):
                    all_chunks.append((chunk, page_num))

            docs, ids, metas = [], [], []
            for chunk_id, (chunk, page_num) in enumerate(all_chunks):
                docs.append(chunk)
                ids.append(make_id(fname, chunk_id))
                metas.append({
                    "source": str(pdf_path),
                    "source_type": "pdf",
                    "filename": fname,
                    "chunk_id": chunk_id,
                    "page": page_num,
                })
            if docs:
                collection.upsert(documents=docs, ids=ids, metadatas=metas)
                added += len(docs)
                print(f"  [PDF] {fname}: {len(docs)} chunks ({enriched_pages} pages enriched)")
        except Exception as e:
            print(f"  [PDF] ERROR {fname}: {e}")
    return added


# ── Forum thread ingestion ─────────────────────────────────────────────────────
def ingest_forum(collection):
    """
    Strategy: post-boundary first, then paragraph-based within each post.

    Each [POST N] is a self-contained conversational unit. Splitting by post
    marker first preserves that unit, then paragraph chunking within each post
    keeps explanations intact ("I tried X and it worked because Y").
    """
    added = 0
    files = list(FORUM_DIR.glob("*.txt")) + list(FORUM_DIR.glob("**/*.txt"))
    for txt_path in files:
        fname = txt_path.name
        if already_ingested(collection, fname):
            continue
        try:
            text = txt_path.read_text(errors="replace")
            posts = re.split(r'\[POST\s+\d+\]', text)
            posts = [p.strip() for p in posts if p.strip()]
            if not posts:
                posts = [text]

            docs, ids, metas = [], [], []
            chunk_id = 0
            for post_idx, post in enumerate(posts):
                for chunk in chunk_by_paragraph(post):
                    docs.append(chunk)
                    ids.append(make_id(fname, chunk_id))
                    metas.append({
                        "source": str(txt_path),
                        "source_type": "forum",
                        "filename": fname,
                        "chunk_id": chunk_id,
                        "post_index": post_idx,
                    })
                    chunk_id += 1
            if docs:
                collection.upsert(documents=docs, ids=ids, metadatas=metas)
                added += len(docs)
                print(f"  [FORUM] {fname}: {len(docs)} chunks")
        except Exception as e:
            print(f"  [FORUM] ERROR {fname}: {e}")
    return added


# ── Wiki HTML ingestion ────────────────────────────────────────────────────────
def ingest_wiki(collection):
    """
    Strategy: section-header boundary first, sentence-based within sections,
    section title prepended to every chunk.

    Prepending the title is the key improvement: a chunk reading "Set to 10kHz
    for a brighter tone" is nearly useless alone. "High Cut Filter: Set to 10kHz
    for a brighter tone" encodes both the concept and its location in the knowledge
    structure — dramatically improving retrieval relevance.
    """
    from bs4 import BeautifulSoup
    added = 0
    all_files = list(WIKI_DIR.rglob("*"))
    files = []
    for f in all_files:
        if not f.is_file():
            continue
        if f.suffix in (".html", ".htm"):
            files.append(f)
        elif f.suffix == "" or f.suffix not in (".png", ".jpg", ".jpeg", ".gif", ".css", ".js", ".tmp", ".txt"):
            try:
                peek = f.read_bytes()[:50].decode("utf-8", errors="replace").strip()
                if peek.startswith("<!") or peek.startswith("<html") or peek.startswith("<"):
                    files.append(f)
            except Exception:
                pass

    for html_path in files:
        fname = html_path.name
        rel_path = str(html_path.relative_to(WIKI_DIR))
        uid = rel_path
        if already_ingested(collection, uid):
            continue
        try:
            raw = html_path.read_text(errors="replace")
            soup = BeautifulSoup(raw, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            # Extract sections as (title, prose_parts, list_items) tuples
            # List items are tracked separately so we can generate synthetic prose
            # for list-heavy sections that would otherwise embed poorly.
            sections = []
            current_title = ""
            current_prose = []
            current_list = []

            for elem in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "pre"]):
                t = elem.get_text(separator=" ", strip=True)
                if not t:
                    continue
                if elem.name in ["h1", "h2", "h3"]:
                    if current_prose or current_list:
                        sections.append((current_title, current_prose[:], current_list[:]))
                    current_title = t
                    current_prose = []
                    current_list = []
                elif elem.name == "li":
                    current_list.append(t)
                else:
                    current_prose.append(t)

            if current_prose or current_list:
                sections.append((current_title, current_prose, current_list))
            if not sections:
                sections = [("", [soup.get_text(separator=" ", strip=True)], [])]

            docs, ids, metas = [], [], []
            chunk_id = 0
            for section_title, prose_parts, list_items in sections:
                prefix = f"{section_title}: " if section_title else ""

                # For list-heavy sections, generate a synthetic prose chunk.
                # A bare list of names embeds poorly — "Digital Mono Digital Stereo..."
                # has almost no semantic signal. A sentence like "The available chorus
                # types are: Digital Mono, Digital Stereo..." embeds richly and matches
                # natural-language queries like "what types of chorus are there?"
                if list_items and len(list_items) >= len(prose_parts):
                    synthetic = f"{prefix}The available options are: {', '.join(list_items)}."
                    docs.append(synthetic)
                    ids.append(make_id(uid, chunk_id))
                    metas.append({
                        "source": str(html_path),
                        "source_type": "wiki",
                        "filename": uid,
                        "chunk_id": chunk_id,
                        "section": section_title,
                    })
                    chunk_id += 1

                # Regular chunking of all content (prose + list items combined)
                all_text = " ".join(prose_parts + list_items)
                if not all_text.strip():
                    continue
                for chunk in chunk_by_sentence(all_text):
                    docs.append(prefix + chunk)
                    ids.append(make_id(uid, chunk_id))
                    metas.append({
                        "source": str(html_path),
                        "source_type": "wiki",
                        "filename": uid,
                        "chunk_id": chunk_id,
                        "section": section_title,
                    })
                    chunk_id += 1
            if docs:
                collection.upsert(documents=docs, ids=ids, metadatas=metas)
                added += len(docs)
                print(f"  [WIKI] {fname}: {len(docs)} chunks")
        except Exception as e:
            print(f"  [WIKI] ERROR {fname}: {e}")
    return added


# ── VTT transcript ingestion ───────────────────────────────────────────────────
def ingest_vtt(collection):
    """
    Strategy: sentence-boundary chunking with video title prefix.

    Transcripts are spoken language — no paragraphs, often minimal punctuation.
    Fixed-size cutting randomly bisects spoken thoughts. Sentence chunking keeps
    complete ideas together. The video title (derived from filename) is prepended
    so the embedding encodes the source context alongside the content.
    """
    added = 0
    files = list(VTT_DIR.glob("*.vtt")) + list(VTT_DIR.glob("**/*.vtt"))
    for vtt_path in files:
        fname = vtt_path.name
        if already_ingested(collection, fname):
            continue
        try:
            text_lines = []
            raw = vtt_path.read_text(errors="replace")
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("WEBVTT"):
                    continue
                if re.match(r'^\d+:\d+', line) or re.match(r'^\d{2}:\d{2}', line):
                    continue
                if '-->' in line:
                    continue
                if re.match(r'^\d+$', line):
                    continue
                line = re.sub(r'<[^>]+>', '', line)
                if line.strip():
                    text_lines.append(line.strip())

            text = " ".join(text_lines)

            # Derive readable title from filename, strip trailing YouTube ID
            title = vtt_path.stem.replace("-", " ").replace("_", " ")
            title = re.sub(r'\s+[A-Za-z0-9_-]{11}$', '', title).strip()
            prefix = f"[Video: {title}] " if title else ""

            docs, ids, metas = [], [], []
            for chunk_id, chunk in enumerate(chunk_by_sentence(text)):
                docs.append(prefix + chunk)
                ids.append(make_id(fname, chunk_id))
                metas.append({
                    "source": str(vtt_path),
                    "source_type": "youtube",
                    "filename": fname,
                    "chunk_id": chunk_id,
                    "video_title": title,
                })
            if docs:
                collection.upsert(documents=docs, ids=ids, metadatas=metas)
                added += len(docs)
                print(f"  [VTT] {fname}: {len(docs)} chunks")
        except Exception as e:
            print(f"  [VTT] ERROR {fname}: {e}")
    return added


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Ingest Axe-Fx III corpus into ChromaDB")
    parser.add_argument("--force", action="store_true", help="Re-ingest all files (clears existing DB)")
    args = parser.parse_args()

    start = datetime.now()
    print(f"[{start}] Starting ingestion...")
    print(f"  DB path: {DB_PATH}")

    client = get_client()
    collection = get_collection(client)

    if args.force:
        print("  --force: deleting existing collection...")
        client.delete_collection(COLLECTION)
        collection = get_collection(client)

    total = 0

    print("\n== PDFs ==")
    n = ingest_pdfs(collection)
    print(f"  Added {n} chunks")
    total += n

    print("\n== Forum threads ==")
    n = ingest_forum(collection)
    print(f"  Added {n} chunks")
    total += n

    print("\n== Wiki HTML ==")
    n = ingest_wiki(collection)
    print(f"  Added {n} chunks")
    total += n

    print("\n== YouTube VTTs ==")
    n = ingest_vtt(collection)
    print(f"  Added {n} chunks")
    total += n

    final_count = collection.count()
    elapsed = datetime.now() - start
    print(f"\nIngestion complete in {elapsed}")
    print(f"  New chunks this run: {total}")
    print(f"  Total docs in DB:    {final_count}")

    meta_path = Path.home() / "rag" / "db_meta.json"
    meta = {"last_updated": datetime.now().isoformat(), "doc_count": final_count}
    meta_path.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
