#!/usr/bin/env python3
"""
Re-ingest Yek's Amp Model Guide and Drive Model Guide with model-aware chunking.
Uses PyMuPDF for clean text extraction, groups pages by amp/drive model entry,
one chunk per complete model entry.
"""

import sys
from pathlib import Path
import fitz  # pymupdf
import chromadb

GUIDES_DIR  = Path.home() / "fractal" / "guides"
DB_PATH     = Path.home() / "rag" / "db"
COLLECTION  = "axefx3"

AMP_PDF   = GUIDES_DIR / "Yeks-Guide-to-Fractal-Audio-Amp-Models.pdf"
DRIVE_PDF = GUIDES_DIR / "Yeks-Guide-to-Fractal-Audio-Drive-Models.pdf"

# Pages to skip at the start of each guide (TOC, intro, disclaimers)
AMP_SKIP_PAGES   = 11   # amp entries start at page 11
DRIVE_SKIP_PAGES = 7    # drive entries start at page 7 (intro pages 0-6 are useful prose, keep from 1)

MAX_CHUNK_WORDS = 1200  # split oversized entries (very long ones)

def make_id(filename, chunk_id):
    import hashlib
    raw = f"{filename}::{chunk_id}"
    return hashlib.md5(raw.encode()).hexdigest()

def chunk_by_model(pdf_path, skip_pages):
    """
    Extract text with PyMuPDF, group pages by model entry (first line = model name).
    Returns list of (model_name, text, first_page_num).
    """
    doc = fitz.open(str(pdf_path))
    entries = []
    current_model = None
    current_pages = []
    current_first_page = skip_pages

    for pg_num in range(skip_pages, len(doc)):
        page_text = doc[pg_num].get_text().strip()
        if not page_text:
            continue

        lines = [l.strip() for l in page_text.split('\n') if l.strip()]
        first_line = lines[0] if lines else ""

        if first_line != current_model:
            # New model entry — save previous
            if current_model and current_pages:
                entries.append((current_model, "\n\n".join(current_pages), current_first_page))
            current_model = first_line
            current_pages = [page_text]
            current_first_page = pg_num
        else:
            current_pages.append(page_text)

    # Save last entry
    if current_model and current_pages:
        entries.append((current_model, "\n\n".join(current_pages), current_first_page))

    return entries

def split_if_large(text, max_words=MAX_CHUNK_WORDS):
    """Split oversized entries at paragraph boundaries."""
    words = text.split()
    if len(words) <= max_words:
        return [text]

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current = []
    current_words = 0

    for para in paragraphs:
        pw = len(para.split())
        if current_words + pw > max_words and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_words = pw
        else:
            current.append(para)
            current_words += pw

    if current:
        chunks.append("\n\n".join(current))

    return chunks if chunks else [text]

def reingest_guide(collection, pdf_path, skip_pages, source_type="pdf"):
    fname = pdf_path.name
    print(f"\n{'='*60}")
    print(f"Processing: {fname}")

    # Delete existing chunks for this file (paginate to avoid SQLite variable limit)
    ids_to_delete = []
    offset = 0
    batch = 1000
    while True:
        existing = collection.get(include=['metadatas'], limit=batch, offset=offset)
        if not existing['ids']:
            break
        for eid, meta in zip(existing['ids'], existing['metadatas']):
            if meta.get('filename') == fname:
                ids_to_delete.append(eid)
        offset += batch
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        print(f"  Deleted {len(ids_to_delete)} old chunks")

    entries = chunk_by_model(pdf_path, skip_pages)
    print(f"  Found {len(entries)} model entries")

    docs, ids, metas = [], [], []
    chunk_id = 0

    for model_name, text, first_page in entries:
        word_count = len(text.split())
        sub_chunks = split_if_large(text)

        for i, sub in enumerate(sub_chunks):
            suffix = f" (part {i+1})" if len(sub_chunks) > 1 else ""
            docs.append(sub)
            ids.append(make_id(fname, chunk_id))
            metas.append({
                "source":      str(pdf_path),
                "source_type": source_type,
                "filename":    fname,
                "chunk_id":    chunk_id,
                "page":        first_page,
                "model_name":  model_name,
            })
            chunk_id += 1

        parts_note = f" → {len(sub_chunks)} chunks" if len(sub_chunks) > 1 else ""
        print(f"  [{first_page:3d}] {model_name[:60]:<60} {word_count:4d}w{parts_note}")

    if docs:
        # Upsert in batches of 100
        for i in range(0, len(docs), 100):
            collection.upsert(
                documents=docs[i:i+100],
                ids=ids[i:i+100],
                metadatas=metas[i:i+100]
            )
        print(f"\n  ✓ Inserted {len(docs)} chunks from {len(entries)} model entries")

    return len(docs)

def main():
    client     = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection(COLLECTION)

    print(f"Collection: {COLLECTION} ({collection.count()} chunks before)")

    total = 0
    total += reingest_guide(collection, AMP_PDF,   skip_pages=AMP_SKIP_PAGES)
    total += reingest_guide(collection, DRIVE_PDF, skip_pages=DRIVE_SKIP_PAGES)

    print(f"\n{'='*60}")
    print(f"Done. {total} total chunks inserted.")
    print(f"Collection now has {collection.count()} chunks.")

if __name__ == "__main__":
    main()
