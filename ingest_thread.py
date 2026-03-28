#!/usr/bin/env python3
"""
ingest_thread.py — Ingest a single Fractal Audio forum thread into ChromaDB.

Usage:
    python ingest_thread.py <thread_url>
    python ingest_thread.py <thread_url> --first-post-only
    python ingest_thread.py <thread_url> --recurse
    python ingest_thread.py <thread_url> --dry-run

Examples:
    python ingest_thread.py https://forum.fractalaudio.com/threads/everything-youve-always-wanted-to-know-about-resetting-an-amp.173317/
    python ingest_thread.py https://forum.fractalaudio.com/threads/if-you-read-nothing-else-read-this.162537/ --first-post-only
    python ingest_thread.py https://forum.fractalaudio.com/threads/axe-fx-iii-official-read-me.170466/ --first-post-only --recurse
"""

import os
import re
import sys
import time
import random
import hashlib
import argparse
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import requests
import chromadb
from bs4 import BeautifulSoup
from chromadb.utils import embedding_functions

# ── Config — must match ingest.py ─────────────────────────────────────────────
DB_PATH    = Path.home() / "rag" / "db"
EMBED_MODEL = "/home/rich/rag/models/axefx-embed/final"
COLLECTION  = "axefx3"
CHUNK_SIZE  = 512
CHUNK_OVERLAP = 64

# Remote embed server (Mac Studio M1 Max — set EMBED_SERVER_URL to enable)
EMBED_SERVER_URL = os.environ.get("EMBED_SERVER_URL", "")

FORUM_BASE = "https://forum.fractalaudio.com"
EMAIL      = os.environ.get("FORUM_EMAIL", "")
PASSWORD   = os.environ.get("FORUM_PASSWORD", "")
HEADERS    = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
MIN_POST_WORDS = 30

# ── DB helpers ─────────────────────────────────────────────────────────────────
def get_collection():
    DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_PATH))
    if EMBED_SERVER_URL:
        from remote_embedder import RemoteEmbeddingFunction
        ef = RemoteEmbeddingFunction(server_url=EMBED_SERVER_URL)
    else:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_or_create_collection(name=COLLECTION, embedding_function=ef)

def make_id(thread_id, post_index, chunk_id):
    raw = f"thread::{thread_id}::p{post_index}::c{chunk_id}"
    return hashlib.md5(raw.encode()).hexdigest()

def already_ingested(col, thread_url):
    results = col.get(where={"thread_url": thread_url}, limit=1)
    return len(results["ids"]) > 0

# ── Chunking (matches ingest.py) ───────────────────────────────────────────────
def chunk_by_paragraph(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    chunks, current_words = [], []
    for para in paragraphs:
        para_words = para.split()
        if len(para_words) > size:
            if current_words:
                chunks.append(" ".join(current_words))
                current_words = current_words[-overlap:]
            sentences = re.split(r'(?<=[.?!])\s+(?=[A-Z"\'])', para)
            buf = []
            for sent in sentences:
                sw = sent.split()
                if len(buf) + len(sw) > size and buf:
                    chunks.append(" ".join(buf))
                    buf = buf[-overlap:]
                buf.extend(sw)
            if buf:
                chunks.append(" ".join(buf))
            current_words = chunks[-1].split()[-overlap:] if chunks else []
            continue
        if len(current_words) + len(para_words) > size and current_words:
            chunks.append(" ".join(current_words))
            current_words = current_words[-overlap:]
        current_words.extend(para_words)
    if current_words:
        chunks.append(" ".join(current_words))
    return [c for c in chunks if len(c.split()) >= 20]

# ── Forum scraping ─────────────────────────────────────────────────────────────
def login(session):
    for attempt in range(3):
        try:
            r = session.get(f"{FORUM_BASE}/login/", headers=HEADERS, timeout=30)
            soup = BeautifulSoup(r.text, "html.parser")
            token = soup.find("input", {"name": "_xfToken"})
            if not token:
                print("ERROR: no CSRF token — login failed")
                sys.exit(1)
            time.sleep(random.uniform(3, 6))
            r = session.post(f"{FORUM_BASE}/login/login", headers=HEADERS, data={
                "login": EMAIL, "password": PASSWORD,
                "_xfToken": token["value"], "remember": "1", "_xfRedirect": "/",
            }, allow_redirects=True, timeout=30)
            if "account/" not in r.text:
                print("ERROR: login failed — check credentials")
                sys.exit(1)
            print("  Logged in.")
            return
        except Exception as e:
            if attempt < 2:
                print(f"  Login connection error ({e.__class__.__name__}), retrying in 20s...")
                time.sleep(20)
            else:
                print(f"  Login failed after 3 attempts: {e}")
                sys.exit(1)

def get_thread_id(url):
    """Extract numeric thread ID from URL."""
    m = re.search(r'\.(\d+)/?$', url.rstrip('/'))
    return m.group(1) if m else hashlib.md5(url.encode()).hexdigest()[:12]

def normalize_thread_url(href):
    """Normalize a forum href to a canonical thread base URL, or None if not a thread link."""
    if not href or "/threads/" not in href:
        return None
    full = href if href.startswith("http") else FORUM_BASE + href
    m = re.match(r'(https://forum\.fractalaudio\.com/threads/[^/?#]+\.\d+)', full)
    return m.group(1) + "/" if m else None

def clean_post(article):
    """Extract clean text from a post article element."""
    body = article.select_one(".bbWrapper")
    if not body:
        return None, None

    author = article.get("data-author", "unknown")

    # Remove quoted blocks and spoilers
    for tag in body.select("blockquote, .bbCodeSpoiler"):
        tag.decompose()

    text = body.get_text(separator="\n")
    text = re.sub(r'Click to expand\.\.\.', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    if len(text.split()) < MIN_POST_WORDS:
        return None, None

    return author, text

def extract_thread_links(articles):
    """Extract all forum thread URLs linked within post bodies (before cleaning)."""
    links = set()
    for article in articles:
        body = article.select_one(".bbWrapper")
        if not body:
            continue
        for a in body.find_all("a", href=True):
            url = normalize_thread_url(a["href"])
            if url:
                links.add(url)
    return links

def fetch_thread(session, url, first_post_only=False, collect_links=False):
    """Fetch all posts from a thread, paginating through all pages.

    Returns (thread_title, posts) or (thread_title, posts, linked_urls) if collect_links=True.
    """
    posts = []   # list of (author, text)
    linked_urls = set()
    thread_title = ""
    page_url = url
    page_num = 0

    while page_url:
        page_num += 1
        time.sleep(random.uniform(3, 6))
        # Retry once on connection drop (forum occasionally closes connections)
        for attempt in range(2):
            try:
                r = session.get(page_url, headers=HEADERS, timeout=30, allow_redirects=True)
                break
            except Exception as e:
                if attempt == 0:
                    print(f"  Connection error on page {page_num}, retrying in 15s... ({e.__class__.__name__})")
                    time.sleep(15)
                else:
                    print(f"  Failed after retry — stopping at page {page_num}")
                    page_url = None
                    break
        if page_url is None:
            break
        if r.status_code != 200:
            print(f"  WARNING: HTTP {r.status_code} on page {page_num}")
            break

        soup = BeautifulSoup(r.text, "html.parser")

        if not thread_title:
            t = soup.select_one("h1.p-title-value")
            thread_title = t.get_text().strip() if t else "Unknown Thread"

        articles = soup.select("article.message")

        if collect_links:
            linked_urls.update(extract_thread_links(articles))

        for article in articles:
            author, text = clean_post(article)
            if author and text:
                posts.append((author, text))

        if first_post_only:
            break

        # Pagination
        next_btn = soup.select_one(".pageNav-jump--next")
        if next_btn:
            next_href = next_btn.get("href", "")
            page_url = next_href if next_href.startswith("http") else FORUM_BASE + next_href
        else:
            page_url = None

        print(f"  Page {page_num}: {len(articles)} posts found ({len(posts)} total kept)")

    if collect_links:
        return thread_title, posts, linked_urls
    return thread_title, posts

# ── Ingest one thread (reusable) ───────────────────────────────────────────────
def ingest_thread(session, col, url, first_post_only=False, dry_run=False, collect_links=False, label=""):
    """Fetch, chunk, and upsert one thread. Returns set of linked thread URLs (if collect_links)."""
    prefix = f"[{label}] " if label else ""

    if not dry_run and already_ingested(col, url):
        print(f"{prefix}Already ingested — skipping.")
        return set()

    print(f"{prefix}Fetching thread...")
    result = fetch_thread(session, url, first_post_only=first_post_only, collect_links=collect_links)
    if collect_links:
        thread_title, posts, linked_urls = result
    else:
        thread_title, posts = result
        linked_urls = set()

    print(f"{prefix}Thread: {thread_title}")
    print(f"{prefix}Posts collected: {len(posts)}, words: {sum(len(t.split()) for _, t in posts):,}")

    if not posts:
        print(f"{prefix}No qualifying posts — skipping.")
        return linked_urls

    thread_id = get_thread_id(url)
    docs, ids, metas = [], [], []
    chunk_counter = 0
    for post_idx, (author, text) in enumerate(posts):
        for chunk in chunk_by_paragraph(text):
            docs.append(chunk)
            ids.append(make_id(thread_id, post_idx, chunk_counter))
            metas.append({
                "source":       url,
                "source_type":  "forum",
                "thread_url":   url,
                "thread_title": thread_title,
                "thread_id":    thread_id,
                "post_author":  author,
                "post_index":   post_idx,
                "chunk_id":     chunk_counter,
                "filename":     f"thread_{thread_id}.txt",
            })
            chunk_counter += 1

    print(f"{prefix}Chunks generated: {len(docs)}")

    if dry_run:
        print(f"{prefix}Dry run — sample chunks:")
        for i, (doc, meta) in enumerate(zip(docs[:3], metas[:3])):
            print(f"  [{i+1}] author={meta['post_author']} post={meta['post_index']}")
            print(f"  {doc[:300]}")
        print(f"  ... {len(docs)} total chunks.")
        return linked_urls

    batch_size = 50
    for i in range(0, len(docs), batch_size):
        col.upsert(
            documents=docs[i:i+batch_size],
            ids=ids[i:i+batch_size],
            metadatas=metas[i:i+batch_size],
        )
    print(f"{prefix}Done. '{thread_title}' → {len(docs)} chunks. Corpus: {col.count():,} total.")
    return linked_urls

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Ingest a forum thread into AxeBot ChromaDB")
    parser.add_argument("url", help="Full URL of the forum thread")
    parser.add_argument("--first-post-only", action="store_true",
                        help="Only ingest the opening post (good for official reference threads)")
    parser.add_argument("--recurse", action="store_true",
                        help="Also ingest forum threads linked within the posts (one level deep)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Scrape and chunk but don't write to the database")
    args = parser.parse_args()

    url = args.url.rstrip("/") + "/"

    print(f"\nAxeBot Thread Ingestor")
    print(f"  URL:    {url}")
    print(f"  Mode:   {'first post only' if args.first_post_only else 'all posts'}" +
          (" + recurse" if args.recurse else ""))
    print(f"  {'DRY RUN — no writes' if args.dry_run else 'LIVE — will write to DB'}\n")

    col = None if args.dry_run else get_collection()

    session = requests.Session()
    print("Logging in...")
    login(session)
    print()

    # ── Primary thread ──
    linked_urls = ingest_thread(
        session, col, url,
        first_post_only=args.first_post_only,
        dry_run=args.dry_run,
        collect_links=args.recurse,
        label="PRIMARY",
    )

    # ── Linked threads (one level deep) ──
    if args.recurse and linked_urls:
        # Remove the primary URL itself and any already-ingested ones
        linked_urls.discard(url)
        print(f"\nFound {len(linked_urls)} linked thread(s) to check.")

        ingested_count = 0
        skipped_count  = 0
        for i, linked_url in enumerate(sorted(linked_urls), 1):
            print(f"\n── Linked {i}/{len(linked_urls)}: {linked_url}")
            time.sleep(random.uniform(5, 10))
            linked_urls_inner = ingest_thread(
                session, col, linked_url,
                first_post_only=False,
                dry_run=args.dry_run,
                collect_links=False,   # no further recursion
                label=f"LINKED {i}/{len(linked_urls)}",
            )
            ingested_count += 1

        print(f"\nRecurse complete: checked {len(linked_urls)} linked threads.")

    if not args.dry_run and col:
        print(f"\nFinal corpus size: {col.count():,} total chunks.")

if __name__ == "__main__":
    main()
