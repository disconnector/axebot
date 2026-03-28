#!/usr/bin/env python3
"""
ingest_search.py — Ingest all threads found in a Fractal Audio forum search results page.

Paginates through the search results, collects unique thread URLs, then ingests each one.
Reuses all core logic from ingest_thread.py.

Usage:
    python ingest_search.py "<search_url>"
    python ingest_search.py "<search_url>" --dry-run
    python ingest_search.py "<search_url>" --first-post-only

Examples:
    # All firmware release notes threads by FractalAudio
    python ingest_search.py "https://forum.fractalaudio.com/search/8134244/?q=Axe-Fx+III+Firmware+Release+Notes&t=post&c[child_nodes]=1&c[nodes][0]=100&c[users]=FractalAudio&o=date"
"""

import re
import sys
import time
import random
import argparse
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import requests
from bs4 import BeautifulSoup

from ingest_thread import (
    get_collection, already_ingested, login, ingest_thread,
    normalize_thread_url, FORUM_BASE, HEADERS,
)


def scrape_search_pages(session, search_url):
    """Paginate through search results and return all unique thread URLs."""
    thread_urls = []
    seen = set()
    page_url = search_url
    page_num = 0

    while page_url:
        page_num += 1
        print(f"  Search page {page_num}: {page_url[:100]}")
        time.sleep(random.uniform(5, 10))

        try:
            r = session.get(page_url, headers=HEADERS, timeout=30)
        except Exception as e:
            print(f"  Connection error on search page {page_num}: {e} — stopping")
            break

        if r.status_code != 200:
            print(f"  HTTP {r.status_code} on search page {page_num} — stopping")
            break

        soup = BeautifulSoup(r.text, "html.parser")

        # Extract thread links from search result titles
        new_this_page = 0
        for a in soup.select("h3.contentRow-title a, .contentRow-title a"):
            href = a.get("href", "")
            url = normalize_thread_url(href)
            if url and url not in seen:
                seen.add(url)
                thread_urls.append(url)
                new_this_page += 1

        print(f"    {new_this_page} new thread URLs (total: {len(thread_urls)})")

        if new_this_page == 0:
            print("  No new results on this page — stopping.")
            break

        # Next page
        next_btn = soup.select_one(".pageNav-jump--next")
        if next_btn:
            next_href = next_btn.get("href", "")
            page_url = next_href if next_href.startswith("http") else FORUM_BASE + next_href
        else:
            page_url = None

    return thread_urls


def main():
    parser = argparse.ArgumentParser(
        description="Ingest all threads from a forum search results page into AxeBot ChromaDB"
    )
    parser.add_argument("url", help="Full URL of the forum search results page")
    parser.add_argument("--first-post-only", action="store_true",
                        help="Only ingest the opening post of each thread")
    parser.add_argument("--dry-run", action="store_true",
                        help="Collect URLs and report counts, but don't write to the database")
    args = parser.parse_args()

    print(f"\nAxeBot Search Ingestor")
    print(f"  Search URL: {args.url[:100]}")
    print(f"  Mode:       {'first post only' if args.first_post_only else 'all posts'}")
    print(f"  {'DRY RUN — no writes' if args.dry_run else 'LIVE — will write to DB'}\n")

    col = None if args.dry_run else get_collection()

    session = requests.Session()
    print("Logging in...")
    login(session)
    print()

    # ── Collect all thread URLs from search results ──
    print("Scraping search results...")
    thread_urls = scrape_search_pages(session, args.url)

    if not thread_urls:
        print("No thread URLs found in search results.")
        sys.exit(0)

    print(f"\nFound {len(thread_urls)} unique threads.")

    # ── Check which are already ingested ──
    if not args.dry_run:
        already_done = [u for u in thread_urls if already_ingested(col, u)]
        to_ingest    = [u for u in thread_urls if not already_ingested(col, u)]
        print(f"  {len(already_done)} already in corpus — will skip")
        print(f"  {len(to_ingest)} new threads to ingest\n")
    else:
        to_ingest = thread_urls
        print(f"  Dry run — would attempt all {len(to_ingest)} threads\n")

    if not to_ingest:
        print("Nothing new to ingest.")
        sys.exit(0)

    # ── Ingest each thread ──
    total_ingested = 0
    for i, url in enumerate(to_ingest, 1):
        print(f"\n── [{i}/{len(to_ingest)}] {url}")
        time.sleep(random.uniform(5, 10))
        ingest_thread(
            session, col, url,
            first_post_only=args.first_post_only,
            dry_run=args.dry_run,
            collect_links=False,
            label=f"{i}/{len(to_ingest)}",
        )
        total_ingested += 1

    print(f"\n{'='*60}")
    print(f"Done. Processed {total_ingested} threads.")
    if not args.dry_run and col:
        print(f"Final corpus size: {col.count():,} total chunks.")


if __name__ == "__main__":
    main()
