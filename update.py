#!/usr/bin/env python3
"""
Axe-Fx III RAG Incremental Update Script
Run weekly to pull new content and ingest into ChromaDB.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

import chromadb
from chromadb.utils import embedding_functions

DB_PATH    = Path.home() / "rag" / "db"
META_PATH  = Path.home() / "rag" / "db_meta.json"
CORPUS     = Path.home() / "fractal"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION  = "axefx3"

# YouTube channels to check for new videos
YT_CHANNELS = [
    "https://www.youtube.com/@LeonTodd",
    "https://www.youtube.com/@CooperCarter",
    "https://www.youtube.com/@AxeFxTutorials",
    "https://www.youtube.com/@FractalAudioSystems",
]

YT_OUTDIR = CORPUS / "youtube-transcripts"

def run(cmd, **kwargs):
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kwargs)
    if result.returncode != 0 and result.stderr:
        print(f"    STDERR: {result.stderr[:200]}")
    return result

def update_youtube():
    """Download only new VTT transcripts since last run."""
    print("\n== Updating YouTube transcripts ==")
    YT_OUTDIR.mkdir(parents=True, exist_ok=True)
    new_files = 0
    before = set(YT_OUTDIR.glob("*.vtt"))
    for channel in YT_CHANNELS:
        print(f"  Checking {channel}...")
        cmd = (
            f"yt-dlp --skip-download --write-auto-sub --sub-format vtt "
            f"--sub-lang en --no-progress "
            f"-o '{YT_OUTDIR}/%(uploader)s_%(id)s.%(ext)s' "
            f"--dateafter now-7days "
            f"'{channel}'"
        )
        run(cmd)
    after = set(YT_OUTDIR.glob("*.vtt"))
    new_files = len(after - before)
    print(f"  New VTT files: {new_files}")
    return new_files

def update_forum():
    """Check for new forum thread files."""
    print("\n== Checking forum threads ==")
    forum_dir = CORPUS / "forum" / "threads"
    if not forum_dir.exists():
        print("  Forum directory not found, skipping")
        return 0
    files = list(forum_dir.glob("*.txt"))
    print(f"  {len(files)} total thread files present")
    return 0  # ingest.py will handle new-only ingestion

def check_firmware():
    """Check Fractal Audio firmware release page for new notes."""
    print("\n== Checking firmware release notes ==")
    try:
        import urllib.request
        url = "https://www.fractalaudio.com/axe-fx-iii-downloads/"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        html = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", errors="replace")
        # Look for firmware version mentions
        import re
        versions = re.findall(r'Firmware\s+[\d.]+', html, re.IGNORECASE)
        if versions:
            print(f"  Found firmware mentions: {', '.join(set(versions[:5]))}")
        else:
            print("  Could not parse firmware versions")
    except Exception as e:
        print(f"  Could not fetch firmware page: {e}")

def run_ingest():
    """Run the ingest script to pick up new files."""
    print("\n== Running incremental ingest ==")
    ingest_py = Path.home() / "rag" / "ingest.py"
    venv_python = Path.home() / "rag" / "venv" / "bin" / "python"
    result = run(f"{venv_python} {ingest_py}")
    return result.stdout

def get_doc_count():
    client = chromadb.PersistentClient(path=str(DB_PATH))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    try:
        col = client.get_collection(name=COLLECTION, embedding_function=ef)
        return col.count()
    except Exception:
        return 0

def main():
    start = datetime.now()
    print(f"[{start}] Starting incremental update...")

    # Get count before
    count_before = get_doc_count()
    print(f"  Doc count before: {count_before}")

    # Update sources
    new_vtt = update_youtube()
    update_forum()
    check_firmware()

    # Re-ingest (dedup handled by ingest.py)
    run_ingest()

    # Get count after
    count_after = get_doc_count()
    new_docs = count_after - count_before

    # Update meta
    meta = {"last_updated": datetime.now().isoformat(), "doc_count": count_after}
    META_PATH.write_text(json.dumps(meta, indent=2))

    elapsed = datetime.now() - start
    print(f"\nUpdate complete in {elapsed}")
    print(f"  New docs added: {new_docs}")
    print(f"  Total docs:     {count_after}")

if __name__ == "__main__":
    main()
