# AxeBot — Axe-Fx III RAG Assistant

A domain-specific AI assistant for the Fractal Audio Axe-Fx III, built on retrieval-augmented generation (RAG). Ask questions about amp models, blocks, parameters, routing, firmware, and community knowledge — all grounded in real documentation.

For a deep dive into the architecture and lessons learned, see [RAG_Framework.md](RAG_Framework.md).

---

## What's Running

| Server | Port | Purpose |
|--------|------|---------|
| `server.py` | 8502 | RAG API — query pipeline, ChromaDB, Claude |
| `admin_server.py` | 8503 | Admin UI — query logs, corpus stats, feature flags |

The user-facing chat UI is served at `http://localhost:8503/`.  
The admin/testing panel is at `http://localhost:8503/test`.

> **Security note:** The admin server has no authentication. Keep port 8503 off the public internet.

---

## Requirements

- Python 3.10+
- ~6 GB disk for the ChromaDB + embedding models
- An [Anthropic API key](https://console.anthropic.com/)

---

## Installation

```bash
git clone <repo-url>
cd rag

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### requirements.txt (key packages)

```
anthropic==0.86.0
chromadb==1.5.5
fastapi==0.135.2
uvicorn==0.42.0
sentence-transformers==5.3.0
torch==2.11.0
httpx==0.28.1
python-dotenv==1.2.2
```

---

## Configuration

Create a `.env` file in the `rag/` directory:

```bash
cp .env.example .env
```

Edit `.env`:

```
ANTHROPIC_API_KEY=your_key_here
```

That's the only required setting. The servers and ingest scripts all load this file automatically via `python-dotenv`.

> **Never commit `.env` to git.** It's in `.gitignore` already — keep it that way.

---

## Building the Corpus

The corpus lives in `~/fractal/` (or wherever you point the paths in `ingest.py`). Structure:

```
fractal/
  guides/          ← PDFs: owner's manual, blocks guide, Yek's amp/drive guides
  wiki/            ← HTML files scraped from the Fractal Audio wiki
  forum/threads/   ← HTML files of forum threads
  youtube-transcripts/ ← .vtt files from YouTube videos
```

You provide the source files. The ingest script handles parsing, chunking, embedding, and loading into ChromaDB.

### Getting source material

- **PDFs:** Download from [Fractal Audio's downloads page](https://www.fractalaudio.com/axe-fx-iii-downloads/)
- **Wiki:** Scrape with `wget --mirror` or a tool like `scrapy`; save individual article HTML files
- **Forum threads:** Save HTML of useful threads (parameter deep-dives, tutorials, Q&A threads)
- **YouTube transcripts:** Download `.vtt` caption files using `yt-dlp --write-sub --sub-lang en`

Quality matters more than quantity — see [RAG_Framework.md](RAG_Framework.md#phase-1-corpus-assembly) for guidance on what to include.

---

## Ingesting the Corpus

With the corpus in place, run:

```bash
source venv/bin/activate
python ingest.py
```

This will:
1. Walk all source directories
2. Parse and chunk each file (strategy varies by source type — PDFs use PyMuPDF, HTML uses BeautifulSoup, VTT files are stripped of timestamps)
3. Embed chunks using `BAAI/bge-large-en-v1.5` (downloads automatically on first run, ~1.3 GB)
4. Upsert into ChromaDB at `~/rag/db/`
5. Skip files already in the database (incremental by default)

To force a full re-ingest from scratch:

```bash
python ingest.py --force
```

> **Note:** First run downloads the embedding model (~1.3 GB) and processes all source files. Depending on corpus size this can take 30–90 minutes. Subsequent incremental runs are fast.

The ingest script uses `EMBED_MODEL = "BAAI/bge-large-en-v1.5"` by default — the same model `server.py` uses for queries. Keep these in sync; if you change one, change the other.

---

## Re-embedding an Existing Corpus

If you change embedding models (e.g. fine-tuning your own, or upgrading to a better base model), use `reembed.py` to migrate without re-parsing the raw files:

```bash
# With a remote embedding server (e.g. GPU machine on LAN)
EMBED_SERVER_URL=http://192.168.1.x:8600 python reembed.py

# Dry run first to verify connectivity
EMBED_SERVER_URL=http://192.168.1.x:8600 python reembed.py --dry-run
```

`reembed.py` reads all documents from the existing ChromaDB collection, re-embeds them using the new model, and writes them to a new collection (`axefx3_bge` by default). The old collection is left untouched for easy rollback.

After re-embedding, update `COLLECTION` in `server.py` to point to the new collection name, then restart the server.

---

## Running the Servers

### Manual (development)

```bash
source venv/bin/activate

# Terminal 1 — RAG API
python server.py

# Terminal 2 — Admin/UI server
python admin_server.py
```

Open `http://localhost:8503` in your browser.

### systemd (production)

A service unit file is included. Install it for the current user:

```bash
cp rag-api.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable rag-api
systemctl --user start rag-api
```

For the admin server, create a matching `rag-admin.service` (same pattern, swap `server.py` for `admin_server.py` and port 8502 for 8503).

Check logs:

```bash
journalctl --user -u rag-api -f
```

---

## Feature Flags

Runtime toggles are stored in `feature_flags.json` and take effect on the next request without a server restart:

| Flag | Default | Effect |
|------|---------|--------|
| `hyde_enabled` | `true` | HyDE query expansion — generates a hypothetical answer to improve retrieval |
| `reranker_enabled` | `true` | Cross-encoder reranking of retrieved chunks before LLM generation |
| `query_type_detection` | `true` | Automatic source-type routing (factual→PDFs/wiki, experiential→forum/youtube) |
| `rate_limiting` | `true` | Per-IP rate limiter (10 req/60s) |

Toggle via the admin UI at `http://localhost:8503/test`, or edit `feature_flags.json` directly.

---

## Advanced: Fine-Tuned Embeddings

The included fine-tuned model (`models/axefx-embed/final`) was trained on thousands of Q&A pairs generated from the corpus. It understands Fractal Audio vocabulary better than the base model — "brown sound", "fizzy", "input trim" map correctly to technical concepts.

To use it, update `EMBED_MODEL` in both `ingest.py` and `server.py` to point to the `models/axefx-embed/final` directory. You must re-ingest (or re-embed) after switching models.

To train your own fine-tuned model on a new corpus:
1. Run `generate_pairs.py` to produce `training_pairs.jsonl` from your corpus
2. Run `train_embeddings.py` to fine-tune a base sentence-transformers model
3. Point `EMBED_MODEL` at the output directory and re-ingest

See [RAG_Framework.md § Phase 4](RAG_Framework.md#phase-4-embedding) for a full explanation of the fine-tuning process.

---

## Corpus Stats

Check corpus health at any time:

```bash
# Quick count
curl http://localhost:8502/health

# Full breakdown by source type (slow — scans all chunks)
curl http://localhost:8502/stats
```

Or use the admin dashboard at `http://localhost:8503`.
