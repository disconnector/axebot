# RAG System Build Framework
## Lessons from AxeBot — A Domain-Specific AI Assistant

---

## What Is a RAG System?

**RAG (Retrieval-Augmented Generation)** is an architecture that gives a large language model (LLM) access to a specific body of knowledge it wasn't trained on. Instead of relying solely on what the model memorized during training, you:

1. Build a searchable database of your domain knowledge
2. At query time, retrieve the most relevant passages
3. Feed those passages to the LLM as context
4. The LLM synthesizes an answer grounded in your actual documents

The result: an AI assistant that knows your specific domain in depth, stays current (you control the corpus), and can cite its sources.

**AxeBot example:** The Axe-Fx III has ~800 amp parameters, 200+ amp models, and a huge community knowledge base. No general LLM knows this in depth. By building a RAG corpus from the official manual, wiki, forum threads, and YouTube transcripts, AxeBot answers questions that would stump GPT-4.

---

## Phase 1: Corpus Assembly

### What Is a Corpus?

Your corpus is the raw knowledge base — all the documents, forum posts, PDFs, videos, etc. that your system will draw from. **Quality here determines everything downstream.**

### Source Types AxeBot Used

| Source | Format | Strength | Weakness |
|--------|--------|----------|----------|
| Official PDFs (owner's manual, blocks guide) | PDF | Authoritative, structured | Dense, requires good chunking |
| Community Wiki | HTML scrape | Comprehensive, curated | Contains navigation boilerplate |
| Forum Threads | HTML scrape | Real-world experience, edge cases | High noise ratio, stubs, off-topic |
| YouTube Transcripts | VTT files | Conversational knowledge, tips | Metadata artifacts, rambling |

### Key Decisions at Corpus Assembly

- **Be selective upfront.** Every noisy document you ingest costs you later in cleanup, wrong answers, and diluted retrieval. It's faster to exclude junk now than delete it after embedding.
- **Preserve metadata.** Store `source_type`, `filename`, `page`, `url`, `author` — you'll use this to filter queries and attribute citations.
- **Domain specificity matters.** A corpus of 46,000 high-quality domain chunks outperforms 100,000 chunks with 50% noise.

---

## Phase 2: Text Extraction

### Why Text Extraction Is Non-Trivial

PDFs especially are notorious — the visual layout doesn't map cleanly to linear text. Page columns, hyphenation, headers/footers, tables, and figures all cause extraction artifacts.

### Tools Compared

| Tool | Quality | Notes |
|------|---------|-------|
| PyPDF2 | Poor | Garbles hyphenation, loses structure |
| PyMuPDF (fitz) | Excellent | Preserves paragraph structure, handles headers cleanly |
| BeautifulSoup | Good for HTML | Works well for wiki/forum scraping |
| VTT parser | Simple | YouTube transcripts are already text, just need timestamp stripping |

**Lesson from AxeBot:** Yek's Amp Guide (the single most valuable document in the corpus) was initially ingested with PyPDF2. Answers about specific amp models were weak and fragmented. Switching to PyMuPDF and re-ingesting dropped 385 chunks down to 141 — each a complete, coherent amp model entry. Answer quality improved dramatically.

---

## Phase 3: Chunking

### What Is Chunking?

LLMs have a context window limit. You can't feed an entire 200-page manual as context. Chunking splits documents into smaller pieces that can each be independently embedded and retrieved.

### Chunk Size Trade-offs

| Chunk Too Small | Chunk Too Large |
|----------------|----------------|
| Loses context (fragment of a sentence, half a definition) | Hard to embed — the embedding averages over too many topics |
| Retrieves irrelevant neighbors | Wastes context window on irrelevant parts |
| Forces LLM to work with incomplete info | Retrieval precision drops |

**Rule of thumb:** 200–600 words works for most domains. For structured reference material (like amp model entries), use **semantic/structural boundaries** instead of word count.

### Chunking Strategies

**Fixed-size (naive):**
- Split every N words with M-word overlap
- Fast and simple, works okay for prose
- Problem: splits mid-definition, mid-explanation

**Semantic/structural (better):**
- Detect natural boundaries: section headers, topic changes, page breaks
- Keep related content together
- AxeBot used this for Yek's guide: detected when the "current amp model" changed between pages, kept each complete amp entry as one chunk

**Model-aware chunking (AxeBot's approach for Yek's guides):**
```
For each page:
  first_line = page.get_text().split('\n')[0].strip()
  if first_line != current_model_name:
      # New amp entry started — save previous chunk, start new one
      current_model_name = first_line
```

### Overlap

When using fixed-size chunking, overlap consecutive chunks by ~10-20% of chunk size. This ensures sentences that fall on a boundary appear in context in both neighbors, preventing retrieval gaps.

---

## Phase 4: Embedding

### What Is an Embedding?

An embedding is a dense vector (list of ~384 numbers) that represents the *semantic meaning* of a chunk of text. Text with similar meanings gets similar vectors. Text with different meanings gets different vectors.

The magic: mathematical operations on vectors correspond to semantic relationships. "Amp" and "amplifier" end up close together. "Input Trim" and "gain reduction at input stage" end up nearby. This is the foundation of semantic search.

### Embedding Models

| Model | Dimensions | Notes |
|-------|-----------|-------|
| General purpose (e.g., all-MiniLM) | 384 | Good baseline, off-the-shelf |
| Fine-tuned domain model | 384 | Trained on your domain Q&A pairs — significantly better for domain-specific queries |

**AxeBot fine-tuned its embedding model** on thousands of generated Q&A pairs from the corpus. The fine-tuned model understands that "brown sound" → amp circuit topology, "fizzy" → high-frequency distortion artifacts, etc. A general embedding model doesn't.

### How to Fine-Tune an Embedding Model

1. **Generate training pairs:** For each chunk, generate 3-5 questions that the chunk answers (use a LLM for this). These become positive pairs: `(question, chunk)`.
2. **Negatives:** Use in-batch negatives (other chunks in the same batch that don't answer the question).
3. **Training framework:** `sentence-transformers` library with `MultipleNegativesRankingLoss`.
4. **Validation:** Hold out 10% of pairs; measure cosine similarity improvement on validation set.

**AxeBot result:** Fine-tuned model on domain Q&A pairs from 98k chunks. The model learned the Fractal Audio vocabulary, making retrieval significantly more precise.

---

## Phase 5: Vector Database (ChromaDB)

### What Is a Vector Database?

A vector database stores embeddings alongside their source text and metadata, and supports efficient **approximate nearest neighbor (ANN)** search — "give me the N chunks whose embeddings are most similar to this query embedding."

### ChromaDB Specifics

AxeBot uses **ChromaDB** with a `PersistentClient` backed by SQLite + HNSW index files on disk.

Key operations:
```python
# Create/get collection
client = chromadb.PersistentClient(path="~/rag/db")
col = client.get_or_create_collection("axefx3")

# Ingest
col.upsert(ids=[...], documents=[...], embeddings=[...], metadatas=[...])

# Query
r = col.query(query_embeddings=[embedding], n_results=20,
              include=['documents', 'metadatas', 'distances'])

# Filter by metadata
r = col.query(..., where={"source_type": "pdf"})

# Paginated read (important for large collections!)
r = col.get(include=['documents','metadatas'], limit=2000, offset=0)
```

**Important:** ChromaDB has a SQLite variable limit. With large collections (~50k+ chunks), always paginate `get()` calls with `limit=2000` or you'll hit `sqlite3.OperationalError: too many SQL variables`.

### HNSW Index — How Similarity Search Actually Works

HNSW (Hierarchical Navigable Small World) is the graph structure that makes fast ANN search possible. It does NOT work like a B-tree.

**The key insight:** Similar vectors cluster spatially in embedding space. If point A and point B are similar, they're neighbors in the HNSW graph. When you search for query Q, you navigate the graph by greedily moving to neighbors that are closer to Q at each step, like a local search on a map.

The hierarchy: the graph has multiple layers. Top layers are coarse (sparse, long-range connections for fast global navigation). Bottom layers are dense (fine-grained, local connections for precise search). You enter at the top, descend to your starting neighborhood, then search the dense bottom layer.

**Speed vs. accuracy:** HNSW is approximate — it may miss the single closest vector in exchange for very fast search. For RAG, this is fine. You're fetching top-20 candidates anyway; one missed neighbor doesn't matter.

---

## Phase 6: The Query Pipeline

### Overview

When a user asks a question, it goes through several stages:

```
User Query
    ↓
[HyDE — Hypothetical Document Expansion]
    ↓
[Embedding — encode the expanded query]
    ↓
[Vector Search — fetch top-K candidates from ChromaDB]
    ↓
[Metadata Filtering — optional source_type filter]
    ↓
[Cross-Encoder Reranking — reorder by semantic match]
    ↓
[Context Assembly — format chunks with citations]
    ↓
[LLM Generation — Claude synthesizes answer]
    ↓
Answer with citations
```

### HyDE (Hypothetical Document Expansion)

**Problem:** A user's question and the answer in your corpus use different words. "How do I fix fizzy tone?" doesn't match "high-frequency distortion artifacts in gain stages." The question embedding and the answer embedding are far apart.

**Solution:** Before embedding the query, use a fast/cheap LLM call to generate a *hypothetical* answer. This hypothetical answer uses the vocabulary of the corpus (it sounds like your documents), so its embedding lands near the real answer chunks.

```python
hypothetical = await quick_llm("Write a technical explanation answering: " + query)
search_text = query + " " + hypothetical
embedding = model.encode(search_text)
```

**AxeBot result:** HyDE significantly improved retrieval for experiential queries like "how do I get the brown sound?" because the hypothetical answer mentioned specific amp models and EQ terminology that the real corpus answers use.

### Cross-Encoder Reranking

**Problem:** Vector search retrieves "approximately nearest" — it's fast but imprecise. The top-20 results include relevant and semi-relevant chunks jumbled together.

**Solution:** Run a cross-encoder model on each `(query, chunk)` pair. The cross-encoder reads both together and outputs a relevance score. Unlike the bi-encoder (which embeds independently), the cross-encoder sees the query and document simultaneously — much more accurate.

Fetch 20 candidates, rerank with cross-encoder, keep top 6-8 for the LLM prompt.

```python
pairs = [(query, chunk) for chunk in candidates]
scores = cross_encoder.predict(pairs)
reranked = sorted(zip(scores, candidates), reverse=True)
top_chunks = reranked[:TOP_K]
```

**Bi-encoder vs. cross-encoder:**
- Bi-encoder: encodes query and doc separately → fast, can precompute doc embeddings → good for retrieval
- Cross-encoder: reads query+doc together → slow, can't precompute → better accuracy → good for reranking

### Context Assembly

Format the top chunks for the LLM with clear structure and source attribution:

```
[1] (wiki) INPUT TRIM: From the Owner's Manual: Amps without an Overdrive...
[2] (pdf, Yek's Amp Guide p.12) The Input Trim parameter reduces the signal...
[3] (forum) FractalAudio said: Turn it all the way up and you'll hear...
```

This lets the LLM cite `[1]`, `[2]`, `[3]` in its answer, and lets your UI inject source tooltips on hover.

---

## Phase 7: The LLM Layer

### System Prompt Design

The system prompt defines the assistant's persona, constraints, and behavior. Key elements for AxeBot:

- **Scope restriction:** "You are an expert Axe-Fx III technician. Only answer questions about Axe-Fx III, FM9, FM3, and related Fractal Audio products."
- **Off-topic handling:** Inject a random funny refusal from a pre-written list (Python `random.choice()` — don't ask the LLM to vary its refusals, it won't).
- **Counting/enumeration limitation:** Tell the model it can only work with retrieved passages, not the whole corpus — prevents hallucinated counts.
- **No interactive preset building:** Suppress offers to "walk you through building a preset" since the model can't actually do that interactively.
- **Citation instructions:** "Reference chunks as [1], [2], etc."

**Important:** Rebuild the system prompt **on every request** if it contains randomized elements. Don't cache it.

### Model Selection

AxeBot uses **Claude Opus** for generation. For a domain expert assistant where answer quality matters more than speed or cost, use the most capable model available. Cheaper models produce noticeably weaker synthesis and more hallucinations.

For the HyDE expansion call (cheap, fast, just vocabulary generation), a smaller model or even a local model is fine.

### Token Cost Tracking

Track input and output tokens per query in your database. Calculate cost at display time using a model-aware pricing dict so costs update when prices change without touching historical data:

```python
MODEL_PRICING = {
    "claude-opus-4-6":   {"input": 15.00, "output": 75.00},  # per 1M tokens
    "claude-sonnet-4-6": {"input":  3.00, "output": 15.00},
}

def cost_usd(model, input_tokens, output_tokens):
    rates = MODEL_PRICING.get(model, DEFAULT_PRICING)
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000
```

---

## Phase 8: Corpus Quality Management

### The Noise Problem

Web-scraped corpora are always dirty. Common noise types:

| Noise Type | Example | Detection |
|-----------|---------|-----------|
| Navigation boilerplate | "Login \| Register \| Jump to navigation \| Jump to search" | 2+ pattern matches from NOISE_PATTERNS list |
| Wiki category pages | "This is the wiki for products made by Fractal Audio" | Strong pattern, 1 match |
| Thread title stubs | "TITLE: Input Trim URL: https://forum.fractalaudio.com/threads/..." | Strong pattern, 1 match |
| Revision history pages | "Revision history of \"Amp block\"" | Strong pattern, 1 match |
| Video metadata artifacts | "Video history: Legend" | Strong pattern, 1 match |
| Too-short chunks | 8 words, fragment of a sentence | Word count < MIN_WORDS |

### Noise Detection Pattern

```python
MIN_WORDS = 30

NOISE_PATTERNS = [          # 2+ hits = noise
    "login", "register", "jump to navigation",
    "retrieved from", "last edited", "privacy policy", ...
]

NOISE_PATTERNS_STRONG = [   # 1 hit = instant noise
    "maintained by members of the community",
    "wiki for products made by fractal audio",
    "url: https://forum.fractalaudio.com",
    "revision history of \"",
    "video history: legend",
]

def is_noise(doc: str) -> bool:
    if not doc: return True
    if len(doc.split()) < MIN_WORDS: return True
    doc_lower = doc.lower()
    if any(p in doc_lower for p in NOISE_PATTERNS_STRONG): return True
    return sum(1 for p in NOISE_PATTERNS if p in doc_lower) >= 2
```

### Benchmark-Driven Cleanup

Build a benchmark with representative queries across three categories:
- **Factual** (should hit PDFs/wiki, low noise expected)
- **Experiential** (forum/youtube, moderate noise expected)
- **Vague** (highest noise risk — "Tell me about the Axe-Fx III")

Measure **noise percentage** before and after cleanup runs. AxeBot's journey:

| Stage | Corpus Size | Overall Noise % |
|-------|------------|----------------|
| Initial ingest | 98,234 chunks | ~46% noise |
| Round 1 cleanup | 52,883 chunks | ~25% noise |
| Round 2 cleanup | 47,462 chunks | ~20% noise |
| Round 3 cleanup | 46,089 chunks | ~16% noise |

### UMAP Visualization — Seeing Your Corpus

UMAP (Uniform Manifold Approximation and Projection) reduces your 384-dimensional embeddings to 2D for visualization. Clusters of similar chunks appear as islands. Noise chunks cluster separately because navigation boilerplate doesn't look like technical content.

**What to look for:**
- Isolated islands far from the main corpus → likely noise or irrelevant content
- Uniform mixing of source types in the main cloud → healthy, diverse corpus
- Tight clusters that are too small → possible duplicate or near-duplicate content

Build an interactive UMAP with Plotly + ScatterGL, colored by source_type. Hover shows chunk preview. This lets you quickly identify entire categories of noise to add to your cleanup patterns.

```python
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                    metric='cosine', random_state=42)
coords = reducer.fit_transform(embeddings)
# → plot with plotly.graph_objects.Scattergl, colored by source_type
```

### Preventive vs. Reactive Cleanup

- **Reactive:** cleanup_noise.py runs daily as a cron job, deletes noise from existing corpus
- **Preventive:** Add the same `is_noise()` check to your ingest pipeline to reject noise at ingest time

Both are needed. Reactive cleanup catches noise that passed initial filters. Preventive filtering keeps new ingests clean.

---

## Phase 9: Infrastructure

### API Server Architecture

Two separate servers:

**RAG API (server.py, port 8502):**
- Handles queries end-to-end: embed → search → rerank → generate
- Streaming endpoint for real-time response display
- Manages ChromaDB connection (single client, reused)
- Exposes `/health` (fast, current doc count) and `/stats` (slow, full breakdown by source)

**Admin Server (admin_server.py, port 8503):**
- Serves user UI (`user.html`) and admin dashboard (`admin.html`)
- Proxies to RAG API, adds authentication layer
- Server-side caches the slow `/stats` response (TTL: 24 hours)
- Logs all queries to SQLite via db_logger.py

**Why split `/health` from `/stats`?**
Full corpus breakdown requires scanning all 46k+ chunks — takes 5-10 seconds. The health endpoint just returns the pre-cached doc count. The admin dashboard polls health every 5s and stats only on load (cached 24h). Before this split, the VM ran at 130% CPU.

### Database Logging

Every query logs to SQLite:
```
id | timestamp | query | response_length | source_filter
   | num_chunks | input_tokens | output_tokens | model | latency_ms
```

Aggregate stats: total queries, avg latency, noise rate, model cost per query, total cost.

### Streaming

Use streaming for the LLM generation endpoint so the user sees tokens appearing immediately rather than waiting for the full response. The Anthropic SDK streams via `async with client.messages.stream(...)`, and the server sends SSE (Server-Sent Events) to the browser.

Token usage comes from `await stream.get_final_message()` after the stream completes — the final message object contains `.usage.input_tokens` and `.usage.output_tokens`.

---

## Phase 10: UI Design

### User Interface Principles for Expert Tools

- **Show your work:** Display source citations with chunk count, source types, and confidence indicators
- **Attribution:** Hover tooltips on `[1]` citations showing "Yek's Amp Guide, p.12" or "Forum — Input Trim thread"
- **Domain branding:** The UI should signal expertise. AxeBot uses a custom robot-with-guitar-axe logo, "Deep Signal NOC" aesthetic — the UI matches the serious, technical domain
- **Streaming:** Tokens should appear immediately. No one wants to wait 5 seconds for a blank screen then see the full answer pop in

### Citation Tooltips

```javascript
// Build citation map from retrieved chunks metadata
let citationMap = {};
chunks.forEach((chunk, i) => {
    citationMap[i + 1] = prettySource(chunk);
});

// Inject span wrappers on [n] references in LLM response
function injectCitationTooltips(html) {
    return html.replace(/\[(\d+)\]/g, (match, num) => {
        const label = citationMap[parseInt(num)];
        if (!label) return match;
        return `<span class="citation-ref" title="${label}">${match}</span>`;
    });
}
```

---

## Lessons Learned & Patterns to Reuse

### What Worked Well

1. **Fine-tuned embeddings** — biggest quality improvement of any single change
2. **HyDE** — closed the vocabulary gap between user queries and corpus language
3. **Cross-encoder reranking** — dramatically improved result precision
4. **UMAP visualization** — identified entire categories of noise that would have taken days to find manually
5. **Structural chunking for reference material** — one chunk per complete entry beats fixed-size every time for structured documents
6. **Benchmark before/after** — made cleanup measurable instead of subjective

### What Caused Problems

1. **PyPDF2** — switch to PyMuPDF immediately; the text quality difference is huge
2. **Not filtering noise at ingest time** — ingested 50k+ noise chunks that had to be deleted later
3. **Caching the slow stats call at the wrong layer** — admin dashboard was polling a 10-second endpoint every 5 seconds
4. **Trusting the LLM to randomize** — it won't; use Python `random.choice()` for any response variation
5. **SQLite variable limit with large collections** — always paginate ChromaDB `get()` calls

### Generalizable Framework for New RAG Projects

1. **Define your corpus carefully** — what sources, what time range, what quality bar?
2. **Extract text properly** — invest in PyMuPDF or appropriate parser upfront
3. **Design chunks around semantic units** — not just word count
4. **Generate training pairs and fine-tune embeddings** — if you have a specific domain, this pays off
5. **Build a benchmark before you do anything else** — you need a baseline to measure against
6. **Visualize with UMAP early** — it reveals corpus structure, islands, and noise patterns in minutes
7. **Implement noise filtering at both ingest and cleanup** — defense in depth
8. **Monitor costs** — track tokens/query from day one; Opus adds up fast
9. **Cache aggressively** — anything that scans the full corpus belongs behind a long TTL cache
10. **Attribute sources** — users trust answers more when they can see where they came from

---

## AxeBot Architecture Summary

```
User Browser
    ↓ HTTPS
Admin Server (port 8503)
    ├── user.html (chat UI, streaming, citations)
    ├── admin.html (query stats, corpus breakdown, cost tracking)
    └── /api/* → proxies to RAG API
                      ↓
              RAG API (port 8502)
                  ├── /query (streaming, full pipeline)
                  ├── /health (fast doc count)
                  └── /stats (full breakdown, slow)
                      ↓
              ChromaDB (46k chunks)
              + Fine-tuned SentenceTransformer
              + Cross-encoder reranker
                      ↓
              Claude Opus API (Anthropic)
```

**Corpus breakdown:**
- PDF: Owner's Manual, Blocks Guide, Yek's Amp/Drive Guide (~3k chunks)
- Wiki: FractalAudio Wiki scraped articles (~18k chunks)
- Forum: Community discussion threads (~20k chunks)
- YouTube: Video transcripts (~5k chunks)

**Daily maintenance:**
- cleanup_noise.py runs at 4am UTC, removes newly ingested noise
- Benchmark available on-demand to measure quality

---

*Built March 2026. AxeBot serves the Fractal Audio community at http://192.168.50.11:8503*
