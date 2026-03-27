#!/usr/bin/env python3
"""
Axe-Fx III RAG — Admin Server  (port 8503)
Two pages:
  /          → Management dashboard (logs, stats, health)
  /test      → Testing interface (module toggles + query harness)
"""

import asyncio
import json
import logging
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import uvicorn

import db_logger
import feature_flags as ff

STATIC_DIR   = Path(__file__).parent / "static"
RAG_API_URL  = "http://localhost:8502"

# Cache the slow /stats corpus breakdown — only re-fetch every 5 minutes
_corpus_cache: dict = {}
_corpus_cache_time: float = 0.0
CORPUS_CACHE_TTL = 86400  # seconds (24 hours)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rag-admin")

app = FastAPI(title="Axe-Fx III RAG Admin", version="1.0.0",
              docs_url=None, redoc_url=None, openapi_url=None)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Pages ──────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def management():
    return FileResponse(str(STATIC_DIR / "admin.html"))

@app.get("/test", include_in_schema=False)
async def testing():
    return FileResponse(str(STATIC_DIR / "test.html"))

# ── API: Logs & Stats ─────────────────────────────────────────────────────────
@app.get("/api/logs")
async def get_logs(limit: int = 100):
    return {"queries": db_logger.recent(limit=limit)}

@app.get("/api/stats")
async def get_stats():
    global _corpus_cache, _corpus_cache_time
    import time

    query_stats = db_logger.stats()

    # Always get doc_count from the fast /health endpoint
    corpus = {}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{RAG_API_URL}/health")
            corpus["total"] = r.json().get("doc_count", 0)
    except Exception:
        pass

    # Only re-fetch the slow breakdown scan every CORPUS_CACHE_TTL seconds
    now = time.monotonic()
    if _corpus_cache and (now - _corpus_cache_time) < CORPUS_CACHE_TTL:
        corpus["by_source_type"] = _corpus_cache
    else:
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.get(f"{RAG_API_URL}/stats")
                breakdown = r.json().get("by_source_type", {})
                if breakdown:
                    _corpus_cache = breakdown
                    _corpus_cache_time = now
                corpus["by_source_type"] = breakdown
        except Exception:
            corpus["by_source_type"] = _corpus_cache  # serve stale on error

    # Fallback: compute total from breakdown if /health call didn't populate it
    if not corpus.get("total") and corpus.get("by_source_type"):
        corpus["total"] = sum(corpus["by_source_type"].values())

    return {"query_stats": query_stats, "corpus": corpus}

@app.get("/api/health")
async def get_health():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{RAG_API_URL}/health")
            return r.json()
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}

# ── API: Feature flags ────────────────────────────────────────────────────────
@app.get("/api/config")
async def get_config():
    return ff.load()

class FlagUpdate(BaseModel):
    hyde_enabled:         Optional[bool] = None
    reranker_enabled:     Optional[bool] = None
    query_type_detection: Optional[bool] = None
    rate_limiting:        Optional[bool] = None

@app.patch("/api/config")
async def update_config(body: FlagUpdate):
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(400, "No valid flags provided")
    result = ff.update(updates)
    log.info(f"Flags updated: {updates}")
    return result

# ── API: Test query (per-request flag overrides) ──────────────────────────────
class TestRequest(BaseModel):
    question:             str
    hyde:                 Optional[bool] = None
    reranker:             Optional[bool] = None
    query_type_detection: Optional[bool] = None
    top_k:                int = 8

@app.post("/api/test")
async def run_test(req: TestRequest):
    """
    Send a test query to the main RAG API with specific flag overrides.
    Flags not specified here fall back to the global feature_flags.json settings.
    """
    payload = {
        "question": req.question,
        "top_k":    req.top_k,
    }
    if req.hyde                 is not None: payload["hyde"]                 = req.hyde
    if req.reranker             is not None: payload["reranker"]             = req.reranker
    if req.query_type_detection is not None: payload["query_type_detection"] = req.query_type_detection

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{RAG_API_URL}/ask",
                                  json=payload,
                                  headers={"Content-Type": "application/json"})
        if r.status_code == 422:
            detail = r.json().get("detail", [])
            msg    = detail[0].get("msg","validation error") if detail else "validation error"
            raise HTTPException(422, msg)
        if r.status_code != 200:
            raise HTTPException(r.status_code, r.text[:200])
        return r.json()
    except httpx.TimeoutException:
        raise HTTPException(504, "RAG API timed out")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8503)
