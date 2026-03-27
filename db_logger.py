"""
Query Logger — SQLite-backed logging for all RAG queries.
Standalone module, no dependencies on other rag modules.
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

DB_PATH = Path.home() / "rag" / "queries.db"

# Pricing in USD per 1M tokens — update here when Anthropic changes rates.
# Source: https://www.anthropic.com/pricing
MODEL_PRICING = {
    "claude-opus-4-6":             {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6":           {"input":  3.00, "output": 15.00},
    "claude-haiku-4-5-20251001":   {"input":  0.80, "output":  4.00},
}
_DEFAULT_PRICING = {"input": 15.00, "output": 75.00}  # fallback to Opus rates

def _cost_usd(model, input_tokens, output_tokens):
    rates = MODEL_PRICING.get(model, _DEFAULT_PRICING)
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000

@contextmanager
def _connect():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT    NOT NULL,
                question      TEXT    NOT NULL,
                answer        TEXT,
                sources       TEXT,
                query_type    TEXT,
                hyde_used     INTEGER,
                rerank_used   INTEGER,
                latency_ms    INTEGER,
                client_ip     TEXT,
                blocked       INTEGER DEFAULT 0,
                block_reason  TEXT,
                input_tokens  INTEGER,
                output_tokens INTEGER,
                model         TEXT
            )
        """)
        # Add new columns to existing databases that predate token tracking
        for col, typedef in [("input_tokens", "INTEGER"), ("output_tokens", "INTEGER"), ("model", "TEXT")]:
            try:
                conn.execute(f"ALTER TABLE queries ADD COLUMN {col} {typedef}")
            except sqlite3.OperationalError:
                pass  # column already exists

def log(question, answer=None, sources=None, query_type=None,
        hyde_used=True, rerank_used=True, latency_ms=None,
        client_ip=None, blocked=False, block_reason=None,
        input_tokens=None, output_tokens=None, model=None):
    with _connect() as conn:
        conn.execute("""
            INSERT INTO queries
              (timestamp, question, answer, sources, query_type,
               hyde_used, rerank_used, latency_ms, client_ip, blocked, block_reason,
               input_tokens, output_tokens, model)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            datetime.utcnow().isoformat(),
            question,
            answer,
            json.dumps(sources) if sources else None,
            query_type,
            int(hyde_used),
            int(rerank_used),
            latency_ms,
            client_ip,
            int(blocked),
            block_reason,
            input_tokens,
            output_tokens,
            model,
        ))

def recent(limit=200):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM queries ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d.get("sources"):
                try:
                    d["sources"] = json.loads(d["sources"])
                except Exception:
                    d["sources"] = []
            result.append(d)
        return result

def stats():
    with _connect() as conn:
        total   = conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
        blocked = conn.execute("SELECT COUNT(*) FROM queries WHERE blocked=1").fetchone()[0]
        avg_lat = conn.execute(
            "SELECT AVG(latency_ms) FROM queries WHERE blocked=0 AND latency_ms IS NOT NULL"
        ).fetchone()[0]
        by_type = conn.execute(
            "SELECT query_type, COUNT(*) cnt FROM queries WHERE blocked=0 GROUP BY query_type"
        ).fetchall()
        last_24h = conn.execute(
            "SELECT COUNT(*) FROM queries WHERE timestamp > datetime('now','-24 hours')"
        ).fetchone()[0]

        # Cost calculations — only rows with token data
        token_rows = conn.execute(
            "SELECT model, input_tokens, output_tokens FROM queries "
            "WHERE blocked=0 AND input_tokens IS NOT NULL AND output_tokens IS NOT NULL"
        ).fetchall()
        total_cost = sum(_cost_usd(r["model"], r["input_tokens"], r["output_tokens"]) for r in token_rows)
        avg_cost   = (total_cost / len(token_rows)) if token_rows else 0

        return {
            "total":          total,
            "blocked":        blocked,
            "allowed":        total - blocked,
            "last_24h":       last_24h,
            "avg_latency_ms": round(avg_lat) if avg_lat else 0,
            "by_query_type":  {r["query_type"]: r["cnt"] for r in by_type},
            "total_cost_usd": round(total_cost, 4),
            "avg_cost_usd":   round(avg_cost, 4),
        }

# Initialise on import
init()
