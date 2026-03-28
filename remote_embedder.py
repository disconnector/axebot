#!/usr/bin/env python3
"""
remote_embedder.py — ChromaDB-compatible embedding function that calls
the Mac Studio embed server over Tailscale instead of running locally.

Drop-in replacement for SentenceTransformerEmbeddingFunction.
Copy this file to ~/rag/ on axebot.

Usage in ingest scripts:
    from remote_embedder import RemoteEmbeddingFunction
    ef = RemoteEmbeddingFunction()
    collection = client.get_or_create_collection(name="axefx3", embedding_function=ef)
"""

import os
import time
import logging
from typing import List

import requests
import numpy as np
from chromadb.utils.embedding_functions import EmbeddingFunction

log = logging.getLogger("remote-embedder")

# ── Config — override via environment variables ────────────────────────────────
EMBED_SERVER_URL = os.environ.get("EMBED_SERVER_URL", "http://localhost:8600")
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "512"))
EMBED_TIMEOUT    = int(os.environ.get("EMBED_TIMEOUT", "120"))   # seconds


class RemoteEmbeddingFunction(EmbeddingFunction):
    """
    ChromaDB EmbeddingFunction that delegates to the Mac Studio MPS embed server.
    """

    @staticmethod
    def name() -> str:
        # Return "default" so ChromaDB skips the stored-vs-provided EF conflict check.
        return "default"

    def __init__(
        self,
        server_url: str = EMBED_SERVER_URL,
        batch_size: int = EMBED_BATCH_SIZE,
        timeout:    int = EMBED_TIMEOUT,
        fallback_local: bool = True,
    ):
        self.server_url  = server_url.rstrip("/")
        self.batch_size  = batch_size
        self.timeout     = timeout
        self.fallback_local = fallback_local
        self._local_ef   = None   # lazy-loaded if server unreachable

        self._check_server()

    def _check_server(self):
        """Verify the embed server is reachable at init time."""
        try:
            r = requests.get(f"{self.server_url}/health", timeout=5)
            info = r.json()
            log.info(f"Embed server: {info['model']} on {info['device']} at {self.server_url}")
            print(f"  Embed server: {info['model']} on {info['device']} ({self.server_url})")
        except Exception as e:
            msg = f"Embed server unreachable at {self.server_url}: {e}"
            if self.fallback_local:
                log.warning(msg + " — will fall back to local CPU embeddings")
                print(f"  WARNING: {msg}")
                print(f"  Falling back to local embeddings.")
            else:
                raise RuntimeError(msg)

    def _embed_remote(self, texts: List[str]) -> List[List[float]]:
        """Send texts to the Mac Studio server in batches."""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            r = requests.post(
                f"{self.server_url}/embed",
                json={"texts": batch, "batch_size": self.batch_size},
                timeout=self.timeout,
            )
            r.raise_for_status()
            all_embeddings.extend(r.json()["embeddings"])

        return all_embeddings

    def _get_local_ef(self):
        """Lazy-load local sentence-transformer as fallback."""
        if self._local_ef is None:
            from chromadb.utils import embedding_functions
            local_model = os.environ.get(
                "EMBED_MODEL", "/home/rich/rag/models/axefx-embed/final"
            )
            log.info(f"Loading local fallback model: {local_model}")
            self._local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=local_model
            )
        return self._local_ef

    def __call__(self, input: List[str]) -> List[List[float]]:
        """ChromaDB calls this with a list of strings, expects list of float vectors."""
        if not input:
            return []

        try:
            t0 = time.perf_counter()
            embeddings = self._embed_remote(input)
            elapsed = time.perf_counter() - t0
            tps = len(input) / elapsed
            log.debug(f"Remote embedded {len(input)} texts in {elapsed:.2f}s ({tps:.0f}/sec)")
            return embeddings

        except Exception as e:
            log.warning(f"Remote embedding failed: {e}")
            if self.fallback_local:
                log.warning("Falling back to local embeddings")
                return self._get_local_ef()(input)
            raise


def health_check(server_url: str = EMBED_SERVER_URL) -> dict:
    """Quick health check — useful for testing."""
    r = requests.get(f"{server_url.rstrip('/')}/health", timeout=5)
    return r.json()


def benchmark(server_url: str = EMBED_SERVER_URL, n: int = 500):
    """Quick throughput benchmark of the remote server."""
    sentences = [
        "The Amp block provides realistic amplifier simulation with detailed tone controls.",
        "Setting the Input Trim adjusts the signal level entering the preamp stage of the model.",
        "Cabinet IRs capture the acoustic signature of real speaker cabinets and microphone placements.",
        "The Gate block provides noise reduction by attenuating signal below a threshold level.",
        "Use scenes to switch between different parameter states within the same preset instantly.",
    ] * (n // 5 + 1)
    sentences = sentences[:n]

    print(f"\nBenchmarking {server_url} with {n} sentences...")
    t0 = time.perf_counter()
    ef = RemoteEmbeddingFunction(server_url=server_url, fallback_local=False)
    vecs = ef(sentences)
    elapsed = time.perf_counter() - t0

    print(f"  {n} embeddings in {elapsed:.2f}s  ({n/elapsed:.0f} texts/sec)")
    print(f"  Vector dim: {len(vecs[0])}")
    print(f"  Sample magnitude: {float(np.linalg.norm(vecs[0])):.4f}")


if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else EMBED_SERVER_URL
    h = health_check(url)
    print(f"Health: {h}")
    benchmark(url)
