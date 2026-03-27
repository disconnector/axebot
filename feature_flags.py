"""
Feature Flags — JSON-file backed, readable/writable by any process.
Any change takes effect on the next request to server.py.
"""

import json
from pathlib import Path
from threading import Lock

FLAGS_PATH = Path(__file__).parent / "feature_flags.json"
_lock = Lock()

DEFAULTS = {
    "hyde_enabled":             True,
    "reranker_enabled":         True,
    "query_type_detection":     True,
    "rate_limiting":            True,
}

def load() -> dict:
    if FLAGS_PATH.exists():
        try:
            return {**DEFAULTS, **json.loads(FLAGS_PATH.read_text())}
        except Exception:
            pass
    return DEFAULTS.copy()

def save(flags: dict):
    with _lock:
        FLAGS_PATH.write_text(json.dumps(flags, indent=2))

def update(updates: dict) -> dict:
    flags = load()
    flags.update({k: v for k, v in updates.items() if k in DEFAULTS})
    save(flags)
    return flags

# Write defaults if file doesn't exist yet
if not FLAGS_PATH.exists():
    save(DEFAULTS)
