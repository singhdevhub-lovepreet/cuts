from __future__ import annotations

import hashlib
import json
from pathlib import Path


def stable_json_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "cuts" / "vlm"
