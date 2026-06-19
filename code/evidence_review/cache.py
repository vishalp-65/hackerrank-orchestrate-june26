"""Content-addressed disk cache for raw perception.

Caching the model's perception (not the final row) lets us re-run the deterministic
adjudication layer — or re-emit output.csv — with zero API calls. The key folds in
the prompt version and model so changing either transparently bypasses stale entries.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from . import config
from .data_loader import ClaimRow


def cache_key(claim: ClaimRow, prompt_version: str, model: str) -> str:
    payload = json.dumps(
        [claim.user_id, claim.image_paths, claim.user_claim, claim.claim_object,
         prompt_version, model],
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _path(key: str) -> Path:
    return config.CACHE_DIR / f"{key}.json"


def load(key: str) -> dict | None:
    p = _path(key)
    if not p.is_file():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save(key: str, perception: dict) -> None:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _path(key).with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(perception, f, ensure_ascii=False)
    os.replace(tmp, _path(key))  # atomic
