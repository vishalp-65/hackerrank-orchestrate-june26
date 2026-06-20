"""Central configuration: paths, model registry, pricing, and tunables.

All paths are derived from this file's location so the package works regardless of
the current working directory.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
# config.py lives at: <repo>/code/evidence_review/config.py
REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
DATASET_DIR = REPO_ROOT / "dataset"
CACHE_DIR = CODE_DIR / ".cache"

CLAIMS_CSV = DATASET_DIR / "claims.csv"
SAMPLE_CLAIMS_CSV = DATASET_DIR / "sample_claims.csv"
USER_HISTORY_CSV = DATASET_DIR / "user_history.csv"
EVIDENCE_REQUIREMENTS_CSV = DATASET_DIR / "evidence_requirements.csv"

# Primary submission artifact lives at the repo root per the problem statement.
DEFAULT_OUTPUT_CSV = REPO_ROOT / "output.csv"
# Mirrored copy kept next to the dataset (where the starter header file lives).
DATASET_OUTPUT_CSV = DATASET_DIR / "output.csv"

EVALUATION_DIR = CODE_DIR / "evaluation"
TEST_RUN_STATS = EVALUATION_DIR / "test_run_stats.json"


# ── Environment (.env autoload + typed getters) ──────────────────────────────
def _load_dotenv(path: Path) -> None:
    """Minimal .env loader: ``KEY=VALUE`` lines → os.environ (never overrides existing).

    Avoids a hard dependency on python-dotenv. Real OS/shell env vars always win.
    """
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except FileNotFoundError:
        pass


_load_dotenv(CODE_DIR / ".env")


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v in (None, ""):
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


# ── Models ───────────────────────────────────────────────────────────────────
MODELS = {
    "opus": "claude-opus-4-8",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5",
}

# Default model resolved from LLM_MODEL (accepts a full id or a short key);
# falls back to sonnet. A full id maps back to its short key when one exists.
_ID_TO_KEY = {v: k for k, v in MODELS.items()}
_LLM_MODEL = _env_str("LLM_MODEL", MODELS["sonnet"])
DEFAULT_MODEL_KEY = _ID_TO_KEY.get(_LLM_MODEL, _LLM_MODEL)

# USD per 1,000,000 tokens: (input, output).
PRICING = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

# ── Tunables ─────────────────────────────────────────────────────────────────
PROMPT_VERSION = "v5"          # v5: NEI-cascade fix, schema trim, distance-1 snapping, image-param cache key
JPEG_QUALITY = 85              # re-encode quality for normalized images

# Env-overridable pipeline knobs (see code/.env and code/.env.example).
MAX_IMAGE_EDGE = _env_int("LLM_MAX_IMAGE_EDGE", 1568)  # downscale longest edge (px); must also be in cache key
MAX_WORKERS = _env_int("LLM_MAX_WORKERS", 8)            # bounded concurrency — raise if provider RPM allows
ENABLE_PROMPT_CACHE = _env_bool("LLM_ENABLE_PROMPT_CACHE", True)   # ephemeral cache_control on system prompt
THINKING_ENABLED = _env_bool("LLM_THINKING_ENABLED", False)       # extended thinking (forces tool_choice=auto)
MAX_TOKENS = _env_int("LLM_MAX_TOKENS", 4096)                     # output cap for the structured tool response
MAX_RETRIES = _env_int("LLM_MAX_RETRIES", 4)                      # exponential-backoff retries on 429 / 5xx
REQUEST_TIMEOUT = _env_float("LLM_TIMEOUT_SECONDS", 30.0)         # seconds per request (tighter tail cap)


def model_id(key_or_id: str) -> str:
    """Resolve a short key ('opus') or a full id ('claude-opus-4-8') to a model id."""
    return MODELS.get(key_or_id, key_or_id)
