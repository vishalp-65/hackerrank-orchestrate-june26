"""Central configuration: paths, model registry, pricing, and tunables.

All paths are derived from this file's location so the package works regardless of
the current working directory.
"""
from __future__ import annotations

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

# ── Models ───────────────────────────────────────────────────────────────────
MODELS = {
    "opus": "claude-opus-4-8",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5",
}
DEFAULT_MODEL_KEY = "opus"

# USD per 1,000,000 tokens: (input, output).
PRICING = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

# ── Tunables ─────────────────────────────────────────────────────────────────
PROMPT_VERSION = "v4"          # v4: precise 4-case contradicted def + authenticity/crack/severity calibration
ENABLE_PROMPT_CACHE = True     # attach ephemeral cache_control to the system prompt
MAX_IMAGE_EDGE = 1568          # downscale longest edge (px) — caps image-token cost
JPEG_QUALITY = 85              # re-encode quality for normalized images
MAX_TOKENS = 2500              # output cap for the structured tool response
MAX_WORKERS = 4                # bounded concurrency (Foundry rate-limit friendly)
MAX_RETRIES = 5                # exponential-backoff retries on 429 / 5xx
REQUEST_TIMEOUT = 120.0        # seconds per request


def model_id(key_or_id: str) -> str:
    """Resolve a short key ('opus') or a full id ('claude-opus-4-8') to a model id."""
    return MODELS.get(key_or_id, key_or_id)
