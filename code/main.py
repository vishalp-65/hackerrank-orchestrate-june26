"""Entry point: run the evidence-review pipeline over dataset/claims.csv → output.csv.

Usage:
    python code/main.py [--model opus|sonnet|haiku] [--workers N] [--no-cache]
                        [--input PATH] [--output PATH] [--limit N]

Credentials are read from the environment (Azure AI Foundry or first-party Anthropic).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make `evidence_review` importable

from evidence_review import config
from evidence_review.data_loader import load_claims
from evidence_review.output_writer import write_output_csv
from evidence_review.pipeline import run_pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-Modal Evidence Review — generate output.csv")
    ap.add_argument("--model", default=config.DEFAULT_MODEL_KEY,
                    help="opus | sonnet | haiku, or a full model id "
                         f"(default from LLM_MODEL env: {config.DEFAULT_MODEL_KEY})")
    ap.add_argument("--workers", type=int, default=config.MAX_WORKERS)
    ap.add_argument("--no-cache", action="store_true", help="ignore the on-disk perception cache")
    ap.add_argument("--input", default=str(config.CLAIMS_CSV))
    ap.add_argument("--output", default=str(config.DEFAULT_OUTPUT_CSV))
    ap.add_argument("--dataset-dir", default=str(config.DATASET_DIR))
    ap.add_argument("--limit", type=int, default=0, help="process only the first N rows (debug)")
    args = ap.parse_args()

    model = config.model_id(args.model)
    claims = load_claims(Path(args.input), Path(args.dataset_dir))
    if args.limit:
        claims = claims[: args.limit]

    print("Multi-Modal Evidence Review")
    print(f"  model={model}  claims={len(claims)}  workers={args.workers}  "
          f"cache={'off' if args.no_cache else 'on'}")
    print(f"  input={args.input}")
    print("-" * 72)

    rows, stats = run_pipeline(
        claims, model=model, use_cache=not args.no_cache, max_workers=args.workers)

    out_path = Path(args.output)
    write_output_csv(rows, out_path)
    # Mirror to dataset/output.csv (where the starter header lives).
    try:
        shutil.copyfile(out_path, config.DATASET_OUTPUT_CSV)
    except Exception:
        pass

    # Persist run stats for the operational analysis. Do NOT clobber real stats with a
    # cache-only re-run (which legitimately spends 0 tokens).
    stats_json = {**stats.to_dict(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if stats.n_api_calls > 0 or not config.TEST_RUN_STATS.exists():
        config.TEST_RUN_STATS.parent.mkdir(parents=True, exist_ok=True)
        with open(config.TEST_RUN_STATS, "w", encoding="utf-8") as f:
            json.dump(stats_json, f, indent=2)
    fmt = stats_json["format_counts"]
    p50, p95 = stats_json["latency_p50_s"], stats_json["latency_p95_s"]

    print("-" * 72)
    print(f"Wrote {len(rows)} rows -> {out_path}")
    print(f"  API calls={stats.n_api_calls}  cache_hits={stats.n_cache_hits}  "
          f"fallbacks={stats.n_fallback_rows}")
    print(f"  images={stats.images_processed} (missing={stats.images_missing})  formats={fmt}")
    print(f"  tokens in={stats.input_tokens} out={stats.output_tokens} "
          f"cache_read={stats.cache_read_tokens} cache_write={stats.cache_creation_tokens}")
    print(f"  est_cost=${stats.cost_usd():.4f}  wall={stats.wall_time_s:.1f}s  "
          f"p50={p50:.1f}s p95={p95:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
