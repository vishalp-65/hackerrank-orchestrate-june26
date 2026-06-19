"""Batch orchestration: load context, call the model (or cache), adjudicate, collect.

Per-claim work runs in a bounded thread pool. A claim that fails after retries is
written as a safe not_enough_information fallback row rather than aborting the batch.
"""
from __future__ import annotations

import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from . import cache, config, prompts, schema
from .adjudicator import adjudicate, fallback_row
from .data_loader import (ClaimRow, default_history, load_evidence_requirements,
                          load_user_history, relevant_requirements)
from .image_utils import normalize_to_jpeg_b64
from .llm_client import call_review, provider_name


@dataclass
class RunStats:
    model: str = ""
    provider: str = ""
    n_claims: int = 0
    n_api_calls: int = 0
    n_cache_hits: int = 0
    n_fallback_rows: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    images_processed: int = 0
    images_missing: int = 0
    format_counts: Counter = field(default_factory=Counter)
    latencies: list[float] = field(default_factory=list)
    wall_time_s: float = 0.0

    def cost_usd(self) -> float:
        # Anthropic cache economics: fresh input 1.0x, cache write 1.25x, cache read 0.1x.
        pin, pout = config.PRICING.get(self.model, (0.0, 0.0))
        input_equiv = (self.input_tokens
                       + 1.25 * self.cache_creation_tokens
                       + 0.1 * self.cache_read_tokens)
        return (input_equiv * pin + self.output_tokens * pout) / 1_000_000

    def to_dict(self) -> dict:
        lat = sorted(self.latencies)
        p50 = lat[len(lat) // 2] if lat else 0.0
        p95 = lat[min(int(len(lat) * 0.95), len(lat) - 1)] if lat else 0.0
        return {
            "model": self.model, "provider": self.provider, "n_claims": self.n_claims,
            "n_api_calls": self.n_api_calls, "n_cache_hits": self.n_cache_hits,
            "n_fallback_rows": self.n_fallback_rows, "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens, "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "images_processed": self.images_processed, "images_missing": self.images_missing,
            "format_counts": dict(self.format_counts), "wall_time_s": round(self.wall_time_s, 2),
            "latency_p50_s": round(p50, 2), "latency_p95_s": round(p95, 2),
            "est_cost_usd": round(self.cost_usd(), 4),
        }


@dataclass
class _Ctx:
    client: object
    model: str
    system_blocks: list
    tool: dict
    history_map: dict
    reqs: list
    use_cache: bool
    prompt_version: str


def _load_images(claim: ClaimRow):
    images, missing_ids, src_formats = [], [], []
    for image_id, path, exists in claim.images:
        if not exists:
            missing_ids.append(image_id)
            continue
        norm = normalize_to_jpeg_b64(path)
        if norm is None:
            missing_ids.append(image_id)
            continue
        norm["image_id"] = image_id
        images.append(norm)
        src_formats.append(norm["src_format"])
    return images, missing_ids, src_formats


def _process(claim: ClaimRow, ctx: _Ctx) -> tuple[dict, dict]:
    """Returns (output_row, meta). Never raises — failures become fallback rows."""
    history = ctx.history_map.get(claim.user_id) or default_history(claim.user_id)
    meta = {"cache_hit": False, "api_call": False, "latency": 0.0,
            "images": 0, "missing": 0, "src_formats": [], "usage": None, "error": None}

    key = cache.cache_key(claim, ctx.prompt_version, ctx.model)
    cached = cache.load(key) if ctx.use_cache else None

    try:
        if cached is not None:
            perception = cached["perception"]
            loaded_ids = cached.get("loaded_image_ids", [])
            meta["src_formats"] = cached.get("src_formats", [])
            meta["images"] = len(loaded_ids)
            meta["missing"] = cached.get("missing", 0)
            meta["cache_hit"] = True
        else:
            images, missing_ids, src_formats = _load_images(claim)
            loaded_ids = [img["image_id"] for img in images]
            relevant = relevant_requirements(ctx.reqs, claim.claim_object)
            user_content = prompts.build_user_content(
                claim, history, relevant, images, missing_ids)
            t0 = time.perf_counter()
            perception, usage = call_review(
                ctx.client, ctx.model, ctx.system_blocks, user_content, ctx.tool)
            meta["latency"] = time.perf_counter() - t0
            meta["api_call"] = True
            meta["usage"] = usage
            meta["images"] = len(images)
            meta["missing"] = len(missing_ids)
            meta["src_formats"] = src_formats
            if ctx.use_cache:
                cache.save(key, {"perception": perception, "loaded_image_ids": loaded_ids,
                                 "src_formats": src_formats, "missing": len(missing_ids)})
        row = adjudicate(perception, history, claim, loaded_ids)
        return row, meta
    except Exception as exc:  # noqa: BLE001 — robustness: never abort the batch
        meta["error"] = f"{type(exc).__name__}: {exc}"
        return fallback_row(claim), meta


def run_pipeline(claims: list[ClaimRow], *, model: str, use_cache: bool = True,
                 max_workers: int = config.MAX_WORKERS,
                 prompt_version: str = config.PROMPT_VERSION,
                 client=None, progress: bool = True) -> tuple[list[dict], RunStats]:
    from .llm_client import make_client
    ctx = _Ctx(
        client=client or make_client(),
        model=model,
        system_blocks=prompts.build_system(load_evidence_requirements()),
        tool=schema.build_tool(),
        history_map=load_user_history(),
        reqs=load_evidence_requirements(),
        use_cache=use_cache,
        prompt_version=prompt_version,
    )
    stats = RunStats(model=model, provider=provider_name(), n_claims=len(claims))
    results: list[dict | None] = [None] * len(claims)

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process, claim, ctx): i for i, claim in enumerate(claims)}
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            row, meta = fut.result()
            results[i] = row
            # accumulate stats (single thread here)
            if meta["error"]:
                stats.n_fallback_rows += 1
            if meta["cache_hit"]:
                stats.n_cache_hits += 1
            if meta["api_call"]:
                stats.n_api_calls += 1
                stats.latencies.append(meta["latency"])
                u = meta["usage"] or {}
                stats.input_tokens += u.get("input_tokens", 0)
                stats.output_tokens += u.get("output_tokens", 0)
                stats.cache_read_tokens += u.get("cache_read_input_tokens", 0)
                stats.cache_creation_tokens += u.get("cache_creation_input_tokens", 0)
            stats.images_processed += meta["images"]
            stats.images_missing += meta["missing"]
            stats.format_counts.update(meta["src_formats"])
            done += 1
            if progress:
                flag = "cache" if meta["cache_hit"] else ("ERR" if meta["error"] else "api")
                print(f"  [{done:>3}/{len(claims)}] {claims[i].user_id:<10} "
                      f"{claims[i].claim_object:<8} {results[i]['claim_status']:<22} ({flag})",
                      flush=True)
    stats.wall_time_s = time.perf_counter() - t_start
    return [r for r in results if r is not None], stats
