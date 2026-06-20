"""Evaluation harness.

Runs the pipeline on the 20 labeled rows in dataset/sample_claims.csv for two model
configurations (Opus 4.8 and Sonnet 4.6 by default), scores both against the gold
labels, compares them, and writes evaluation/evaluation_report.md (with the required
operational analysis). Uses the same disk cache as production, so re-runs are free.

Operational stats (tokens / cost / latency) are persisted per config from the first
real (cache-miss) run, so a cached re-run still reports the true processing cost.

Usage:
    python code/evaluation/main.py [--models opus sonnet] [--no-cache] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # add code/ to path

from evidence_review import config
from evidence_review.data_loader import load_claims
from evidence_review.output_writer import write_output_csv
from evidence_review.pipeline import run_pipeline
from evaluation import metrics

CMP_FIELDS = ["evidence_standard_met", "valid_image", "claim_status", "issue_type",
              "object_part", "severity", "risk_flags", "supporting_image_ids"]


def _gold_distribution(golds: list[dict]) -> dict:
    flag_tokens = Counter()
    for g in golds:
        flag_tokens.update(metrics.parse_set(g.get("risk_flags")) or {"none"})
    return {
        "claim_object": dict(Counter(g["claim_object"] for g in golds)),
        "claim_status": dict(Counter(g["claim_status"] for g in golds)),
        "risk_flag_tokens": dict(flag_tokens.most_common()),
    }


def run_config(model_key: str, claims, golds, use_cache: bool) -> dict:
    model = config.model_id(model_key)
    print(f"\n=== Config: {model_key} ({model}) ===")
    preds, stats = run_pipeline(claims, model=model, use_cache=use_cache,
                                max_workers=config.MAX_WORKERS)
    write_output_csv(preds, config.EVALUATION_DIR / f"predictions_sample_{model_key}.csv")

    # Persist real operational stats; reuse them when this run was served from cache.
    sd = stats.to_dict()
    stats_path = config.EVALUATION_DIR / f"sample_stats_{model_key}.json"
    if stats.n_api_calls > 0 or not stats_path.is_file():
        stats_path.write_text(json.dumps(sd, indent=2), encoding="utf-8")
    else:
        try:
            sd = json.loads(stats_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    m = metrics.compute_all(preds, golds)
    diffs = metrics.disagreements(preds, golds, CMP_FIELDS)
    print(f"  composite={m['composite']:.3f}  claim_status_acc={m['claim_status_acc']:.3f}  "
          f"risk_f1={m['risk_flags_f1']:.3f}  est_cost=${sd['est_cost_usd']:.4f}")
    return {"model_key": model_key, "model": model, "metrics": m, "diffs": diffs,
            "stats": sd, "preds": preds, "golds": golds}


def _fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def build_report(results: list[dict], gold_dist: dict, n: int) -> str:
    winner = max(results, key=lambda r: r["metrics"]["composite"])
    L = []
    L.append("# Evaluation Report — Multi-Modal Evidence Review\n")
    L.append(f"Provider: **{results[0]['stats']['provider']}**  |  "
             f"Prompt version: **{config.PROMPT_VERSION}**  |  "
             f"Configs compared: {', '.join(r['model'] for r in results)}\n")

    L.append("## 1. Dataset (sample / dev split)\n")
    L.append(f"- Rows: **{n}** labeled claims from `dataset/sample_claims.csv`")
    L.append(f"- claim_object: {gold_dist['claim_object']}")
    L.append(f"- claim_status (gold): {gold_dist['claim_status']}")
    L.append(f"- risk-flag tokens (gold): {gold_dist['risk_flag_tokens']}\n")

    L.append("## 2. Configuration comparison\n")
    keys = [
        ("composite", "Composite score"),
        ("claim_status_acc", "claim_status accuracy"),
        ("claim_status_macro_f1", "claim_status macro-F1"),
        ("risk_flags_f1", "risk_flags micro-F1"),
        ("risk_flags_exact", "risk_flags exact-match"),
        ("evidence_standard_met_acc", "evidence_standard_met acc"),
        ("valid_image_acc", "valid_image acc"),
        ("issue_type_macro_f1", "issue_type macro-F1 (union labels)"),
        ("object_part_macro_f1", "object_part macro-F1 (union labels)"),
        ("severity_macro_f1", "severity macro-F1 (union labels)"),
        ("supporting_image_ids_exact", "supporting_image_ids exact"),
    ]
    L.append("| Metric | " + " | ".join(r["model_key"] for r in results) + " |")
    L.append("|" + "---|" * (len(results) + 1))
    for k, label in keys:
        L.append("| " + label + " | " + " | ".join(_fmt(r["metrics"][k]) for r in results) + " |")
    for label, key in [
        ("API calls (sample)", "n_api_calls"),
        ("Input tokens (sample)", "input_tokens"),
        ("Output tokens (sample)", "output_tokens"),
        ("Cache-read tokens (sample)", "cache_read_tokens"),
    ]:
        L.append("| " + label + " | " + " | ".join(str(r["stats"][key]) for r in results) + " |")
    L.append("| Est. cost USD (sample) | " + " | ".join(f"${r['stats']['est_cost_usd']:.4f}" for r in results) + " |")
    L.append("| Wall time s (sample) | " + " | ".join(f"{r['stats']['wall_time_s']:.1f}" for r in results) + " |")
    L.append(f"\n**Selected configuration: `{winner['model']}`** "
             f"(highest composite = {winner['metrics']['composite']:.3f}).\n")

    L.append(f"## 3. Per-field accuracy — {winner['model_key']}\n")
    wm = winner["metrics"]
    L.append("| Field | Accuracy |")
    L.append("|---|---|")
    for f in ["evidence_standard_met_acc", "valid_image_acc", "claim_status_acc",
              "issue_type_acc", "object_part_acc", "severity_acc"]:
        L.append(f"| {f.replace('_acc','')} | {wm[f]:.3f} |")
    L.append(f"| risk_flags (micro-F1 / exact) | {wm['risk_flags_f1']:.3f} / {wm['risk_flags_exact']:.3f} |")
    L.append(f"| supporting_image_ids (F1 / exact) | {wm['supporting_image_ids_f1']:.3f} / {wm['supporting_image_ids_exact']:.3f} |\n")

    # Confusion matrices for the two weakest fields
    L.append(f"### Confusion matrices — {winner['model_key']}\n")
    wp, wg = winner.get("preds", []), winner.get("golds", [])
    if wp and wg:
        for field_label in [("issue_type", "issue\\_type"), ("severity", "severity")]:
            fkey, flabel = field_label
            L.append(f"**{flabel}** (rows=gold, cols=pred):\n")
            L.append(metrics.confusion_markdown(wp, wg, fkey))
            L.append("")
    L.append("")

    L.append(f"## 4. Error analysis — {winner['model_key']} ({len(winner['diffs'])} rows with diffs)\n")
    if not winner["diffs"]:
        L.append("No field-level disagreements with gold.\n")
    for d in winner["diffs"]:
        parts = "; ".join(f"{f}: pred=`{v['pred']}` vs gold=`{v['gold']}`"
                          for f, v in d["diffs"].items())
        L.append(f"- row {d['row']} ({d['user_id']}, {d['claim_object']}): {parts}")
    L.append("")

    L.append("## 5. Operational analysis\n")
    L.append("Pricing assumptions (USD per 1M tokens): "
             + ", ".join(f"{m} = ${i}/{o}" for m, (i, o) in config.PRICING.items())
             + ". Cache economics: fresh input ×1.0, cache write ×1.25, cache read ×0.1. "
             "Image tokens ≈ width×height/750; images are downscaled to a "
             f"{config.MAX_IMAGE_EDGE}px long edge before sending.\n")
    for r in results:
        s = r["stats"]
        no_cache_cost = s.get("est_cost_no_cache_usd", "n/a")
        L.append(f"### Sample run — {r['model']}")
        L.append(f"- Model calls: {s['n_api_calls']} (cache hits: {s['n_cache_hits']}); "
                 f"fallback rows: {s['n_fallback_rows']}")
        L.append(f"- Tokens — input {s['input_tokens']}, output {s['output_tokens']}, "
                 f"cache-read {s['cache_read_tokens']}, cache-write {s['cache_creation_tokens']}")
        L.append(f"- Images processed: {s['images_processed']} "
                 f"(missing {s['images_missing']}, undecodable {s.get('images_undecodable', 0)}); "
                 f"source formats: {s['format_counts']}")
        L.append(f"- Est. cost: ${s['est_cost_usd']:.4f} "
                 f"(without prompt-caching: ${no_cache_cost}); "
                 f"wall {s['wall_time_s']:.1f}s; "
                 f"per-call latency p50 {s['latency_p50_s']:.1f}s / p95 {s['latency_p95_s']:.1f}s\n")

    if config.TEST_RUN_STATS.is_file():
        ts = json.loads(config.TEST_RUN_STATS.read_text(encoding="utf-8"))
        no_cache_ts = ts.get("est_cost_no_cache_usd", "n/a")
        L.append("### Test run — actuals (dataset/claims.csv → output.csv)")
        if ts.get("model") != winner["model"]:
            L.append(f"> **NOTE:** production run used `{ts['model']}` ({ts['provider']}) "
                     f"but eval winner is `{winner['model']}`. Numbers below reflect the actual "
                     "production run that generated output.csv.")
        L.append(f"- Model: {ts['model']} ({ts['provider']}); claims: {ts['n_claims']}; "
                 f"API calls: {ts['n_api_calls']}; cache hits: {ts['n_cache_hits']}; "
                 f"fallbacks: {ts['n_fallback_rows']}")
        L.append(f"- Tokens — input {ts['input_tokens']}, output {ts['output_tokens']}, "
                 f"cache-read {ts['cache_read_tokens']}, cache-write {ts['cache_creation_tokens']}")
        L.append(f"- Images processed: {ts['images_processed']} "
                 f"(missing {ts['images_missing']}, undecodable {ts.get('images_undecodable', 0)}); "
                 f"formats: {ts['format_counts']}")
        L.append(f"- Est. cost: ${ts['est_cost_usd']} "
                 f"(without prompt-caching: ${no_cache_ts}); "
                 f"wall {ts['wall_time_s']}s; "
                 f"latency p50 {ts['latency_p50_s']}s / p95 {ts['latency_p95_s']}s\n")
    else:
        s = winner["stats"]
        calls = max(1, s["n_api_calls"])
        in_per, out_per = s["input_tokens"] / calls, s["output_tokens"] / calls
        pin, pout = config.PRICING[winner["model"]]
        proj = (44 * in_per * pin + 44 * out_per * pout) / 1_000_000
        L.append("### Test run — projection (44 claims; main.py not yet run)")
        L.append(f"- ~44 model calls on {winner['model']}; est. cost ≈ ${proj:.3f} "
                 f"(~{in_per:.0f} input + {out_per:.0f} output tokens/call).\n")

    L.append("### TPM/RPM, batching, throttling, caching, retries\n")
    L.append(f"- **Batching/throttling:** bounded `ThreadPoolExecutor(max_workers={config.MAX_WORKERS})` "
             "keeps concurrent requests well under provider RPM limits; the modest token volume "
             "(~thousands of input tokens/call) stays far below typical TPM ceilings.")
    L.append("- **Caching:** (1) a content-addressed **disk cache** of raw perception keyed by "
             "inputs+prompt_version+model — re-runs and policy edits cost **zero** API calls; "
             "(2) **prompt caching** (ephemeral `cache_control`) on the static system+requirements "
             "prefix, served from cache after the first call (see cache-read tokens above).")
    L.append(f"- **Retries:** exponential backoff with jitter (honors `Retry-After` header on 429) — "
             f"up to {config.MAX_RETRIES + 1} attempts ({config.MAX_RETRIES} retries); "
             "non-429 4xx surface immediately. A claim that "
             "still fails is written as a safe not_enough_information fallback row so the batch completes.")
    L.append("- **Determinism:** no `temperature`/`thinking` (rejected on Opus 4.8); all policy is in "
             "the deterministic adjudicator, so output is reproducible from the cache.\n")

    L.append("## 6. Recommendation\n")
    others = [r for r in results if r is not winner]
    delta = winner["metrics"]["composite"] - max((r["metrics"]["composite"] for r in others), default=0.0)
    L.append(f"Use **`{winner['model']}`** for the production run — highest composite "
             f"({winner['metrics']['composite']:.3f}"
             + (f", +{delta:.3f} over the next config" if others else "")
             + "). The perception/adjudication split means the chosen model can be swapped without "
             "touching the policy layer.\n")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate + compare configurations on the sample set.")
    ap.add_argument("--models", nargs="+", default=[config.DEFAULT_MODEL_KEY],
                    help="model keys/ids to evaluate (default: the configured LLM_MODEL). "
                         "Pass multiple to compare, e.g. --models opus sonnet")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sample = load_claims(config.SAMPLE_CLAIMS_CSV, config.DATASET_DIR, with_expected=True)
    if args.limit:
        sample = sample[: args.limit]
    golds = [c.expected for c in sample]
    gold_dist = _gold_distribution(golds)

    results = [run_config(mk, sample, golds, use_cache=not args.no_cache) for mk in args.models]

    report = build_report(results, gold_dist, len(sample))
    out = config.EVALUATION_DIR / "evaluation_report.md"
    out.write_text(report, encoding="utf-8")

    print("\n" + "=" * 72)
    print("Comparison (composite):",
          ", ".join(f"{r['model_key']}={r['metrics']['composite']:.3f}" for r in results))
    print(f"Report written -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
