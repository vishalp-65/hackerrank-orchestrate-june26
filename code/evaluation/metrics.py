"""Scoring metrics for predictions vs. the labeled sample set (stdlib only).

Predictions and gold rows are aligned by position (both in sample-file order).
"""
from __future__ import annotations

from collections import Counter


def _norm(v: str) -> str:
    return str(v or "").strip().lower()


def parse_set(value: str) -> set[str]:
    """Semicolon list → set; 'none'/'' is the empty set."""
    v = _norm(value)
    if v in ("", "none"):
        return set()
    return {t.strip() for t in v.split(";") if t.strip()}


def field_accuracy(preds: list[dict], golds: list[dict], field: str) -> float:
    if not preds:
        return 0.0
    hits = sum(_norm(p.get(field)) == _norm(g.get(field)) for p, g in zip(preds, golds))
    return hits / len(preds)


def macro_f1(preds: list[dict], golds: list[dict], field: str) -> float:
    pairs = [(_norm(p.get(field)), _norm(g.get(field))) for p, g in zip(preds, golds)]
    # Use union of predicted AND gold labels so over-predicted classes are penalised.
    labels = ({p for p, _ in pairs} | {g for _, g in pairs}) - {""}
    if not labels:
        return 0.0
    f1s = []
    for lab in labels:
        tp = sum(p == lab and g == lab for p, g in pairs)
        fp = sum(p == lab and g != lab for p, g in pairs)
        fn = sum(p != lab and g == lab for p, g in pairs)
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom else 0.0)
    return sum(f1s) / len(f1s)


def set_metrics(preds: list[dict], golds: list[dict], field: str) -> dict:
    """Micro precision/recall/F1 over set tokens, plus exact-set-match rate."""
    tp = fp = fn = exact = 0
    for p, g in zip(preds, golds):
        ps, gs = parse_set(p.get(field)), parse_set(g.get(field))
        tp += len(ps & gs)
        fp += len(ps - gs)
        fn += len(gs - ps)
        exact += (ps == gs)
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1,
            "exact_match": exact / len(preds) if preds else 0.0}


def confusion(preds: list[dict], golds: list[dict], field: str) -> Counter:
    """Return Counter keyed by (gold_norm, pred_norm) for confusion-matrix analysis."""
    return Counter((_norm(g.get(field)), _norm(p.get(field))) for p, g in zip(preds, golds))


def confusion_markdown(preds: list[dict], golds: list[dict], field: str) -> str:
    """Render a compact markdown confusion matrix for the given field."""
    matrix = confusion(preds, golds, field)
    all_labels = sorted({lab for pair in matrix for lab in pair} - {""})
    if not all_labels:
        return f"*No data for {field}*"
    header = "| gold \\ pred | " + " | ".join(all_labels) + " |"
    sep = "|---|" + "---|" * len(all_labels)
    rows = [header, sep]
    for gold in all_labels:
        cells = [str(matrix.get((gold, pred), 0)) for pred in all_labels]
        rows.append(f"| {gold} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


# composite weights (must sum to 1.0)
WEIGHTS = {
    "claim_status_macro_f1": 0.25,
    "risk_flags_f1": 0.20,
    "evidence_standard_met_acc": 0.15,
    "issue_type_macro_f1": 0.15,
    "severity_macro_f1": 0.05,    # trimmed 0.10→0.05 to make room for valid_image
    "supporting_image_ids_exact": 0.10,
    "object_part_macro_f1": 0.05,
    "valid_image_acc": 0.05,       # authenticity gate — was missing from composite
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, f"WEIGHTS must sum to 1.0, got {sum(WEIGHTS.values())}"


def compute_all(preds: list[dict], golds: list[dict]) -> dict:
    m = {
        "n": len(preds),
        # exact-match accuracy
        "evidence_standard_met_acc": field_accuracy(preds, golds, "evidence_standard_met"),
        "valid_image_acc": field_accuracy(preds, golds, "valid_image"),
        "claim_status_acc": field_accuracy(preds, golds, "claim_status"),
        "issue_type_acc": field_accuracy(preds, golds, "issue_type"),
        "object_part_acc": field_accuracy(preds, golds, "object_part"),
        "severity_acc": field_accuracy(preds, golds, "severity"),
        # macro-F1 (union of predicted+gold labels so over-predicted classes are penalised)
        "claim_status_macro_f1": macro_f1(preds, golds, "claim_status"),
        "issue_type_macro_f1": macro_f1(preds, golds, "issue_type"),
        "object_part_macro_f1": macro_f1(preds, golds, "object_part"),
        "severity_macro_f1": macro_f1(preds, golds, "severity"),
    }
    rf = set_metrics(preds, golds, "risk_flags")
    sup = set_metrics(preds, golds, "supporting_image_ids")
    m["risk_flags_f1"] = rf["f1"]
    m["risk_flags_exact"] = rf["exact_match"]
    m["supporting_image_ids_f1"] = sup["f1"]
    m["supporting_image_ids_exact"] = sup["exact_match"]
    m["composite"] = sum(w * m[k] for k, w in WEIGHTS.items())
    return m


_SET_FIELDS = {"risk_flags", "supporting_image_ids"}


def _differs(field: str, pv, gv) -> bool:
    if field in _SET_FIELDS:  # order-independent comparison for set-valued fields
        return parse_set(pv) != parse_set(gv)
    return _norm(pv) != _norm(gv)


def disagreements(preds: list[dict], golds: list[dict], fields: list[str]) -> list[dict]:
    """Per-row field-level mismatches for error analysis."""
    out = []
    for i, (p, g) in enumerate(zip(preds, golds)):
        diffs = {f: {"pred": p.get(f), "gold": g.get(f)}
                 for f in fields if _differs(f, p.get(f), g.get(f))}
        if diffs:
            out.append({"row": i, "user_id": g.get("user_id"),
                        "claim_object": g.get("claim_object"), "diffs": diffs})
    return out
