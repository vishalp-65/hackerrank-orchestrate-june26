"""Risk-flag derivation and the co-occurrence policy.

Risk flags come from three sources — per-image quality, claim/authenticity signals,
and the user_history.csv ``history_flags`` column — unioned, then reconciled with the
policy observed in the labeled samples.
"""
from __future__ import annotations

from . import schema

# Flags that describe *evidence insufficiency*. The labeled samples only attach these
# to non-supported claims, so they are suppressed when the claim is supported (a single
# weak/extra image should not flag a claim whose evidence is otherwise clear).
INSUFFICIENCY_FLAGS = frozenset({
    "damage_not_visible", "wrong_angle", "cropped_or_obstructed",
    "wrong_object", "wrong_object_part",
})


def derive_risk_flags(perception: dict, history: dict, claim_status: str) -> list[str]:
    flags: set[str] = set()

    # 1) Per-image quality + authenticity.
    for obs in perception.get("per_image_observations", []) or []:
        for raw in obs.get("quality_flags", []) or []:
            norm = schema.normalize_risk_flag(raw)
            if norm and norm != "none":
                flags.add(norm)
        if obs.get("is_original_photo") is False:
            flags.add("non_original_image")

    # 2) Instruction injection (conversation or in-image text).
    if perception.get("text_instruction_detected"):
        flags.add("text_instruction_present")

    # 3) Claim-vs-evidence relationship.
    match = (perception.get("claim_match_assessment") or "").strip().lower()
    if match == "mismatch":
        flags.add("claim_mismatch")
    elif match == "no_evidence":
        flags.add("damage_not_visible")

    # 4) User-history flags (verbatim from the history row; never derived from raw
    #    stats so as not to over-flag relative to the labeled convention).
    for raw in (history.get("history_flags") or "none").split(";"):
        norm = schema.normalize_risk_flag(raw)
        if norm and norm != "none":
            flags.add(norm)

    # Policy: a supported claim does not carry evidence-insufficiency flags.
    if claim_status == "supported":
        flags -= INSUFFICIENCY_FLAGS

    # Policy: a flagged user history always implies manual review.
    if "user_history_risk" in flags:
        flags.add("manual_review_required")

    return sorted(flags) if flags else ["none"]
