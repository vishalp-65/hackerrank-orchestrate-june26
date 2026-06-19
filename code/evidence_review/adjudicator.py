"""Deterministic adjudication: raw perception → the 14 output columns.

All policy lives here (not in the prompt), so the labeled-sample conventions are
encoded as auditable rules:

* evidence not met OR raw status NEI  ⟹  not_enough_information, issue/severity
  ``unknown``, supporting_image_ids ``none`` (the NEI cascade).
* valid_image is ``false`` only for authenticity problems (non-original / manipulated).
* user_history_risk  ⟹  manual_review_required (in :mod:`risk_rules`).
"""
from __future__ import annotations

import re

from . import schema
from .data_loader import ClaimRow
from .risk_rules import derive_risk_flags


def _b(v) -> str:
    return "true" if v else "false"


def _oneline(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _img_sort_key(image_id: str):
    try:
        return int(str(image_id).rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return 9999


def adjudicate(perception: dict, history: dict, claim: ClaimRow,
               loaded_image_ids: list[str]) -> dict:
    loaded = set(loaded_image_ids)
    obs = perception.get("per_image_observations") or []

    # ── evidence_standard_met ────────────────────────────────────────────────
    evidence_met = bool(perception.get("evidence_standard_assessment"))
    if obs and not any(o.get("is_relevant") for o in obs):
        evidence_met = False
    if not loaded:
        evidence_met = False

    claim_status_raw = schema.normalize_claim_status(perception.get("claim_status_raw"))
    is_nei = (not evidence_met) or (claim_status_raw == "not_enough_information")

    # ── valid_image (authenticity only) ──────────────────────────────────────
    non_original = any(o.get("is_original_photo") is False for o in obs)
    manipulated = any("possible_manipulation" in (o.get("quality_flags") or []) for o in obs)
    valid_image = not (non_original or manipulated)

    # ── object_part (visible first, else claimed) ────────────────────────────
    object_part = schema.normalize_object_part(
        perception.get("overall_visible_object_part"), claim.claim_object)
    if object_part == "unknown":
        claimed = schema.normalize_object_part(perception.get("claimed_part"), claim.claim_object)
        object_part = claimed if claimed != "unknown" else "unknown"

    # ── status-dependent fields ──────────────────────────────────────────────
    if is_nei:
        claim_status = "not_enough_information"
        issue_type = "unknown"
        severity = "unknown"
        supporting = "none"
    else:
        claim_status = claim_status_raw
        issue_type = schema.normalize_issue_type(perception.get("overall_visible_issue_type"))
        severity = schema.normalize_severity(perception.get("overall_severity"))
        ids = [str(i).strip() for i in (perception.get("supporting_image_ids") or [])]
        ids = [i for i in ids if i in loaded]
        if not ids:  # fall back to relevant images that actually show damage
            ids = [o.get("image_id") for o in obs
                   if o.get("image_id") in loaded and o.get("is_relevant") and o.get("damage_visible")]
            if not ids:
                ids = [o.get("image_id") for o in obs
                       if o.get("image_id") in loaded and o.get("is_relevant")]
        ids = sorted({i for i in ids if i}, key=_img_sort_key)
        supporting = ";".join(ids) if ids else "none"

    # ── risk_flags (depends on the final claim_status) ───────────────────────
    risk_flags = ";".join(derive_risk_flags(perception, history, claim_status))

    # ── free-text fields ─────────────────────────────────────────────────────
    evidence_reason = _oneline(perception.get("evidence_standard_reason")) or (
        "Assessed from the submitted images.")
    justification = _oneline(perception.get("claim_status_justification")) or (
        "Decision based on the visible image evidence.")
    if perception.get("text_instruction_detected") and "instruction" not in justification.lower():
        justification += " Instruction-like text in the submission was disregarded."

    return {
        "user_id": claim.user_id,
        "image_paths": claim.image_paths,
        "user_claim": claim.user_claim,
        "claim_object": claim.claim_object,
        "evidence_standard_met": _b(evidence_met),
        "evidence_standard_met_reason": evidence_reason,
        "risk_flags": risk_flags,
        "issue_type": issue_type,
        "object_part": object_part,
        "claim_status": claim_status,
        "claim_status_justification": justification,
        "supporting_image_ids": supporting,
        "valid_image": _b(valid_image),
        "severity": severity,
    }


def fallback_row(claim: ClaimRow, reason: str = "Automated review could not process this claim.") -> dict:
    """Safe NEI row used when a claim cannot be processed (e.g. repeated API failure)."""
    return {
        "user_id": claim.user_id,
        "image_paths": claim.image_paths,
        "user_claim": claim.user_claim,
        "claim_object": claim.claim_object,
        "evidence_standard_met": "false",
        "evidence_standard_met_reason": _oneline(reason),
        "risk_flags": "manual_review_required",
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": "The claim could not be assessed automatically and was routed for manual review.",
        "supporting_image_ids": "none",
        "valid_image": "false",
        "severity": "unknown",
    }
