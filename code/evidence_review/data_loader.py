"""CSV loading, image-path resolution, and per-claim context assembly."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from . import config


@dataclass
class ClaimRow:
    user_id: str
    image_paths: str            # raw semicolon-joined string (echoed verbatim to output)
    user_claim: str
    claim_object: str
    # (image_id, absolute_path, exists) tuples, in submission order
    images: list[tuple[str, Path, bool]] = field(default_factory=list)
    expected: dict | None = None  # gold output columns (sample set only)


def _read_dicts(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def resolve_image_path(relative_path: str, dataset_dir: Path = config.DATASET_DIR) -> Path:
    """Resolve a dataset-relative path like 'images/test/case_001/img_1.jpg'."""
    rel = relative_path.strip().replace("\\", "/")
    return (dataset_dir / rel).resolve()


def _parse_images(image_paths: str, dataset_dir: Path) -> list[tuple[str, Path, bool]]:
    out: list[tuple[str, Path, bool]] = []
    for p in (s for s in image_paths.split(";") if s.strip()):
        abs_path = resolve_image_path(p, dataset_dir)
        image_id = abs_path.stem  # filenames are already img_1, img_2, ...
        out.append((image_id, abs_path, abs_path.is_file()))
    return out


def load_claims(path: Path = config.CLAIMS_CSV,
                dataset_dir: Path = config.DATASET_DIR,
                with_expected: bool = False) -> list[ClaimRow]:
    rows: list[ClaimRow] = []
    for r in _read_dicts(path):
        claim = ClaimRow(
            user_id=r["user_id"].strip(),
            image_paths=r["image_paths"].strip(),
            user_claim=r["user_claim"],
            claim_object=r["claim_object"].strip().lower(),
            images=_parse_images(r["image_paths"], dataset_dir),
        )
        if with_expected:
            from .schema import OUTPUT_COLUMNS
            claim.expected = {c: r[c] for c in OUTPUT_COLUMNS if c in r}
        rows.append(claim)
    return rows


def load_user_history(path: Path = config.USER_HISTORY_CSV) -> dict[str, dict]:
    return {r["user_id"].strip(): r for r in _read_dicts(path)}


def default_history(user_id: str) -> dict:
    """Neutral history for a user absent from user_history.csv."""
    return {
        "user_id": user_id, "past_claim_count": "0", "accept_claim": "0",
        "manual_review_claim": "0", "rejected_claim": "0", "last_90_days_claim_count": "0",
        "history_flags": "none", "history_summary": "No prior history on record.",
    }


def load_evidence_requirements(path: Path = config.EVIDENCE_REQUIREMENTS_CSV) -> list[dict]:
    return _read_dicts(path)


def relevant_requirements(reqs: list[dict], claim_object: str) -> list[dict]:
    """Requirements applying to this object type plus the 'all' rows."""
    co = claim_object.strip().lower()
    return [r for r in reqs if r.get("claim_object", "").strip().lower() in (co, "all")]
