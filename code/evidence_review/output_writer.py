"""CSV writer for the 14-column prediction schema."""
from __future__ import annotations

import csv
from pathlib import Path

from .schema import OUTPUT_COLUMNS

# LLM-authored or user-supplied free-text that could contain spreadsheet formulas.
_FREE_TEXT_COLUMNS = frozenset({
    "user_claim", "evidence_standard_met_reason",
    "claim_status_justification", "image_paths",
})
_FORMULA_CHARS = frozenset("=+-@\t\r")


def _sanitize(value: str) -> str:
    """Prefix a tab to cells whose first non-space char is a formula trigger."""
    s = str(value) if value else ""
    stripped = s.lstrip()
    if stripped and stripped[0] in _FORMULA_CHARS:
        return "\t" + s
    return s


def write_output_csv(rows: list[dict], path: Path) -> None:
    """Write rows in the exact schema order, quoting every field (RFC 4180)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(OUTPUT_COLUMNS), quoting=csv.QUOTE_ALL,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                c: (_sanitize(row.get(c, "")) if c in _FREE_TEXT_COLUMNS else row.get(c, ""))
                for c in OUTPUT_COLUMNS
            })
