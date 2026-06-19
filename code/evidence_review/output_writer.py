"""CSV writer for the 14-column prediction schema."""
from __future__ import annotations

import csv
from pathlib import Path

from .schema import OUTPUT_COLUMNS


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
            writer.writerow({c: row.get(c, "") for c in OUTPUT_COLUMNS})
