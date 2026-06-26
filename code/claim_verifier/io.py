"""CSV IO and atomic output writing."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

from claim_verifier.constants import OUTPUT_COLUMNS
from claim_verifier.models import Claim, ClaimDecision


def read_claims(path: str | Path, limit: int | None = None) -> list[Claim]:
    claims = []
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            raw_paths = [item.strip() for item in row["image_paths"].split(";") if item.strip()]
            image_paths = [_resolve_image_path(csv_path.parent, item) for item in raw_paths]
            claims.append(
                Claim(
                    row_index=index,
                    user_id=row["user_id"],
                    image_paths=image_paths,
                    image_paths_raw=row["image_paths"],
                    user_claim=row["user_claim"],
                    claim_object=row["claim_object"],
                )
            )
            if limit is not None and len(claims) >= limit:
                break
    return claims


def _resolve_image_path(csv_dir: Path, raw_path: str) -> str:
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return str(path)
    dataset_relative = csv_dir / path
    return str(dataset_relative)


def write_output_atomic(path: str | Path, decisions: list[ClaimDecision]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        newline="",
        encoding="utf-8",
        dir=str(destination.parent or Path(".")),
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for decision in decisions:
            writer.writerow(decision.to_csv_row())
    os.replace(tmp_path, destination)
