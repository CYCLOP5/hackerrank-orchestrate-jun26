"""Evaluation metrics for sample predictions."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from claim_verifier.constants import OUTPUT_COLUMNS

LIST_FIELDS = {"risk_flags", "supporting_image_ids"}
WEIGHTS = {
    "claim_status": 5.0,
    "evidence_standard_met": 3.0,
    "issue_type": 2.0,
    "object_part": 2.0,
    "supporting_image_ids": 2.0,
    "risk_flags": 1.0,
    "valid_image": 1.0,
    "severity": 1.0,
}


def evaluate_predictions(
    expected_path: str | Path,
    predictions_path: str | Path,
    report_path: str | Path | None = None,
    mismatches_path: str | Path | None = None,
) -> dict[str, Any]:
    expected = _read_rows(expected_path)
    predicted = _read_rows(predictions_path)
    if len(expected) != len(predicted):
        raise ValueError(f"Row count mismatch: expected {len(expected)}, got {len(predicted)}")

    field_scores: dict[str, list[float]] = {field: [] for field in OUTPUT_COLUMNS}
    mismatches: list[dict[str, str]] = []

    for row_index, (exp, pred) in enumerate(zip(expected, predicted, strict=True), start=1):
        for field in OUTPUT_COLUMNS:
            score = _score_field(field, exp[field], pred[field])
            field_scores[field].append(score)
            if score < 1.0:
                mismatches.append(
                    {
                        "row_index": str(row_index),
                        "field": field,
                        "expected": exp[field],
                        "predicted": pred[field],
                        "score": f"{score:.4f}",
                    }
                )

    field_summary = {
        field: sum(scores) / len(scores) if scores else 0.0
        for field, scores in field_scores.items()
    }
    weighted_total = 0.0
    weight_sum = 0.0
    for field, score in field_summary.items():
        weight = WEIGHTS.get(field, 0.5)
        weighted_total += score * weight
        weight_sum += weight

    report = {
        "row_count": len(expected),
        "field_scores": field_summary,
        "headline_score": weighted_total / weight_sum if weight_sum else 0.0,
        "mismatch_count": len(mismatches),
    }
    if report_path:
        Path(report_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if mismatches_path:
        _write_mismatches(mismatches_path, mismatches)
    return report


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    missing = [field for field in OUTPUT_COLUMNS if rows and field not in rows[0]]
    if missing:
        raise ValueError(f"Missing output columns in {path}: {missing}")
    return rows


def _score_field(field: str, expected: str, predicted: str) -> float:
    if field in LIST_FIELDS:
        return _jaccard(_split_semicolon(expected), _split_semicolon(predicted))
    return 1.0 if _normalize_scalar(expected) == _normalize_scalar(predicted) else 0.0


def _split_semicolon(value: str) -> set[str]:
    if not value or value.strip() == "none":
        return set()
    return {item.strip() for item in value.split(";") if item.strip() and item.strip() != "none"}


def _jaccard(expected: set[str], predicted: set[str]) -> float:
    if not expected and not predicted:
        return 1.0
    union = expected | predicted
    return len(expected & predicted) / len(union)


def _normalize_scalar(value: str) -> str:
    return value.strip().lower()


def _write_mismatches(path: str | Path, rows: list[dict[str, str]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_index", "field", "expected", "predicted", "score"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
