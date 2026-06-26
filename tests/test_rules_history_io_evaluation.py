import csv
from pathlib import Path

from claim_verifier.evaluation import evaluate_predictions
from claim_verifier.history import UserHistory
from claim_verifier.io import read_claims, write_output_atomic
from claim_verifier.models import ClaimDecision
from claim_verifier.rules import EvidenceRules
from claim_verifier.focus import build_claimed_focus


def test_evidence_rules_select_general_plus_matching_object_rules():
    claim = read_claims("dataset/sample_claims.csv", limit=1)[0]
    rules = EvidenceRules.from_csv("dataset/evidence_requirements.csv")

    selected = rules.for_claim(claim, build_claimed_focus(claim))
    ids = [rule.requirement_id for rule in selected]

    assert "REQ_GENERAL_OBJECT_PART" in ids
    assert "REQ_REVIEW_TRUST" in ids
    assert "REQ_CAR_BODY_PANEL" in ids


def test_claim_image_paths_resolve_relative_to_dataset_csv():
    claim = read_claims("dataset/claims.csv", limit=1)[0]

    assert claim.image_paths[0] == "dataset/images/test/case_001/img_1.jpg"
    assert claim.image_paths_raw.startswith("images/test/case_001/img_1.jpg")


def test_user_history_preserves_authoritative_flags_and_adds_manual_review():
    history = UserHistory.from_csv("dataset/user_history.csv")

    record = history.get("user_005")

    assert record.history_flags == ["user_history_risk", "manual_review_required"]


def test_atomic_writer_uses_required_columns(tmp_path):
    output = tmp_path / "output.csv"
    decision = _decision()

    write_output_atomic(output, [decision])

    with output.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "user_id",
            "image_paths",
            "user_claim",
            "claim_object",
            "evidence_standard_met",
            "evidence_standard_met_reason",
            "risk_flags",
            "issue_type",
            "object_part",
            "claim_status",
            "claim_status_justification",
            "supporting_image_ids",
            "valid_image",
            "severity",
        ]
        row = next(reader)
    assert row["evidence_standard_met"] == "true"
    assert row["risk_flags"] == "none"
    assert b"\r\n" not in output.read_bytes()


def test_evaluation_scores_lists_with_jaccard(tmp_path):
    expected = tmp_path / "expected.csv"
    predicted = tmp_path / "predicted.csv"
    _write_rows(expected, [_decision().to_csv_row() | {"risk_flags": "a;b"}])
    _write_rows(predicted, [_decision().to_csv_row() | {"risk_flags": "a"}])

    report = evaluate_predictions(expected, predicted)

    assert report["field_scores"]["risk_flags"] == 0.5
    assert report["headline_score"] < 1.0


def _decision():
    return ClaimDecision(
        user_id="user_001",
        image_paths="images/test/case_001/img_1.jpg",
        user_claim="claim",
        claim_object="car",
        evidence_standard_met=True,
        evidence_standard_met_reason="reason",
        risk_flags=[],
        issue_type="dent",
        object_part="door",
        claim_status="supported",
        claim_status_justification="justification",
        supporting_image_ids=["img_1"],
        valid_image=True,
        severity="medium",
    )


def _write_rows(path: Path, rows: list[dict[str, str]]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
