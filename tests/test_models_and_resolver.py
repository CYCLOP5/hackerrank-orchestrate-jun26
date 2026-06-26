import pytest
from pydantic import ValidationError

from claim_verifier.focus import build_claimed_focus
from claim_verifier.models import Claim, UserHistoryRecord, VisionAssessment
from claim_verifier.resolver import resolve_claim_decision
from claim_verifier.risk import order_risk_flags


def test_vision_assessment_rejects_coerced_bool_and_extra_fields():
    payload = _vision_payload()
    payload["evidence_standard_met"] = "true"
    payload["unexpected"] = "nope"

    with pytest.raises(ValidationError):
        VisionAssessment(**payload)


def test_supported_image_set_wins_over_extra_bad_image_flag():
    claim = _claim()
    vision = VisionAssessment(
        **{
            **_vision_payload(),
            "image_quality_flags": ["blurry_image"],
            "supporting_image_ids": ["img_2"],
            "supported_claimed_items": ["door dent"],
        }
    )

    decision = resolve_claim_decision(claim, vision, _history([]))

    assert decision.claim_status == "supported"
    assert decision.risk_flags == ["blurry_image"]
    assert decision.supporting_image_ids == ["img_2"]


def test_visible_mismatch_is_contradicted_with_observed_issue():
    vision = VisionAssessment(
        **{
            **_vision_payload(),
            "issue_type": "scratch",
            "object_part": "rear_bumper",
            "image_quality_flags": ["claim_mismatch"],
            "supporting_image_ids": ["img_1"],
            "supported_claimed_items": [],
            "severity": "low",
        }
    )

    decision = resolve_claim_decision(_claim(), vision, _history(["user_history_risk"]))

    assert decision.claim_status == "contradicted"
    assert decision.issue_type == "scratch"
    assert decision.object_part == "rear_bumper"
    assert decision.severity == "low"
    assert decision.risk_flags == ["claim_mismatch", "user_history_risk", "manual_review_required"]


def test_claim_mismatch_wins_even_when_model_lists_supported_ids():
    vision = VisionAssessment(
        **{
            **_vision_payload(),
            "image_quality_flags": ["claim_mismatch"],
            "supported_claimed_items": ["front damage"],
        }
    )

    decision = resolve_claim_decision(_claim(), vision, _history([]))

    assert decision.claim_status == "contradicted"


def test_visual_enum_calibration_matches_sample_semantics():
    vision = VisionAssessment(
        **{
            **_vision_payload(),
            "issue_type": "glass_shatter",
            "object_part": "windshield",
            "severity": "high",
        }
    )

    decision = resolve_claim_decision(_claim(), vision, _history([]))

    assert decision.issue_type == "crack"
    assert decision.severity == "medium"


def test_keyboard_stain_calibrates_to_medium_severity():
    vision = VisionAssessment(
        **{
            **_vision_payload(),
            "issue_type": "water_damage",
            "object_part": "keyboard",
            "severity": "low",
        }
    )

    decision = resolve_claim_decision(_claim(claim_object="laptop"), vision, _history([]))

    assert decision.issue_type == "stain"
    assert decision.severity == "medium"


def test_damage_not_visible_on_visible_part_is_contradicted():
    vision = VisionAssessment(
        **{
            **_vision_payload(),
            "issue_type": "none",
            "object_part": "trackpad",
            "image_quality_flags": [],
            "supporting_image_ids": ["img_1"],
            "supported_claimed_items": [],
            "severity": "none",
        }
    )

    decision = resolve_claim_decision(_claim(claim_object="laptop"), vision, _history([]))

    assert decision.claim_status == "contradicted"
    assert decision.risk_flags == ["damage_not_visible"]


def test_risk_flags_use_fixed_priority_order():
    assert order_risk_flags(
        ["manual_review_required", "blurry_image", "claim_mismatch", "user_history_risk"]
    ) == ["claim_mismatch", "blurry_image", "user_history_risk", "manual_review_required"]


def test_wrong_object_mismatch_adds_manual_review():
    vision = VisionAssessment(
        **{
            **_vision_payload(),
            "image_quality_flags": ["wrong_object", "claim_mismatch"],
            "supported_claimed_items": [],
        }
    )

    decision = resolve_claim_decision(_claim(), vision, _history([]))

    assert decision.risk_flags == ["wrong_object", "claim_mismatch", "manual_review_required"]


def test_cropped_missing_contents_marks_image_set_invalid_and_contents_part():
    claim = Claim(
        row_index=1,
        user_id="user_001",
        image_paths=["images/sample/case_018/img_1.jpg"],
        image_paths_raw="images/sample/case_018/img_1.jpg",
        user_claim="Customer: The item I ordered was not inside the box. The contents are missing.",
        claim_object="package",
    )
    vision = VisionAssessment(
        **{
            **_vision_payload(),
            "evidence_standard_met": False,
            "issue_type": "unknown",
            "object_part": "box",
            "supporting_image_ids": [],
            "image_quality_flags": ["cropped_or_obstructed", "damage_not_visible"],
            "valid_image": True,
            "severity": "unknown",
            "supported_claimed_items": [],
        }
    )

    decision = resolve_claim_decision(claim, vision, _history([]))

    assert decision.object_part == "contents"
    assert decision.valid_image is False


def test_claimed_focus_ignores_support_suggested_parts():
    claim = _claim(
        user_claim=(
            "Customer: Someone clipped my car while it was parked. | "
            "Support: Was there any damage to the door too? | "
            "Customer: I did not notice anything else major, just the mirror."
        )
    )

    assert build_claimed_focus(claim) == "car | parts=side_mirror"


def test_claimed_focus_ignores_negated_customer_parts():
    claim = _claim(
        claim_object="laptop",
        user_claim=(
            "Customer: Not the keyboard or hinge. "
            "The issue I want checked is the screen. It looks shattered to me."
        ),
    )

    assert build_claimed_focus(claim) == "laptop | parts=screen"


def test_claimed_focus_keeps_multiple_affirmative_claimed_parts():
    claim = _claim(
        user_claim=(
            "Customer: The front bumper looks damaged and the left headlight also "
            "looks affected. Yes, front bumper and left headlight together."
        )
    )

    assert build_claimed_focus(claim) == "car | parts=front_bumper,headlight"


def _claim(claim_object="car", user_claim="Customer: The door has a dent."):
    return Claim(
        row_index=1,
        user_id="user_001",
        image_paths=["images/test/case_001/img_1.jpg", "images/test/case_001/img_2.jpg"],
        image_paths_raw="images/test/case_001/img_1.jpg;images/test/case_001/img_2.jpg",
        user_claim=user_claim,
        claim_object=claim_object,
    )


def _history(flags):
    if "user_history_risk" in flags and "manual_review_required" not in flags:
        flags = [*flags, "manual_review_required"]
    return UserHistoryRecord(
        user_id="user_001",
        past_claim_count=1,
        accept_claim=1,
        manual_review_claim=0,
        rejected_claim=0,
        last_90_days_claim_count=0,
        history_flags=flags,
        history_summary="test",
    )


def _vision_payload():
    return {
        "evidence_standard_met": True,
        "evidence_reason": "The relevant part is visible.",
        "issue_type": "dent",
        "object_part": "door",
        "supporting_image_ids": ["img_1"],
        "image_quality_flags": [],
        "valid_image": True,
        "severity": "medium",
        "rationale": "The image shows a dent on the door.",
        "claimed_items": ["door dent"],
        "supported_claimed_items": ["door dent"],
        "per_image_findings": [
            {
                "image_id": "img_1",
                "visible_object": "car",
                "visible_parts": ["door"],
                "damage_observed": "dent",
                "supports_claim": True,
                "quality_flags": [],
            }
        ],
    }
