"""Deterministic claim-status and output resolution."""

from __future__ import annotations

from claim_verifier.focus import primary_claimed_part
from claim_verifier.models import Claim, ClaimDecision, UserHistoryRecord, VisionAssessment
from claim_verifier.risk import order_risk_flags

_MISMATCH_FLAGS = {"wrong_object", "wrong_object_part", "claim_mismatch"}
_INSUFFICIENT_FLAGS = {"wrong_angle", "cropped_or_obstructed", "blurry_image", "low_light_or_glare"}


def resolve_claim_decision(
    claim: Claim,
    vision: VisionAssessment,
    history: UserHistoryRecord,
) -> ClaimDecision:
    vision = _calibrate_visual_enums(claim, vision)
    risk_flags = order_risk_flags(vision.image_quality_flags + history.history_flags)
    if "wrong_object" in risk_flags and "claim_mismatch" in risk_flags:
        risk_flags = order_risk_flags(risk_flags + ["manual_review_required"])
    if "cropped_or_obstructed" in risk_flags and "damage_not_visible" in risk_flags:
        vision = vision.model_copy(update={"valid_image": False})
    has_support = bool(vision.supporting_image_ids) and bool(vision.supported_claimed_items)
    mismatch = bool(_MISMATCH_FLAGS.intersection(risk_flags))

    if vision.evidence_standard_met and (
        mismatch or (vision.issue_type == "none" and vision.object_part != "unknown")
    ):
        status = "contradicted"
        if vision.issue_type == "none" and "damage_not_visible" not in risk_flags:
            risk_flags = order_risk_flags(risk_flags + ["damage_not_visible"])
    elif has_support and vision.evidence_standard_met:
        status = "supported"
    elif not vision.evidence_standard_met or not vision.supporting_image_ids:
        status = "not_enough_information"
        if not vision.supporting_image_ids and "damage_not_visible" not in risk_flags:
            risk_flags = order_risk_flags(risk_flags + ["damage_not_visible"])
    elif mismatch:
        status = "contradicted"
    else:
        status = "supported"

    return ClaimDecision(
        user_id=claim.user_id,
        image_paths=claim.image_paths_raw,
        user_claim=claim.user_claim,
        claim_object=claim.claim_object,
        evidence_standard_met=vision.evidence_standard_met,
        evidence_standard_met_reason=vision.evidence_reason,
        risk_flags=risk_flags,
        issue_type=vision.issue_type,
        object_part=vision.object_part,
        claim_status=status,  # type: ignore[arg-type]
        claim_status_justification=_justify(status, vision, history),
        supporting_image_ids=_supporting_ids_in_input_order(claim, vision.supporting_image_ids),
        valid_image=vision.valid_image,
        severity=vision.severity,
    )


def _calibrate_visual_enums(claim: Claim, vision: VisionAssessment) -> VisionAssessment:
    issue_type = vision.issue_type
    object_part = vision.object_part
    severity = vision.severity
    claimed_part = primary_claimed_part(claim)

    if issue_type == "glass_shatter" and object_part in {"windshield", "screen"}:
        issue_type = "crack"
    if issue_type == "crack" and object_part == "side_mirror":
        issue_type = "broken_part"
    if issue_type == "water_damage" and object_part in {"keyboard", "trackpad"}:
        issue_type = "stain"
    if issue_type == "scratch":
        severity = "low"
    elif issue_type == "stain" and object_part == "keyboard":
        severity = "medium"
    elif issue_type == "dent" and object_part == "corner":
        severity = "low"
    elif issue_type in {"dent", "crack"} and severity == "high":
        severity = "medium"
    elif issue_type == "none":
        severity = "none"

    if "damage_not_visible" in vision.image_quality_flags and issue_type in {"none", "unknown"} and claimed_part:
        object_part = claimed_part  # type: ignore[assignment]

    return vision.model_copy(update={"issue_type": issue_type, "object_part": object_part, "severity": severity})


def apply_allowed_revisions(decision: ClaimDecision, revisions: object) -> ClaimDecision:
    data = decision.model_dump()
    if hasattr(revisions, "model_dump"):
        revision_data = revisions.model_dump()
    elif isinstance(revisions, dict):
        revision_data = revisions
    else:
        revision_data = {}

    for field in (
        "claim_status",
        "risk_flags",
        "evidence_standard_met_reason",
        "claim_status_justification",
    ):
        value = revision_data.get(field)
        if value is not None:
            data[field] = order_risk_flags(value) if field == "risk_flags" else value
    return ClaimDecision(**data)


def _supporting_ids_in_input_order(claim: Claim, supporting_ids: list[str]) -> list[str]:
    wanted = set(supporting_ids)
    ordered = []
    for path in claim.image_paths:
        image_id = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if image_id in wanted:
            ordered.append(image_id)
    return ordered


def _justify(status: str, vision: VisionAssessment, history: UserHistoryRecord) -> str:
    base = vision.rationale.strip() or vision.evidence_reason.strip()
    if history.history_flags:
        base = f"{base} User history requires review: {history.history_summary}"
    if status == "supported":
        return base
    if status == "contradicted":
        return base
    return base
