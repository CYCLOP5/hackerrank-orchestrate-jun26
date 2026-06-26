"""Typed interfaces for the claim verification modules."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr

from claim_verifier.constants import (
    CLAIM_OBJECTS,
    CLAIM_STATUSES,
    ISSUE_TYPES,
    OBJECT_PARTS,
    RISK_FLAGS,
    SEVERITIES,
)

ClaimObject = Literal["car", "laptop", "package"]
ClaimStatus = Literal["supported", "contradicted", "not_enough_information"]
IssueType = Literal[
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "broken_part",
    "missing_part",
    "torn_packaging",
    "crushed_packaging",
    "water_damage",
    "stain",
    "none",
    "unknown",
]
ObjectPart = Literal[
    "front_bumper",
    "rear_bumper",
    "door",
    "hood",
    "windshield",
    "side_mirror",
    "headlight",
    "taillight",
    "fender",
    "quarter_panel",
    "body",
    "screen",
    "keyboard",
    "trackpad",
    "hinge",
    "lid",
    "corner",
    "port",
    "base",
    "box",
    "package_corner",
    "package_side",
    "seal",
    "label",
    "contents",
    "item",
    "unknown",
]
RiskFlag = Literal[
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
]
Severity = Literal["none", "low", "medium", "high", "unknown"]
AuditVerdict = Literal["pass", "needs_revision", "rerun_vision", "fail_loud"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Claim(StrictModel):
    row_index: StrictInt = Field(ge=1)
    user_id: StrictStr
    image_paths: list[StrictStr] = Field(min_length=1)
    image_paths_raw: StrictStr
    user_claim: StrictStr
    claim_object: ClaimObject


class PerImageFinding(StrictModel):
    image_id: StrictStr
    visible_object: StrictStr
    visible_parts: list[ObjectPart]
    damage_observed: StrictStr
    supports_claim: StrictBool
    quality_flags: list[RiskFlag]


class VisionAssessment(StrictModel):
    evidence_standard_met: StrictBool
    evidence_reason: StrictStr
    issue_type: IssueType
    object_part: ObjectPart
    supporting_image_ids: list[StrictStr]
    image_quality_flags: list[RiskFlag]
    valid_image: StrictBool
    severity: Severity
    rationale: StrictStr
    claimed_items: list[StrictStr]
    supported_claimed_items: list[StrictStr]
    per_image_findings: list[PerImageFinding]


class AuditRevisions(StrictModel):
    claim_status: ClaimStatus | None = None
    risk_flags: list[RiskFlag] | None = None
    evidence_standard_met_reason: StrictStr | None = None
    claim_status_justification: StrictStr | None = None


class AuditResult(StrictModel):
    verdict: AuditVerdict
    allowed_revisions: AuditRevisions
    visual_inconsistency_reason: StrictStr | None
    clarification_prompt: StrictStr | None
    audit_reason: StrictStr


class UserHistoryRecord(StrictModel):
    user_id: StrictStr
    past_claim_count: StrictInt
    accept_claim: StrictInt
    manual_review_claim: StrictInt
    rejected_claim: StrictInt
    last_90_days_claim_count: StrictInt
    history_flags: list[RiskFlag]
    history_summary: StrictStr


class ImageDiagnostics(StrictModel):
    image_id: StrictStr
    path: StrictStr
    file_size_bytes: StrictInt
    width: StrictInt
    height: StrictInt
    format: StrictStr
    mean_luminance: StrictFloat
    contrast: StrictFloat
    blur_score: StrictFloat
    perceptual_hash: StrictStr
    preprocessed_width: StrictInt
    preprocessed_height: StrictInt
    preprocessed_size_bytes: StrictInt


class ClaimDecision(StrictModel):
    user_id: StrictStr
    image_paths: StrictStr
    user_claim: StrictStr
    claim_object: ClaimObject
    evidence_standard_met: StrictBool
    evidence_standard_met_reason: StrictStr
    risk_flags: list[RiskFlag]
    issue_type: IssueType
    object_part: ObjectPart
    claim_status: ClaimStatus
    claim_status_justification: StrictStr
    supporting_image_ids: list[StrictStr]
    valid_image: StrictBool
    severity: Severity

    def to_csv_row(self) -> dict[str, str]:
        return {
            "user_id": self.user_id,
            "image_paths": self.image_paths,
            "user_claim": self.user_claim,
            "claim_object": self.claim_object,
            "evidence_standard_met": _bool_to_csv(self.evidence_standard_met),
            "evidence_standard_met_reason": self.evidence_standard_met_reason,
            "risk_flags": _join_semicolon(self.risk_flags),
            "issue_type": self.issue_type,
            "object_part": self.object_part,
            "claim_status": self.claim_status,
            "claim_status_justification": self.claim_status_justification,
            "supporting_image_ids": _join_semicolon(self.supporting_image_ids),
            "valid_image": _bool_to_csv(self.valid_image),
            "severity": self.severity,
        }


def _bool_to_csv(value: bool) -> str:
    return "true" if value else "false"


def _join_semicolon(values: list[str]) -> str:
    return ";".join(values) if values else "none"


def allowed_values_snapshot() -> dict[str, tuple[str, ...]]:
    return {
        "claim_object": CLAIM_OBJECTS,
        "claim_status": CLAIM_STATUSES,
        "issue_type": ISSUE_TYPES,
        "object_part": OBJECT_PARTS,
        "risk_flags": RISK_FLAGS,
        "severity": SEVERITIES,
    }
