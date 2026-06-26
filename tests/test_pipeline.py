import asyncio

from claim_verifier.auditor import AuditCall
from claim_verifier.constants import AUDIT_PROMPT_VERSION, AUDIT_SCHEMA_VERSION, VISION_PROMPT_VERSION, VISION_SCHEMA_VERSION
from claim_verifier.history import UserHistory
from claim_verifier.models import AuditResult, AuditRevisions, Claim, ImageDiagnostics, UserHistoryRecord, VisionAssessment
from claim_verifier.pipeline import VerificationAdapters, verify_claim
from claim_verifier.rules import EvidenceRules
from claim_verifier.vision import VisionCall


def test_verify_claim_uses_auditor_allowed_revision():
    claim = _claim()
    adapters = VerificationAdapters(
        vision_judge=FakeVisionJudge(_vision()),
        evidence_rules=EvidenceRules([]),
        user_history=UserHistory({"user_001": _history(["user_history_risk", "manual_review_required"])}),
        auditor=FakeAuditor(
            AuditResult(
                verdict="needs_revision",
                allowed_revisions=AuditRevisions(
                    claim_status="supported",
                    claim_status_justification="Auditor confirms support with review.",
                ),
                visual_inconsistency_reason=None,
                clarification_prompt=None,
                audit_reason="History risk does not override visual support.",
            )
        ),
    )

    decision = asyncio.run(verify_claim(claim, adapters))

    assert decision.claim_status == "supported"
    assert decision.claim_status_justification == "Auditor confirms support with review."
    assert decision.risk_flags == ["user_history_risk", "manual_review_required"]


class FakeVisionJudge:
    def __init__(self, assessment):
        self.assessment = assessment

    async def judge(self, **kwargs):
        return VisionCall(
            assessment=self.assessment,
            image_diagnostics=[
                ImageDiagnostics(
                    image_id="img_1",
                    path="images/test/case_001/img_1.jpg",
                    file_size_bytes=1,
                    width=10,
                    height=10,
                    format="JPEG",
                    mean_luminance=1.0,
                    contrast=1.0,
                    blur_score=1.0,
                    perceptual_hash="00",
                    preprocessed_width=10,
                    preprocessed_height=10,
                    preprocessed_size_bytes=1,
                )
            ],
            response_id="resp_test",
            model="fake",
            prompt_version=VISION_PROMPT_VERSION,
            schema_version=VISION_SCHEMA_VERSION,
            request_settings={},
            timing_ms=1.0,
            prompt="fake",
        )


class FakeAuditor:
    def __init__(self, result):
        self.result = result

    async def audit(self, **kwargs):
        return AuditCall(
            result=self.result,
            response_id="audit_test",
            model="fake-audit",
            prompt_version=AUDIT_PROMPT_VERSION,
            schema_version=AUDIT_SCHEMA_VERSION,
            request_settings={},
            timing_ms=1.0,
            prompt="fake",
        )


def _claim():
    return Claim(
        row_index=1,
        user_id="user_001",
        image_paths=["images/test/case_001/img_1.jpg"],
        image_paths_raw="images/test/case_001/img_1.jpg",
        user_claim="Customer: The front bumper is cracked.",
        claim_object="car",
    )


def _history(flags):
    return UserHistoryRecord(
        user_id="user_001",
        past_claim_count=1,
        accept_claim=1,
        manual_review_claim=0,
        rejected_claim=0,
        last_90_days_claim_count=0,
        history_flags=flags,
        history_summary="history",
    )


def _vision():
    return VisionAssessment(
        evidence_standard_met=True,
        evidence_reason="Front bumper is visible.",
        issue_type="crack",
        object_part="front_bumper",
        supporting_image_ids=["img_1"],
        image_quality_flags=[],
        valid_image=True,
        severity="medium",
        rationale="The image supports the crack claim.",
        claimed_items=["front bumper crack"],
        supported_claimed_items=["front bumper crack"],
        per_image_findings=[
            {
                "image_id": "img_1",
                "visible_object": "car",
                "visible_parts": ["front_bumper"],
                "damage_observed": "crack",
                "supports_claim": True,
                "quality_flags": [],
            }
        ],
    )
