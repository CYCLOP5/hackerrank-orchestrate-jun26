"""Text-only assessment auditor adapter."""

from __future__ import annotations

from dataclasses import dataclass

from claim_verifier.constants import AUDIT_PROMPT_VERSION, AUDIT_SCHEMA_VERSION
from claim_verifier.model_client import OpenAIStructuredModelClient
from claim_verifier.models import AuditResult, Claim, ClaimDecision, UserHistoryRecord, VisionAssessment
from claim_verifier.rules import EvidenceRule, render_rules


@dataclass(frozen=True)
class AuditCall:
    result: AuditResult
    response_id: str | None
    model: str
    prompt_version: str
    schema_version: str
    request_settings: dict[str, object]
    timing_ms: float
    prompt: str


class OpenAIAssessmentAuditor:
    def __init__(
        self,
        client: OpenAIStructuredModelClient,
        model: str,
        temperature: float | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature

    async def audit(
        self,
        *,
        claim: Claim,
        claimed_focus: str,
        evidence_rules: list[EvidenceRule],
        history: UserHistoryRecord,
        vision: VisionAssessment,
        decision: ClaimDecision,
    ) -> AuditCall:
        prompt = _audit_prompt(claim, claimed_focus, evidence_rules, history, vision, decision)
        response = await self._client.parse(
            model=self._model,
            input_payload=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            schema_model=AuditResult,
            temperature=self._temperature,
            row_context=f"row={claim.row_index} user={claim.user_id} auditor",
        )
        return AuditCall(
            result=response.parsed,  # type: ignore[arg-type]
            response_id=response.response_id,
            model=self._model,
            prompt_version=AUDIT_PROMPT_VERSION,
            schema_version=AUDIT_SCHEMA_VERSION,
            request_settings={"temperature": self._temperature},
            timing_ms=response.timing_ms,
            prompt=prompt,
        )


def should_audit(
    *,
    decision: ClaimDecision,
    vision: VisionAssessment,
    history: UserHistoryRecord,
) -> bool:
    return any(
        [
            decision.claim_status != "supported",
            bool(decision.risk_flags),
            not decision.evidence_standard_met,
            not decision.valid_image,
            len(vision.claimed_items) > len(vision.supported_claimed_items) > 0,
            decision.issue_type == "unknown",
            decision.object_part == "unknown",
            bool(history.history_flags),
        ]
    )


def _audit_prompt(
    claim: Claim,
    claimed_focus: str,
    evidence_rules: list[EvidenceRule],
    history: UserHistoryRecord,
    vision: VisionAssessment,
    decision: ClaimDecision,
) -> str:
    return f"""You are auditing a deterministic insurance-claim verifier.

You do not see images. Do not invent or change visual facts. You may only review whether
the proposed ClaimDecision follows from the validated VisionAssessment, evidence rules,
claimed focus, and user-history flags.

Allowed actions:
- pass: decision is consistent.
- needs_revision: only deterministic fields may change: claim_status, risk_flags,
  evidence_standard_met_reason, claim_status_justification.
- rerun_vision: use only if visual fields are internally inconsistent; provide a concise
  clarification_prompt for one and only one VisionJudge rerun.
- fail_loud: use for irreconcilable inconsistencies.

Claim row: {claim.row_index}
User: {claim.user_id}
Claimed focus: {claimed_focus}
Conversation: {claim.user_claim}
Evidence rules:
{render_rules(evidence_rules)}
User history: {history.model_dump(mode="json")}
VisionAssessment: {vision.model_dump(mode="json")}
Proposed ClaimDecision: {decision.model_dump(mode="json")}
"""
