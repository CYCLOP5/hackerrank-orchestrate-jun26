"""Deep verification interface and orchestration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

from claim_verifier.auditor import AuditCall, should_audit
from claim_verifier.focus import build_claimed_focus
from claim_verifier.history import UserHistory
from claim_verifier.models import AuditResult, Claim, ClaimDecision
from claim_verifier.resolver import apply_allowed_revisions, resolve_claim_decision
from claim_verifier.rules import EvidenceRules
from claim_verifier.tracing import TraceWriter
from claim_verifier.vision import VisionCall


class VisionJudge(Protocol):
    async def judge(
        self,
        *,
        claim: Claim,
        evidence_rules: object,
        claimed_focus: str,
        clarification_prompt: str | None = None,
        attempt: int = 1,
    ) -> VisionCall: ...


class AssessmentAuditor(Protocol):
    async def audit(
        self,
        *,
        claim: Claim,
        claimed_focus: str,
        evidence_rules: object,
        history: object,
        vision: object,
        decision: ClaimDecision,
    ) -> AuditCall: ...


@dataclass(frozen=True)
class VerificationAdapters:
    vision_judge: VisionJudge
    evidence_rules: EvidenceRules
    user_history: UserHistory
    auditor: AssessmentAuditor | None = None
    trace_writer: TraceWriter | None = None


async def verify_claim(
    claim: Claim,
    adapters: VerificationAdapters,
    *,
    no_audit: bool = False,
) -> ClaimDecision:
    claimed_focus = build_claimed_focus(claim)
    evidence_rules = adapters.evidence_rules.for_claim(claim, claimed_focus)
    history = adapters.user_history.get(claim.user_id)

    vision_call = await adapters.vision_judge.judge(
        claim=claim,
        evidence_rules=evidence_rules,
        claimed_focus=claimed_focus,
        attempt=1,
    )
    decision = resolve_claim_decision(claim, vision_call.assessment, history)
    _trace_vision(adapters, claim, vision_call, decision, attempt=1)

    if no_audit or adapters.auditor is None or not should_audit(
        decision=decision,
        vision=vision_call.assessment,
        history=history,
    ):
        return decision

    audit_call = await adapters.auditor.audit(
        claim=claim,
        claimed_focus=claimed_focus,
        evidence_rules=evidence_rules,
        history=history,
        vision=vision_call.assessment,
        decision=decision,
    )
    _trace_audit(adapters, claim, audit_call, decision)

    if audit_call.result.verdict == "pass":
        return decision
    if audit_call.result.verdict == "needs_revision":
        revised = apply_allowed_revisions(decision, audit_call.result.allowed_revisions)
        _trace_decision(adapters, claim, revised, phase="audit_revision")
        return revised
    if audit_call.result.verdict == "rerun_vision":
        return await _rerun_once(
            claim=claim,
            adapters=adapters,
            claimed_focus=claimed_focus,
            evidence_rules=evidence_rules,
            history=history,
            clarification=audit_call.result.clarification_prompt,
        )
    raise RuntimeError(f"Audit failed for row {claim.row_index}: {audit_call.result.audit_reason}")


async def verify_claims_concurrently(
    claims: list[Claim],
    adapters: VerificationAdapters,
    *,
    concurrency: int,
    no_audit: bool = False,
) -> list[ClaimDecision]:
    results: list[ClaimDecision | None] = [None] * len(claims)
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(index: int, claim: Claim) -> None:
        async with semaphore:
            results[index] = await verify_claim(claim, adapters, no_audit=no_audit)

    async with asyncio.TaskGroup() as task_group:
        for index, claim in enumerate(claims):
            task_group.create_task(worker(index, claim))

    if any(result is None for result in results):
        raise RuntimeError("Concurrent verification completed with missing rows")
    return [result for result in results if result is not None]


async def _rerun_once(
    *,
    claim: Claim,
    adapters: VerificationAdapters,
    claimed_focus: str,
    evidence_rules: object,
    history: object,
    clarification: str | None,
) -> ClaimDecision:
    if not clarification:
        raise RuntimeError(f"Audit requested rerun for row {claim.row_index} without clarification")
    vision_call = await adapters.vision_judge.judge(
        claim=claim,
        evidence_rules=evidence_rules,
        claimed_focus=claimed_focus,
        clarification_prompt=clarification,
        attempt=2,
    )
    decision = resolve_claim_decision(claim, vision_call.assessment, history)  # type: ignore[arg-type]
    _trace_vision(adapters, claim, vision_call, decision, attempt=2)
    if adapters.auditor is None:
        return decision
    audit_call = await adapters.auditor.audit(
        claim=claim,
        claimed_focus=claimed_focus,
        evidence_rules=evidence_rules,
        history=history,
        vision=vision_call.assessment,
        decision=decision,
    )
    _trace_audit(adapters, claim, audit_call, decision)
    return _apply_second_audit(claim, decision, audit_call.result)


def _apply_second_audit(claim: Claim, decision: ClaimDecision, result: AuditResult) -> ClaimDecision:
    if result.verdict == "pass":
        return decision
    if result.verdict == "needs_revision":
        return apply_allowed_revisions(decision, result.allowed_revisions)
    raise RuntimeError(f"Audit still inconsistent after rerun for row {claim.row_index}: {result.audit_reason}")


def _trace_vision(
    adapters: VerificationAdapters,
    claim: Claim,
    vision_call: VisionCall,
    decision: ClaimDecision,
    attempt: int,
) -> None:
    if adapters.trace_writer is None:
        return
    adapters.trace_writer.write(
        row_index=claim.row_index,
        phase="vision",
        attempt=attempt,
        model=vision_call.model,
        prompt_version=vision_call.prompt_version,
        request_settings=vision_call.request_settings | {"response_id": vision_call.response_id},
        image_diagnostics=vision_call.image_diagnostics,
        vision_assessment=vision_call.assessment,
        claim_decision=decision,
        timing_ms=vision_call.timing_ms,
        prompt=vision_call.prompt,
    )


def _trace_audit(
    adapters: VerificationAdapters,
    claim: Claim,
    audit_call: AuditCall,
    decision: ClaimDecision,
) -> None:
    if adapters.trace_writer is None:
        return
    adapters.trace_writer.write(
        row_index=claim.row_index,
        phase="audit",
        attempt=None,
        model=audit_call.model,
        prompt_version=audit_call.prompt_version,
        request_settings=audit_call.request_settings | {"response_id": audit_call.response_id},
        audit_result=audit_call.result,
        claim_decision=decision,
        timing_ms=audit_call.timing_ms,
        prompt=audit_call.prompt,
    )


def _trace_decision(
    adapters: VerificationAdapters,
    claim: Claim,
    decision: ClaimDecision,
    phase: str,
) -> None:
    if adapters.trace_writer is None:
        return
    adapters.trace_writer.write(
        row_index=claim.row_index,
        phase=phase,
        claim_decision=decision,
    )
