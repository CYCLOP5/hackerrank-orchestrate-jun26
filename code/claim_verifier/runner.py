"""Executable wiring for prediction and sample generation."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from claim_verifier.auditor import OpenAIAssessmentAuditor
from claim_verifier.history import UserHistory
from claim_verifier.io import read_claims, write_output_atomic
from claim_verifier.model_client import OpenAIStructuredModelClient
from claim_verifier.pipeline import VerificationAdapters, verify_claims_concurrently
from claim_verifier.preflight import preflight
from claim_verifier.rules import EvidenceRules
from claim_verifier.tracing import TraceWriter, make_run_id
from claim_verifier.vision import OpenAIVisionJudge


async def run_predictions(
    *,
    claims_path: str,
    history_path: str,
    requirements_path: str,
    output_path: str,
    runs_dir: str,
    limit: int | None,
    no_audit: bool,
    trace_prompts: bool,
    run_prefix: str = "run",
    run_id: str | None = None,
) -> Path:
    load_dotenv()
    preflight(
        claims_path=claims_path,
        history_path=history_path,
        requirements_path=requirements_path,
        output_path=output_path,
        require_openai=True,
        limit=limit,
    )
    run_id = run_id or make_run_id(run_prefix)
    run_dir = Path(runs_dir) / run_id
    trace_writer = TraceWriter(run_dir, run_id, trace_prompts=trace_prompts)

    api_key = os.environ["OPENAI_API_KEY"]
    timeout = float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "90"))
    concurrency = max(1, int(os.environ.get("OPENAI_CONCURRENCY", "2")))
    vision_model = os.environ.get("OPENAI_VISION_MODEL", "gpt-5.5")
    auditor_model = os.environ.get("OPENAI_AUDITOR_MODEL", "gpt-5.4-mini")

    client = OpenAIStructuredModelClient(api_key=api_key, timeout_seconds=timeout)
    adapters = VerificationAdapters(
        vision_judge=OpenAIVisionJudge(client=client, model=vision_model),
        evidence_rules=EvidenceRules.from_csv(requirements_path),
        user_history=UserHistory.from_csv(history_path),
        auditor=None if no_audit else OpenAIAssessmentAuditor(client=client, model=auditor_model),
        trace_writer=trace_writer,
    )
    claims = read_claims(claims_path, limit=limit)
    decisions = await verify_claims_concurrently(
        claims,
        adapters,
        concurrency=concurrency,
        no_audit=no_audit,
    )
    write_output_atomic(output_path, decisions)
    return run_dir


def run_predictions_sync(**kwargs: object) -> Path:
    return asyncio.run(run_predictions(**kwargs))  # type: ignore[arg-type]
