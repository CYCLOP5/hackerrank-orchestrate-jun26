"""JSONL tracing for verification runs."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from claim_verifier.constants import TRACE_SCHEMA_VERSION


class TraceWriter:
    def __init__(self, run_dir: str | Path, run_id: str, trace_prompts: bool = False) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.trace_prompts = trace_prompts
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "traces.jsonl"
        self._lock = threading.Lock()

    def write(
        self,
        *,
        row_index: int,
        phase: str,
        attempt: int | None = None,
        model: str | None = None,
        prompt_version: str | None = None,
        request_settings: dict[str, Any] | None = None,
        image_diagnostics: Any = None,
        vision_assessment: Any = None,
        audit_result: Any = None,
        claim_decision: Any = None,
        timing_ms: float | None = None,
        error: str | None = None,
        prompt: str | None = None,
    ) -> None:
        record = {
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "row_index": row_index,
            "phase": phase,
            "attempt": attempt,
            "model": model,
            "prompt_version": prompt_version,
            "request_settings": request_settings or {},
            "image_diagnostics": _jsonable(image_diagnostics),
            "vision_assessment": _jsonable(vision_assessment),
            "audit_result": _jsonable(audit_result),
            "claim_decision": _jsonable(claim_decision),
            "timing_ms": timing_ms,
            "error": error,
        }
        if self.trace_prompts and prompt is not None:
            record["prompt"] = prompt
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")


def make_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value
