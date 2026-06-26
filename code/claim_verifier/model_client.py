"""OpenAI structured output client with retries."""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class StructuredModelResponse:
    parsed: BaseModel
    response_id: str | None
    timing_ms: float


class OpenAIStructuredModelClient:
    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = 90.0,
        max_attempts: int = 3,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, timeout=timeout_seconds)
        self.max_attempts = max_attempts

    async def parse(
        self,
        *,
        model: str,
        input_payload: list[dict[str, Any]],
        schema_model: type[ModelT],
        temperature: float | None,
        row_context: str,
    ) -> StructuredModelResponse:
        last_error: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            started = time.perf_counter()
            try:
                kwargs = {
                    "model": model,
                    "input": input_payload,
                    "text_format": schema_model,
                }
                if temperature is not None:
                    kwargs["temperature"] = temperature
                response = await asyncio.to_thread(self._client.responses.parse, **kwargs)
                parsed = _extract_parsed(response, schema_model)
                timing_ms = (time.perf_counter() - started) * 1000
                return StructuredModelResponse(
                    parsed=parsed,
                    response_id=getattr(response, "id", None),
                    timing_ms=timing_ms,
                )
            except Exception as exc:  # noqa: BLE001 - SDK error classes vary by version.
                last_error = exc
                if attempt >= self.max_attempts or not _is_transient(exc):
                    break
                await asyncio.sleep(_retry_delay(exc, attempt))
        raise RuntimeError(f"OpenAI structured call failed for {row_context}: {last_error}") from last_error


def _extract_parsed(response: Any, schema_model: type[ModelT]) -> BaseModel:
    parsed = getattr(response, "output_parsed", None)
    if parsed is not None:
        return parsed if isinstance(parsed, BaseModel) else schema_model.model_validate(parsed)

    for output in getattr(response, "output", []) or []:
        for content in getattr(output, "content", []) or []:
            parsed = getattr(content, "parsed", None)
            if parsed is not None:
                return parsed if isinstance(parsed, BaseModel) else schema_model.model_validate(parsed)
    raise ValueError("OpenAI response did not contain parsed structured output")


def _is_transient(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status in {408, 409, 429, 500, 502, 503, 504}:
        return True
    name = exc.__class__.__name__.lower()
    return any(token in name for token in ("timeout", "ratelimit", "connection", "server"))


def _retry_delay(exc: Exception, attempt: int) -> float:
    retry_after = _retry_after(exc)
    if retry_after is not None:
        return retry_after
    cap = min(60.0, 2.0**attempt)
    return random.uniform(0.5, cap)


def _retry_after(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None
