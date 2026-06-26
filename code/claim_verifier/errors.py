"""Error formatting for fail-loud CLI output."""

from __future__ import annotations


def summarize_exception(exc: BaseException) -> str:
    leaves = _leaf_messages(exc)
    if not leaves:
        return str(exc)
    return " | ".join(dict.fromkeys(leaves))


def _leaf_messages(exc: BaseException) -> list[str]:
    if isinstance(exc, BaseExceptionGroup):
        messages: list[str] = []
        for child in exc.exceptions:
            messages.extend(_leaf_messages(child))
        return messages
    return [str(exc)]
