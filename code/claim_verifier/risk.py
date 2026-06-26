"""Risk flag normalization and ordering."""

from __future__ import annotations

from claim_verifier.constants import RISK_FLAG_PRIORITY, RISK_FLAGS
from claim_verifier.models import RiskFlag

_ALLOWED = set(RISK_FLAGS)
_PRIORITY = {flag: index for index, flag in enumerate(RISK_FLAG_PRIORITY)}


def parse_flags(raw: str) -> list[RiskFlag]:
    raw = (raw or "").strip()
    if not raw or raw == "none":
        return []
    flags = []
    for item in raw.split(";"):
        flag = item.strip()
        if not flag or flag == "none":
            continue
        if flag not in _ALLOWED:
            raise ValueError(f"Unknown risk flag: {flag}")
        flags.append(flag)  # type: ignore[arg-type]
    return order_risk_flags(flags)


def order_risk_flags(flags: list[RiskFlag]) -> list[RiskFlag]:
    deduped = list(dict.fromkeys(flags))
    return sorted(deduped, key=lambda flag: (_PRIORITY.get(flag, 999), flag))


def merge_history_flags(flags: list[RiskFlag]) -> list[RiskFlag]:
    merged = list(flags)
    if "user_history_risk" in merged and "manual_review_required" not in merged:
        merged.append("manual_review_required")
    return order_risk_flags(merged)
