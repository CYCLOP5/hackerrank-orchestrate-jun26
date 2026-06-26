"""Deterministic extraction of an advisory claimed focus."""

from __future__ import annotations

import re

from claim_verifier.constants import ISSUE_TYPES, OBJECT_PARTS
from claim_verifier.models import Claim


def build_claimed_focus(claim: Claim) -> str:
    text = _claimant_text(claim)
    parts = _matches(text, OBJECT_PARTS)
    issues = _matches(text, ISSUE_TYPES)

    chunks = [claim.claim_object]
    if parts:
        chunks.append("parts=" + ",".join(parts))
    if issues:
        chunks.append("issues=" + ",".join(issues))
    if len(chunks) == 1:
        chunks.append("general damage review")
    return " | ".join(chunks)


def primary_claimed_part(claim: Claim) -> str | None:
    text = _claimant_text(claim)
    if claim.claim_object == "package":
        if any(term in text for term in ("content", "contents", "inside", "item", "product", "missing")):
            return "contents"
        if "seal" in text:
            return "seal"
        if "label" in text:
            return "label"
    parts = _matches(text, OBJECT_PARTS)
    return parts[0] if parts else None


def _matches(text: str, values: tuple[str, ...]) -> list[str]:
    found = []
    for value in values:
        if value in {"none", "unknown", "body"}:
            continue
        terms = {value, value.replace("_", " ")}
        if value == "glass_shatter":
            terms.update({"shattered glass", "glass shatter", "broken glass"})
        if value == "broken_part":
            terms.update({"broken", "broke", "breakage"})
        if value == "missing_part":
            terms.update({"missing", "missing part"})
        if value == "water_damage":
            terms.update({"water damage", "wet", "water"})
        if value == "side_mirror":
            terms.update({"mirror", "side mirror"})
        if value == "front_bumper":
            terms.add("front bumper")
        if value == "rear_bumper":
            terms.add("rear bumper")
        if value == "taillight":
            terms.update({"back light", "rear light", "tail light"})
        if any(_has_affirmative_mention(text, term) for term in terms):
            found.append(value)
    return found


def _claimant_text(claim: Claim) -> str:
    claimant_segments = []
    for segment in claim.user_claim.split("|"):
        segment = segment.strip()
        if not segment:
            continue
        if ":" not in segment:
            claimant_segments.append(segment)
            continue
        speaker, text = segment.split(":", 1)
        if speaker.strip().lower() in {"customer", "claimant", "user"}:
            claimant_segments.append(text.strip())
    text = " | ".join(claimant_segments) or claim.user_claim
    return _normalize(text)


def _has_affirmative_mention(text: str, term: str) -> bool:
    pattern = rf"(?P<prefix>^|[^a-z0-9])(?P<term>{re.escape(term)})(?P<suffix>[^a-z0-9]|$)"
    for match in re.finditer(pattern, text):
        if not _is_negated(text, match.start("term"), term):
            return True
    return False


def _is_negated(text: str, start: int, term: str) -> bool:
    before = text[max(0, start - 60) : start]
    return bool(
        re.search(r"\b(?:not|no)\s+(?:the\s+)?$", before)
        or re.search(r"\bnot\s+(?:the\s+)?[a-z0-9_ ]{1,40}\bor\s+(?:the\s+)?$", before)
    )


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower())
