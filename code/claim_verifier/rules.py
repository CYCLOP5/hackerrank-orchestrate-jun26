"""Evidence requirement lookup adapter."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from claim_verifier.models import Claim


@dataclass(frozen=True)
class EvidenceRule:
    requirement_id: str
    claim_object: str
    applies_to: str
    minimum_image_evidence: str


class EvidenceRules:
    def __init__(self, rules: list[EvidenceRule]) -> None:
        self._rules = rules

    @classmethod
    def from_csv(cls, path: str | Path) -> "EvidenceRules":
        rules = []
        with Path(path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rules.append(
                    EvidenceRule(
                        requirement_id=row["requirement_id"],
                        claim_object=row["claim_object"],
                        applies_to=row["applies_to"],
                        minimum_image_evidence=row["minimum_image_evidence"],
                    )
                )
        return cls(rules)

    def for_claim(self, claim: Claim, claimed_focus: str) -> list[EvidenceRule]:
        general = [rule for rule in self._rules if rule.claim_object == "all"]
        object_rules = [rule for rule in self._rules if rule.claim_object == claim.claim_object]
        matched = [rule for rule in object_rules if _matches_applies_to(rule.applies_to, claimed_focus)]
        return general + (matched if matched else object_rules)


def render_rules(rules: list[EvidenceRule]) -> str:
    return "\n".join(
        f"- {rule.requirement_id} ({rule.applies_to}): {rule.minimum_image_evidence}"
        for rule in rules
    )


def _matches_applies_to(applies_to: str, claimed_focus: str) -> bool:
    focus = claimed_focus.lower().replace("_", " ")
    tokens = [
        token
        for token in re.split(r"[^a-z0-9]+", applies_to.lower())
        if len(token) >= 4 and token not in {"when", "with", "should", "claim", "review"}
    ]
    return any(token in focus for token in tokens)
