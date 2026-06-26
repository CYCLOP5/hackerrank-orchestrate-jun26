"""User history lookup adapter."""

from __future__ import annotations

import csv
from pathlib import Path

from claim_verifier.models import UserHistoryRecord
from claim_verifier.risk import merge_history_flags, parse_flags


class UserHistory:
    def __init__(self, records: dict[str, UserHistoryRecord]) -> None:
        self._records = records

    @classmethod
    def from_csv(cls, path: str | Path) -> "UserHistory":
        records: dict[str, UserHistoryRecord] = {}
        with Path(path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                flags = merge_history_flags(parse_flags(row["history_flags"]))
                record = UserHistoryRecord(
                    user_id=row["user_id"],
                    past_claim_count=int(row["past_claim_count"]),
                    accept_claim=int(row["accept_claim"]),
                    manual_review_claim=int(row["manual_review_claim"]),
                    rejected_claim=int(row["rejected_claim"]),
                    last_90_days_claim_count=int(row["last_90_days_claim_count"]),
                    history_flags=flags,
                    history_summary=row["history_summary"],
                )
                records[record.user_id] = record
        return cls(records)

    def get(self, user_id: str) -> UserHistoryRecord:
        try:
            return self._records[user_id]
        except KeyError as exc:
            raise KeyError(f"Missing user history for {user_id}") from exc
