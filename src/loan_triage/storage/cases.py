"""Case records: the review queue and the dashboard's read model."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from loan_triage.domain.enums import CaseStatus
from loan_triage.storage.db import Database


def _now() -> str:
    return datetime.now(UTC).isoformat()


class CaseRow(BaseModel):
    case_id: str
    status: CaseStatus
    route: str | None = None
    priority: int | None = None
    risk_tier: str | None = None
    risk_score: int | None = None
    confidence: float | None = None
    applicant: str | None = None
    loan_amount: str | None = None
    flag_count: int = 0
    decided_by: str | None = None
    error: str | None = None
    received_at: str
    updated_at: str
    state: dict[str, Any] = {}


class CaseRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    def create(self, case_id: str, received_at: str) -> None:
        with self._database.write() as connection:
            connection.execute(
                "INSERT INTO cases (case_id, status, received_at, updated_at) VALUES (?, ?, ?, ?)",
                (case_id, str(CaseStatus.PROCESSING), received_at, _now()),
            )

    def update(self, case_id: str, *, status: CaseStatus, **columns: Any) -> None:
        values: dict[str, Any] = {**columns, "status": str(status), "updated_at": _now()}
        if "state" in values:
            values["state_json"] = json.dumps(values.pop("state"), default=str)
        assignments = ", ".join(f"{name} = :{name}" for name in values)  # names are internal
        with self._database.write() as connection:
            connection.execute(
                f"UPDATE cases SET {assignments} WHERE case_id = :case_id",  # noqa: S608
                {**values, "case_id": case_id},
            )

    def claim(self, case_id: str, expected: CaseStatus) -> bool:
        """Move a case to PROCESSING only if it is still in `expected`.

        Two reviewers acting on the same file cannot both win, because the status check and the
        update happen in one statement.
        """
        with self._database.write() as connection:
            cursor = connection.execute(
                "UPDATE cases SET status = ?, updated_at = ? WHERE case_id = ? AND status = ?",
                (str(CaseStatus.PROCESSING), _now(), case_id, str(expected)),
            )
        return cursor.rowcount == 1

    def get(self, case_id: str) -> CaseRow | None:
        with self._database.read() as connection:
            row = connection.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        return _to_case(row) if row else None

    def list(self, statuses: list[CaseStatus] | None = None) -> list[CaseRow]:
        query = "SELECT * FROM cases"
        parameters: list[str] = []
        if statuses:
            query += f" WHERE status IN ({', '.join('?' * len(statuses))})"
            parameters = [str(status) for status in statuses]
        query += " ORDER BY COALESCE(priority, 9), received_at, case_id"
        with self._database.read() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [_to_case(row) for row in rows]


def _to_case(row: Any) -> CaseRow:
    data = dict(row)
    data["state"] = json.loads(data.pop("state_json"))
    return CaseRow(**data)
