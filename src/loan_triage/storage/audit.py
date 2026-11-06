"""Append-only, hash-chained record of every agent and human decision.

Each row's hash covers its content and the previous row's hash, so editing, reordering, or
removing any row breaks verification from that point on. Triggers refuse UPDATE and DELETE as a
first line of defence; the chain catches anyone who gets past them. Truncating the newest rows
needs an external anchor to detect: publish `head_hash()` somewhere the database owner cannot edit.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from loan_triage.domain.enums import ActorType
from loan_triage.domain.normalize import mask_ssn
from loan_triage.storage.db import Database

GENESIS_HASH = "0" * 64


class AuditEvent(BaseModel):
    seq: int
    event_id: str
    case_id: str
    recorded_at: str
    actor_type: ActorType
    actor: str
    stage: str
    action: str
    detail: dict[str, Any]
    prev_hash: str
    hash: str


class ChainReport(BaseModel):
    ok: bool
    events_checked: int
    first_bad_seq: int | None = None
    problem: str | None = None


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return mask_ssn(value)
    if isinstance(value, dict):
        return {key: _redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _digest(row: dict[str, Any], prev_hash: str) -> str:
    body = {key: row[key] for key in sorted(row) if key not in {"seq", "hash", "prev_hash"}}
    return hashlib.sha256(f"{prev_hash}|{_canonical(body)}".encode()).hexdigest()


def _to_event(row: sqlite3.Row) -> AuditEvent:
    return AuditEvent(**{**dict(row), "detail": json.loads(row["detail"])})


class AuditLog:
    def __init__(self, database: Database) -> None:
        self._database = database

    def record(
        self,
        *,
        case_id: str,
        actor_type: ActorType,
        actor: str,
        stage: str,
        action: str,
        detail: dict[str, Any] | None = None,
    ) -> str:
        row = {
            "event_id": uuid.uuid4().hex,
            "case_id": case_id,
            "recorded_at": datetime.now(UTC).isoformat(),
            "actor_type": str(actor_type),
            "actor": actor,
            "stage": stage,
            "action": action,
            "detail": _canonical(_redact(detail or {})),
        }
        with self._database.write() as connection:
            last = connection.execute(
                "SELECT hash FROM audit_events ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            prev_hash = last["hash"] if last else GENESIS_HASH
            digest = _digest(row, prev_hash)
            connection.execute(
                """INSERT INTO audit_events
                   (event_id, case_id, recorded_at, actor_type, actor, stage, action, detail,
                    prev_hash, hash)
                   VALUES (:event_id, :case_id, :recorded_at, :actor_type, :actor, :stage,
                           :action, :detail, :prev_hash, :hash)""",
                {**row, "prev_hash": prev_hash, "hash": digest},
            )
        return digest

    def for_case(self, case_id: str) -> list[AuditEvent]:
        with self._database.read() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_events WHERE case_id = ? ORDER BY seq", (case_id,)
            ).fetchall()
        return [_to_event(row) for row in rows]

    def recent(self, limit: int = 200) -> list[AuditEvent]:
        with self._database.read() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_events ORDER BY seq DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_to_event(row) for row in rows]

    def head_hash(self) -> str:
        with self._database.read() as connection:
            row = connection.execute(
                "SELECT hash FROM audit_events ORDER BY seq DESC LIMIT 1"
            ).fetchone()
        return row["hash"] if row else GENESIS_HASH

    def verify(self) -> ChainReport:
        with self._database.read() as connection:
            rows = connection.execute("SELECT * FROM audit_events ORDER BY seq").fetchall()

        prev_hash = GENESIS_HASH
        expected_seq = rows[0]["seq"] if rows else 1
        for count, row in enumerate(rows):
            data = dict(row)
            problem = None
            if data["seq"] != expected_seq:
                problem = f"sequence jumps from {expected_seq - 1} to {data['seq']}"
            elif data["prev_hash"] != prev_hash:
                problem = "link to the previous event is broken"
            elif data["hash"] != _digest(data, prev_hash):
                problem = "event content does not match its hash"
            if problem:
                return ChainReport(
                    ok=False, events_checked=count, first_bad_seq=data["seq"], problem=problem
                )
            prev_hash = data["hash"]
            expected_seq += 1
        return ChainReport(ok=True, events_checked=len(rows))
