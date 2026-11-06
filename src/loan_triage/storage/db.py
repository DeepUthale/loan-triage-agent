"""SQLite access. A fresh connection per unit of work keeps this safe across graph threads."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id      TEXT PRIMARY KEY,
    status       TEXT NOT NULL,
    route        TEXT,
    priority     INTEGER,
    risk_tier    TEXT,
    risk_score   INTEGER,
    confidence   REAL,
    applicant    TEXT,
    loan_amount  TEXT,
    flag_count   INTEGER NOT NULL DEFAULT 0,
    decided_by   TEXT,
    error        TEXT,
    received_at  TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    state_json   TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status, priority, received_at);

CREATE TABLE IF NOT EXISTS mail_messages (
    message_id  TEXT PRIMARY KEY,
    thread_id   TEXT NOT NULL,
    outcome     TEXT NOT NULL,
    case_id     TEXT,
    detail      TEXT NOT NULL DEFAULT '',
    recorded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mail_thread ON mail_messages(thread_id, recorded_at);

CREATE TABLE IF NOT EXISTS audit_events (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id    TEXT NOT NULL UNIQUE,
    case_id     TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    actor_type  TEXT NOT NULL,
    actor       TEXT NOT NULL,
    stage       TEXT NOT NULL,
    action      TEXT NOT NULL,
    detail      TEXT NOT NULL,
    prev_hash   TEXT NOT NULL,
    hash        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_case ON audit_events(case_id, seq);

CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_events
BEGIN SELECT RAISE(ABORT, 'audit_events is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_events
BEGIN SELECT RAISE(ABORT, 'audit_events is append-only'); END;
"""


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        """Take the write lock up front, so read-then-append cannot interleave."""
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
