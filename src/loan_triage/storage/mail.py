"""Which emails have been imported.

Read-only access means Gmail itself cannot be marked, so this ledger is what stops a scheduled
sync or a double click from triaging one email twice. It also maps threads to cases, which is how
a reply carrying a missing document finds its way back to the right file.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from pydantic import BaseModel

from loan_triage.domain.enums import MailOutcome
from loan_triage.storage.db import Database

# Long enough to outlast a real import, so only a crashed run's claim ever expires.
_STALE_CLAIM = timedelta(minutes=30)


class MailRecord(BaseModel):
    message_id: str
    thread_id: str
    outcome: MailOutcome
    case_id: str | None
    detail: str
    recorded_at: str


class MailLedger:
    def __init__(self, database: Database) -> None:
        self._database = database

    def claim(self, message_id: str, thread_id: str) -> bool:
        """Reserve an email for import. Only one caller can win, however many race for it."""
        now = datetime.now(UTC)
        with self._database.write() as connection:
            connection.execute(
                "DELETE FROM mail_messages "
                "WHERE message_id = ? AND outcome = ? AND recorded_at < ?",
                (message_id, str(MailOutcome.IMPORTING), (now - _STALE_CLAIM).isoformat()),
            )
            cursor = connection.execute(
                """INSERT OR IGNORE INTO mail_messages
                   (message_id, thread_id, outcome, recorded_at) VALUES (?, ?, ?, ?)""",
                (message_id, thread_id, str(MailOutcome.IMPORTING), now.isoformat()),
            )
        return cursor.rowcount == 1

    def finish(
        self,
        message_id: str,
        outcome: MailOutcome,
        *,
        case_id: str | None = None,
        detail: str = "",
    ) -> None:
        with self._database.write() as connection:
            connection.execute(
                """UPDATE mail_messages
                   SET outcome = ?, case_id = ?, detail = ?, recorded_at = ?
                   WHERE message_id = ?""",
                (str(outcome), case_id, detail, datetime.now(UTC).isoformat(), message_id),
            )

    def release(self, message_id: str) -> None:
        """Forget a claim, which leaves the email available to import again."""
        with self._database.write() as connection:
            connection.execute("DELETE FROM mail_messages WHERE message_id = ?", (message_id,))

    def get(self, message_id: str) -> MailRecord | None:
        with self._database.read() as connection:
            row = connection.execute(
                "SELECT * FROM mail_messages WHERE message_id = ?", (message_id,)
            ).fetchone()
        return MailRecord(**dict(row)) if row else None

    def case_for_thread(self, thread_id: str) -> str | None:
        with self._database.read() as connection:
            row = connection.execute(
                """SELECT case_id FROM mail_messages
                   WHERE thread_id = ? AND case_id IS NOT NULL
                   ORDER BY recorded_at LIMIT 1""",
                (thread_id,),
            ).fetchone()
        return row["case_id"] if row else None
