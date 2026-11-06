"""Turn emails into cases.

Sender, subject, and filenames are written by whoever sent the email, so they are hostile input.
They are stored and shown as data, and neither the subject nor the body is ever sent to Claude:
triage reads the attachments only.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from loan_triage.config import Settings
from loan_triage.domain.enums import ActorType, CaseStatus, MailOutcome
from loan_triage.inbox.base import (
    AttachmentRef,
    Mailbox,
    MailboxError,
    MailboxNotConnected,
    MailSummary,
)
from loan_triage.inbox.demo import DemoMailbox
from loan_triage.inbox.gmail import LOGIN_HINT, GmailAuth, connect_gmail
from loan_triage.ingest.pdf import InvalidDocument
from loan_triage.service import ReviewRejected, TriageService, Upload
from loan_triage.storage.cases import CaseRow

logger = logging.getLogger(__name__)

_MAX_TEXT = 200
_MAX_FILENAME = 120
# Audit events need a case, and an email with nothing usable never gets one.
NO_CASE = "no-case"


class InboxState(StrEnum):
    NEW = "new"
    FOLLOW_UP = "follow_up"
    BLOCKED = "blocked"
    DONE = "done"


@dataclass(frozen=True)
class InboxItem:
    message: MailSummary
    state: InboxState
    case_id: str | None = None
    note: str = ""

    @property
    def actionable(self) -> bool:
        return self.state in {InboxState.NEW, InboxState.FOLLOW_UP}


@dataclass(frozen=True)
class ImportResult:
    message: MailSummary
    outcome: MailOutcome
    case_id: str | None = None
    case_status: CaseStatus | None = None
    detail: str = ""


def open_mailbox(settings: Settings) -> Mailbox:
    if settings.mailbox_mode == "demo":
        return DemoMailbox()
    auth = GmailAuth(settings.gmail_client_secrets_path, settings.gmail_token_path)
    if not auth.connected:
        raise MailboxNotConnected(f"Gmail is not connected. {LOGIN_HINT}")
    return connect_gmail(auth)


def clean_text(value: str, limit: int = _MAX_TEXT) -> str:
    """One line and a sane length, so a header cannot forge rows in a log or swamp a table."""
    return " ".join("".join(c if c.isprintable() else " " for c in value).split())[:limit]


def clean_filename(name: str) -> str:
    return clean_text(re.split(r"[\\/]", name)[-1], _MAX_FILENAME) or "attachment.pdf"


def _looks_like_pdf(attachment: AttachmentRef) -> bool:
    return attachment.mime_type == "application/pdf" or attachment.filename.lower().endswith(".pdf")


class InboxImporter:
    def __init__(self, service: TriageService, mailbox: Mailbox) -> None:
        self._service = service
        self._mailbox = mailbox
        self._ledger = service.mail_ledger

    def preview(self) -> list[InboxItem]:
        settings = self._service.settings
        messages = self._mailbox.search(settings.gmail_query, settings.gmail_max_messages)
        return [self._describe(item) for item in sorted(messages, key=_by_arrival)]

    def _describe(self, message: MailSummary) -> InboxItem:
        record = self._ledger.get(message.message_id)
        if record is not None:
            return InboxItem(message, InboxState.DONE, record.case_id, record.detail)
        case_id = self._ledger.case_for_thread(message.thread_id)
        if case_id is None:
            return InboxItem(message, InboxState.NEW)
        status = self._service.case(case_id).status
        if status is CaseStatus.AWAITING_DOCUMENTS:
            return InboxItem(message, InboxState.FOLLOW_UP, case_id)
        return InboxItem(message, InboxState.BLOCKED, case_id, _not_waiting(status))

    def run(
        self,
        messages: Iterable[MailSummary],
        on_result: Callable[[ImportResult], None] | None = None,
    ) -> list[ImportResult]:
        """Oldest first, so a reply lands after the application it belongs to."""
        results = []
        for message in sorted(messages, key=_by_arrival):
            result = self.import_message(message)
            results.append(result)
            if on_result is not None:
                on_result(result)
        return results

    def import_message(self, message: MailSummary) -> ImportResult:
        if not self._ledger.claim(message.message_id, message.thread_id):
            record = self._ledger.get(message.message_id)
            case_id = record.case_id if record else None
            return ImportResult(message, MailOutcome.DUPLICATE, case_id, detail="Already imported.")
        try:
            return self._process(message)
        except MailboxError as error:
            self._ledger.release(message.message_id)
            return ImportResult(message, MailOutcome.ERROR, detail=str(error))
        except Exception:
            self._ledger.release(message.message_id)
            raise

    def _process(self, message: MailSummary) -> ImportResult:
        uploads, rejected = self._collect(message)
        source = self._source(message, rejected)
        if not uploads:
            return self._reject(message, source, rejected)
        case_id = self._ledger.case_for_thread(message.thread_id)
        if case_id is not None:
            return self._attach(message, source, uploads, case_id)
        return self._open(message, source, uploads)

    def _collect(self, message: MailSummary) -> tuple[list[Upload], list[dict[str, str]]]:
        """Keep the good attachments. One bad file should not cost the applicant the rest."""
        uploads: list[Upload] = []
        rejected: list[dict[str, str]] = []
        for attachment in message.attachments:
            filename = clean_filename(attachment.filename)
            if not _looks_like_pdf(attachment):
                rejected.append({"filename": filename, "reason": "not a PDF"})
                continue
            upload = Upload(filename, self._mailbox.fetch(message, attachment))
            try:
                self._service.check_upload(upload)
            except InvalidDocument as error:
                rejected.append({"filename": filename, "reason": str(error)})
            else:
                uploads.append(upload)
        return uploads, rejected

    def _source(self, message: MailSummary, rejected: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "channel": self._mailbox.name,
            "message_id": message.message_id,
            "thread_id": message.thread_id,
            "sender": clean_text(message.sender),
            "subject": clean_text(message.subject),
            "received_at": message.received_at.isoformat(),
            "rejected_attachments": rejected,
        }

    def _audit(self, case_id: str, action: str, source: dict[str, Any]) -> None:
        self._service.audit.record(
            case_id=case_id,
            actor_type=ActorType.SYSTEM,
            actor="inbox",
            stage="intake",
            action=action,
            detail=source,
        )

    def _reject(
        self, message: MailSummary, source: dict[str, Any], rejected: list[dict[str, str]]
    ) -> ImportResult:
        detail = "; ".join(f"{item['filename']}: {item['reason']}" for item in rejected)
        detail = detail or "No attachments."
        self._ledger.finish(message.message_id, MailOutcome.REJECTED, detail=detail)
        self._audit(NO_CASE, "email_rejected", source)
        return ImportResult(message, MailOutcome.REJECTED, detail=detail)

    def _open(
        self, message: MailSummary, source: dict[str, Any], uploads: list[Upload]
    ) -> ImportResult:
        case = self._service.submit(uploads, received_on=message.received_at.date(), source=source)
        if case.status is CaseStatus.FAILED:
            # The failed case stays visible, and the email stays available to try again.
            self._ledger.release(message.message_id)
            return ImportResult(
                message, MailOutcome.ERROR, case.case_id, case.status, case.error or ""
            )
        self._ledger.finish(message.message_id, MailOutcome.OPENED, case_id=case.case_id)
        return self._result(message, MailOutcome.OPENED, case)

    def _attach(
        self,
        message: MailSummary,
        source: dict[str, Any],
        uploads: list[Upload],
        case_id: str,
    ) -> ImportResult:
        status = self._service.case(case_id).status
        if status is not CaseStatus.AWAITING_DOCUMENTS:
            return self._skip(message, case_id, status)
        try:
            case = self._service.add_documents(case_id, uploads)
        except ReviewRejected:
            # Another session resumed this case first.
            return self._skip(message, case_id, self._service.case(case_id).status)
        self._ledger.finish(message.message_id, MailOutcome.ATTACHED, case_id=case_id)
        self._audit(case_id, "email_documents_attached", source)
        return self._result(message, MailOutcome.ATTACHED, case)

    def _skip(self, message: MailSummary, case_id: str, status: CaseStatus) -> ImportResult:
        # Released, not finished: if this case later asks for documents, the email still counts.
        self._ledger.release(message.message_id)
        return ImportResult(
            message, MailOutcome.SKIPPED, case_id, status, detail=_not_waiting(status)
        )

    @staticmethod
    def _result(message: MailSummary, outcome: MailOutcome, case: CaseRow) -> ImportResult:
        return ImportResult(message, outcome, case.case_id, case.status)


def _by_arrival(message: MailSummary) -> Any:
    return message.received_at


def _not_waiting(status: CaseStatus) -> str:
    return f"Reply to a case that is {status.replace('_', ' ')}, so it was not attached."
