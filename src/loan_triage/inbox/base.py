"""What every mailbox provides, so the importer never depends on Gmail directly."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class AttachmentRef:
    filename: str
    mime_type: str
    # The provider's handle for fetching the content, or the content itself if it came inline.
    reference: str = ""
    inline_data: str = ""


@dataclass(frozen=True)
class MailSummary:
    message_id: str
    thread_id: str
    sender: str
    subject: str
    received_at: datetime
    attachments: tuple[AttachmentRef, ...] = ()


class MailboxError(Exception):
    """A failure reading the mailbox. The message is safe to show on screen."""


class MailboxNotConnected(MailboxError):
    """There is no working authorization, so nothing can be read until someone signs in."""


class Mailbox(Protocol):
    name: str

    def address(self) -> str: ...

    def search(self, query: str, limit: int) -> list[MailSummary]: ...

    def fetch(self, message: MailSummary, attachment: AttachmentRef) -> bytes: ...
