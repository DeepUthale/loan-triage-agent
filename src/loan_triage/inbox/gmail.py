"""Gmail, through the official API and a read-only grant.

`gmail.readonly` is the narrowest scope that can read attachments. The app can see mail but cannot
send, change, or delete it, so a bug or a stolen token cannot act on the mailbox.
"""

from __future__ import annotations

import base64
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from google.auth.exceptions import RefreshError, TransportError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from httplib2 import HttpLib2Error

from loan_triage.inbox.base import AttachmentRef, MailboxError, MailboxNotConnected, MailSummary

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

LOGIN_HINT = "Run `uv run loan-triage gmail-login` to connect."
_API_RETRIES = 3


class GmailAuth:
    def __init__(self, client_secrets: Path, token_path: Path) -> None:
        self._client_secrets = client_secrets
        self._token_path = token_path

    @property
    def connected(self) -> bool:
        return self._token_path.exists()

    def authorize(self) -> None:
        """Run the browser sign-in. Being interactive, it belongs in the CLI, not a web request."""
        if not self._client_secrets.exists():
            raise MailboxError(
                f"No OAuth client at {self._client_secrets}. See Email intake in the README."
            )
        flow = InstalledAppFlow.from_client_secrets_file(str(self._client_secrets), SCOPES)
        self._save(flow.run_local_server(port=0, timeout_seconds=300))

    def disconnect(self) -> None:
        self._token_path.unlink(missing_ok=True)

    def credentials(self) -> Credentials:
        if not self.connected:
            raise MailboxNotConnected(f"Gmail is not connected. {LOGIN_HINT}")
        credentials = Credentials.from_authorized_user_file(str(self._token_path), SCOPES)
        if not credentials.valid:
            credentials.refresh(Request())
            self._save(credentials)
        return credentials

    def _save(self, credentials: Credentials) -> None:
        """The token reads this person's mail, so only the owner may read the file."""
        self._token_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self._token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w") as token_file:
            token_file.write(credentials.to_json())
        self._token_path.chmod(0o600)


@contextmanager
def _gmail_errors() -> Iterator[None]:
    try:
        yield
    except RefreshError as exc:
        raise MailboxNotConnected(f"Gmail access expired or was revoked. {LOGIN_HINT}") from exc
    except HttpError as exc:
        if exc.status_code == 401:
            raise MailboxNotConnected(f"Gmail rejected the sign-in. {LOGIN_HINT}") from exc
        raise MailboxError(f"Gmail returned an error: {exc.reason}") from exc
    except (OSError, HttpLib2Error, TransportError) as exc:
        raise MailboxError("Could not reach Gmail.") from exc


def _walk(part: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield part
    for child in part.get("parts", []):
        yield from _walk(child)


def _attachments(payload: dict[str, Any]) -> tuple[AttachmentRef, ...]:
    references = []
    for part in _walk(payload):
        body = part.get("body", {})
        if part.get("filename") and (body.get("attachmentId") or body.get("data")):
            references.append(
                AttachmentRef(
                    filename=part["filename"],
                    mime_type=part.get("mimeType", ""),
                    reference=body.get("attachmentId", ""),
                    inline_data=body.get("data", ""),
                )
            )
    return tuple(references)


def parse_message(message: dict[str, Any]) -> MailSummary:
    payload = message["payload"]
    headers = {item["name"].lower(): item["value"] for item in payload.get("headers", [])}
    return MailSummary(
        message_id=message["id"],
        thread_id=message["threadId"],
        sender=headers.get("from", "Unknown sender"),
        subject=headers.get("subject", "(no subject)"),
        received_at=datetime.fromtimestamp(int(message["internalDate"]) / 1000, UTC),
        attachments=_attachments(payload),
    )


def _decode(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


class GmailMailbox:
    name = "gmail"

    def __init__(self, build_api: Callable[[], Any], retries: int = _API_RETRIES) -> None:
        self._build_api = build_api
        self._retries = retries
        self._cached_api: Any = None

    def _api(self) -> Any:
        # Built on first use, so opening the page costs no network call.
        if self._cached_api is None:
            self._cached_api = self._build_api()
        return self._cached_api

    def _run(self, request: Any) -> dict[str, Any]:
        response: dict[str, Any] = request.execute(num_retries=self._retries)
        return response

    def address(self) -> str:
        with _gmail_errors():
            return str(self._run(self._api().users().getProfile(userId="me"))["emailAddress"])

    def search(self, query: str, limit: int) -> list[MailSummary]:
        with _gmail_errors():
            messages = self._api().users().messages()
            listing = self._run(messages.list(userId="me", q=query, maxResults=limit))
            return [
                parse_message(self._run(messages.get(userId="me", id=item["id"], format="full")))
                for item in listing.get("messages", [])
            ]

    def fetch(self, message: MailSummary, attachment: AttachmentRef) -> bytes:
        if attachment.inline_data:
            return _decode(attachment.inline_data)
        with _gmail_errors():
            request = (
                self._api()
                .users()
                .messages()
                .attachments()
                .get(userId="me", messageId=message.message_id, id=attachment.reference)
            )
            return _decode(self._run(request)["data"])


def connect_gmail(auth: GmailAuth) -> GmailMailbox:
    return GmailMailbox(
        lambda: build("gmail", "v1", credentials=auth.credentials(), cache_discovery=False)
    )
