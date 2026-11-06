"""Exercise the real Google client against a scripted transport.

Requests go through `googleapiclient` and responses come back through its parsing, with no network
and no Google account, so a wrong method name or parameter fails here rather than in production.
"""

from __future__ import annotations

import base64
import json
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httplib2
import pytest
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import HttpMockSequence

from loan_triage.inbox.base import AttachmentRef, MailboxError, MailboxNotConnected, MailSummary
from loan_triage.inbox.gmail import SCOPES, GmailAuth, GmailMailbox, parse_message

PDF = b"%PDF-1.7 attachment \xfb\xff bytes"


def encoded(data: bytes) -> str:
    """Gmail sends URL-safe base64 with the padding removed."""
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def gmail_message(**overrides: Any) -> dict[str, Any]:
    message = {
        "id": "m1",
        "threadId": "t1",
        "internalDate": "1793000000000",
        "payload": {
            "mimeType": "multipart/mixed",
            "headers": [
                {"name": "FROM", "value": "Maya Rivera <maya@example.com>"},
                {"name": "subject", "value": "My application"},
            ],
            "parts": [
                {
                    "mimeType": "multipart/alternative",
                    "parts": [{"mimeType": "text/plain", "body": {"data": encoded(b"Hello")}}],
                },
                {
                    "mimeType": "application/pdf",
                    "filename": "paystub.pdf",
                    "body": {"attachmentId": "att-1", "size": 100},
                },
            ],
        },
    }
    return {**message, **overrides}


def ok(payload: dict[str, Any]) -> tuple[dict[str, str], str]:
    return {"status": "200"}, json.dumps(payload)


def failure(status: int, reason: str = "denied") -> tuple[dict[str, str], str]:
    return {"status": str(status)}, json.dumps({"error": {"code": status, "message": reason}})


def scripted(*responses: tuple[dict[str, str], str]) -> tuple[GmailMailbox, HttpMockSequence]:
    http = HttpMockSequence(list(responses))
    api = build("gmail", "v1", http=http, static_discovery=True)
    return GmailMailbox(lambda: api, retries=0), http


SUMMARY = MailSummary("m1", "t1", "a", "b", parse_message(gmail_message()).received_at)


def test_search_reads_the_fields_triage_needs() -> None:
    mailbox, _http = scripted(
        ok({"messages": [{"id": "m1", "threadId": "t1"}]}), ok(gmail_message())
    )
    message = mailbox.search("label:loans", 10)[0]
    assert (message.message_id, message.thread_id) == ("m1", "t1")
    assert message.sender == "Maya Rivera <maya@example.com>"
    assert message.subject == "My application"
    assert message.received_at.tzinfo is not None
    assert message.received_at.year == 2026


def test_search_sends_the_filter_and_limit() -> None:
    mailbox, http = scripted(ok({"messages": []}))
    mailbox.search("label:loan-applications has:attachment", 7)
    uri = http.request_sequence[0][0]
    query = parse_qs(urlparse(uri).query)
    assert query["q"] == ["label:loan-applications has:attachment"]
    assert query["maxResults"] == ["7"]
    assert "/users/me/messages" in uri


def test_empty_mailbox_returns_nothing() -> None:
    mailbox, _http = scripted(ok({"resultSizeEstimate": 0}))
    assert mailbox.search("label:x", 10) == []


def test_attachments_are_found_inside_nested_parts() -> None:
    nested = gmail_message()
    nested["payload"]["parts"][0]["parts"].append(
        {"filename": "w2.pdf", "mimeType": "application/pdf", "body": {"attachmentId": "att-2"}}
    )
    names = [item.filename for item in parse_message(nested).attachments]
    assert sorted(names) == ["paystub.pdf", "w2.pdf"]


def test_inline_body_parts_are_not_mistaken_for_attachments() -> None:
    assert len(parse_message(gmail_message()).attachments) == 1


def test_missing_headers_do_not_break_parsing() -> None:
    message = gmail_message()
    message["payload"]["headers"] = []
    summary = parse_message(message)
    assert summary.sender == "Unknown sender"
    assert summary.subject == "(no subject)"


def test_attachment_download_decodes_gmail_base64() -> None:
    mailbox, http = scripted(ok({"data": encoded(PDF), "size": len(PDF)}))
    reference = AttachmentRef("paystub.pdf", "application/pdf", "att-1")
    assert mailbox.fetch(SUMMARY, reference) == PDF
    assert "/messages/m1/attachments/att-1" in http.request_sequence[0][0]


def test_small_attachment_sent_inline_needs_no_request() -> None:
    mailbox, http = scripted()
    reference = AttachmentRef("a.pdf", "application/pdf", inline_data=encoded(PDF))
    assert mailbox.fetch(SUMMARY, reference) == PDF
    assert http.request_sequence == []


def test_rejected_sign_in_asks_to_reconnect() -> None:
    mailbox, _http = scripted(failure(401))
    with pytest.raises(MailboxNotConnected, match="gmail-login"):
        mailbox.search("q", 1)


@pytest.mark.parametrize("status", [403, 404, 429, 500])
def test_api_errors_become_safe_messages(status: int) -> None:
    mailbox, _http = scripted(failure(status, "Gmail API has not been enabled"))
    with pytest.raises(MailboxError, match="Gmail API has not been enabled"):
        mailbox.search("q", 1)


@pytest.mark.parametrize(
    "error",
    [httplib2.ServerNotFoundError("no dns"), TimeoutError("slow"), ConnectionResetError()],
)
def test_network_failures_become_safe_messages(error: Exception) -> None:
    def fail() -> Any:
        raise error

    with pytest.raises(MailboxError, match="Could not reach Gmail"):
        GmailMailbox(fail).search("q", 1)


def test_revoked_access_asks_to_reconnect() -> None:
    def revoked() -> Any:
        raise RefreshError("invalid_grant")

    with pytest.raises(MailboxNotConnected, match="revoked"):
        GmailMailbox(revoked).address()


def test_address_reads_the_profile() -> None:
    mailbox, _http = scripted(ok({"emailAddress": "loans@example.com"}))
    assert mailbox.address() == "loans@example.com"


def test_opening_a_mailbox_makes_no_request() -> None:
    def fail() -> Any:
        raise AssertionError("the page must not touch the network just by rendering")

    GmailMailbox(fail)


def test_the_app_only_ever_asks_for_read_access() -> None:
    assert SCOPES == ["https://www.googleapis.com/auth/gmail.readonly"]


def auth_in(tmp_path: Path) -> GmailAuth:
    return GmailAuth(tmp_path / "client.json", tmp_path / "secrets" / "token.json")


def fake_credentials(token: str = "access") -> Credentials:
    # google-auth reads a token with no expiry as expired, and would try to refresh it for real.
    return Credentials(
        token=token,
        refresh_token="refresh",
        token_uri="https://oauth2.googleapis.com/token",
        client_id="id",
        client_secret="secret",
        scopes=SCOPES,
        expiry=datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=1),
    )


def test_not_connected_without_a_token(tmp_path: Path) -> None:
    auth = auth_in(tmp_path)
    assert not auth.connected
    with pytest.raises(MailboxNotConnected):
        auth.credentials()


def test_token_file_is_private_to_its_owner(tmp_path: Path) -> None:
    auth = auth_in(tmp_path)
    auth._save(fake_credentials())
    mode = stat.S_IMODE((tmp_path / "secrets" / "token.json").stat().st_mode)
    assert mode == 0o600


def test_saving_over_a_loose_token_tightens_it(tmp_path: Path) -> None:
    token = tmp_path / "secrets" / "token.json"
    token.parent.mkdir()
    token.write_text("{}")
    token.chmod(0o644)
    auth_in(tmp_path)._save(fake_credentials())
    assert stat.S_IMODE(token.stat().st_mode) == 0o600


def test_saved_token_loads_back(tmp_path: Path) -> None:
    auth = auth_in(tmp_path)
    auth._save(fake_credentials("saved"))
    assert auth.connected
    assert auth.credentials().token == "saved"


def test_expired_token_is_refreshed_and_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auth = auth_in(tmp_path)
    auth._save(fake_credentials("old"))
    token = tmp_path / "secrets" / "token.json"
    stale = json.loads(token.read_text())
    stale["expiry"] = "2020-01-01T00:00:00Z"
    token.write_text(json.dumps(stale))

    def refreshed(self: Credentials, _request: Any) -> None:
        self.token = "new"

    monkeypatch.setattr(Credentials, "refresh", refreshed)
    assert auth.credentials().token == "new"
    assert json.loads(token.read_text())["token"] == "new"


def test_disconnect_removes_the_token(tmp_path: Path) -> None:
    auth = auth_in(tmp_path)
    auth._save(fake_credentials())
    auth.disconnect()
    auth.disconnect()  # safe to repeat
    assert not auth.connected


def test_login_without_a_client_file_explains_itself(tmp_path: Path) -> None:
    with pytest.raises(MailboxError, match="README"):
        auth_in(tmp_path).authorize()
