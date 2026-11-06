"""A mailbox of synthetic emails, for trying the workflow without a Google account.

Like the demo reading engine, it is an explicit choice and labelled wherever it appears. The set
covers what a real inbox produces: clean files, a defect, an attack, an applicant who forgot a
document and replies with it, and an email with nothing usable attached.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from loan_triage.domain.enums import DocumentType
from loan_triage.inbox.base import AttachmentRef, MailboxError, MailSummary
from loan_triage.synth.generator import (
    SCENARIOS,
    LoanFile,
    SyntheticDocument,
    generate_file,
    render_document,
    unplanted_file,
)

_SEED = 11
_APPLICATIONS = (
    ("clean_prime", "Mortgage application documents", 26),
    ("missing_w2", "Application for pre-qualification", 20),
    ("inflated_income", "Loan application - please review", 14),
    ("visible_injection", "Urgent: home loan application", 8),
)
_REPLY_HOURS_AGO = 3
_REJECTED_HOURS_AGO = 1


def _sender(loan_file: LoanFile) -> str:
    name = loan_file.document(DocumentType.LOAN_APPLICATION).fields["applicant_name"]
    first, *_, last = name.split()
    return f"{name} <{first.lower()}.{last.lower()}@example.com>"


class DemoMailbox:
    name = "demo"

    def __init__(self) -> None:
        self._contents: dict[tuple[str, str], bytes] = {}
        self._loaded: list[MailSummary] | None = None

    def _messages(self) -> list[MailSummary]:
        # Built on first use: Streamlit runs the tab on every rerun, and rendering PDFs takes time.
        if self._loaded is None:
            self._loaded = self._build()
        return self._loaded

    def _add_document(self, message_id: str, document: SyntheticDocument) -> AttachmentRef:
        self._contents[message_id, document.filename] = render_document(document)
        return AttachmentRef(document.filename, "application/pdf", document.filename)

    def _build(self) -> list[MailSummary]:
        now = datetime.now(UTC)
        today = now.date()
        scenarios = {scenario.key: scenario for scenario in SCENARIOS}
        messages = []
        for key, subject, hours_ago in _APPLICATIONS:
            loan_file = generate_file(scenarios[key], _SEED, today)
            message_id = f"demo-{key}"
            references = tuple(self._add_document(message_id, d) for d in loan_file.documents)
            messages.append(
                MailSummary(
                    message_id,
                    f"thread-{key}",
                    _sender(loan_file),
                    subject,
                    now - timedelta(hours=hours_ago),
                    references,
                )
            )

        applicant = unplanted_file(scenarios["missing_w2"], _SEED, today)
        reply_id = "demo-missing_w2-reply"
        messages.append(
            MailSummary(
                reply_id,
                "thread-missing_w2",
                _sender(applicant),
                "Re: Application for pre-qualification",
                now - timedelta(hours=_REPLY_HOURS_AGO),
                (self._add_document(reply_id, applicant.document(DocumentType.W2)),),
            )
        )

        unusable_id = "demo-unusable"
        self._contents[unusable_id, "bank_statement.pdf"] = b"this file was not really a PDF"
        messages.append(
            MailSummary(
                unusable_id,
                "thread-unusable",
                "Sam Applicant <sam.applicant@example.com>",
                "Here are my documents",
                now - timedelta(hours=_REJECTED_HOURS_AGO),
                (
                    AttachmentRef("bank_statement.pdf", "application/pdf", "bank_statement.pdf"),
                    AttachmentRef("selfie.jpg", "image/jpeg", "selfie.jpg"),
                ),
            )
        )
        return messages

    def address(self) -> str:
        return "demo inbox"

    def search(self, query: str, limit: int) -> list[MailSummary]:
        return self._messages()[:limit]

    def fetch(self, message: MailSummary, attachment: AttachmentRef) -> bytes:
        self._messages()
        try:
            return self._contents[message.message_id, attachment.reference]
        except KeyError as exc:
            raise MailboxError(f"No attachment named {attachment.filename}.") from exc
