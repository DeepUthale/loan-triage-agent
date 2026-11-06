"""Lookups across a case's extractions."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

from loan_triage.domain.enums import DocumentType
from loan_triage.domain.models import DocumentExtraction

# A file may hold several of one type, such as two paystubs. Decisions use the newest.
_RECENCY_FIELD = {
    DocumentType.PAYSTUB: "pay_period_end",
    DocumentType.BANK_STATEMENT: "statement_period_end",
    DocumentType.CREDIT_REPORT: "report_date",
}


class CaseView:
    def __init__(self, extractions: Iterable[DocumentExtraction]) -> None:
        self.extractions = list(extractions)

    def of_type(self, document_type: DocumentType) -> list[DocumentExtraction]:
        return [e for e in self.extractions if e.document_type is document_type]

    def primary(self, document_type: DocumentType) -> DocumentExtraction | None:
        matches = self.of_type(document_type)
        if not matches:
            return None
        if document_type is DocumentType.W2:
            return max(matches, key=lambda e: e.integer("tax_year") or 0)
        recency = _RECENCY_FIELD.get(document_type)
        if recency is None:
            return matches[0]
        return max(matches, key=lambda e: e.day(recency) or date.min)
