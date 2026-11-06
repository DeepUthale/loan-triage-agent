"""Vocabulary shared across the pipeline."""

from __future__ import annotations

from enum import StrEnum


class DocumentType(StrEnum):
    LOAN_APPLICATION = "loan_application"
    PAYSTUB = "paystub"
    W2 = "w2"
    BANK_STATEMENT = "bank_statement"
    CREDIT_REPORT = "credit_report"
    UNKNOWN = "unknown"

    @property
    def label(self) -> str:
        return "W-2" if self is DocumentType.W2 else self.value.replace("_", " ")


class FieldKind(StrEnum):
    TEXT = "text"
    NAME = "name"
    MONEY = "money"
    DATE = "date"
    INTEGER = "integer"
    PERCENT = "percent"
    PAY_FREQUENCY = "pay_frequency"
    SSN_LAST4 = "ssn_last4"


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return list(Severity).index(self)


class FlagCode(StrEnum):
    # Completeness
    MISSING_DOCUMENT = "MISSING_DOCUMENT"
    MISSING_FIELD = "MISSING_FIELD"
    UNPARSEABLE_FIELD = "UNPARSEABLE_FIELD"
    UNRECOGNIZED_DOCUMENT = "UNRECOGNIZED_DOCUMENT"
    # Cross-document consistency
    NAME_MISMATCH = "NAME_MISMATCH"
    SSN_MISMATCH = "SSN_MISMATCH"
    EMPLOYER_MISMATCH = "EMPLOYER_MISMATCH"
    INCOME_MISMATCH = "INCOME_MISMATCH"
    W2_INCOME_DIVERGENCE = "W2_INCOME_DIVERGENCE"
    # Internal arithmetic
    PAYSTUB_ARITHMETIC = "PAYSTUB_ARITHMETIC"
    PAYSTUB_YTD_IMPLAUSIBLE = "PAYSTUB_YTD_IMPLAUSIBLE"
    BANK_STATEMENT_ARITHMETIC = "BANK_STATEMENT_ARITHMETIC"
    # Timeliness
    STALE_DOCUMENT = "STALE_DOCUMENT"
    FUTURE_DATED_DOCUMENT = "FUTURE_DATED_DOCUMENT"
    # Underwriting attention
    LARGE_UNSOURCED_DEPOSIT = "LARGE_UNSOURCED_DEPOSIT"
    # Extraction quality
    UNGROUNDED_FIELD = "UNGROUNDED_FIELD"
    LOW_CONFIDENCE_FIELD = "LOW_CONFIDENCE_FIELD"
    UNVERIFIABLE_DOCUMENT = "UNVERIFIABLE_DOCUMENT"
    EXTRACTION_FAILED = "EXTRACTION_FAILED"
    # Document integrity
    PROMPT_INJECTION_SUSPECTED = "PROMPT_INJECTION_SUSPECTED"
    HIDDEN_TEXT_DETECTED = "HIDDEN_TEXT_DETECTED"
    DUPLICATE_DOCUMENT = "DUPLICATE_DOCUMENT"
    MODEL_REPORTED_ANOMALY = "MODEL_REPORTED_ANOMALY"


class Route(StrEnum):
    FAST_TRACK = "fast_track"
    UNDERWRITER_REVIEW = "underwriter_review"
    REQUEST_DOCUMENTS = "request_documents"


class RiskTier(StrEnum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"
    # A core metric could not be computed, so the file cannot be called low risk.
    INDETERMINATE = "indeterminate"


class CaseStatus(StrEnum):
    PROCESSING = "processing"
    PENDING_REVIEW = "pending_review"
    AWAITING_DOCUMENTS = "awaiting_documents"
    FAST_TRACKED = "fast_tracked"
    ADVANCED = "advanced"
    DECLINED = "declined"
    FAILED = "failed"

    @property
    def is_open(self) -> bool:
        return self in {
            CaseStatus.PROCESSING,
            CaseStatus.PENDING_REVIEW,
            CaseStatus.AWAITING_DOCUMENTS,
        }


class ReviewAction(StrEnum):
    ADVANCE = "advance"
    DECLINE = "decline"
    REQUEST_DOCUMENTS = "request_documents"
    # Fix extracted values and re-run validation and risk; the case stays open.
    CORRECT = "correct"


class MailOutcome(StrEnum):
    IMPORTING = "importing"
    OPENED = "opened"
    ATTACHED = "attached"
    REJECTED = "rejected"
    # Never stored: these leave the email free to be imported later.
    SKIPPED = "skipped"
    DUPLICATE = "duplicate"
    ERROR = "error"


class ActorType(StrEnum):
    SYSTEM = "system"
    AGENT = "agent"
    HUMAN = "human"
