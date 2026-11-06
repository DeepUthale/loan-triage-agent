"""Records the pipeline passes around.

Graph state holds these as plain JSON (`model_dump(mode="json")`), which keeps checkpoints
portable and readable. Decimals therefore travel as strings and are re-parsed at each boundary.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, Field

from loan_triage.domain.enums import (
    CaseStatus,
    DocumentType,
    FieldKind,
    FlagCode,
    ReviewAction,
    RiskTier,
    Route,
    Severity,
)


class Flag(BaseModel):
    code: FlagCode
    severity: Severity
    message: str
    document_ids: list[str] = Field(default_factory=list)
    fields: list[str] = Field(default_factory=list)


class IntegrityFinding(BaseModel):
    code: FlagCode
    detail: str
    page: int
    excerpt: str = ""


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    path: str
    sha256: str
    content_sha256: str
    page_count: int
    has_text_layer: bool
    findings: list[IntegrityFinding] = Field(default_factory=list)


class Classification(BaseModel):
    document_type: DocumentType
    confidence: float
    rationale: str


class FieldResult(BaseModel):
    name: str
    kind: FieldKind
    required: bool
    present: bool
    raw: str = ""
    value: str | None = None
    quote: str = ""
    page: int = 0
    model_confidence: float = 0.0
    confidence: float = 0.0
    # True: quote found in the file. False: not found. None: no text layer to check against.
    grounded: bool | None = None
    parse_error: str | None = None
    corrected_by: str | None = None
    correction_reason: str | None = None
    original_value: str | None = None


class DocumentExtraction(BaseModel):
    document_id: str
    document_type: DocumentType
    fields: dict[str, FieldResult]
    anomalies: list[str] = Field(default_factory=list)
    attempts: int = 1

    def _value(self, name: str) -> str | None:
        result = self.fields.get(name)
        return result.value if result is not None else None

    def text(self, name: str) -> str | None:
        return self._value(name)

    def money(self, name: str) -> Decimal | None:
        value = self._value(name)
        return Decimal(value) if value is not None else None

    def integer(self, name: str) -> int | None:
        value = self._value(name)
        return int(value) if value is not None else None

    def day(self, name: str) -> date | None:
        value = self._value(name)
        return date.fromisoformat(value) if value is not None else None


class IncomeAnalysis(BaseModel):
    stated_monthly: Decimal | None = None
    paystub_monthly: Decimal | None = None
    w2_monthly: Decimal | None = None
    verified_monthly: Decimal | None = None
    basis: str = "unverified"


class RiskFactor(BaseModel):
    key: str
    label: str
    reason_code: str
    adverse_action_reason: str
    value: Decimal | None
    points: int


class RiskAssessment(BaseModel):
    score: int
    tier: RiskTier
    factors: list[RiskFactor]
    knockouts: list[str] = Field(default_factory=list)
    missing_core_factors: list[str] = Field(default_factory=list)


class RoutingDecision(BaseModel):
    route: Route
    priority: int
    reasons: list[str]
    case_confidence: float


class FieldCorrection(BaseModel):
    document_id: str
    field: str
    value: str


class ReviewDecision(BaseModel):
    action: ReviewAction
    reviewer: str
    rationale: str
    reason_codes: list[str] = Field(default_factory=list)
    corrections: list[FieldCorrection] = Field(default_factory=list)
    requested_documents: list[str] = Field(default_factory=list)


class Outcome(BaseModel):
    status: CaseStatus
    decided_by: str
    rationale: str
    reason_codes: list[str] = Field(default_factory=list)
