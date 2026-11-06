"""The contract every reading engine meets, so the graph never depends on a provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel

from loan_triage.domain.enums import DocumentType
from loan_triage.domain.models import DocumentRecord
from loan_triage.domain.schemas import DocumentClassification, DocumentFields, UnderwriterBrief


class LLMError(Exception):
    """A document-level failure. The file continues, flagged, to a human."""

    def __init__(self, kind: str, detail: str = "") -> None:
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind = kind
        self.detail = detail


class LLMConfigurationError(Exception):
    """Credentials or request shape are wrong. No document would succeed, so stop the run."""


class CallMetrics(BaseModel):
    engine: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_ms: int = 0
    cost_usd: float | None = None
    stop_reason: str = "end_turn"
    served_by_fallback: bool = False


@dataclass(frozen=True)
class LLMResult[T]:
    output: T
    metrics: CallMetrics


@dataclass(frozen=True)
class DocumentInput:
    record: DocumentRecord
    data: bytes
    pages: list[str]


class ReadingEngine(Protocol):
    name: str

    def classify(self, document: DocumentInput) -> LLMResult[DocumentClassification]: ...

    def extract(
        self,
        document: DocumentInput,
        document_type: DocumentType,
        focus: list[str] | None = None,
    ) -> LLMResult[DocumentFields]: ...

    def write_brief(self, facts: dict[str, Any]) -> LLMResult[UnderwriterBrief]: ...
