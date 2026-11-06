"""Offline reading engine.

A label parser that understands only this project's synthetic documents. It lets the workflow,
dashboard, and tests run with no API key, and it is labelled as such everywhere it appears. Its
accuracy says nothing about Claude's: it exercises the pipeline around the model, not the model.
"""

from __future__ import annotations

import re
from typing import Any

from loan_triage.domain.enums import DocumentType
from loan_triage.domain.schemas import (
    EXTRACTION_SCHEMAS,
    DocumentClassification,
    DocumentFields,
    ExtractedField,
    UnderwriterBrief,
    field_specs,
)
from loan_triage.llm.base import CallMetrics, DocumentInput, LLMError, LLMResult
from loan_triage.synth.layout import LABELS, TITLES

_ABSENT = ExtractedField(present=False, value="", quote="", page=0, confidence=0.95)


class DemoEngine:
    name = "demo"

    def _metrics(self) -> CallMetrics:
        return CallMetrics(engine=self.name, model="label-parser")

    def classify(self, document: DocumentInput) -> LLMResult[DocumentClassification]:
        if not document.record.has_text_layer:
            raise LLMError("unreadable", "the demo engine has no OCR")
        text = "\n".join(document.pages)
        for document_type, title in TITLES.items():
            if title in text:
                return LLMResult(
                    DocumentClassification(
                        document_type=document_type.value,
                        confidence=0.99,
                        rationale=f"Headed '{title}'.",
                    ),
                    self._metrics(),
                )
        return LLMResult(
            DocumentClassification(
                document_type="unknown", confidence=0.6, rationale="No known heading."
            ),
            self._metrics(),
        )

    def extract(
        self,
        document: DocumentInput,
        document_type: DocumentType,
        focus: list[str] | None = None,
    ) -> LLMResult[DocumentFields]:
        schema = EXTRACTION_SCHEMAS[document_type]
        values = {
            name: self._read(LABELS[document_type][name], document.pages)
            for name in field_specs(schema)
        }
        return LLMResult(schema(anomalies=[], **values), self._metrics())

    @staticmethod
    def _read(label: str, pages: list[str]) -> ExtractedField:
        pattern = re.compile(rf"^{re.escape(label)}\s+(.+)$", re.MULTILINE)
        for number, page in enumerate(pages, start=1):
            match = pattern.search(page)
            if match:
                value = match.group(1).strip()
                confidence = 0.40 if "?" in value else 0.97
                return ExtractedField(
                    present=True,
                    value=value,
                    quote=match.group(0),
                    page=number,
                    confidence=confidence,
                )
        return _ABSENT

    def write_brief(self, facts: dict[str, Any]) -> LLMResult[UnderwriterBrief]:
        reasons = facts["routing_reasons"]
        summary = (
            f"Offline demo brief. Risk score {facts['risk_score']} ({facts['risk_tier']}). "
            f"Routed here for {len(reasons)} reason(s)."
        )
        return LLMResult(
            UnderwriterBrief(
                summary=summary,
                key_concerns=list(reasons),
                verification_steps=["Compare the flagged fields against the source documents."],
            ),
            self._metrics(),
        )
