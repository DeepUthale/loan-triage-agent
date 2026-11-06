"""Graph nodes.

Three rules keep this workflow safe to operate:

* A failure on one document becomes a flag on that document. The file carries on to a person.
* Each node writes its decision to the audit log, tagged with the policy and prompt in force.
* `interrupt()` is the first statement in a node. LangGraph re-runs the whole node on resume, so
  anything before the interrupt would happen twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from langgraph.types import Send, interrupt
from pydantic import BaseModel

from loan_triage.domain.enums import (
    ActorType,
    CaseStatus,
    DocumentType,
    FlagCode,
    ReviewAction,
    Route,
)
from loan_triage.domain.models import (
    Classification,
    DocumentExtraction,
    DocumentRecord,
    FieldResult,
    Flag,
    IncomeAnalysis,
    Outcome,
    ReviewDecision,
    RiskAssessment,
    RoutingDecision,
)
from loan_triage.domain.normalize import normalize_value
from loan_triage.graph.state import DocumentTask, Json, TriageState
from loan_triage.ingest.injection import scan_pages
from loan_triage.ingest.pdf import load_pdf
from loan_triage.llm.base import CallMetrics, DocumentInput, LLMError, ReadingEngine
from loan_triage.llm.prompts import PROMPT_VERSION
from loan_triage.rules.case_view import CaseView
from loan_triage.rules.grounding import (
    case_confidence,
    merge_reextraction,
    verify_extraction,
    weak_fields,
)
from loan_triage.rules.income import analyze_income
from loan_triage.rules.policy import Policy
from loan_triage.rules.risk import assess_risk
from loan_triage.rules.routing import route_case
from loan_triage.rules.validation import ValidationContext, validate_case
from loan_triage.storage.audit import AuditLog

AGENT = "triage-agent"


def _dump(model: BaseModel) -> Json:
    return model.model_dump(mode="json")


def _read_input(record: DocumentRecord) -> DocumentInput:
    path = Path(record.path)
    return DocumentInput(record=record, data=path.read_bytes(), pages=load_pdf(path).pages)


def _field_audit(extraction: DocumentExtraction) -> dict[str, Json]:
    """Per-field trace, without quotes: the case record keeps the evidence."""
    return {
        name: {
            "value": field.value,
            "confidence": round(field.confidence, 3),
            "model_confidence": round(field.model_confidence, 3),
            "grounded": field.grounded,
            "parse_error": field.parse_error,
        }
        for name, field in extraction.fields.items()
    }


@dataclass
class TriageNodes:
    engine: ReadingEngine
    audit: AuditLog
    policy: Policy

    def _log(
        self,
        case_id: str,
        stage: str,
        action: str,
        detail: Json | None = None,
        *,
        actor_type: ActorType = ActorType.AGENT,
        actor: str = AGENT,
    ) -> None:
        self.audit.record(
            case_id=case_id,
            actor_type=actor_type,
            actor=actor,
            stage=stage,
            action=action,
            detail={
                **(detail or {}),
                "engine": self.engine.name,
                "policy_version": self.policy.version,
                "policy_sha256": self.policy.source_sha256,
                "prompt_version": PROMPT_VERSION,
            },
        )

    def _log_failure(self, case_id: str, stage: str, document_id: str, error: LLMError) -> None:
        self._log(
            case_id,
            stage,
            "document_failed",
            {"document_id": document_id, "kind": error.kind, "detail": error.detail},
        )

    # Intake -----------------------------------------------------------------------------------

    def ingest(self, state: TriageState) -> TriageState:
        records: dict[str, Json] = {}
        for upload in state.get("uploads", []):
            parsed = load_pdf(Path(upload["path"]))
            record = DocumentRecord(
                document_id=upload["document_id"],
                filename=upload["filename"],
                path=upload["path"],
                sha256=parsed.sha256,
                content_sha256=parsed.content_sha256,
                page_count=len(parsed.pages),
                has_text_layer=parsed.has_text_layer,
                findings=[*parsed.findings, *scan_pages(parsed.pages)],
            )
            records[record.document_id] = _dump(record)
            self._log(
                state["case_id"],
                "ingest",
                "document_received",
                {
                    "document_id": record.document_id,
                    "filename": record.filename,
                    "sha256": record.sha256,
                    "pages": record.page_count,
                    "has_text_layer": record.has_text_layer,
                    "findings": [str(finding.code) for finding in record.findings],
                },
                actor_type=ActorType.SYSTEM,
                actor="intake",
            )
        return {
            "documents": records,
            "new_document_ids": list(records),
            "uploads": [],
            "reextraction_passes": 0,
            "weak_fields": {},
        }

    def dispatch_classification(self, state: TriageState) -> list[Send] | str:
        ids = state.get("new_document_ids", [])
        if not ids:
            return "assess_quality"
        return [
            Send(
                "classify_document",
                DocumentTask(case_id=state["case_id"], document=state["documents"][document_id]),
            )
            for document_id in ids
        ]

    def classify_document(self, task: DocumentTask) -> TriageState:
        record = DocumentRecord.model_validate(task["document"])
        document_id = record.document_id
        try:
            result = self.engine.classify(_read_input(record))
        except LLMError as error:
            self._log_failure(task["case_id"], "classify", document_id, error)
            fallback = Classification(
                document_type=DocumentType.UNKNOWN, confidence=0.0, rationale=str(error)
            )
            return {
                "classifications": {document_id: _dump(fallback)},
                "failures": {document_id: error.kind},
            }
        classification = Classification(
            document_type=DocumentType(result.output.document_type),
            confidence=result.output.confidence,
            rationale=result.output.rationale,
        )
        self._log(
            task["case_id"],
            "classify",
            "document_classified",
            {"document_id": document_id, **_dump(classification), "call": _dump(result.metrics)},
        )
        return {"classifications": {document_id: _dump(classification)}}

    def plan_extraction(self, _state: TriageState) -> TriageState:
        """Fan-in point: runs once, after every classification task has finished."""
        return {}

    def dispatch_extraction(self, state: TriageState) -> list[Send] | str:
        tasks = []
        for document_id in state.get("new_document_ids", []):
            document_type = state["classifications"][document_id]["document_type"]
            if document_type != DocumentType.UNKNOWN:
                tasks.append(
                    Send(
                        "extract_document",
                        DocumentTask(
                            case_id=state["case_id"],
                            document=state["documents"][document_id],
                            document_type=document_type,
                        ),
                    )
                )
        return tasks or "assess_quality"

    def _read_fields(
        self, task: DocumentTask, focus: list[str] | None
    ) -> tuple[DocumentExtraction, CallMetrics]:
        record = DocumentRecord.model_validate(task["document"])
        document_type = DocumentType(task["document_type"])
        document = _read_input(record)
        result = self.engine.extract(document, document_type, focus)
        extraction = verify_extraction(
            record.document_id, document_type, result.output, document.pages, self.policy.quality
        )
        return extraction, result.metrics

    def extract_document(self, task: DocumentTask) -> TriageState:
        document_id = task["document"]["document_id"]
        try:
            extraction, metrics = self._read_fields(task, None)
        except LLMError as error:
            self._log_failure(task["case_id"], "extract", document_id, error)
            return {"failures": {document_id: error.kind}}
        self._log(
            task["case_id"],
            "extract",
            "fields_extracted",
            {
                "document_id": document_id,
                "document_type": str(extraction.document_type),
                "fields": _field_audit(extraction),
                "anomalies": extraction.anomalies,
                "call": _dump(metrics),
            },
        )
        return {"extractions": {document_id: _dump(extraction)}}

    # Self-correction loop ---------------------------------------------------------------------

    def assess_quality(self, state: TriageState) -> TriageState:
        passes = state.get("reextraction_passes", 0)
        # Only this round's documents: earlier ones had their re-read when they first arrived.
        current = set(state.get("new_document_ids", []))
        weak = {
            extraction.document_id: fields
            for extraction in _extractions(state)
            if extraction.document_id in current
            and (fields := weak_fields(extraction, self.policy.quality))
        }
        retry = weak if passes < self.policy.quality.max_reextraction_passes else {}
        self._log(
            state["case_id"],
            "quality",
            "reextraction_planned" if retry else "quality_gate_passed",
            {"weak_fields": weak, "pass": passes, "will_retry": bool(retry)},
        )
        return {"weak_fields": retry, "reextraction_passes": passes + 1 if retry else passes}

    def dispatch_reextraction(self, state: TriageState) -> list[Send] | str:
        weak = state.get("weak_fields", {})
        if not weak:
            return "validate"
        return [
            Send(
                "reextract_document",
                DocumentTask(
                    case_id=state["case_id"],
                    document=state["documents"][document_id],
                    document_type=state["extractions"][document_id]["document_type"],
                    focus=fields,
                    previous=state["extractions"][document_id],
                ),
            )
            for document_id, fields in weak.items()
        ]

    def reextract_document(self, task: DocumentTask) -> TriageState:
        document_id = task["document"]["document_id"]
        previous = DocumentExtraction.model_validate(task["previous"])
        try:
            fresh, metrics = self._read_fields(task, task["focus"])
        except LLMError as error:
            # The first-pass reading is still valid, so keep it rather than losing the document.
            self._log_failure(task["case_id"], "reextract", document_id, error)
            return {}
        merged = merge_reextraction(previous, fresh, task["focus"])
        self._log(
            task["case_id"],
            "reextract",
            "fields_reread",
            {
                "document_id": document_id,
                "focus": task["focus"],
                "fields": {name: _field_audit(merged)[name] for name in task["focus"]},
                "call": _dump(metrics),
            },
        )
        return {"extractions": {document_id: _dump(merged)}}

    # Deterministic decisioning ----------------------------------------------------------------

    def validate(self, state: TriageState) -> TriageState:
        view = CaseView(_extractions(state))
        income = analyze_income(view)
        flags = validate_case(
            ValidationContext(
                view=view,
                documents={
                    key: DocumentRecord.model_validate(value)
                    for key, value in state["documents"].items()
                },
                classifications={
                    key: Classification.model_validate(value)
                    for key, value in state["classifications"].items()
                },
                failures=state.get("failures", {}),
                income=income,
                policy=self.policy,
                as_of=date.fromisoformat(state["received_on"]),
            )
        )
        self._log(
            state["case_id"],
            "validate",
            "validation_completed",
            {
                "flags": [
                    {"code": str(flag.code), "severity": str(flag.severity)} for flag in flags
                ],
                "income": _dump(income),
            },
            actor_type=ActorType.SYSTEM,
            actor="rules-engine",
        )
        return {"income": _dump(income), "flags": [_dump(flag) for flag in flags]}

    def assess_risk(self, state: TriageState) -> TriageState:
        risk = assess_risk(
            CaseView(_extractions(state)),
            IncomeAnalysis.model_validate(state["income"]),
            date.fromisoformat(state["received_on"]),
            self.policy.risk,
        )
        self._log(
            state["case_id"],
            "risk",
            "risk_scored",
            _dump(risk),
            actor_type=ActorType.SYSTEM,
            actor="rules-engine",
        )
        return {"risk": _dump(risk)}

    def route(self, state: TriageState) -> TriageState:
        decision = route_case(
            [Flag.model_validate(flag) for flag in state["flags"]],
            RiskAssessment.model_validate(state["risk"]),
            case_confidence(_extractions(state)),
            self.policy,
            reviewed=state.get("review_cycles", 0) > 0,
        )
        self._log(
            state["case_id"],
            "route",
            "case_routed",
            _dump(decision),
            actor_type=ActorType.SYSTEM,
            actor="rules-engine",
        )
        return {"routing": _dump(decision)}

    def next_after_route(self, state: TriageState) -> str:
        match RoutingDecision.model_validate(state["routing"]).route:
            case Route.FAST_TRACK:
                return "finalize"
            case Route.REQUEST_DOCUMENTS:
                return "request_documents"
            case Route.UNDERWRITER_REVIEW:
                return "prepare_review"

    # Human in the loop ------------------------------------------------------------------------

    def prepare_review(self, state: TriageState) -> TriageState:
        brief: Json | None = None
        try:
            result = self.engine.write_brief(_brief_facts(state))
        except LLMError as error:
            # A brief is a convenience. The review itself must never depend on it.
            self._log(state["case_id"], "review", "brief_failed", {"kind": error.kind})
        else:
            brief = _dump(result.output)
            self._log(state["case_id"], "review", "brief_written", {"call": _dump(result.metrics)})
        self._log(
            state["case_id"],
            "review",
            "queued_for_underwriter",
            {"priority": state["routing"]["priority"], "reasons": state["routing"]["reasons"]},
        )
        return {"brief": brief}

    def human_review(self, state: TriageState) -> TriageState:
        payload = interrupt({"kind": "review", "case_id": state["case_id"]})
        decision = ReviewDecision.model_validate(payload)
        unresolved = [flag["code"] for flag in state["flags"] if flag["severity"] == "critical"]
        self._log(
            state["case_id"],
            "review",
            f"human_{decision.action}",
            {**_dump(decision), "critical_flags_at_decision": unresolved},
            actor_type=ActorType.HUMAN,
            actor=decision.reviewer,
        )
        return {
            "reviews": [_dump(decision)],
            "review_cycles": state.get("review_cycles", 0) + 1,
        }

    def next_after_review(self, state: TriageState) -> str:
        decision = ReviewDecision.model_validate(state["reviews"][-1])
        match decision.action:
            case ReviewAction.CORRECT:
                return "apply_corrections"
            case ReviewAction.REQUEST_DOCUMENTS:
                return "request_documents"
            case ReviewAction.ADVANCE | ReviewAction.DECLINE:
                return "finalize"

    def apply_corrections(self, state: TriageState) -> TriageState:
        decision = ReviewDecision.model_validate(state["reviews"][-1])
        updated: dict[str, DocumentExtraction] = {}
        for correction in decision.corrections:
            extraction = updated.get(correction.document_id) or DocumentExtraction.model_validate(
                state["extractions"][correction.document_id]
            )
            field = extraction.fields[correction.field]
            revised: FieldResult = field.model_copy(
                update={
                    "present": True,
                    "raw": correction.value,
                    "value": normalize_value(field.kind, correction.value),
                    "confidence": 1.0,
                    "parse_error": None,
                    "corrected_by": decision.reviewer,
                    "correction_reason": decision.rationale,
                    "original_value": field.value if field.value is not None else field.raw,
                }
            )
            fields = {**extraction.fields, correction.field: revised}
            updated[correction.document_id] = extraction.model_copy(update={"fields": fields})
        return {"extractions": {key: _dump(value) for key, value in updated.items()}}

    def request_documents(self, state: TriageState) -> TriageState:
        decisions = state.get("reviews", [])
        requested = (
            decisions[-1].get("requested_documents", [])
            if decisions and decisions[-1]["action"] == ReviewAction.REQUEST_DOCUMENTS
            else []
        )
        if not requested:
            requested = [
                flag["message"]
                for flag in state["flags"]
                if flag["code"] == FlagCode.MISSING_DOCUMENT
            ]
        self._log(state["case_id"], "documents", "documents_requested", {"requested": requested})
        return {"requested_documents": requested}

    def await_documents(self, state: TriageState) -> TriageState:
        uploads = interrupt({"kind": "documents", "case_id": state["case_id"]})
        return {"uploads": uploads, "requested_documents": []}

    # Close-out --------------------------------------------------------------------------------

    def finalize(self, state: TriageState) -> TriageState:
        decisions = state.get("reviews", [])
        if decisions:
            last = ReviewDecision.model_validate(decisions[-1])
            outcome = Outcome(
                status=(
                    CaseStatus.DECLINED
                    if last.action is ReviewAction.DECLINE
                    else CaseStatus.ADVANCED
                ),
                decided_by=last.reviewer,
                rationale=last.rationale,
                reason_codes=last.reason_codes,
            )
            self._log(
                state["case_id"],
                "finalize",
                "case_closed",
                _dump(outcome),
                actor_type=ActorType.SYSTEM,
                actor="workflow",
            )
        else:
            outcome = Outcome(
                status=CaseStatus.FAST_TRACKED,
                decided_by=AGENT,
                rationale=state["routing"]["reasons"][0],
            )
            self._log(state["case_id"], "finalize", "case_fast_tracked", _dump(outcome))
        return {"outcome": _dump(outcome)}


def _extractions(state: TriageState) -> list[DocumentExtraction]:
    return [
        DocumentExtraction.model_validate(item) for item in state.get("extractions", {}).values()
    ]


def _brief_facts(state: TriageState) -> Json:
    """Only what the underwriter's brief needs. No names, addresses, or identifiers."""
    risk = RiskAssessment.model_validate(state["risk"])
    return {
        "routing_reasons": state["routing"]["reasons"],
        "risk_score": risk.score,
        "risk_tier": str(risk.tier),
        "knockouts": risk.knockouts,
        "risk_factors": [
            {"factor": factor.label, "value": str(factor.value), "points": factor.points}
            for factor in risk.factors
            if factor.points or factor.value is None
        ],
        "income": state["income"],
        "flags": [
            {"severity": flag["severity"], "message": flag["message"]} for flag in state["flags"]
        ],
    }
