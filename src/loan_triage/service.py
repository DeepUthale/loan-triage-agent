"""Application service: the one entry point the dashboard, CLI, and evals all share."""

from __future__ import annotations

import logging
import secrets
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from loan_triage.config import Settings
from loan_triage.domain.enums import ActorType, CaseStatus, DocumentType, ReviewAction
from loan_triage.domain.models import ReviewDecision
from loan_triage.domain.normalize import normalize_value
from loan_triage.graph.builder import build_graph
from loan_triage.graph.nodes import TriageNodes
from loan_triage.graph.state import TriageState
from loan_triage.ingest.pdf import parse_pdf, validate_pdf_bytes
from loan_triage.llm.anthropic_engine import AnthropicEngine
from loan_triage.llm.base import LLMConfigurationError, ReadingEngine
from loan_triage.llm.demo import DemoEngine
from loan_triage.rules.policy import Policy, load_policy
from loan_triage.storage.audit import AuditLog
from loan_triage.storage.cases import CaseRepository, CaseRow
from loan_triage.storage.db import Database
from loan_triage.storage.mail import MailLedger

logger = logging.getLogger(__name__)


class ReviewRejected(ValueError):
    """The decision is invalid or the case is no longer open for it. Safe to show the user."""


@dataclass(frozen=True)
class Upload:
    filename: str
    data: bytes


def build_engine(settings: Settings) -> ReadingEngine:
    return AnthropicEngine(settings) if settings.llm_mode == "anthropic" else DemoEngine()


class TriageService:
    def __init__(self, settings: Settings, engine: ReadingEngine | None = None) -> None:
        self.settings = settings
        self.policy: Policy = load_policy(settings.policy_path)
        self.engine = engine or build_engine(settings)
        database = Database(settings.database_path)
        self.audit = AuditLog(database)
        self.cases = CaseRepository(database)
        self.mail_ledger = MailLedger(database)

        settings.documents_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_connection = sqlite3.connect(
            settings.checkpoint_path, check_same_thread=False
        )
        saver = SqliteSaver(self._checkpoint_connection)
        saver.setup()
        self.graph = build_graph(TriageNodes(self.engine, self.audit, self.policy), saver)

    def close(self) -> None:
        self._checkpoint_connection.close()

    # Commands ---------------------------------------------------------------------------------

    def check_upload(self, upload: Upload) -> None:
        """Raise `InvalidDocument` unless this is a readable PDF within the size limit."""
        validate_pdf_bytes(upload.data, max_bytes=self.settings.max_upload_bytes)
        parse_pdf(upload.data)

    def submit(
        self,
        uploads: list[Upload],
        *,
        received_on: date | None = None,
        source: dict[str, Any] | None = None,
    ) -> CaseRow:
        if not uploads:
            raise ReviewRejected("Add at least one document.")
        for upload in uploads:
            self.check_upload(upload)

        now = datetime.now(UTC)
        case_id = f"LN-{now:%Y%m%d}-{secrets.token_hex(3).upper()}"
        self.cases.create(case_id, now.isoformat())
        self.audit.record(
            case_id=case_id,
            actor_type=ActorType.SYSTEM,
            actor="intake",
            stage="intake",
            action="case_opened",
            detail={
                "documents": len(uploads),
                "engine": self.engine.name,
                "source": source or {"channel": "upload"},
            },
        )
        initial: TriageState = {
            "case_id": case_id,
            "received_on": (received_on or now.date()).isoformat(),
            "uploads": self._store(case_id, uploads),
            "review_cycles": 0,
            "source": source or {"channel": "upload"},
        }
        return self._run(case_id, initial)

    def decide(self, case_id: str, decision: ReviewDecision) -> CaseRow:
        self._check_decision(case_id, decision)
        if not self.cases.claim(case_id, CaseStatus.PENDING_REVIEW):
            raise ReviewRejected(f"{case_id} is no longer awaiting review.")
        return self._run(case_id, Command(resume=decision.model_dump(mode="json")))

    def add_documents(self, case_id: str, uploads: list[Upload]) -> CaseRow:
        if not uploads:
            raise ReviewRejected("Add at least one document.")
        for upload in uploads:
            self.check_upload(upload)
        if not self.cases.claim(case_id, CaseStatus.AWAITING_DOCUMENTS):
            raise ReviewRejected(f"{case_id} is not waiting for documents.")
        return self._run(case_id, Command(resume=self._store(case_id, uploads)))

    # Queries ----------------------------------------------------------------------------------

    def case(self, case_id: str) -> CaseRow:
        row = self.cases.get(case_id)
        if row is None:
            raise KeyError(case_id)
        return row

    # Internals --------------------------------------------------------------------------------

    def _store(self, case_id: str, uploads: list[Upload]) -> list[dict[str, str]]:
        directory = self.settings.documents_dir / case_id
        directory.mkdir(parents=True, exist_ok=True)
        stored = []
        for upload in uploads:
            # Generated name on disk: the uploaded filename is shown to people, never trusted.
            document_id = f"doc-{uuid.uuid4().hex[:10]}"
            path = directory / f"{document_id}.pdf"
            path.write_bytes(upload.data)
            stored.append(
                {"document_id": document_id, "filename": upload.filename, "path": str(path)}
            )
        return stored

    def _check_decision(self, case_id: str, decision: ReviewDecision) -> None:
        if not decision.reviewer.strip():
            raise ReviewRejected("A reviewer name is required.")
        if not decision.rationale.strip():
            raise ReviewRejected("A rationale is required.")
        if decision.action is ReviewAction.DECLINE:
            if not decision.reason_codes:
                raise ReviewRejected("A decline needs at least one adverse-action reason.")
            unknown = set(decision.reason_codes) - set(self.policy.adverse_action_reasons)
            if unknown:
                raise ReviewRejected(f"Unknown reason codes: {sorted(unknown)}")
        if decision.action is ReviewAction.REQUEST_DOCUMENTS and not decision.requested_documents:
            raise ReviewRejected("Say which documents are needed.")
        if decision.action is ReviewAction.CORRECT:
            if not decision.corrections:
                raise ReviewRejected("Enter at least one correction.")
            extractions = self.case(case_id).state.get("extractions", {})
            for correction in decision.corrections:
                fields = extractions.get(correction.document_id, {}).get("fields", {})
                if correction.field not in fields:
                    raise ReviewRejected(f"Unknown field: {correction.field}")
                try:
                    normalize_value(fields[correction.field]["kind"], correction.value)
                except ValueError as exc:
                    raise ReviewRejected(f"{correction.field}: {exc}") from exc

    def _run(self, case_id: str, payload: Any) -> CaseRow:
        config = {"configurable": {"thread_id": case_id}}
        try:
            self.graph.invoke(payload, config)
        except Exception as exc:
            # Whatever went wrong, the file must end up in front of a person, visibly.
            logger.error("triage failed for %s: %s", case_id, exc, exc_info=_is_unexpected(exc))
            self.audit.record(
                case_id=case_id,
                actor_type=ActorType.SYSTEM,
                actor="workflow",
                stage="workflow",
                action="pipeline_error",
                detail={"error": type(exc).__name__, "message": str(exc)[:500]},
            )
            self.cases.update(case_id, status=CaseStatus.FAILED, priority=1, error=str(exc)[:500])
            return self.case(case_id)
        return self._sync(case_id, config)

    def _sync(self, case_id: str, config: dict[str, Any]) -> CaseRow:
        snapshot = self.graph.get_state(config)
        state: TriageState = snapshot.values
        routing = state.get("routing", {})
        risk = state.get("risk", {})
        outcome = state.get("outcome", {})

        if snapshot.interrupts:
            kind = snapshot.interrupts[0].value["kind"]
            status = (
                CaseStatus.PENDING_REVIEW if kind == "review" else (CaseStatus.AWAITING_DOCUMENTS)
            )
        else:
            status = CaseStatus(outcome["status"])

        self.cases.update(
            case_id,
            status=status,
            route=routing.get("route"),
            priority=routing.get("priority"),
            risk_tier=risk.get("tier"),
            risk_score=risk.get("score"),
            confidence=routing.get("case_confidence"),
            applicant=_headline(state, "applicant_name"),
            loan_amount=_headline(state, "loan_amount"),
            flag_count=len(state.get("flags", [])),
            decided_by=outcome.get("decided_by"),
            state=state,
        )
        return self.case(case_id)


def _is_unexpected(error: Exception) -> bool:
    """A configuration problem already says what to fix. Anything else needs its traceback."""
    return not isinstance(error, LLMConfigurationError)


def _headline(state: TriageState, field: str) -> str | None:
    for extraction in state.get("extractions", {}).values():
        if extraction["document_type"] == DocumentType.LOAN_APPLICATION:
            value = extraction["fields"].get(field, {}).get("value")
            if value is not None:
                return str(value)
    return None
