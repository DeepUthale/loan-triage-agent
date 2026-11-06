"""End-to-end runs through the real graph, checkpointer, and storage."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from loan_triage.domain.enums import ActorType, CaseStatus, DocumentType, ReviewAction, Route
from loan_triage.domain.models import FieldCorrection, ReviewDecision
from loan_triage.domain.schemas import DocumentFields, ExtractedField
from loan_triage.ingest.pdf import InvalidDocument
from loan_triage.llm.base import (
    CallMetrics,
    DocumentInput,
    LLMConfigurationError,
    LLMError,
    LLMResult,
)
from loan_triage.llm.demo import DemoEngine
from loan_triage.service import ReviewRejected, TriageService, Upload
from loan_triage.synth.generator import SCENARIOS, render_document, unplanted_file
from tests.conftest import AS_OF, ServiceFactory, uploads_for


def submit(service: TriageService, key: str):
    return service.submit(uploads_for(key), received_on=AS_OF)


def decision(action: ReviewAction, **overrides: Any) -> ReviewDecision:
    values = {"action": action, "reviewer": "j.chen", "rationale": "Reviewed the file."}
    return ReviewDecision(**{**values, **overrides})


def actions(service: TriageService, case_id: str) -> list[str]:
    return [event.action for event in service.audit.for_case(case_id)]


def document_id(case, document_type: DocumentType) -> str:
    return next(
        key
        for key, value in case.state["extractions"].items()
        if value["document_type"] == document_type
    )


def test_clean_file_fast_tracks_without_a_person(service: TriageService) -> None:
    case = submit(service, "clean_prime")
    assert case.status is CaseStatus.FAST_TRACKED
    assert case.route == Route.FAST_TRACK
    assert case.decided_by == "triage-agent"
    assert case.state.get("reviews", []) == []
    assert case.applicant and case.loan_amount


def test_risky_file_pauses_for_review(service: TriageService) -> None:
    case = submit(service, "high_dti")
    assert case.status is CaseStatus.PENDING_REVIEW
    assert case.state["brief"]["summary"]
    assert "human_advance" not in actions(service, case.case_id)


def test_review_resumes_and_closes_the_case(service: TriageService) -> None:
    case = submit(service, "high_dti")
    closed = service.decide(case.case_id, decision(ReviewAction.ADVANCE))
    assert closed.status is CaseStatus.ADVANCED
    assert closed.decided_by == "j.chen"


def test_interrupt_is_not_replayed_into_the_audit_log(service: TriageService) -> None:
    """LangGraph re-runs a node on resume, so writes before `interrupt()` would double up."""
    case = submit(service, "high_dti")
    service.decide(case.case_id, decision(ReviewAction.ADVANCE))
    recorded = actions(service, case.case_id)
    assert recorded.count("queued_for_underwriter") == 1
    assert recorded.count("human_advance") == 1


def test_decline_records_adverse_action_reasons(service: TriageService) -> None:
    case = submit(service, "high_dti")
    closed = service.decide(case.case_id, decision(ReviewAction.DECLINE, reason_codes=["RC01"]))
    assert closed.status is CaseStatus.DECLINED
    assert closed.state["outcome"]["reason_codes"] == ["RC01"]


def test_a_machine_never_declines(service: TriageService) -> None:
    """Even the worst file in the set stops at a person."""
    for key in ("dti_knockout", "low_credit", "visible_injection"):
        case = submit(service, key)
        assert case.status is CaseStatus.PENDING_REVIEW


def test_decline_without_reasons_is_refused(service: TriageService) -> None:
    case = submit(service, "high_dti")
    with pytest.raises(ReviewRejected, match="adverse-action"):
        service.decide(case.case_id, decision(ReviewAction.DECLINE))
    assert service.case(case.case_id).status is CaseStatus.PENDING_REVIEW


def test_unknown_reason_code_is_refused(service: TriageService) -> None:
    case = submit(service, "high_dti")
    with pytest.raises(ReviewRejected, match="Unknown"):
        service.decide(case.case_id, decision(ReviewAction.DECLINE, reason_codes=["RC99"]))


def test_rationale_is_mandatory(service: TriageService) -> None:
    case = submit(service, "high_dti")
    with pytest.raises(ReviewRejected, match="rationale"):
        service.decide(case.case_id, decision(ReviewAction.ADVANCE, rationale="  "))


def test_a_decided_case_cannot_be_decided_again(service: TriageService) -> None:
    case = submit(service, "high_dti")
    service.decide(case.case_id, decision(ReviewAction.ADVANCE))
    with pytest.raises(ReviewRejected, match="no longer"):
        service.decide(case.case_id, decision(ReviewAction.DECLINE, reason_codes=["RC01"]))


def test_two_reviewers_cannot_both_win(service: TriageService) -> None:
    case = submit(service, "high_dti")

    def attempt(reviewer: str) -> str:
        try:
            service.decide(case.case_id, decision(ReviewAction.ADVANCE, reviewer=reviewer))
        except ReviewRejected:
            return "rejected"
        return "accepted"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, ["a", "b"]))
    assert sorted(results) == ["accepted", "rejected"]
    assert actions(service, case.case_id).count("human_advance") == 1


def test_correction_reruns_policy_and_stays_with_the_reviewer(service: TriageService) -> None:
    case = submit(service, "tampered_paystub")
    assert any(flag["code"] == "PAYSTUB_ARITHMETIC" for flag in case.state["flags"])
    stub = case.state["extractions"][document_id(case, DocumentType.PAYSTUB)]
    gross = float(stub["fields"]["gross_pay_current"]["value"])
    deductions = float(stub["fields"]["total_deductions_current"]["value"])

    corrected = service.decide(
        case.case_id,
        decision(
            ReviewAction.CORRECT,
            corrections=[
                FieldCorrection(
                    document_id=stub["document_id"],
                    field="net_pay_current",
                    value=f"{gross - deductions:.2f}",
                )
            ],
        ),
    )

    assert corrected.status is CaseStatus.PENDING_REVIEW
    assert not any(flag["code"] == "PAYSTUB_ARITHMETIC" for flag in corrected.state["flags"])
    field = corrected.state["extractions"][stub["document_id"]]["fields"]["net_pay_current"]
    assert field["corrected_by"] == "j.chen"
    assert field["original_value"] is not None
    assert corrected.state["routing"]["reasons"][0] == "Already with an underwriter."
    assert actions(service, case.case_id).count("risk_scored") == 2


def test_invalid_correction_is_refused_before_anything_runs(service: TriageService) -> None:
    case = submit(service, "tampered_paystub")
    stub = document_id(case, DocumentType.PAYSTUB)
    with pytest.raises(ReviewRejected, match="net_pay_current"):
        service.decide(
            case.case_id,
            decision(
                ReviewAction.CORRECT,
                corrections=[
                    FieldCorrection(document_id=stub, field="net_pay_current", value="lots")
                ],
            ),
        )
    assert service.case(case.case_id).status is CaseStatus.PENDING_REVIEW


def test_missing_document_loop_completes_the_file(service: TriageService) -> None:
    case = submit(service, "missing_w2")
    assert case.status is CaseStatus.AWAITING_DOCUMENTS
    assert case.route == Route.REQUEST_DOCUMENTS
    assert "W-2" in case.state["requested_documents"][0]

    # The W-2 that was held back when this file was planted.
    w2 = next(
        document
        for document in unplanted_file(
            next(item for item in SCENARIOS if item.key == "missing_w2"), 1, AS_OF
        ).documents
        if document.document_type is DocumentType.W2
    )

    updated = service.add_documents(case.case_id, [Upload("w2.pdf", render_document(w2))])
    assert updated.status is CaseStatus.FAST_TRACKED
    assert not any(flag["code"] == "MISSING_DOCUMENT" for flag in updated.state["flags"])
    assert len(updated.state["documents"]) == 5


def test_reviewer_can_ask_for_more_documents(service: TriageService) -> None:
    case = submit(service, "high_dti")
    waiting = service.decide(
        case.case_id,
        decision(ReviewAction.REQUEST_DOCUMENTS, requested_documents=["Letter of explanation"]),
    )
    assert waiting.status is CaseStatus.AWAITING_DOCUMENTS
    assert waiting.state["requested_documents"] == ["Letter of explanation"]

    resumed = service.add_documents(case.case_id, uploads_for("clean_prime")[:1])
    assert resumed.status is CaseStatus.PENDING_REVIEW  # a person opened it; a person closes it


def test_documents_for_a_case_that_is_not_waiting_are_refused(service: TriageService) -> None:
    case = submit(service, "high_dti")
    with pytest.raises(ReviewRejected, match="not waiting"):
        service.add_documents(case.case_id, uploads_for("clean_prime")[:1])


def test_every_stage_reaches_the_audit_log(service: TriageService) -> None:
    case = submit(service, "high_dti")
    service.decide(case.case_id, decision(ReviewAction.ADVANCE))
    events = service.audit.for_case(case.case_id)
    assert {event.stage for event in events} >= {
        "intake", "ingest", "classify", "extract", "quality", "validate", "risk", "route",
        "review", "finalize",
    }  # fmt: skip
    assert {event.actor_type for event in events} == set(ActorType)
    human = next(event for event in events if event.actor_type is ActorType.HUMAN)
    assert human.actor == "j.chen"
    assert human.detail["policy_version"] == service.policy.version
    assert human.detail["policy_sha256"] == service.policy.source_sha256
    assert service.audit.verify().ok


def test_injection_is_escalated_whatever_the_rest_of_the_file_says(
    service: TriageService,
) -> None:
    case = submit(service, "visible_injection")
    assert case.status is CaseStatus.PENDING_REVIEW
    assert case.priority == 1
    assert case.risk_tier == "low"  # otherwise spotless, which is exactly the point


# Failure injection ----------------------------------------------------------------------------


class FlakyEngine(DemoEngine):
    """Fails on request, so each failure path can be driven through the real graph."""

    def __init__(self, fail_on: DocumentType | None = None, error: Exception | None = None) -> None:
        self.fail_on = fail_on
        self.error = error or LLMError("refusal", "general_harms")
        self.extract_calls: list[tuple[DocumentType, list[str] | None]] = []

    def extract(
        self,
        document: DocumentInput,
        document_type: DocumentType,
        focus: list[str] | None = None,
    ) -> LLMResult[DocumentFields]:
        self.extract_calls.append((document_type, focus))
        if document_type is self.fail_on:
            raise self.error
        return super().extract(document, document_type, focus)


def test_failed_document_reaches_a_person_not_a_dead_end(make_service: ServiceFactory) -> None:
    service = make_service(FlakyEngine(DocumentType.PAYSTUB))
    case = submit(service, "clean_prime")
    assert case.status is CaseStatus.PENDING_REVIEW
    assert any(flag["code"] == "EXTRACTION_FAILED" for flag in case.state["flags"])
    assert "document_failed" in actions(service, case.case_id)


def test_classification_failure_reaches_a_person(make_service: ServiceFactory) -> None:
    class Engine(DemoEngine):
        def classify(self, document: DocumentInput) -> Any:
            raise LLMError("api_error", "status 500")

    case = submit(make_service(Engine()), "clean_prime")
    assert case.status is CaseStatus.PENDING_REVIEW


def test_brief_failure_does_not_block_review(make_service: ServiceFactory) -> None:
    class Engine(DemoEngine):
        def write_brief(self, facts: dict[str, Any]) -> Any:
            raise LLMError("truncated")

    service = make_service(Engine())
    case = submit(service, "high_dti")
    assert case.status is CaseStatus.PENDING_REVIEW
    assert case.state["brief"] is None
    assert "brief_failed" in actions(service, case.case_id)


def test_bad_credentials_fail_the_case_visibly(make_service: ServiceFactory) -> None:
    service = make_service(FlakyEngine(DocumentType.PAYSTUB, LLMConfigurationError("bad key")))
    case = submit(service, "clean_prime")
    assert case.status is CaseStatus.FAILED
    assert "bad key" in (case.error or "")
    assert "pipeline_error" in actions(service, case.case_id)


class WeakFirstPass(FlakyEngine):
    """Misquotes one field on the first read and gets it right when asked to look again."""

    def extract(
        self,
        document: DocumentInput,
        document_type: DocumentType,
        focus: list[str] | None = None,
    ) -> LLMResult[DocumentFields]:
        result = super().extract(document, document_type, focus)
        if document_type is DocumentType.PAYSTUB and not focus:
            good: ExtractedField = result.output.gross_pay_ytd  # type: ignore[attr-defined]
            bad = good.model_copy(update={"quote": "YTD Gross $1.00", "value": "$1.00"})
            return LLMResult(
                result.output.model_copy(update={"gross_pay_ytd": bad}),
                CallMetrics(engine="demo", model="label-parser"),
            )
        return result


def test_self_correction_recovers_a_bad_first_read(make_service: ServiceFactory) -> None:
    engine = WeakFirstPass()
    service = make_service(engine)
    case = submit(service, "clean_prime")
    assert engine.extract_calls.count((DocumentType.PAYSTUB, ["gross_pay_ytd"])) == 1
    assert case.status is CaseStatus.FAST_TRACKED
    assert "fields_reread" in actions(service, case.case_id)


def test_reextraction_is_bounded(make_service: ServiceFactory) -> None:
    class AlwaysWeak(FlakyEngine):
        def extract(
            self,
            document: DocumentInput,
            document_type: DocumentType,
            focus: list[str] | None = None,
        ) -> LLMResult[DocumentFields]:
            result = super().extract(document, document_type, focus)
            if document_type is DocumentType.PAYSTUB:
                good: ExtractedField = result.output.gross_pay_ytd  # type: ignore[attr-defined]
                bad = good.model_copy(update={"quote": "invented", "value": "$1.00"})
                output = result.output.model_copy(update={"gross_pay_ytd": bad})
                return LLMResult(output, result.metrics)
            return result

    engine = AlwaysWeak()
    case = submit(make_service(engine), "clean_prime")
    assert sum(call[0] is DocumentType.PAYSTUB for call in engine.extract_calls) == 2
    assert case.status is CaseStatus.PENDING_REVIEW
    assert any(flag["code"] == "UNGROUNDED_FIELD" for flag in case.state["flags"])


def test_new_documents_do_not_trigger_rereads_of_old_ones(make_service: ServiceFactory) -> None:
    class BadBankRead(FlakyEngine):
        def extract(
            self,
            document: DocumentInput,
            document_type: DocumentType,
            focus: list[str] | None = None,
        ) -> LLMResult[DocumentFields]:
            result = super().extract(document, document_type, focus)
            if document_type is DocumentType.BANK_STATEMENT:
                bad = result.output.closing_balance.model_copy(  # type: ignore[attr-defined]
                    update={"quote": "invented", "value": "$1.00"}
                )
                return LLMResult(
                    result.output.model_copy(update={"closing_balance": bad}), result.metrics
                )
            return result

    engine = BadBankRead()
    service = make_service(engine)
    case = submit(service, "missing_w2")
    assert case.status is CaseStatus.PENDING_REVIEW  # the bad read outranks the missing W-2
    before = len(engine.extract_calls)
    service.decide(
        case.case_id, decision(ReviewAction.REQUEST_DOCUMENTS, requested_documents=["W-2"])
    )
    service.add_documents(case.case_id, uploads_for("clean_prime")[2:3])
    assert [call[0] for call in engine.extract_calls[before:]] == [DocumentType.W2]


def test_failed_reread_keeps_the_first_reading(make_service: ServiceFactory) -> None:
    class RereadFails(WeakFirstPass):
        def extract(
            self,
            document: DocumentInput,
            document_type: DocumentType,
            focus: list[str] | None = None,
        ) -> LLMResult[DocumentFields]:
            if focus:
                raise LLMError("connection_error")
            return super().extract(document, document_type, focus)

    case = submit(make_service(RereadFails()), "clean_prime")
    assert case.status is CaseStatus.PENDING_REVIEW
    assert any(flag["code"] == "UNGROUNDED_FIELD" for flag in case.state["flags"])


def test_case_survives_a_process_restart(make_service: ServiceFactory) -> None:
    """Review state lives in the checkpoint, so a pending file outlives the app that queued it."""
    first = make_service(None)
    case = submit(first, "high_dti")
    first.close()

    restarted = make_service(None)
    closed = restarted.decide(case.case_id, decision(ReviewAction.ADVANCE))
    assert closed.status is CaseStatus.ADVANCED


def test_submission_with_no_documents_is_refused(service: TriageService) -> None:
    with pytest.raises(ReviewRejected):
        service.submit([])


def test_non_pdf_upload_is_refused(service: TriageService) -> None:
    with pytest.raises(InvalidDocument):
        service.submit([Upload("notes.txt", b"hello")])
