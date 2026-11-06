"""Measure the pipeline against planted outcomes.

The headline number is `unsafe_fast_tracks`: files that needed a person and did not get one. It
is the one failure a triage system cannot afford, so CI fails if it is ever non-zero. Accuracy
for its own sake comes second, because an over-cautious route only costs review time.
"""

from __future__ import annotations

import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from loan_triage.config import Settings
from loan_triage.domain.enums import CaseStatus, FieldKind, FlagCode, Route
from loan_triage.domain.models import DocumentExtraction
from loan_triage.domain.normalize import normalize_value
from loan_triage.domain.schemas import EXTRACTION_SCHEMAS, field_specs
from loan_triage.llm.base import ReadingEngine
from loan_triage.service import TriageService, Upload
from loan_triage.storage.cases import CaseRow
from loan_triage.synth.generator import (
    SCENARIOS,
    LoanFile,
    Scenario,
    generate_file,
    render_document,
)

EVAL_AS_OF = date(2026, 10, 30)


@dataclass
class CaseResult:
    case_key: str
    scenario: str
    expected_route: Route
    actual_route: Route | None
    missing_flags: list[str]
    status: str

    @property
    def route_correct(self) -> bool:
        return self.expected_route == self.actual_route

    @property
    def unsafe(self) -> bool:
        return self.actual_route is Route.FAST_TRACK and self.expected_route is not Route.FAST_TRACK


@dataclass
class EvalReport:
    engine: str
    model: str
    cases: list[CaseResult] = field(default_factory=list)
    fields_correct: int = 0
    fields_total: int = 0
    field_misses: Counter[str] = field(default_factory=Counter)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def unsafe_fast_tracks(self) -> int:
        return sum(case.unsafe for case in self.cases)

    @property
    def route_accuracy(self) -> float:
        return sum(c.route_correct for c in self.cases) / len(self.cases) if self.cases else 0.0

    @property
    def flag_recall(self) -> float:
        expected = sum(
            len(next(s for s in SCENARIOS if s.key == c.scenario).expected_flags)
            for c in self.cases
        )
        missed = sum(len(c.missing_flags) for c in self.cases)
        return (expected - missed) / expected if expected else 1.0

    @property
    def field_accuracy(self) -> float:
        return self.fields_correct / self.fields_total if self.fields_total else 0.0

    def confusion(self) -> dict[str, Counter[str]]:
        matrix: dict[str, Counter[str]] = {}
        for case in self.cases:
            actual = str(case.actual_route) if case.actual_route else "failed"
            matrix.setdefault(str(case.expected_route), Counter())[actual] += 1
        return matrix

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "model": self.model,
            "cases": len(self.cases),
            "unsafe_fast_tracks": self.unsafe_fast_tracks,
            "route_accuracy": round(self.route_accuracy, 4),
            "flag_recall": round(self.flag_recall, 4),
            "field_accuracy": round(self.field_accuracy, 4),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 4),
            "confusion": {key: dict(value) for key, value in self.confusion().items()},
            "failures": [
                {
                    "case": case.case_key,
                    "expected": case.expected_route,
                    "actual": case.actual_route,
                    "missing_flags": case.missing_flags,
                }
                for case in self.cases
                if not case.route_correct or case.missing_flags
            ],
            "top_field_misses": dict(self.field_misses.most_common(10)),
        }


def _uploads(loan_file: LoanFile) -> list[Upload]:
    return [Upload(doc.filename, render_document(doc)) for doc in loan_file.documents]


def _score_fields(report: EvalReport, loan_file: LoanFile, case: CaseRow) -> None:
    """Compare what was extracted with what was printed, for every planted document."""
    extractions = [
        DocumentExtraction.model_validate(item)
        for item in case.state.get("extractions", {}).values()
    ]
    for document in loan_file.documents:
        if document.scanned:
            continue
        specs = field_specs(EXTRACTION_SCHEMAS[document.document_type])
        extraction = next(
            (e for e in extractions if e.document_type is document.document_type), None
        )
        for name, spec in specs.items():
            printed = document.fields.get(name)
            if printed is None:
                continue
            report.fields_total += 1
            got = extraction.fields[name].value if extraction else None
            if got is not None and got == _expected(spec.kind, printed):
                report.fields_correct += 1
            else:
                report.field_misses[f"{document.document_type}.{name}"] += 1


def _expected(kind: FieldKind, printed: str) -> str | None:
    try:
        return normalize_value(kind, printed)
    except ValueError:
        return None


def _usage(report: EvalReport, service: TriageService, case_id: str) -> None:
    for event in service.audit.for_case(case_id):
        call = event.detail.get("call")
        if call:
            report.input_tokens += call["input_tokens"] + call["cache_read_tokens"]
            report.output_tokens += call["output_tokens"]
            report.cost_usd += call["cost_usd"] or 0.0


def run_scenario(
    service: TriageService, scenario: Scenario, seed: int, report: EvalReport
) -> CaseResult:
    loan_file = generate_file(scenario, seed, EVAL_AS_OF)
    case = service.submit(_uploads(loan_file), received_on=EVAL_AS_OF)
    flags = {FlagCode(flag["code"]) for flag in case.state.get("flags", [])}
    if case.status is not CaseStatus.FAILED:
        _score_fields(report, loan_file, case)
    _usage(report, service, case.case_id)
    return CaseResult(
        case_key=loan_file.case_key,
        scenario=scenario.key,
        expected_route=scenario.expected_route,
        actual_route=Route(case.route) if case.route else None,
        missing_flags=sorted(code.value for code in scenario.expected_flags - flags),
        status=str(case.status),
    )


def run_evaluation(
    settings: Settings,
    *,
    seeds: int = 3,
    engine: ReadingEngine | None = None,
    scenarios: tuple[Scenario, ...] = SCENARIOS,
) -> EvalReport:
    """Run in a throwaway data directory, so eval files never mix with real cases."""
    workspace = Path(tempfile.mkdtemp(prefix="loan-triage-eval-"))
    isolated = settings.model_copy(update={"data_dir": workspace})
    service = TriageService(isolated, engine=engine)
    report = EvalReport(engine=service.engine.name, model=settings.model)
    try:
        for scenario in scenarios:
            for seed in range(seeds):
                report.cases.append(run_scenario(service, scenario, seed, report))
    finally:
        service.close()
        shutil.rmtree(workspace, ignore_errors=True)
    return report
