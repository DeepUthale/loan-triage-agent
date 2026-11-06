"""A sample book of files that fills every part of the dashboard."""

from __future__ import annotations

from datetime import UTC, datetime

from loan_triage.service import TriageService, Upload
from loan_triage.storage.cases import CaseRow
from loan_triage.synth.generator import SCENARIOS, Scenario, generate_file, render_document

_SEED = 7
_PORTFOLIO = (
    "clean_prime",
    "clean_near_prime",
    "duplicate_upload",
    "high_dti",
    "dti_knockout",
    "inflated_income",
    "name_mismatch",
    "tampered_paystub",
    "visible_injection",
    "hidden_injection",
    "smudged_value",
    "scanned_paystub",
    "missing_w2",
)


def submit_scenario(service: TriageService, scenario: Scenario, seed: int = _SEED) -> CaseRow:
    today = datetime.now(UTC).date()
    loan_file = generate_file(scenario, seed, today)
    uploads = [Upload(doc.filename, render_document(doc)) for doc in loan_file.documents]
    return service.submit(uploads, received_on=today)


def seed_portfolio(service: TriageService) -> list[tuple[str, CaseRow]]:
    chosen = [scenario for scenario in SCENARIOS if scenario.key in _PORTFOLIO]
    return [(scenario.key, submit_scenario(service, scenario)) for scenario in chosen]
