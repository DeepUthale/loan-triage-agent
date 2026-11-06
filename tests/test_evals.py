from datetime import date

import pytest

from loan_triage.config import Settings
from loan_triage.domain.enums import Route
from loan_triage.evals.runner import CaseResult, EvalReport, run_evaluation
from loan_triage.synth.generator import SCENARIOS, generate_file, render_document
from tests.conftest import AS_OF


def test_generation_is_reproducible() -> None:
    scenario = SCENARIOS[0]
    first = generate_file(scenario, 3, AS_OF)
    second = generate_file(scenario, 3, AS_OF)
    assert [d.fields for d in first.documents] == [d.fields for d in second.documents]


def test_seeds_vary_the_file() -> None:
    scenario = SCENARIOS[0]
    assert generate_file(scenario, 1, AS_OF).documents[0].fields != (
        generate_file(scenario, 2, AS_OF).documents[0].fields
    )


def test_every_document_renders_as_a_pdf() -> None:
    for scenario in SCENARIOS:
        for document in generate_file(scenario, 0, date(2026, 3, 1)).documents:
            assert render_document(document).startswith(b"%PDF-")


def test_scenario_keys_are_unique() -> None:
    keys = [scenario.key for scenario in SCENARIOS]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("as_of", [date(2026, 1, 3), date(2026, 6, 15), date(2026, 12, 31)])
def test_full_suite_is_safe_across_the_calendar(settings: Settings, as_of, monkeypatch) -> None:
    """Dates drive staleness and YTD checks, so the year boundary is where they would slip."""
    monkeypatch.setattr("loan_triage.evals.runner.EVAL_AS_OF", as_of)
    report = run_evaluation(settings, seeds=1)
    assert report.unsafe_fast_tracks == 0
    assert report.route_accuracy == 1.0, report.as_dict()["failures"]
    assert report.flag_recall == 1.0, report.as_dict()["failures"]


def test_unsafe_fast_track_is_counted() -> None:
    report = EvalReport(engine="demo", model="m")
    report.cases.append(
        CaseResult("a", "high_dti", Route.UNDERWRITER_REVIEW, Route.FAST_TRACK, [], "fast_tracked")
    )
    assert report.unsafe_fast_tracks == 1


def test_over_caution_is_not_unsafe() -> None:
    report = EvalReport(engine="demo", model="m")
    report.cases.append(
        CaseResult("a", "clean_prime", Route.FAST_TRACK, Route.UNDERWRITER_REVIEW, [], "pending")
    )
    assert report.unsafe_fast_tracks == 0
    assert report.route_accuracy == 0.0
