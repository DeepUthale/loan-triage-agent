from decimal import Decimal

from hypothesis import given
from hypothesis import strategies as st

from loan_triage.domain.enums import FlagCode, RiskTier, Route, Severity
from loan_triage.domain.models import Flag, RiskAssessment
from loan_triage.rules.policy import Policy, load_policy
from loan_triage.rules.routing import route_case
from tests.conftest import ROOT

POLICY: Policy = load_policy(ROOT / "config" / "policy.yaml")
CLEAN_RISK = RiskAssessment(score=5, tier=RiskTier.LOW, factors=[])


def flag(code: FlagCode, severity: Severity = Severity.ERROR) -> Flag:
    return Flag(code=code, severity=severity, message=str(code))


def route(flags: list[Flag] | None = None, risk=CLEAN_RISK, confidence=0.95, **options):
    return route_case(flags or [], risk, confidence, POLICY, **options)


def test_clean_file_fast_tracks() -> None:
    assert route().route is Route.FAST_TRACK


def test_moderate_risk_is_allowed_to_fast_track() -> None:
    risk = RiskAssessment(score=35, tier=RiskTier.MODERATE, factors=[])
    assert route(risk=risk).route is Route.FAST_TRACK


def test_one_warning_is_tolerated_two_are_not() -> None:
    warning = flag(FlagCode.DUPLICATE_DOCUMENT, Severity.WARNING)
    assert route([warning]).route is Route.FAST_TRACK
    assert route([warning, warning]).route is Route.UNDERWRITER_REVIEW


def test_low_confidence_goes_to_review_with_a_reason() -> None:
    decision = route(confidence=0.5)
    assert decision.route is Route.UNDERWRITER_REVIEW
    assert "0.50" in decision.reasons[0]


def test_missing_documents_are_requested_without_troubling_an_underwriter() -> None:
    decision = route([flag(FlagCode.MISSING_DOCUMENT)])
    assert decision.route is Route.REQUEST_DOCUMENTS


def test_a_red_flag_outranks_a_missing_document() -> None:
    flags = [flag(FlagCode.MISSING_DOCUMENT), flag(FlagCode.NAME_MISMATCH)]
    assert route(flags).route is Route.UNDERWRITER_REVIEW


def test_every_trigger_is_reported_not_only_the_first() -> None:
    risk = RiskAssessment(score=60, tier=RiskTier.HIGH, factors=[], knockouts=["Credit score"])
    decision = route([flag(FlagCode.NAME_MISMATCH)], risk=risk, confidence=0.5)
    assert len(decision.reasons) == 4


def test_a_file_with_a_person_stays_with_a_person() -> None:
    assert route(reviewed=True).route is Route.UNDERWRITER_REVIEW


def test_critical_flags_are_the_top_priority() -> None:
    assert route([flag(FlagCode.PROMPT_INJECTION_SUSPECTED, Severity.CRITICAL)]).priority == 1


_FLAGS = st.builds(
    Flag,
    code=st.sampled_from(FlagCode),
    severity=st.sampled_from(Severity),
    message=st.just("x"),
)
_TIERS = st.sampled_from(RiskTier)


@given(st.lists(_FLAGS, max_size=6), _TIERS, st.floats(min_value=0, max_value=1))
def test_fast_track_only_when_every_gate_passes(flags, tier, confidence) -> None:
    """The safety invariant, over arbitrary inputs rather than chosen examples."""
    risk = RiskAssessment(score=50, tier=tier, factors=[])
    decision = route_case(flags, risk, confidence, POLICY)
    if decision.route is Route.FAST_TRACK:
        assert all(flag.severity.rank < Severity.ERROR.rank for flag in flags)
        assert tier in {RiskTier.LOW, RiskTier.MODERATE}
        assert confidence >= POLICY.quality.min_field_confidence


@given(st.lists(_FLAGS, min_size=1, max_size=6), _TIERS, st.floats(min_value=0, max_value=1))
def test_critical_flag_always_reaches_a_person(flags, tier, confidence) -> None:
    flags = [*flags, flag(FlagCode.HIDDEN_TEXT_DETECTED, Severity.CRITICAL)]
    risk = RiskAssessment(score=0, tier=tier, factors=[])
    assert route_case(flags, risk, confidence, POLICY).route is Route.UNDERWRITER_REVIEW


@given(st.sampled_from([RiskTier.HIGH, RiskTier.SEVERE, RiskTier.INDETERMINATE]))
def test_high_or_unknown_risk_never_fast_tracks(tier) -> None:
    risk = RiskAssessment(score=int(Decimal(90)), tier=tier, factors=[])
    assert route(risk=risk).route is not Route.FAST_TRACK
