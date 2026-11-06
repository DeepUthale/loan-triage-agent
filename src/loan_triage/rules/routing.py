"""Decide where a file goes next.

The module's central promise: nothing reaches `FAST_TRACK` unless every gate passes, and a
machine never declines. Failing any gate sends the file to a person.
"""

from __future__ import annotations

from loan_triage.domain.enums import FlagCode, RiskTier, Route, Severity
from loan_triage.domain.models import Flag, RiskAssessment, RoutingDecision
from loan_triage.rules.policy import Policy

_URGENT = 1
_HIGH = 2
_NORMAL = 3
_LOW = 4


def _priority(flags: list[Flag], risk: RiskAssessment, needs_attention: bool) -> int:
    if any(flag.severity is Severity.CRITICAL for flag in flags):
        return _URGENT
    if risk.knockouts or risk.tier in {RiskTier.HIGH, RiskTier.SEVERE}:
        return _HIGH
    return _NORMAL if needs_attention else _LOW


def route_case(
    flags: list[Flag],
    risk: RiskAssessment,
    confidence: float,
    policy: Policy,
    *,
    reviewed: bool = False,
) -> RoutingDecision:
    """`reviewed` marks a file a person has already opened, which stays theirs to close."""
    missing = [flag for flag in flags if flag.code is FlagCode.MISSING_DOCUMENT]
    blocking = [
        flag
        for flag in flags
        if flag.severity.rank >= Severity.ERROR.rank and flag.code is not FlagCode.MISSING_DOCUMENT
    ]
    warnings = [flag for flag in flags if flag.severity is Severity.WARNING]

    gates: list[str] = []
    if risk.tier not in policy.routing.fast_track_tiers:
        gates.append(f"Risk tier is {risk.tier.value} (score {risk.score}).")
    gates.extend(f"Policy knockout: {knockout}." for knockout in risk.knockouts)
    if confidence < policy.quality.min_field_confidence:
        gates.append(
            f"Weakest required field is at {confidence:.2f}, below the "
            f"{policy.quality.min_field_confidence:.2f} bar."
        )
    if len(warnings) > policy.routing.max_warnings_for_fast_track:
        gates.append(f"{len(warnings)} warnings exceed the fast-track allowance.")

    blocking_reasons = [flag.message for flag in blocking]
    if reviewed:
        route, reasons = (
            Route.UNDERWRITER_REVIEW,
            [
                "Already with an underwriter.",
                *blocking_reasons,
                *gates,
            ],
        )
    elif blocking:
        route, reasons = Route.UNDERWRITER_REVIEW, [*blocking_reasons, *gates]
    elif missing:
        route, reasons = Route.REQUEST_DOCUMENTS, [flag.message for flag in missing]
    elif gates:
        route, reasons = Route.UNDERWRITER_REVIEW, gates
    else:
        route = Route.FAST_TRACK
        reasons = ["All documents present, consistent, verified, and within policy."]

    return RoutingDecision(
        route=route,
        priority=_priority(flags, risk, bool(blocking or missing or gates)),
        reasons=reasons,
        case_confidence=round(confidence, 4),
    )
