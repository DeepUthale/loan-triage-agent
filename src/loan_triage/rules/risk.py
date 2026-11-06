"""Score a file against credit policy.

Pure arithmetic with `Decimal`. The same inputs and policy always give the same score, every
point traces to a named factor, and each factor carries the wording an adverse-action notice
would need. Name, address, and other identity fields are never read here.
"""

from __future__ import annotations

from datetime import date
from decimal import ROUND_HALF_UP, Decimal

from loan_triage.domain.enums import DocumentType, RiskTier
from loan_triage.domain.models import IncomeAnalysis, RiskAssessment, RiskFactor
from loan_triage.rules.case_view import CaseView
from loan_triage.rules.policy import RiskPolicy

_RATIO = Decimal("0.0001")
_TENTHS = Decimal("0.1")


def _ratio(numerator: Decimal | None, denominator: Decimal | None) -> Decimal | None:
    if numerator is None or not denominator:
        return None
    return (numerator / denominator).quantize(_RATIO, ROUND_HALF_UP)


def _months_between(start: date, end: date) -> int:
    months = (end.year - start.year) * 12 + end.month - start.month
    return max(months - (end.day < start.day), 0)


def _as_decimal(value: int | None) -> Decimal | None:
    return Decimal(value) if value is not None else None


def compute_metrics(
    view: CaseView, income: IncomeAnalysis, as_of: date
) -> dict[str, Decimal | None]:
    application = view.primary(DocumentType.LOAN_APPLICATION)
    credit = view.primary(DocumentType.CREDIT_REPORT)
    bank = view.primary(DocumentType.BANK_STATEMENT)

    housing = application.money("proposed_monthly_housing") if application else None
    debts = credit.money("total_monthly_debt_payments") if credit else None
    loan = application.money("loan_amount") if application else None
    value = application.money("property_value") if application else None
    closing = bank.money("closing_balance") if bank else None
    started = application.day("employment_start_date") if application else None

    obligations = housing + debts if housing is not None and debts is not None else None
    reserves = _ratio(closing, housing)
    return {
        "dti": _ratio(obligations, income.verified_monthly),
        "ltv": _ratio(loan, value),
        "credit_score": _as_decimal(credit.integer("credit_score")) if credit else None,
        "reserves_months": reserves.quantize(_TENTHS, ROUND_HALF_UP) if reserves else reserves,
        "employment_tenure_months": _as_decimal(_months_between(started, as_of))
        if started
        else None,
        "delinquencies_24m": _as_decimal(credit.integer("delinquencies_24m")) if credit else None,
        "public_records": _as_decimal(credit.integer("public_records")) if credit else None,
        "nsf_count": _as_decimal(bank.integer("nsf_count")) if bank else None,
    }


def _tier_for(score: int, policy: RiskPolicy) -> RiskTier:
    cutoffs = policy.tier_cutoffs
    if score >= cutoffs.severe:
        return RiskTier.SEVERE
    if score >= cutoffs.high:
        return RiskTier.HIGH
    if score >= cutoffs.moderate:
        return RiskTier.MODERATE
    return RiskTier.LOW


def score_metrics(metrics: dict[str, Decimal | None], policy: RiskPolicy) -> RiskAssessment:
    factors: list[RiskFactor] = []
    for key, factor in policy.factors.items():
        value = metrics.get(key)
        factors.append(
            RiskFactor(
                key=key,
                label=factor.label,
                reason_code=factor.reason_code,
                adverse_action_reason=factor.adverse_action_reason,
                value=value,
                points=factor.points_for(value) if value is not None else 0,
            )
        )

    score = min(sum(factor.points for factor in factors), 100)
    knockouts: list[str] = []
    for knockout in policy.knockouts:
        value = metrics.get(knockout.factor)
        if value is not None and knockout.triggered_by(value):
            knockouts.append(policy.factors[knockout.factor].label)

    missing = [key for key in policy.core_factors if metrics.get(key) is None]
    tier = _tier_for(score, policy)
    if missing:
        tier = RiskTier.INDETERMINATE
    elif knockouts and tier in {RiskTier.LOW, RiskTier.MODERATE}:
        tier = RiskTier.HIGH
    return RiskAssessment(
        score=score,
        tier=tier,
        factors=sorted(factors, key=lambda factor: -factor.points),
        knockouts=knockouts,
        missing_core_factors=missing,
    )


def assess_risk(
    view: CaseView, income: IncomeAnalysis, as_of: date, policy: RiskPolicy
) -> RiskAssessment:
    return score_metrics(compute_metrics(view, income, as_of), policy)
