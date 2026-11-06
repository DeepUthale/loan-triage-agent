"""Derive verified income from source documents rather than from what the applicant states."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from loan_triage.domain.enums import DocumentType
from loan_triage.domain.models import DocumentExtraction, IncomeAnalysis
from loan_triage.domain.normalize import PERIODS_PER_YEAR
from loan_triage.rules.case_view import CaseView

_CENTS = Decimal("0.01")


def monthly_from_paystub(paystub: DocumentExtraction) -> Decimal | None:
    gross = paystub.money("gross_pay_current")
    frequency = paystub.text("pay_frequency")
    if gross is None or frequency is None:
        return None
    return (gross * PERIODS_PER_YEAR[frequency] / 12).quantize(_CENTS, ROUND_HALF_UP)


def analyze_income(view: CaseView) -> IncomeAnalysis:
    application = view.primary(DocumentType.LOAN_APPLICATION)
    paystub = view.primary(DocumentType.PAYSTUB)
    w2 = view.primary(DocumentType.W2)

    stated = application.money("stated_monthly_income") if application else None
    paystub_monthly = monthly_from_paystub(paystub) if paystub else None
    wages = w2.money("wages_box1") if w2 else None
    w2_monthly = (wages / 12).quantize(_CENTS, ROUND_HALF_UP) if wages is not None else None

    # Underwrite to the lower figure: a raise can be documented, an overstatement cannot.
    sources = {"paystub": paystub_monthly, "w2": w2_monthly}
    known = {name: value for name, value in sources.items() if value is not None}
    if not known:
        return IncomeAnalysis(stated_monthly=stated)
    basis = min(known, key=lambda name: known[name])
    return IncomeAnalysis(
        stated_monthly=stated,
        paystub_monthly=paystub_monthly,
        w2_monthly=w2_monthly,
        verified_monthly=known[basis],
        basis=basis,
    )
