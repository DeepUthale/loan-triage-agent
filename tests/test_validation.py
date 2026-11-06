from datetime import date
from decimal import Decimal

import pytest

from loan_triage.domain.enums import DocumentType as DT
from loan_triage.domain.enums import FlagCode, Severity
from loan_triage.domain.models import (
    Classification,
    DocumentExtraction,
    Flag,
    IntegrityFinding,
)
from loan_triage.rules.case_view import CaseView
from loan_triage.rules.income import analyze_income
from loan_triage.rules.policy import Policy
from loan_triage.rules.validation import (
    NameMatch,
    ValidationContext,
    compare_names,
    employers_match,
    validate_case,
)
from tests.conftest import AS_OF, make_extraction, make_record


def application(**overrides: object) -> DocumentExtraction:
    values = {
        "applicant_name": "Maya L. Rivera",
        "ssn_last4": "6789",
        "application_date": "2026-10-28",
        "loan_amount": "400000.00",
        "loan_purpose": "Purchase",
        "loan_term_months": "360",
        "interest_rate": "6.25",
        "property_address": "1 Main St",
        "property_value": "500000.00",
        "proposed_monthly_housing": "2500.00",
        "stated_monthly_income": "10000.00",
        "employer_name": "Bluepeak Software Corp",
        "employment_start_date": "2020-01-01",
    }
    return make_extraction(DT.LOAN_APPLICATION, **{**values, **overrides})


def paystub(**overrides: object) -> DocumentExtraction:
    values = {
        "employee_name": "Maya L. Rivera",
        "employer_name": "Bluepeak Software Corp",
        "pay_period_end": "2026-10-15",
        "pay_date": "2026-10-20",
        "pay_frequency": "semimonthly",
        "gross_pay_current": "5000.00",
        "total_deductions_current": "1350.00",
        "net_pay_current": "3650.00",
        "gross_pay_ytd": "95000.00",
    }
    return make_extraction(DT.PAYSTUB, **{**values, **overrides})


def w2(**overrides: object) -> DocumentExtraction:
    values = {
        "employee_name": "Maya L. Rivera",
        "employer_name": "Bluepeak Software Corp",
        "ssn_last4": "6789",
        "tax_year": "2025",
        "wages_box1": "116400.00",
    }
    return make_extraction(DT.W2, **{**values, **overrides})


def bank(**overrides: object) -> DocumentExtraction:
    values = {
        "account_holder_name": "Maya L. Rivera",
        "statement_period_end": "2026-10-15",
        "opening_balance": "20000.00",
        "total_deposits": "7300.00",
        "total_withdrawals": "6200.00",
        "closing_balance": "21100.00",
        "largest_deposit": "3650.00",
        "nsf_count": "0",
    }
    return make_extraction(DT.BANK_STATEMENT, **{**values, **overrides})


def credit(**overrides: object) -> DocumentExtraction:
    values = {
        "consumer_name": "Maya L. Rivera",
        "ssn_last4": "6789",
        "report_date": "2026-10-10",
        "credit_score": "760",
        "total_monthly_debt_payments": "600.00",
        "delinquencies_24m": "0",
        "public_records": "0",
    }
    return make_extraction(DT.CREDIT_REPORT, **{**values, **overrides})


def run(policy: Policy, *replacements: DocumentExtraction, **context: object) -> list[Flag]:
    defaults = {e.document_type: e for e in (application(), paystub(), w2(), bank(), credit())}
    for item in replacements:
        defaults[item.document_type] = item
    extractions = list(defaults.values())
    view = CaseView(extractions)
    records = {e.document_id: make_record(e.document_id) for e in extractions}
    classifications = {
        e.document_id: Classification(document_type=e.document_type, confidence=1, rationale="")
        for e in extractions
    }
    arguments = {
        "view": view,
        "documents": records,
        "classifications": classifications,
        "failures": {},
        "income": analyze_income(view),
        "policy": policy,
        "as_of": AS_OF,
    }
    return validate_case(ValidationContext(**{**arguments, **context}))


def codes(flags: list[Flag]) -> set[FlagCode]:
    return {flag.code for flag in flags}


def test_consistent_file_raises_nothing(policy: Policy) -> None:
    assert run(policy) == []


def test_missing_document_is_named(policy: Policy) -> None:
    only_application = {
        "doc-loan_application": Classification(
            document_type=DT.LOAN_APPLICATION, confidence=1, rationale=""
        )
    }
    flags = run(policy, classifications=only_application)
    assert sum(f.code is FlagCode.MISSING_DOCUMENT for f in flags) == 4


def test_missing_required_field(policy: Policy) -> None:
    flags = run(policy, credit(credit_score=None))
    assert FlagCode.MISSING_FIELD in codes(flags)


def test_missing_optional_field_is_fine(policy: Policy) -> None:
    assert run(policy, bank(nsf_count=None)) == []


def test_unknown_document_is_flagged(policy: Policy) -> None:
    documents = {"mystery": make_record("mystery")}
    classifications = {
        "mystery": Classification(document_type=DT.UNKNOWN, confidence=0.4, rationale=""),
        **{
            document_type: Classification(document_type=document_type, confidence=1, rationale="")
            for document_type in DT
            if document_type is not DT.UNKNOWN
        },
    }
    flags = run(policy, documents=documents, classifications=classifications)
    assert FlagCode.UNRECOGNIZED_DOCUMENT in codes(flags)


def test_extraction_failure_is_an_error(policy: Policy) -> None:
    record = make_record("doc-paystub")
    flags = run(policy, documents={"doc-paystub": record}, failures={"doc-paystub": "refusal"})
    failed = next(f for f in flags if f.code is FlagCode.EXTRACTION_FAILED)
    assert failed.severity is Severity.ERROR
    assert "refusal" in failed.message


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("Maya L. Rivera", "Maya Rivera", NameMatch.SAME),
        ("Rivera, Maya L.", "Maya Rivera", NameMatch.SAME),
        ("Dr. Maya Rivera Jr.", "MAYA RIVERA", NameMatch.SAME),
        ("M. Rivera", "Maya Rivera", NameMatch.CLOSE),
        ("Maya Rivara", "Maya Rivera", NameMatch.CLOSE),
        ("Maya Rivera", "Theodore Ashworth", NameMatch.DIFFERENT),
        ("Maya Rivera", "Mark Rivera", NameMatch.DIFFERENT),
    ],
)
def test_compare_names(left: str, right: str, expected: NameMatch) -> None:
    assert compare_names(left, right, 0.88) is expected


def test_a_different_person_is_an_error(policy: Policy) -> None:
    flags = run(policy, w2(employee_name="Theodore Ashworth"))
    mismatch = next(f for f in flags if f.code is FlagCode.NAME_MISMATCH)
    assert mismatch.severity is Severity.ERROR


def test_a_likely_typo_is_only_a_warning(policy: Policy) -> None:
    flags = run(policy, bank(account_holder_name="Maya Rivara"))
    assert next(f for f in flags if f.code is FlagCode.NAME_MISMATCH).severity is Severity.WARNING


def test_ssn_mismatch(policy: Policy) -> None:
    assert FlagCode.SSN_MISMATCH in codes(run(policy, credit(ssn_last4="0007")))


def test_employer_suffixes_are_ignored() -> None:
    assert employers_match("Bluepeak Software Corp", "BLUEPEAK SOFTWARE, INC.", 0.8)
    assert not employers_match("Bluepeak Software", "Ironclad Staffing Partners", 0.8)


def test_paystub_employer_mismatch_is_an_error(policy: Policy) -> None:
    flags = run(policy, paystub(employer_name="Ironclad Staffing"))
    assert next(f for f in flags if f.code is FlagCode.EMPLOYER_MISMATCH).severity is Severity.ERROR


def test_w2_from_a_previous_job_is_only_a_warning(policy: Policy) -> None:
    flags = run(policy, w2(employer_name="Ironclad Staffing"))
    assert (
        next(f for f in flags if f.code is FlagCode.EMPLOYER_MISMATCH).severity is Severity.WARNING
    )


def test_income_overstatement_severity_scales(policy: Policy) -> None:
    mild = run(policy, application(stated_monthly_income="11000.00"))
    assert next(f for f in mild if f.code is FlagCode.INCOME_MISMATCH).severity is Severity.WARNING
    gross = run(policy, application(stated_monthly_income="14000.00"))
    assert next(f for f in gross if f.code is FlagCode.INCOME_MISMATCH).severity is Severity.ERROR


def test_understating_income_is_not_a_problem(policy: Policy) -> None:
    assert run(policy, application(stated_monthly_income="8000.00")) == []


def test_verified_income_uses_the_lower_source() -> None:
    income = analyze_income(CaseView([paystub(), w2(wages_box1="96000.00")]))
    assert income.paystub_monthly == Decimal("10000.00")
    assert (income.verified_monthly, income.basis) == (Decimal("8000.00"), "w2")


def test_no_income_documents_leaves_income_unverified() -> None:
    income = analyze_income(CaseView([application()]))
    assert income.verified_monthly is None
    assert income.stated_monthly == Decimal("10000.00")


def test_paystub_that_does_not_foot(policy: Policy) -> None:
    assert FlagCode.PAYSTUB_ARITHMETIC in codes(run(policy, paystub(net_pay_current="4550.00")))


def test_paystub_within_rounding_tolerance(policy: Policy) -> None:
    assert run(policy, paystub(net_pay_current="3650.75")) == []


def test_ytd_below_a_single_period(policy: Policy) -> None:
    flags = run(policy, paystub(gross_pay_ytd="100.00"))
    assert next(f for f in flags if f.code is FlagCode.PAYSTUB_YTD_IMPLAUSIBLE).severity is (
        Severity.ERROR
    )


def test_ytd_above_what_the_calendar_allows(policy: Policy) -> None:
    flags = run(policy, paystub(gross_pay_ytd="300000.00"))
    assert next(f for f in flags if f.code is FlagCode.PAYSTUB_YTD_IMPLAUSIBLE).severity is (
        Severity.WARNING
    )


def test_bank_statement_that_does_not_foot(policy: Policy) -> None:
    flags = run(policy, bank(closing_balance="26100.00"))
    assert FlagCode.BANK_STATEMENT_ARITHMETIC in codes(flags)


def test_stale_document(policy: Policy) -> None:
    assert FlagCode.STALE_DOCUMENT in codes(run(policy, credit(report_date="2026-05-01")))


def test_age_limit_is_inclusive(policy: Policy) -> None:
    limit = policy.validation.max_document_age_days[DT.BANK_STATEMENT]
    dated = date.fromordinal(AS_OF.toordinal() - limit).isoformat()
    assert run(policy, bank(statement_period_end=dated)) == []


def test_future_dated_document(policy: Policy) -> None:
    flags = run(policy, bank(statement_period_end="2026-12-01"))
    assert FlagCode.FUTURE_DATED_DOCUMENT in codes(flags)


def test_large_deposit(policy: Policy) -> None:
    assert FlagCode.LARGE_UNSOURCED_DEPOSIT in codes(run(policy, bank(largest_deposit="30000.00")))


def test_payroll_deposit_is_not_large(policy: Policy) -> None:
    assert run(policy, bank(largest_deposit="9000.00")) == []


@pytest.mark.parametrize(
    "code", [FlagCode.PROMPT_INJECTION_SUSPECTED, FlagCode.HIDDEN_TEXT_DETECTED]
)
def test_each_integrity_finding_is_critical_on_its_own(policy: Policy, code: FlagCode) -> None:
    record = make_record(
        "doc-paystub", findings=[IntegrityFinding(code=code, detail="x", page=1, excerpt="y")]
    )
    flags = run(policy, documents={"doc-paystub": record})
    assert flags[0].severity is Severity.CRITICAL


def test_duplicates_match_on_content_not_bytes(policy: Policy) -> None:
    records = {
        "a": make_record("a", sha256="1", content_sha256="same"),
        "b": make_record("b", sha256="2", content_sha256="same"),
    }
    flags = run(policy, documents=records)
    assert FlagCode.DUPLICATE_DOCUMENT in codes(flags)


def test_anomaly_reported_by_the_model_reaches_the_reviewer(policy: Policy) -> None:
    stub = paystub().model_copy(update={"anomalies": ["Font differs on the net pay line."]})
    flags = run(policy, stub)
    assert FlagCode.MODEL_REPORTED_ANOMALY in codes(flags)


def test_no_text_layer_is_flagged(policy: Policy) -> None:
    flags = run(policy, documents={"scan": make_record("scan", has_text_layer=False)})
    assert FlagCode.UNVERIFIABLE_DOCUMENT in codes(flags)


def test_flags_are_sorted_most_severe_first(policy: Policy) -> None:
    flags = run(
        policy, w2(employee_name="Theodore Ashworth"), bank(account_holder_name="Maya Rivara")
    )
    ranks = [flag.severity.rank for flag in flags]
    assert ranks == sorted(ranks, reverse=True)
