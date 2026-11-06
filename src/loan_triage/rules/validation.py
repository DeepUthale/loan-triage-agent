"""Deterministic checks: is the file complete, and does it agree with itself?

Each rule is a small generator of `Flag`s, so adding one means adding one function to `RULES`.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from difflib import SequenceMatcher
from enum import Enum, auto

from loan_triage.domain.enums import DocumentType, FlagCode, Severity
from loan_triage.domain.models import (
    Classification,
    DocumentExtraction,
    DocumentRecord,
    Flag,
    IncomeAnalysis,
)
from loan_triage.domain.normalize import PERIODS_PER_YEAR
from loan_triage.rules.case_view import CaseView
from loan_triage.rules.policy import Policy

_NAME_NOISE = {"mr", "mrs", "ms", "dr", "jr", "sr", "ii", "iii", "iv"}
_EMPLOYER_NOISE = {
    "inc", "incorporated", "llc", "llp", "lp", "ltd", "limited",
    "corp", "corporation", "co", "company", "the",
}  # fmt: skip

_INTEGRITY_SEVERITY = {
    FlagCode.PROMPT_INJECTION_SUSPECTED: Severity.CRITICAL,
    FlagCode.HIDDEN_TEXT_DETECTED: Severity.CRITICAL,
}


@dataclass(frozen=True)
class ValidationContext:
    view: CaseView
    documents: dict[str, DocumentRecord]
    classifications: dict[str, Classification]
    failures: dict[str, str]
    income: IncomeAnalysis
    policy: Policy
    as_of: date


class NameMatch(Enum):
    SAME = auto()
    CLOSE = auto()
    DIFFERENT = auto()


def _name_tokens(name: str) -> list[str]:
    if "," in name:  # "Rivera, Maya L." -> "Maya L. Rivera"
        last, _, rest = name.partition(",")
        name = f"{rest} {last}"
    tokens = re.findall(r"[^\W\d_]+", name.casefold())
    return [token for token in tokens if token not in _NAME_NOISE]


def compare_names(left: str, right: str, threshold: float) -> NameMatch:
    a, b = _name_tokens(left), _name_tokens(right)
    if not a or not b:
        return NameMatch.DIFFERENT
    if a[0] == b[0] and a[-1] == b[-1]:
        return NameMatch.SAME
    initial_only = a[-1] == b[-1] and (len(a[0]) == 1 or len(b[0]) == 1) and a[0][0] == b[0][0]
    # First and last only, so a middle initial on one document cannot sink a near match.
    first_last = SequenceMatcher(None, f"{a[0]} {a[-1]}", f"{b[0]} {b[-1]}").ratio()
    if initial_only or first_last >= threshold:
        return NameMatch.CLOSE
    return NameMatch.DIFFERENT


def _employer_key(name: str) -> str:
    tokens = re.findall(r"[^\W_]+", name.casefold())
    return " ".join(token for token in tokens if token not in _EMPLOYER_NOISE)


def employers_match(left: str, right: str, threshold: float) -> bool:
    a, b = _employer_key(left), _employer_key(right)
    return bool(a and b) and SequenceMatcher(None, a, b).ratio() >= threshold


def _flag(
    code: FlagCode,
    severity: Severity,
    message: str,
    *extractions: DocumentExtraction | None,
    fields: Iterable[str] = (),
) -> Flag:
    return Flag(
        code=code,
        severity=severity,
        message=message,
        document_ids=[e.document_id for e in extractions if e is not None],
        fields=list(fields),
    )


def _money(amount: Decimal) -> str:
    return f"${amount:,.2f}"


def check_required_documents(ctx: ValidationContext) -> Iterator[Flag]:
    received = {c.document_type for c in ctx.classifications.values()}
    for required in ctx.policy.required_documents:
        if required not in received:
            yield Flag(
                code=FlagCode.MISSING_DOCUMENT,
                severity=Severity.ERROR,
                message=f"Required document not received: {required.label}.",
            )


def check_unrecognized(ctx: ValidationContext) -> Iterator[Flag]:
    for document_id, classification in ctx.classifications.items():
        if classification.document_type is DocumentType.UNKNOWN:
            yield Flag(
                code=FlagCode.UNRECOGNIZED_DOCUMENT,
                severity=Severity.WARNING,
                message=f"Could not identify {ctx.documents[document_id].filename}.",
                document_ids=[document_id],
            )


def check_extraction_failures(ctx: ValidationContext) -> Iterator[Flag]:
    for document_id, reason in ctx.failures.items():
        yield Flag(
            code=FlagCode.EXTRACTION_FAILED,
            severity=Severity.ERROR,
            message=f"{ctx.documents[document_id].filename} could not be processed: {reason}.",
            document_ids=[document_id],
        )


def check_field_quality(ctx: ValidationContext) -> Iterator[Flag]:
    quality = ctx.policy.quality
    for extraction in ctx.view.extractions:
        label = extraction.document_type.label
        for key, field in extraction.fields.items():
            name = key.replace("_", " ").capitalize()
            if not field.present:
                if field.required:
                    yield _flag(
                        FlagCode.MISSING_FIELD,
                        Severity.ERROR,
                        f"{name} is missing from the {label}.",
                        extraction,
                        fields=[key],
                    )
            elif field.parse_error is not None:
                yield _flag(
                    FlagCode.UNPARSEABLE_FIELD,
                    Severity.ERROR,
                    f"{name} on the {label} could not be read as printed: {field.raw!r}.",
                    extraction,
                    fields=[key],
                )
            elif field.grounded is False:
                yield _flag(
                    FlagCode.UNGROUNDED_FIELD,
                    Severity.ERROR if field.required else Severity.WARNING,
                    f"The quote given for {name.lower()} on the {label} was not found in the file.",
                    extraction,
                    fields=[key],
                )
            elif field.required and field.confidence < quality.min_field_confidence:
                yield _flag(
                    FlagCode.LOW_CONFIDENCE_FIELD,
                    Severity.WARNING,
                    f"{name} on the {label} is below the confidence bar ({field.confidence:.2f}).",
                    extraction,
                    fields=[name],
                )


def check_text_layer(ctx: ValidationContext) -> Iterator[Flag]:
    for document_id, record in ctx.documents.items():
        if not record.has_text_layer:
            yield Flag(
                code=FlagCode.UNVERIFIABLE_DOCUMENT,
                severity=Severity.WARNING,
                message=f"{record.filename} has no text layer, so quotes cannot be verified.",
                document_ids=[document_id],
            )


def check_integrity(ctx: ValidationContext) -> Iterator[Flag]:
    for document_id, record in ctx.documents.items():
        for finding in record.findings:
            excerpt = f" Excerpt: {finding.excerpt!r}" if finding.excerpt else ""
            yield Flag(
                code=finding.code,
                severity=_INTEGRITY_SEVERITY[finding.code],
                message=f"{record.filename}, page {finding.page}: {finding.detail}{excerpt}",
                document_ids=[document_id],
            )


def check_duplicates(ctx: ValidationContext) -> Iterator[Flag]:
    hashes = Counter(record.content_sha256 for record in ctx.documents.values())
    for digest, count in hashes.items():
        if count > 1:
            ids = [i for i, record in ctx.documents.items() if record.content_sha256 == digest]
            yield Flag(
                code=FlagCode.DUPLICATE_DOCUMENT,
                severity=Severity.WARNING,
                message=f"The same document was uploaded {count} times.",
                document_ids=ids,
            )


def check_model_anomalies(ctx: ValidationContext) -> Iterator[Flag]:
    for extraction in ctx.view.extractions:
        for anomaly in extraction.anomalies:
            yield _flag(
                FlagCode.MODEL_REPORTED_ANOMALY,
                Severity.ERROR,
                f"Reported while reading the file: {anomaly}",
                extraction,
            )


def check_identity(ctx: ValidationContext) -> Iterator[Flag]:
    application = ctx.view.primary(DocumentType.LOAN_APPLICATION)
    anchor = application.text("applicant_name") if application else None
    if anchor is None:
        return
    threshold = ctx.policy.validation.name_similarity_threshold
    sources = (
        (DocumentType.PAYSTUB, "employee_name"),
        (DocumentType.W2, "employee_name"),
        (DocumentType.BANK_STATEMENT, "account_holder_name"),
        (DocumentType.CREDIT_REPORT, "consumer_name"),
    )
    for document_type, field in sources:
        for extraction in ctx.view.of_type(document_type):
            name = extraction.text(field)
            if name is None:
                continue
            match compare_names(anchor, name, threshold):
                case NameMatch.CLOSE:
                    severity = Severity.WARNING
                case NameMatch.DIFFERENT:
                    severity = Severity.ERROR
                case NameMatch.SAME:
                    continue
            yield _flag(
                FlagCode.NAME_MISMATCH,
                severity,
                f"Name on the {document_type.label} ({name}) does not match "
                f"the application ({anchor}).",
                application,
                extraction,
                fields=[field],
            )

    claimed = application.text("ssn_last4") if application else None
    for document_type in (DocumentType.W2, DocumentType.CREDIT_REPORT):
        for extraction in ctx.view.of_type(document_type):
            found = extraction.text("ssn_last4")
            if claimed and found and claimed != found:
                yield _flag(
                    FlagCode.SSN_MISMATCH,
                    Severity.ERROR,
                    f"SSN last four on the {document_type.label} do not match the application.",
                    application,
                    extraction,
                    fields=["ssn_last4"],
                )


def check_employer(ctx: ValidationContext) -> Iterator[Flag]:
    application = ctx.view.primary(DocumentType.LOAN_APPLICATION)
    declared = application.text("employer_name") if application else None
    if declared is None:
        return
    threshold = ctx.policy.validation.employer_similarity_threshold
    # A last-year W-2 can legitimately name a previous employer, so it only warrants a warning.
    for document_type, severity in (
        (DocumentType.PAYSTUB, Severity.ERROR),
        (DocumentType.W2, Severity.WARNING),
    ):
        for extraction in ctx.view.of_type(document_type):
            employer = extraction.text("employer_name")
            if employer and not employers_match(declared, employer, threshold):
                yield _flag(
                    FlagCode.EMPLOYER_MISMATCH,
                    severity,
                    f"Employer on the {document_type.label} ({employer}) does "
                    f"not match the application ({declared}).",
                    application,
                    extraction,
                    fields=["employer_name"],
                )


def check_income(ctx: ValidationContext) -> Iterator[Flag]:
    income, rules = ctx.income, ctx.policy.validation
    application = ctx.view.primary(DocumentType.LOAN_APPLICATION)
    paystub = ctx.view.primary(DocumentType.PAYSTUB)
    w2 = ctx.view.primary(DocumentType.W2)

    documented = income.paystub_monthly or income.w2_monthly
    if income.stated_monthly is not None and documented:
        overstatement = (income.stated_monthly - documented) / documented
        if overstatement > rules.income_warn_tolerance:
            severity = (
                Severity.ERROR if overstatement > rules.income_error_tolerance else Severity.WARNING
            )
            yield _flag(
                FlagCode.INCOME_MISMATCH,
                severity,
                f"Stated income {_money(income.stated_monthly)}/mo is {overstatement:.0%} above "
                f"the documented {_money(documented)}/mo.",
                application,
                paystub or w2,
                fields=["stated_monthly_income"],
            )

    if income.paystub_monthly and income.w2_monthly:
        divergence = abs(income.w2_monthly - income.paystub_monthly) / income.paystub_monthly
        if divergence > rules.w2_divergence_tolerance:
            yield _flag(
                FlagCode.W2_INCOME_DIVERGENCE,
                Severity.WARNING,
                f"W-2 income {_money(income.w2_monthly)}/mo differs from the paystub "
                f"{_money(income.paystub_monthly)}/mo by {divergence:.0%}.",
                paystub,
                w2,
                fields=["wages_box1"],
            )


def check_paystub_arithmetic(ctx: ValidationContext) -> Iterator[Flag]:
    rules = ctx.policy.validation
    for stub in ctx.view.of_type(DocumentType.PAYSTUB):
        gross = stub.money("gross_pay_current")
        deductions = stub.money("total_deductions_current")
        net = stub.money("net_pay_current")
        if gross is not None and deductions is not None and net is not None:
            expected = gross - deductions
            if abs(expected - net) > rules.arithmetic_tolerance:
                yield _flag(
                    FlagCode.PAYSTUB_ARITHMETIC,
                    Severity.ERROR,
                    f"Gross less deductions is {_money(expected)}, but net pay reads "
                    f"{_money(net)}.",
                    stub,
                    fields=["gross_pay_current", "total_deductions_current", "net_pay_current"],
                )

        ytd = stub.money("gross_pay_ytd")
        frequency = stub.text("pay_frequency")
        pay_date = stub.day("pay_date")
        if gross is None or ytd is None or frequency is None or pay_date is None:
            continue
        periods = PERIODS_PER_YEAR[frequency]
        elapsed = max(1, round(pay_date.timetuple().tm_yday / (365 / periods)))
        ceiling = gross * elapsed * (1 + rules.ytd_overage_tolerance)
        if ytd < gross:
            yield _flag(
                FlagCode.PAYSTUB_YTD_IMPLAUSIBLE,
                Severity.ERROR,
                f"Year-to-date gross {_money(ytd)} is below this period's {_money(gross)}.",
                stub,
                fields=["gross_pay_ytd"],
            )
        elif ytd > ceiling:
            yield _flag(
                FlagCode.PAYSTUB_YTD_IMPLAUSIBLE,
                Severity.WARNING,
                f"Year-to-date gross {_money(ytd)} is more than {elapsed} pay periods can explain.",
                stub,
                fields=["gross_pay_ytd"],
            )


def check_bank_arithmetic(ctx: ValidationContext) -> Iterator[Flag]:
    tolerance = ctx.policy.validation.arithmetic_tolerance
    for statement in ctx.view.of_type(DocumentType.BANK_STATEMENT):
        opening = statement.money("opening_balance")
        deposits = statement.money("total_deposits")
        withdrawals = statement.money("total_withdrawals")
        closing = statement.money("closing_balance")
        if opening is None or deposits is None or withdrawals is None or closing is None:
            continue
        expected = opening + deposits - withdrawals
        if abs(expected - closing) > tolerance:
            yield _flag(
                FlagCode.BANK_STATEMENT_ARITHMETIC,
                Severity.ERROR,
                f"Balances do not foot: expected {_money(expected)}, statement shows "
                f"{_money(closing)}.",
                statement,
                fields=[
                    "opening_balance",
                    "total_deposits",
                    "total_withdrawals",
                    "closing_balance",
                ],
            )


def check_document_age(ctx: ValidationContext) -> Iterator[Flag]:
    date_fields = {
        DocumentType.PAYSTUB: "pay_period_end",
        DocumentType.BANK_STATEMENT: "statement_period_end",
        DocumentType.CREDIT_REPORT: "report_date",
    }
    limits = ctx.policy.validation.max_document_age_days
    for document_type, field in date_fields.items():
        for extraction in ctx.view.of_type(document_type):
            dated = extraction.day(field)
            if dated is None:
                continue
            age = (ctx.as_of - dated).days
            label = document_type.label
            if age < 0:
                yield _flag(
                    FlagCode.FUTURE_DATED_DOCUMENT,
                    Severity.ERROR,
                    f"The {label} is dated {dated}, after the file was received.",
                    extraction,
                    fields=[field],
                )
            elif age > limits[document_type]:
                yield _flag(
                    FlagCode.STALE_DOCUMENT,
                    Severity.ERROR,
                    f"The {label} is {age} days old; the limit is {limits[document_type]}.",
                    extraction,
                    fields=[field],
                )


def check_large_deposit(ctx: ValidationContext) -> Iterator[Flag]:
    verified = ctx.income.verified_monthly
    if not verified:
        return
    limit = verified * ctx.policy.validation.large_deposit_income_ratio
    for statement in ctx.view.of_type(DocumentType.BANK_STATEMENT):
        deposit = statement.money("largest_deposit")
        if deposit is not None and deposit > limit:
            yield _flag(
                FlagCode.LARGE_UNSOURCED_DEPOSIT,
                Severity.ERROR,
                f"Deposit of {_money(deposit)} exceeds {_money(limit)} and needs a documented "
                "source.",
                statement,
                fields=["largest_deposit"],
            )


RULES: tuple[Callable[[ValidationContext], Iterable[Flag]], ...] = (
    check_integrity,
    check_required_documents,
    check_unrecognized,
    check_extraction_failures,
    check_field_quality,
    check_text_layer,
    check_duplicates,
    check_model_anomalies,
    check_identity,
    check_employer,
    check_income,
    check_paystub_arithmetic,
    check_bank_arithmetic,
    check_document_age,
    check_large_deposit,
)


def validate_case(ctx: ValidationContext) -> list[Flag]:
    flags = [flag for rule in RULES for flag in rule(ctx)]
    return sorted(flags, key=lambda flag: -flag.severity.rank)
