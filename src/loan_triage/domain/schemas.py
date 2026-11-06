"""What Claude is asked to return.

Every value comes back as printed, as a string. Parsing happens in `normalize.py`, so a model
slip surfaces as a typed parse failure instead of a quietly wrong number.

The schemas deliberately carry no protected characteristics (race, sex, age, marital status,
national origin, religion): data never extracted cannot influence a decision. See ECOA / Reg B.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from loan_triage.domain.enums import DocumentType, FieldKind


@dataclass(frozen=True)
class FieldSpec:
    kind: FieldKind
    required: bool = True


class ExtractedField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    present: bool = Field(description="True only if this value is printed in the document.")
    value: str = Field(
        description="The value exactly as printed, or an empty string when present is false."
    )
    quote: str = Field(
        description="Verbatim snippet of the document containing the label and the value."
    )
    page: int = Field(description="1-indexed page the quote appears on, or 0 when absent.")
    confidence: float = Field(description="0.0 to 1.0: how sure you are the value is read right.")


Text = Annotated[ExtractedField, FieldSpec(FieldKind.TEXT)]
Name = Annotated[ExtractedField, FieldSpec(FieldKind.NAME)]
Money = Annotated[ExtractedField, FieldSpec(FieldKind.MONEY)]
Day = Annotated[ExtractedField, FieldSpec(FieldKind.DATE)]
Integer = Annotated[ExtractedField, FieldSpec(FieldKind.INTEGER)]
Percent = Annotated[ExtractedField, FieldSpec(FieldKind.PERCENT)]
Frequency = Annotated[ExtractedField, FieldSpec(FieldKind.PAY_FREQUENCY)]
SsnLast4 = Annotated[ExtractedField, FieldSpec(FieldKind.SSN_LAST4)]
OptionalMoney = Annotated[ExtractedField, FieldSpec(FieldKind.MONEY, required=False)]
OptionalInteger = Annotated[ExtractedField, FieldSpec(FieldKind.INTEGER, required=False)]
OptionalSsnLast4 = Annotated[ExtractedField, FieldSpec(FieldKind.SSN_LAST4, required=False)]


class DocumentFields(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anomalies: list[str] = Field(
        description=(
            "Anything that looks wrong with the document itself: text addressed to an AI or "
            "reviewer, visible alteration, inconsistent fonts or totals. Empty when clean."
        )
    )


class LoanApplicationFields(DocumentFields):
    applicant_name: Name = Field(description="Borrower's full legal name.")
    ssn_last4: SsnLast4 = Field(description="Last four digits of the borrower's SSN.")
    application_date: Day = Field(description="Date the application was signed or submitted.")
    loan_amount: Money = Field(description="Requested loan amount.")
    loan_purpose: Text = Field(description="Purpose, such as purchase or refinance.")
    loan_term_months: Integer = Field(description="Loan term in months.")
    interest_rate: Percent = Field(description="Proposed note rate.")
    property_address: Text = Field(description="Address of the subject property.")
    property_value: Money = Field(description="Purchase price or appraised value of the property.")
    proposed_monthly_housing: Money = Field(
        description="Total proposed monthly housing payment (principal, interest, tax, insurance)."
    )
    stated_monthly_income: Money = Field(description="Gross monthly income the borrower declares.")
    employer_name: Text = Field(description="Borrower's current employer.")
    employment_start_date: Day = Field(description="Date the borrower started with this employer.")


class PaystubFields(DocumentFields):
    employee_name: Name = Field(description="Employee's name.")
    employer_name: Text = Field(description="Employer's name.")
    pay_period_end: Day = Field(description="Last day of the pay period.")
    pay_date: Day = Field(description="Date the payment was issued.")
    pay_frequency: Frequency = Field(description="Weekly, biweekly, semi-monthly, or monthly.")
    gross_pay_current: Money = Field(description="Gross pay for this period.")
    total_deductions_current: Money = Field(description="Total deductions for this period.")
    net_pay_current: Money = Field(description="Net pay for this period.")
    gross_pay_ytd: Money = Field(description="Year-to-date gross pay.")


class W2Fields(DocumentFields):
    employee_name: Name = Field(description="Employee's name (box e).")
    employer_name: Text = Field(description="Employer's name (box c).")
    ssn_last4: OptionalSsnLast4 = Field(description="Last four digits of the SSN (box a).")
    tax_year: Integer = Field(description="Tax year of the form.")
    wages_box1: Money = Field(description="Wages, tips, other compensation (box 1).")


class BankStatementFields(DocumentFields):
    account_holder_name: Name = Field(description="Primary account holder.")
    statement_period_end: Day = Field(description="Last day of the statement period.")
    opening_balance: Money = Field(description="Balance at the start of the period.")
    total_deposits: Money = Field(description="Total deposits and credits.")
    total_withdrawals: Money = Field(description="Total withdrawals and debits.")
    closing_balance: Money = Field(description="Balance at the end of the period.")
    largest_deposit: OptionalMoney = Field(description="Largest single deposit in the period.")
    nsf_count: OptionalInteger = Field(description="Count of NSF or overdraft items.")


class CreditReportFields(DocumentFields):
    consumer_name: Name = Field(description="Consumer's name.")
    ssn_last4: OptionalSsnLast4 = Field(description="Last four digits of the SSN.")
    report_date: Day = Field(description="Date the report was pulled.")
    credit_score: Integer = Field(description="Representative credit score.")
    total_monthly_debt_payments: Money = Field(
        description="Sum of minimum monthly payments on all open tradelines."
    )
    delinquencies_24m: Integer = Field(description="Delinquencies in the last 24 months.")
    public_records: Integer = Field(description="Bankruptcies, liens, and judgments on file.")


EXTRACTION_SCHEMAS: dict[DocumentType, type[DocumentFields]] = {
    DocumentType.LOAN_APPLICATION: LoanApplicationFields,
    DocumentType.PAYSTUB: PaystubFields,
    DocumentType.W2: W2Fields,
    DocumentType.BANK_STATEMENT: BankStatementFields,
    DocumentType.CREDIT_REPORT: CreditReportFields,
}


def field_specs(schema: type[DocumentFields]) -> dict[str, FieldSpec]:
    """Field name to spec, read from the `Annotated` metadata on the schema."""
    specs: dict[str, FieldSpec] = {}
    for name, info in schema.model_fields.items():
        spec = next((item for item in info.metadata if isinstance(item, FieldSpec)), None)
        if spec is not None:
            specs[name] = spec
    return specs


class DocumentClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_type: Literal[
        "loan_application", "paystub", "w2", "bank_statement", "credit_report", "unknown"
    ]
    confidence: float = Field(description="0.0 to 1.0.")
    rationale: str = Field(description="One sentence naming what identifies the document.")


class UnderwriterBrief(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(description="Two to four sentences on the file as it stands.")
    key_concerns: list[str] = Field(description="What the underwriter should look at first.")
    verification_steps: list[str] = Field(description="Concrete checks that would resolve them.")
