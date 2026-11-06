from loan_triage.domain.enums import DocumentType
from loan_triage.domain.schemas import ExtractedField, W2Fields
from loan_triage.rules.grounding import (
    case_confidence,
    merge_reextraction,
    verify_extraction,
    weak_fields,
)
from loan_triage.rules.policy import Policy
from tests.conftest import make_extraction, make_field

PAGES = [
    "Wage and Tax Statement\nEmployee Name Maya L. Rivera\nEmployer Name Bluepeak Software Corp\n"
    "Employee SSN 123-45-6789\nTax Year 2025\n"
    "Box 1 Wages, Tips, Other Compensation $98,450.00"
]


def field(value: str, quote: str, confidence: float = 0.97) -> ExtractedField:
    return ExtractedField(present=True, value=value, quote=quote, page=1, confidence=confidence)


def absent() -> ExtractedField:
    return ExtractedField(present=False, value="", quote="", page=0, confidence=0.9)


def w2(**overrides: ExtractedField) -> W2Fields:
    values = {
        "employee_name": field("Maya L. Rivera", "Employee Name Maya L. Rivera"),
        "employer_name": field("Bluepeak Software Corp", "Employer Name Bluepeak Software Corp"),
        "ssn_last4": field("6789", "Employee SSN 123-45-6789"),
        "tax_year": field("2025", "Tax Year 2025"),
        "wages_box1": field("$98,450.00", "Compensation $98,450.00"),
    }
    return W2Fields(anomalies=[], **{**values, **overrides})


def verify(output: W2Fields, policy: Policy, pages: list[str] = PAGES):
    return verify_extraction("doc-1", DocumentType.W2, output, pages, policy.quality)


def test_accurate_quotes_keep_the_model_confidence(policy: Policy) -> None:
    result = verify(w2(), policy)
    assert all(item.grounded for item in result.fields.values())
    assert result.fields["wages_box1"].value == "98450.00"
    assert result.fields["wages_box1"].confidence == 0.97


def test_invented_quote_is_capped_however_confident_the_model_is(policy: Policy) -> None:
    result = verify(w2(wages_box1=field("$120,000.00", "Compensation $120,000.00", 0.99)), policy)
    wages = result.fields["wages_box1"]
    assert wages.grounded is False
    assert wages.confidence == policy.quality.ungrounded_confidence_cap


def test_real_quote_that_does_not_contain_the_value_fails(policy: Policy) -> None:
    result = verify(w2(wages_box1=field("$120,000.00", "Compensation $98,450.00")), policy)
    assert result.fields["wages_box1"].grounded is False


def test_quote_survives_pdf_whitespace_differences(policy: Policy) -> None:
    pages = [PAGES[0].replace("Maya L. Rivera", "Maya  L.\nRivera")]
    assert verify(w2(), policy, pages).fields["employee_name"].grounded is True


def test_money_matches_by_amount_not_by_digits(policy: Policy) -> None:
    result = verify(w2(wages_box1=field("$98,450", "Compensation $98,450.00")), policy)
    assert result.fields["wages_box1"].grounded is True
    # "$9,845" is a substring of the digits but a different amount.
    result = verify(w2(wages_box1=field("$9,845", "Compensation $98,450.00")), policy)
    assert result.fields["wages_box1"].grounded is False


def test_no_text_layer_is_unverifiable_and_capped_below_the_bar(policy: Policy) -> None:
    result = verify(w2(), policy, pages=[""])
    assert result.fields["wages_box1"].grounded is None
    assert result.fields["wages_box1"].confidence < policy.quality.min_field_confidence


def test_full_ssn_never_survives_into_stored_quotes(policy: Policy) -> None:
    result = verify(w2(), policy)
    assert "123-45" not in result.fields["ssn_last4"].quote
    assert result.fields["ssn_last4"].value == "6789"


def test_unparseable_value_scores_zero(policy: Policy) -> None:
    pages = [PAGES[0].replace("$98,450.00", "$98,4?0.00")]
    result = verify(w2(wages_box1=field("$98,4?0.00", "Compensation $98,4?0.00")), policy, pages)
    assert result.fields["wages_box1"].parse_error is not None
    assert result.fields["wages_box1"].confidence == 0.0


def test_out_of_range_model_confidence_is_clamped(policy: Policy) -> None:
    result = verify(w2(tax_year=field("2025", "Tax Year 2025", 7.0)), policy)
    assert result.fields["tax_year"].model_confidence == 1.0


def test_weak_fields_finds_what_a_reread_could_fix(policy: Policy) -> None:
    output = w2(
        tax_year=absent(),
        wages_box1=field("$120,000.00", "Compensation $120,000.00"),
    )
    assert set(weak_fields(verify(output, policy), policy.quality)) == {"tax_year", "wages_box1"}


def test_weak_fields_skips_optional_fields(policy: Policy) -> None:
    assert weak_fields(verify(w2(ssn_last4=absent()), policy), policy.quality) == []


def test_weak_fields_skips_a_scan_since_rereading_cannot_verify_it(policy: Policy) -> None:
    assert weak_fields(verify(w2(), policy, pages=[""]), policy.quality) == []


def test_merge_takes_a_better_reading() -> None:
    previous = make_extraction(DocumentType.W2, tax_year="2025")
    previous.fields["tax_year"] = make_field("tax_year", "2025", confidence=0.4)
    fresh = make_extraction(DocumentType.W2, tax_year="2024")
    merged = merge_reextraction(previous, fresh, ["tax_year"])
    assert merged.fields["tax_year"].value == "2024"
    assert merged.attempts == 2


def test_merge_never_loses_a_stronger_reading() -> None:
    previous = make_extraction(DocumentType.W2, tax_year="2025")
    fresh = make_extraction(DocumentType.W2, tax_year="2024")
    fresh.fields["tax_year"] = make_field("tax_year", "2024", confidence=0.2)
    assert merge_reextraction(previous, fresh, ["tax_year"]).fields["tax_year"].value == "2025"


def test_merge_ignores_fields_outside_the_focus() -> None:
    previous = make_extraction(DocumentType.W2, tax_year="2025", employee_name="A B")
    fresh = make_extraction(DocumentType.W2, tax_year="2024", employee_name="C D")
    merged = merge_reextraction(previous, fresh, ["tax_year"])
    assert merged.fields["employee_name"].value == "A B"


def test_case_confidence_is_the_weakest_required_field() -> None:
    extraction = make_extraction(DocumentType.W2, tax_year="2025")
    extraction.fields["tax_year"] = make_field("tax_year", "2025", confidence=0.61)
    extraction.fields["ssn_last4"] = make_field("ssn_last4", "1", required=False, confidence=0.1)
    assert case_confidence([extraction]) == 0.0  # the absent required fields score zero
    for name, item in extraction.fields.items():
        if name not in {"tax_year", "ssn_last4"}:
            extraction.fields[name] = item.model_copy(update={"confidence": 0.9})
    assert case_confidence([extraction]) == 0.61


def test_case_confidence_with_nothing_extracted_is_zero() -> None:
    assert case_confidence([]) == 0.0
