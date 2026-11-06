"""Check Claude's transcription against the file itself.

Self-reported confidence is a weak signal, and Citations cannot be combined with structured
outputs. So each field carries a verbatim quote, and code confirms that the quote really occurs
in the PDF and really contains the value. This turns "the model says so" into "the file says so".
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from loan_triage.domain.enums import DocumentType, FieldKind
from loan_triage.domain.models import DocumentExtraction, FieldResult
from loan_triage.domain.normalize import mask_ssn, normalize_value, parse_money
from loan_triage.domain.schemas import DocumentFields, ExtractedField, FieldSpec, field_specs
from loan_triage.rules.policy import QualityPolicy

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_NUMERIC_KINDS = {FieldKind.MONEY, FieldKind.INTEGER, FieldKind.PERCENT}


def _squash(text: str) -> str:
    """Compare on letters and digits only: PDF text extraction mangles spacing and punctuation."""
    return re.sub(r"[\W_]+", "", text).casefold()


def _numbers_in(text: str) -> set[Decimal]:
    numbers: set[Decimal] = set()
    for token in _NUMBER.findall(text):
        try:
            numbers.add(Decimal(token.replace(",", "")))
        except InvalidOperation:  # pragma: no cover - the pattern only yields numeric tokens
            continue
    return numbers


def _value_in_quote(kind: FieldKind, value: str, quote: str) -> bool:
    if kind in _NUMERIC_KINDS:
        try:
            target = abs(parse_money(value.rstrip("% ")))
        except ValueError:
            return False
        return target in _numbers_in(quote)
    if kind is FieldKind.SSN_LAST4:
        return re.sub(r"\D", "", value)[-4:] in re.sub(r"\D", "", quote)
    return bool(_squash(value)) and _squash(value) in _squash(quote)


def locate_quote(
    field: ExtractedField, kind: FieldKind, pages: list[str]
) -> tuple[bool | None, int]:
    """Return (grounded, page). `grounded` is None when the file has no text to check against."""
    if not any(page.strip() for page in pages):
        return None, field.page
    if not field.quote.strip() or not _value_in_quote(kind, field.value, field.quote):
        return False, field.page
    needle = _squash(field.quote)
    for number, page in enumerate(pages, start=1):
        if needle in _squash(page):
            return True, number
    return False, field.page


def _clamp(confidence: float) -> float:
    return max(0.0, min(1.0, confidence))


def _verify_field(
    name: str,
    spec: FieldSpec,
    field: ExtractedField,
    pages: list[str],
    quality: QualityPolicy,
) -> FieldResult:
    if not field.present or not field.value.strip():
        return FieldResult(name=name, kind=spec.kind, required=spec.required, present=False)

    grounded, page = locate_quote(field, spec.kind, pages)
    model_confidence = _clamp(field.confidence)
    match grounded:
        case False:
            confidence = min(model_confidence, quality.ungrounded_confidence_cap)
        case None:
            confidence = min(model_confidence, quality.unverifiable_confidence_cap)
        case _:
            confidence = model_confidence

    value: str | None = None
    parse_error: str | None = None
    try:
        value = normalize_value(spec.kind, field.value)
    except ValueError as exc:
        parse_error = str(exc)
        confidence = 0.0

    return FieldResult(
        name=name,
        kind=spec.kind,
        required=spec.required,
        present=True,
        raw=mask_ssn(field.value),
        value=value,
        quote=mask_ssn(field.quote),
        page=page,
        model_confidence=model_confidence,
        confidence=confidence,
        grounded=grounded,
        parse_error=parse_error,
    )


def verify_extraction(
    document_id: str,
    document_type: DocumentType,
    output: DocumentFields,
    pages: list[str],
    quality: QualityPolicy,
) -> DocumentExtraction:
    results: dict[str, FieldResult] = {}
    for name, spec in field_specs(type(output)).items():
        field: ExtractedField = getattr(output, name)
        results[name] = _verify_field(name, spec, field, pages, quality)
    return DocumentExtraction(
        document_id=document_id,
        document_type=document_type,
        fields=results,
        anomalies=[mask_ssn(item) for item in output.anomalies],
    )


def weak_fields(extraction: DocumentExtraction, quality: QualityPolicy) -> list[str]:
    """Required fields worth a second read: missing, unparsed, ungrounded, or low confidence.

    A field held down only by the no-text-layer cap is left out, since no re-read can lift it.
    """

    def improvable(field: FieldResult) -> bool:
        if not field.present or field.parse_error is not None:
            return True
        return field.grounded is not None and field.confidence < quality.min_field_confidence

    return [
        name for name, field in extraction.fields.items() if field.required and improvable(field)
    ]


def merge_reextraction(
    previous: DocumentExtraction, fresh: DocumentExtraction, focus: list[str]
) -> DocumentExtraction:
    """Adopt a re-read field only if it improves on the first pass; never lose a good value."""
    fields = dict(previous.fields)
    for name in focus:
        candidate, current = fresh.fields.get(name), fields.get(name)
        if candidate is None:
            continue
        if current is None or (candidate.present, candidate.confidence) > (
            current.present,
            current.confidence,
        ):
            fields[name] = candidate
    return previous.model_copy(
        update={
            "fields": fields,
            "anomalies": sorted({*previous.anomalies, *fresh.anomalies}),
            "attempts": previous.attempts + 1,
        }
    )


def case_confidence(extractions: list[DocumentExtraction]) -> float:
    """A file is only as trustworthy as its weakest required field."""
    scores = [
        field.confidence
        for extraction in extractions
        for field in extraction.fields.values()
        if field.required
    ]
    return min(scores, default=0.0)
