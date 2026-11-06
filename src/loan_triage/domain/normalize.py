"""Turn printed strings into canonical values. Every parser raises `ValueError` on doubt."""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from loan_triage.domain.enums import FieldKind

PERIODS_PER_YEAR = {"weekly": 52, "biweekly": 26, "semimonthly": 24, "monthly": 12}

# US-format first: these are US lending documents, so 03/04/2026 means March 4.
_DATE_FORMATS = (
    "%m/%d/%Y",
    "%Y-%m-%d",
    "%B %d, %Y",
    "%b %d, %Y",
    "%m-%d-%Y",
    "%m/%d/%y",
    "%d %B %Y",
    "%d %b %Y",
)
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def parse_money(raw: str) -> Decimal:
    text = raw.strip().upper().replace("USD", "").replace("$", "").replace(",", "").strip()
    negative = (
        text.startswith("-") or text.endswith("-") or (text.startswith("(") and text.endswith(")"))
    )
    text = text.strip("()-").replace("$", "").strip()
    if not re.fullmatch(r"\d+(\.\d+)?", text):
        raise ValueError(f"not a money amount: {raw!r}")
    try:
        amount = Decimal(text)
    except InvalidOperation as exc:  # pragma: no cover - the regex above already screens this
        raise ValueError(f"not a money amount: {raw!r}") from exc
    return -amount if negative else amount


def parse_date(raw: str) -> date:
    text = re.sub(r"\s+", " ", raw.strip().replace("Sept", "Sep"))
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()  # noqa: DTZ007 - calendar dates, no zone
        except ValueError:
            continue
    raise ValueError(f"not a date: {raw!r}")


def parse_integer(raw: str) -> int:
    text = raw.strip().replace(",", "")
    if not re.fullmatch(r"\d+", text):
        raise ValueError(f"not an integer: {raw!r}")
    return int(text)


def parse_percent(raw: str) -> Decimal:
    text = raw.strip().rstrip("%").strip()
    if not re.fullmatch(r"\d+(\.\d+)?", text):
        raise ValueError(f"not a percentage: {raw!r}")
    return Decimal(text)


def parse_pay_frequency(raw: str) -> str:
    text = re.sub(r"[\s\-_]", "", raw.strip().lower())
    aliases = {
        "weekly": "weekly",
        "biweekly": "biweekly",
        "everytwoweeks": "biweekly",
        "semimonthly": "semimonthly",
        "twicemonthly": "semimonthly",
        "monthly": "monthly",
    }
    if text not in aliases:
        raise ValueError(f"not a pay frequency: {raw!r}")
    return aliases[text]


def parse_ssn_last4(raw: str) -> str:
    digits = re.sub(r"\D", "", raw)
    if len(digits) < 4:
        raise ValueError("fewer than four digits")
    return digits[-4:]


def parse_name(raw: str) -> str:
    text = re.sub(r"\s+", " ", raw.strip())
    if not re.search(r"[^\W\d_]{2,}", text):
        raise ValueError(f"not a name: {raw!r}")
    return text


def parse_text(raw: str) -> str:
    text = re.sub(r"\s+", " ", raw.strip())
    if not text:
        raise ValueError("empty value")
    return text


def normalize_value(kind: FieldKind, raw: str) -> str:
    """Canonical JSON-safe form of `raw`, ready for storage in graph state."""
    match kind:
        case FieldKind.MONEY:
            return str(parse_money(raw))
        case FieldKind.DATE:
            return parse_date(raw).isoformat()
        case FieldKind.INTEGER:
            return str(parse_integer(raw))
        case FieldKind.PERCENT:
            return str(parse_percent(raw))
        case FieldKind.PAY_FREQUENCY:
            return parse_pay_frequency(raw)
        case FieldKind.SSN_LAST4:
            return parse_ssn_last4(raw)
        case FieldKind.NAME:
            return parse_name(raw)
        case FieldKind.TEXT:
            return parse_text(raw)


def mask_ssn(text: str) -> str:
    """Drop the first five digits of anything shaped like a full SSN."""
    return _SSN.sub(lambda match: f"XXX-XX-{match.group()[-4:]}", text)
