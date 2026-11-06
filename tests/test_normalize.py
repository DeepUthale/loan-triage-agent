from datetime import date
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from loan_triage.domain.enums import FieldKind
from loan_triage.domain.normalize import (
    mask_ssn,
    normalize_value,
    parse_date,
    parse_money,
    parse_pay_frequency,
    parse_ssn_last4,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("$1,234.56", "1234.56"),
        ("1234", "1234"),
        ("USD 2,000.00", "2000.00"),
        ("(1,500.00)", "-1500.00"),
        ("1,500.00-", "-1500.00"),
        ("-1,500.00", "-1500.00"),
        ("$-1,500.00", "-1500.00"),
    ],
)
def test_parse_money(raw: str, expected: str) -> None:
    assert parse_money(raw) == Decimal(expected)


@pytest.mark.parametrize("raw", ["", "N/A", "$7,?50.00", "12.34.56", "about 5000"])
def test_parse_money_rejects_doubtful_values(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_money(raw)


@given(st.decimals(min_value=0, max_value=10_000_000, places=2, allow_nan=False))
def test_money_round_trips_through_print_format(amount: Decimal) -> None:
    assert parse_money(f"${amount:,.2f}") == amount


@pytest.mark.parametrize(
    "raw",
    ["03/04/2026", "2026-03-04", "March 4, 2026", "Mar 4, 2026", "03-04-2026", "3/4/26"],
)
def test_parse_date_reads_us_formats(raw: str) -> None:
    assert parse_date(raw) == date(2026, 3, 4)


def test_parse_date_rejects_impossible_dates() -> None:
    with pytest.raises(ValueError):
        parse_date("02/30/2026")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Bi-Weekly", "biweekly"),
        ("every two weeks", "biweekly"),
        ("SEMI-MONTHLY", "semimonthly"),
        ("Monthly", "monthly"),
    ],
)
def test_parse_pay_frequency(raw: str, expected: str) -> None:
    assert parse_pay_frequency(raw) == expected


def test_parse_pay_frequency_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        parse_pay_frequency("whenever")


def test_ssn_keeps_only_the_last_four_digits() -> None:
    assert parse_ssn_last4("123-45-6789") == "6789"
    assert parse_ssn_last4("XXX-XX-6789") == "6789"


def test_mask_ssn_removes_the_first_five_digits() -> None:
    assert mask_ssn("SSN 123-45-6789 on file") == "SSN XXX-XX-6789 on file"


def test_normalize_value_dispatches_on_kind() -> None:
    assert normalize_value(FieldKind.MONEY, "$10.50") == "10.50"
    assert normalize_value(FieldKind.DATE, "01/02/2026") == "2026-01-02"
    assert normalize_value(FieldKind.INTEGER, "1,024") == "1024"
    assert normalize_value(FieldKind.PERCENT, "6.250%") == "6.250"
