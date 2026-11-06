from datetime import date
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from loan_triage.domain.enums import RiskTier
from loan_triage.rules.policy import Policy, RiskPolicy, load_policy
from loan_triage.rules.risk import _months_between, score_metrics
from tests.conftest import ROOT

CLEAN: dict[str, Decimal | None] = {
    "dti": Decimal("0.30"),
    "ltv": Decimal("0.75"),
    "credit_score": Decimal(780),
    "reserves_months": Decimal(12),
    "employment_tenure_months": Decimal(60),
    "delinquencies_24m": Decimal(0),
    "public_records": Decimal(0),
    "nsf_count": Decimal(0),
}
RISK: RiskPolicy = load_policy(ROOT / "config" / "policy.yaml").risk


def score(**overrides: Decimal | None):
    return score_metrics({**CLEAN, **overrides}, RISK)


def test_clean_file_scores_zero() -> None:
    result = score()
    assert (result.score, result.tier) == (0, RiskTier.LOW)


@pytest.mark.parametrize(
    ("value", "points"),
    [("0.36", 0), ("0.3601", 15), ("0.43", 15), ("0.50", 30), ("0.5001", 45)],
)
def test_dti_band_edges_are_inclusive(value: str, points: int) -> None:
    factor = next(f for f in score(dti=Decimal(value)).factors if f.key == "dti")
    assert factor.points == points


@pytest.mark.parametrize(
    ("value", "points"), [(740, 0), (739, 10), (680, 10), (620, 25), (619, 45)]
)
def test_credit_band_edges_are_inclusive(value: int, points: int) -> None:
    factor = next(f for f in score(credit_score=Decimal(value)).factors if f.key == "credit_score")
    assert factor.points == points


@pytest.mark.parametrize(
    ("overrides", "label"),
    [
        ({"dti": Decimal("0.51")}, "Debt-to-income ratio"),
        ({"ltv": Decimal("1.01")}, "Loan-to-value ratio"),
        ({"credit_score": Decimal(579)}, "Credit score"),
        ({"public_records": Decimal(1)}, "Public records on file"),
    ],
)
def test_knockouts_guarantee_at_least_high(overrides, label: str) -> None:
    result = score(**overrides)
    assert label in result.knockouts
    assert result.tier in {RiskTier.HIGH, RiskTier.SEVERE}


@pytest.mark.parametrize("key", ["dti", "ltv", "credit_score"])
def test_missing_core_metric_is_indeterminate_never_low(key: str) -> None:
    result = score(**{key: None})
    assert result.tier is RiskTier.INDETERMINATE
    assert key in result.missing_core_factors


def test_missing_non_core_metric_scores_no_points() -> None:
    result = score(nsf_count=None)
    assert result.tier is RiskTier.LOW
    assert next(f for f in result.factors if f.key == "nsf_count").value is None


def test_score_is_capped_at_100() -> None:
    result = score(
        dti=Decimal("0.9"),
        ltv=Decimal("1.2"),
        credit_score=Decimal(500),
        reserves_months=Decimal(0),
        delinquencies_24m=Decimal(5),
        public_records=Decimal(3),
    )
    assert (result.score, result.tier) == (100, RiskTier.SEVERE)


def test_factors_lead_with_the_largest_contributor() -> None:
    result = score(credit_score=Decimal(600), dti=Decimal("0.40"))
    assert result.factors[0].key == "credit_score"


def test_every_factor_carries_adverse_action_wording(policy: Policy) -> None:
    for factor in score().factors:
        assert factor.reason_code in policy.adverse_action_reasons
        assert factor.adverse_action_reason


@given(
    st.decimals(min_value=0, max_value=2, places=4),
    st.decimals(min_value=0, max_value=2, places=4),
)
def test_more_debt_never_lowers_the_score(a: Decimal, b: Decimal) -> None:
    low, high = sorted((a, b))
    assert score(dti=low).score <= score(dti=high).score


@given(st.integers(min_value=300, max_value=849), st.integers(min_value=1, max_value=50))
def test_better_credit_never_raises_the_score(value: int, gain: int) -> None:
    worse = score(credit_score=Decimal(value))
    better = score(credit_score=Decimal(min(value + gain, 850)))
    assert better.score <= worse.score


def test_tenure_counts_whole_months() -> None:
    assert _months_between(date(2024, 1, 15), date(2026, 1, 15)) == 24
    assert _months_between(date(2024, 1, 15), date(2026, 1, 14)) == 23
    assert _months_between(date(2026, 2, 1), date(2026, 1, 1)) == 0
