"""Typed, validated view of `config/policy.yaml`."""

from __future__ import annotations

import hashlib
from decimal import Decimal
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator

from loan_triage.domain.enums import DocumentType, RiskTier


class Band(BaseModel):
    limit: Decimal | None
    points: int


class FactorPolicy(BaseModel):
    label: str
    reason_code: str
    adverse_action_reason: str
    direction: Literal["higher_is_worse", "lower_is_worse"]
    bands: list[Band]

    @model_validator(mode="after")
    def _bands_are_ordered(self) -> FactorPolicy:
        limits = [band.limit for band in self.bands[:-1]]
        if None in limits or self.bands[-1].limit is not None:
            raise ValueError(f"{self.label}: only the last band may be open-ended")
        ordered = sorted(limits, reverse=self.direction == "lower_is_worse")  # type: ignore[type-var]
        if limits != ordered:
            raise ValueError(f"{self.label}: bands are out of order")
        points = [band.points for band in self.bands]
        if points != sorted(points):
            raise ValueError(f"{self.label}: points must rise as bands get worse")
        return self

    def points_for(self, value: Decimal) -> int:
        for band in self.bands:
            if band.limit is None:
                return band.points
            within = (
                value >= band.limit if self.direction == "lower_is_worse" else (value <= band.limit)
            )
            if within:
                return band.points
        raise AssertionError("bands always end with an open-ended band")  # pragma: no cover


class Knockout(BaseModel):
    factor: str
    above: Decimal | None = None
    below: Decimal | None = None

    def triggered_by(self, value: Decimal) -> bool:
        return (self.above is not None and value > self.above) or (
            self.below is not None and value < self.below
        )


class TierCutoffs(BaseModel):
    moderate: int
    high: int
    severe: int


class RiskPolicy(BaseModel):
    tier_cutoffs: TierCutoffs
    core_factors: list[str]
    factors: dict[str, FactorPolicy]
    knockouts: list[Knockout]

    @model_validator(mode="after")
    def _references_exist(self) -> RiskPolicy:
        unknown = {*self.core_factors, *(k.factor for k in self.knockouts)} - set(self.factors)
        if unknown:
            raise ValueError(f"policy references unknown factors: {sorted(unknown)}")
        return self


class QualityPolicy(BaseModel):
    min_field_confidence: float
    max_reextraction_passes: int
    unverifiable_confidence_cap: float
    ungrounded_confidence_cap: float


class ValidationPolicy(BaseModel):
    income_warn_tolerance: Decimal
    income_error_tolerance: Decimal
    w2_divergence_tolerance: Decimal
    arithmetic_tolerance: Decimal
    ytd_overage_tolerance: Decimal
    large_deposit_income_ratio: Decimal
    name_similarity_threshold: float
    employer_similarity_threshold: float
    max_document_age_days: dict[DocumentType, int]


class RoutingPolicy(BaseModel):
    fast_track_tiers: list[RiskTier]
    max_warnings_for_fast_track: int


class Policy(BaseModel):
    version: str
    required_documents: list[DocumentType]
    quality: QualityPolicy
    validation: ValidationPolicy
    routing: RoutingPolicy
    risk: RiskPolicy
    source_sha256: str = ""

    @property
    def adverse_action_reasons(self) -> dict[str, str]:
        return {f.reason_code: f.adverse_action_reason for f in self.risk.factors.values()}


def load_policy(path: Path) -> Policy:
    raw = path.read_bytes()
    policy = Policy.model_validate(yaml.safe_load(raw))
    return policy.model_copy(update={"source_sha256": hashlib.sha256(raw).hexdigest()})
