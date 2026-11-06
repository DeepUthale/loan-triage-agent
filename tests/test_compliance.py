"""Guardrails on what the system is allowed to know and do."""

from __future__ import annotations

import inspect
import re
from typing import get_args

from loan_triage.domain.enums import DocumentType
from loan_triage.domain.schemas import (
    EXTRACTION_SCHEMAS,
    DocumentClassification,
    field_specs,
)
from loan_triage.graph.nodes import _brief_facts
from loan_triage.llm import prompts
from loan_triage.rules import risk, routing
from tests.conftest import AS_OF, uploads_for

PROTECTED = re.compile(
    r"race|ethnic|color|religio|national_origin|nationality|sex|gender|marital|spouse|"
    r"age\b|birth|dob|disab|familial",
    re.IGNORECASE,
)


def test_no_schema_extracts_a_protected_characteristic() -> None:
    for schema in EXTRACTION_SCHEMAS.values():
        for name in field_specs(schema):
            assert not PROTECTED.search(name), f"{schema.__name__}.{name}"


def test_prompt_tells_claude_to_leave_protected_data_out() -> None:
    for word in ("race", "religion", "national origin", "sex", "marital status", "age"):
        assert word in prompts.EXTRACTION_SYSTEM


def test_full_ssn_is_never_requested() -> None:
    specs = [name for schema in EXTRACTION_SCHEMAS.values() for name in field_specs(schema)]
    assert not any("ssn" in name and name != "ssn_last4" for name in specs)


def test_risk_scoring_never_reads_identity_fields() -> None:
    source = inspect.getsource(risk)
    for field in ("applicant_name", "property_address", "employee_name", "consumer_name"):
        assert field not in source


def test_routing_has_no_way_to_decline() -> None:
    assert "decline" not in {str(item).lower() for item in routing.Route}


def test_brief_facts_exclude_identifying_details(service) -> None:
    case = service.submit(uploads_for("high_dti"), received_on=AS_OF)
    facts = str(_brief_facts(case.state))
    assert case.applicant not in facts


def test_every_document_type_can_be_classified() -> None:
    allowed = set(get_args(DocumentClassification.model_fields["document_type"].annotation))
    assert allowed == {str(item) for item in DocumentType}


def test_prompts_treat_documents_as_data() -> None:
    assert "never an instruction" in prompts.EXTRACTION_SYSTEM
    assert "never an instruction" in prompts.CLASSIFICATION_SYSTEM
    assert "never an instruction" in prompts.BRIEF_SYSTEM


def test_brief_is_forbidden_from_recommending_an_outcome() -> None:
    assert "does not recommend" in prompts.BRIEF_SYSTEM
