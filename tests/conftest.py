from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from loan_triage.config import Settings
from loan_triage.domain.enums import DocumentType
from loan_triage.domain.models import DocumentExtraction, DocumentRecord, FieldResult
from loan_triage.domain.schemas import EXTRACTION_SCHEMAS, field_specs
from loan_triage.llm.base import ReadingEngine
from loan_triage.rules.policy import Policy, load_policy
from loan_triage.service import TriageService, Upload
from loan_triage.synth.generator import SCENARIOS, LoanFile, generate_file, render_document

ROOT = Path(__file__).parent.parent
AS_OF = date(2026, 10, 30)


@pytest.fixture(scope="session")
def policy() -> Policy:
    return load_policy(ROOT / "config" / "policy.yaml")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        llm_mode="demo",
        data_dir=tmp_path / "data",
        policy_path=ROOT / "config" / "policy.yaml",
    )


ServiceFactory = Callable[[ReadingEngine | None], TriageService]


@pytest.fixture
def make_service(settings: Settings) -> Iterator[ServiceFactory]:
    services: list[TriageService] = []

    def factory(engine: ReadingEngine | None = None) -> TriageService:
        service = TriageService(settings, engine=engine)
        services.append(service)
        return service

    yield factory
    for service in services:
        service.close()


@pytest.fixture
def service(make_service: ServiceFactory) -> TriageService:
    return make_service(None)


def scenario_file(key: str, seed: int = 1) -> LoanFile:
    scenario = next(item for item in SCENARIOS if item.key == key)
    return generate_file(scenario, seed, AS_OF)


def uploads_for(key: str, seed: int = 1) -> list[Upload]:
    return [
        Upload(doc.filename, render_document(doc)) for doc in scenario_file(key, seed).documents
    ]


def make_field(name: str, value: str | None, **overrides: Any) -> FieldResult:
    defaults: dict[str, Any] = {"name": name, "kind": "text", "required": True, "present": False}
    if value is not None:
        defaults |= {
            "present": True,
            "raw": value,
            "value": value,
            "quote": f"{name} {value}",
            "page": 1,
            "model_confidence": 0.97,
            "confidence": 0.97,
            "grounded": True,
        }
    return FieldResult(**{**defaults, **overrides})


def make_extraction(
    document_type: DocumentType, document_id: str | None = None, **values: Any
) -> DocumentExtraction:
    """An extraction with every schema field populated by `values`, others absent."""
    specs = field_specs(EXTRACTION_SCHEMAS[document_type])
    fields = {}
    for name, spec in specs.items():
        value = values.get(name)
        fields[name] = make_field(
            name,
            None if value is None else str(value),
            kind=spec.kind,
            required=spec.required,
        )
    return DocumentExtraction(
        document_id=document_id or f"doc-{document_type}",
        document_type=document_type,
        fields=fields,
    )


def make_record(document_id: str, **overrides: Any) -> DocumentRecord:
    defaults: dict[str, Any] = {
        "document_id": document_id,
        "filename": f"{document_id}.pdf",
        "path": f"/tmp/{document_id}.pdf",
        "sha256": f"hash-{document_id}",
        "content_sha256": f"content-{document_id}",
        "page_count": 1,
        "has_text_layer": True,
    }
    return DocumentRecord(**{**defaults, **overrides})
