"""Graph state. Values are plain JSON so checkpoints stay portable and inspectable."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

Json = dict[str, Any]


def merge_maps(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Reducer for fan-out nodes: parallel document tasks each contribute their own key."""
    return {**left, **right}


class TriageState(TypedDict, total=False):
    case_id: str
    received_on: str
    source: Json
    uploads: list[Json]
    new_document_ids: list[str]

    documents: Annotated[dict[str, Json], merge_maps]
    classifications: Annotated[dict[str, Json], merge_maps]
    extractions: Annotated[dict[str, Json], merge_maps]
    failures: Annotated[dict[str, str], merge_maps]

    reextraction_passes: int
    weak_fields: dict[str, list[str]]

    income: Json
    flags: list[Json]
    risk: Json
    routing: Json
    brief: Json | None

    review_cycles: int
    reviews: Annotated[list[Json], operator.add]
    requested_documents: list[str]
    outcome: Json


class DocumentTask(TypedDict, total=False):
    case_id: str
    document: Json
    document_type: str
    focus: list[str]
    previous: Json
