"""Claude-backed reading engine."""

from __future__ import annotations

import base64
import logging
import time
from typing import Any

import anthropic
from anthropic import transform_schema
from pydantic import BaseModel, ValidationError

from loan_triage.config import Effort, Settings
from loan_triage.domain.enums import DocumentType
from loan_triage.domain.schemas import (
    EXTRACTION_SCHEMAS,
    DocumentClassification,
    DocumentFields,
    UnderwriterBrief,
)
from loan_triage.llm import prompts
from loan_triage.llm.base import (
    CallMetrics,
    DocumentInput,
    LLMConfigurationError,
    LLMError,
    LLMResult,
)

logger = logging.getLogger(__name__)

# Retries `cyber` and `frontier_llm` refusals server-side. Any other refusal comes back as
# `stop_reason: "refusal"` and is handled like every other per-document failure.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# USD per million tokens: input, output, cache read, cache write.
_PRICING = {"claude-sonnet-5-5": (2.00, 10.00, 0.20, 2.50)}


def _cost_usd(model: str, metrics: CallMetrics) -> float | None:
    rates = _PRICING.get(model)
    if rates is None:
        return None
    input_rate, output_rate, read_rate, write_rate = rates
    return round(
        (
            metrics.input_tokens * input_rate
            + metrics.output_tokens * output_rate
            + metrics.cache_read_tokens * read_rate
            + metrics.cache_write_tokens * write_rate
        )
        / 1_000_000,
        6,
    )


class AnthropicEngine:
    name = "anthropic"

    def __init__(self, settings: Settings, client: anthropic.Anthropic | None = None) -> None:
        self._settings = settings
        key = settings.anthropic_api_key
        # With no key here, the SDK resolves its own sources, such as an `ant auth login` profile.
        self._client = client or anthropic.Anthropic(
            api_key=key.get_secret_value() if key else None,
            timeout=settings.request_timeout_s,
            max_retries=settings.max_retries,
        )

    def classify(self, document: DocumentInput) -> LLMResult[DocumentClassification]:
        return self._call(
            schema=DocumentClassification,
            system=prompts.CLASSIFICATION_SYSTEM,
            content=[_document_block(document), _text(prompts.classification_request())],
            effort=self._settings.classification_effort,
        )

    def extract(
        self,
        document: DocumentInput,
        document_type: DocumentType,
        focus: list[str] | None = None,
    ) -> LLMResult[DocumentFields]:
        schema = EXTRACTION_SCHEMAS[document_type]
        return self._call(
            schema=schema,
            system=prompts.EXTRACTION_SYSTEM,
            content=[
                _document_block(document),
                _text(prompts.extraction_request(document_type, schema, focus)),
            ],
            effort=self._settings.extraction_effort,
        )

    def write_brief(self, facts: dict[str, Any]) -> LLMResult[UnderwriterBrief]:
        return self._call(
            schema=UnderwriterBrief,
            system=prompts.BRIEF_SYSTEM,
            content=[_text(prompts.brief_request(facts))],
            effort=self._settings.brief_effort,
        )

    def _call[T: BaseModel](
        self,
        *,
        schema: type[T],
        system: str,
        content: list[dict[str, Any]],
        effort: Effort,
    ) -> LLMResult[T]:
        # `create` rather than `parse`: `parse` validates the text before the caller can see
        # `stop_reason`, which turns a refusal or truncation into an opaque validation error.
        request: Any = {
            "model": self._settings.model,
            "max_tokens": self._settings.max_output_tokens,
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": content}],
            "output_config": {
                "effort": effort,
                "format": {"type": "json_schema", "schema": transform_schema(schema)},
            },
        }
        started = time.perf_counter()
        try:
            response = self._client.beta.messages.create(**request)
        except TypeError as exc:
            # The SDK raises this before any network call when no credential source resolves.
            raise LLMConfigurationError(
                "No Claude credentials found. Set ANTHROPIC_API_KEY or run `ant auth login`."
            ) from exc
        except (
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
            anthropic.NotFoundError,
            anthropic.BadRequestError,
        ) as exc:
            raise LLMConfigurationError(f"Claude API rejected the request: {exc}") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("rate_limited", "retries exhausted") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError("api_error", f"status {exc.status_code}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("connection_error", str(exc)) from exc

        metrics = self._metrics(response, started)
        logger.info(
            "claude call: stop=%s input=%d output=%d cache_read=%d",
            metrics.stop_reason,
            metrics.input_tokens,
            metrics.output_tokens,
            metrics.cache_read_tokens,
        )
        return LLMResult(output=_parse(response, schema), metrics=metrics)

    def _metrics(self, response: Any, started: float) -> CallMetrics:
        usage = response.usage
        served_by_fallback = any(
            entry.type == "fallback_message" for entry in (usage.iterations or [])
        )
        metrics = CallMetrics(
            engine=self.name,
            model=response.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_input_tokens or 0,
            cache_write_tokens=usage.cache_creation_input_tokens or 0,
            latency_ms=round((time.perf_counter() - started) * 1000),
            stop_reason=response.stop_reason or "unknown",
            served_by_fallback=served_by_fallback,
        )
        return metrics.model_copy(update={"cost_usd": _cost_usd(response.model, metrics)})


def _parse[T: BaseModel](response: Any, schema: type[T]) -> T:
    if response.stop_reason == "refusal":
        details = response.stop_details
        raise LLMError("refusal", getattr(details, "category", None) or "unspecified")
    if response.stop_reason == "max_tokens":
        raise LLMError("truncated", "output hit max_tokens")
    if response.stop_reason != "end_turn":
        raise LLMError("unexpected_stop", str(response.stop_reason))
    text = next((block.text for block in response.content if block.type == "text"), None)
    if text is None:
        raise LLMError("invalid_output", "no text block in response")
    try:
        return schema.model_validate_json(text)
    except ValidationError as exc:
        raise LLMError("invalid_output", f"{exc.error_count()} schema errors") from exc


def _text(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _document_block(document: DocumentInput) -> dict[str, Any]:
    return {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.standard_b64encode(document.data).decode("ascii"),
        },
    }
