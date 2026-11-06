"""Exercise the real SDK against a fake transport.

No network and no key, but every request is serialized and every response parsed by the actual
`anthropic` client, so wrong parameters or shapes fail here rather than in production.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest

from loan_triage.config import Settings
from loan_triage.domain.enums import DocumentType
from loan_triage.domain.schemas import EXTRACTION_SCHEMAS, PaystubFields, field_specs
from loan_triage.llm.anthropic_engine import FALLBACK_BETA, AnthropicEngine
from loan_triage.llm.base import DocumentInput, LLMConfigurationError, LLMError
from loan_triage.llm.prompts import EXTRACTION_SYSTEM, field_guide
from tests.conftest import make_record

DOCUMENT = DocumentInput(record=make_record("doc-1"), data=b"%PDF-1.7 fake", pages=["text"])


def message(text: str = "{}", stop_reason: str = "end_turn", **overrides: Any) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5-5",
        "content": [{"type": "text", "text": text}] if text else [],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 1_000,
            "output_tokens": 500,
            "cache_read_input_tokens": 2_000,
            "cache_creation_input_tokens": 0,
        },
        **overrides,
    }


def paystub_json() -> str:
    empty = {"present": False, "value": "", "quote": "", "page": 0, "confidence": 0.9}
    return json.dumps({"anomalies": [], **dict.fromkeys(field_specs(PaystubFields), empty)})


class Harness:
    def __init__(self, respond: Callable[[httpx2.Request], httpx2.Response]) -> None:
        self.requests: list[httpx2.Request] = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            self.requests.append(request)
            return respond(request)

        client = anthropic.Anthropic(
            api_key="test-key",
            max_retries=0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
        )
        self.engine = AnthropicEngine(Settings(_env_file=None), client=client)

    @property
    def body(self) -> dict[str, Any]:
        return json.loads(self.requests[-1].content)


def respond_with(payload: dict[str, Any], status: int = 200):
    return lambda _request: httpx2.Response(status, json=payload)


def api_error(status: int, kind: str):
    return respond_with({"type": "error", "error": {"type": kind, "message": "nope"}}, status)


def test_request_shape() -> None:
    harness = Harness(respond_with(message(paystub_json())))
    harness.engine.extract(DOCUMENT, DocumentType.PAYSTUB)
    body = harness.body

    assert body["model"] == "claude-sonnet-5-5"
    assert body["fallbacks"] == "default"
    assert FALLBACK_BETA in harness.requests[-1].headers["anthropic-beta"]
    assert "tool_choice" not in body  # forced tool use is a 400 on this model
    assert "temperature" not in body
    assert "thinking" not in body  # adaptive by default

    assert body["output_config"]["effort"] == "medium"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["system"][0]["text"] == EXTRACTION_SYSTEM

    document_block, text_block = body["messages"][0]["content"]
    assert document_block["type"] == "document"
    assert document_block["source"]["media_type"] == "application/pdf"
    assert text_block["type"] == "text"


@pytest.mark.parametrize("document_type", list(EXTRACTION_SCHEMAS))
def test_every_field_description_reaches_claude(document_type: DocumentType) -> None:
    """`transform_schema` drops descriptions beside a `$ref`, so the prompt has to carry them."""
    schema = EXTRACTION_SCHEMAS[document_type]
    harness = Harness(respond_with(message(paystub_json())))
    # The canned reply only fits the paystub schema, and the request is what matters here.
    with contextlib.suppress(LLMError):
        harness.engine.extract(DOCUMENT, document_type)
    prompt = harness.body["messages"][0]["content"][1]["text"]
    for name in field_specs(schema):
        description = schema.model_fields[name].description
        assert description and description in prompt


def test_output_schema_is_strict_and_free_of_unsupported_keywords() -> None:
    harness = Harness(respond_with(message(paystub_json())))
    harness.engine.extract(DOCUMENT, DocumentType.PAYSTUB)
    schema = harness.body["output_config"]["format"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(field_specs(PaystubFields)) <= set(schema["required"])
    serialized = json.dumps(schema)
    for keyword in ("minimum", "maximum", "minLength", "maxLength"):
        assert keyword not in serialized


def test_focus_fields_are_named_in_the_retry() -> None:
    harness = Harness(respond_with(message(paystub_json())))
    harness.engine.extract(DOCUMENT, DocumentType.PAYSTUB, focus=["gross_pay_ytd"])
    assert "`gross_pay_ytd`" in harness.body["messages"][0]["content"][1]["text"]


def test_field_guide_marks_optional_fields() -> None:
    assert "Optional" in field_guide(EXTRACTION_SCHEMAS[DocumentType.BANK_STATEMENT])


def test_response_is_parsed_and_costed() -> None:
    harness = Harness(respond_with(message(paystub_json())))
    result = harness.engine.extract(DOCUMENT, DocumentType.PAYSTUB)
    assert isinstance(result.output, PaystubFields)
    metrics = result.metrics
    assert (metrics.input_tokens, metrics.output_tokens) == (1_000, 500)
    assert metrics.cache_read_tokens == 2_000
    # 1,000 x $2/M + 500 x $10/M + 2,000 x $0.20/M
    assert metrics.cost_usd == pytest.approx(0.0074)


def test_refusal_is_a_typed_failure_not_a_parse_error() -> None:
    payload = message(
        "",
        "refusal",
        stop_details={
            "type": "refusal",
            "category": "general_harms",
            "explanation": None,
        },
    )
    harness = Harness(respond_with(payload))
    with pytest.raises(LLMError) as error:
        harness.engine.extract(DOCUMENT, DocumentType.PAYSTUB)
    assert (error.value.kind, error.value.detail) == ("refusal", "general_harms")


def test_truncated_output_is_a_typed_failure() -> None:
    harness = Harness(respond_with(message('{"anomalies": [', "max_tokens")))
    with pytest.raises(LLMError) as error:
        harness.engine.extract(DOCUMENT, DocumentType.PAYSTUB)
    assert error.value.kind == "truncated"


def test_output_that_violates_the_schema_is_a_typed_failure() -> None:
    harness = Harness(respond_with(message('{"anomalies": []}')))
    with pytest.raises(LLMError) as error:
        harness.engine.extract(DOCUMENT, DocumentType.PAYSTUB)
    assert error.value.kind == "invalid_output"


def test_rate_limit_is_a_document_level_failure() -> None:
    harness = Harness(api_error(429, "rate_limit_error"))
    with pytest.raises(LLMError) as error:
        harness.engine.extract(DOCUMENT, DocumentType.PAYSTUB)
    assert error.value.kind == "rate_limited"


def test_server_error_is_a_document_level_failure() -> None:
    harness = Harness(api_error(500, "api_error"))
    with pytest.raises(LLMError) as error:
        harness.engine.classify(DOCUMENT)
    assert error.value.kind == "api_error"


def test_connection_failure_is_a_document_level_failure() -> None:
    def fail(_request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("no route to host")

    with pytest.raises(LLMError) as error:
        Harness(fail).engine.classify(DOCUMENT)
    assert error.value.kind == "connection_error"


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (400, "invalid_request_error"),
        (401, "authentication_error"),
        (403, "permission_error"),
        (404, "not_found_error"),
    ],
)
def test_configuration_errors_stop_the_run(status: int, kind: str) -> None:
    harness = Harness(api_error(status, kind))
    with pytest.raises(LLMConfigurationError):
        harness.engine.classify(DOCUMENT)


def test_api_key_in_dotenv_reaches_the_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The documented quick start: the key lives in `.env`, not in the shell environment."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("LOAN_TRIAGE_LLM_MODE=anthropic\nANTHROPIC_API_KEY=sk-from-dotenv\n")

    settings = Settings(_env_file=env_file)

    assert settings.llm_mode == "anthropic"
    assert AnthropicEngine(settings)._client.api_key == "sk-from-dotenv"


def test_key_is_not_exposed_when_settings_are_printed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    assert "sk-secret" not in repr(Settings(_env_file=None))


def test_shell_environment_key_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-shell")
    assert AnthropicEngine(Settings(_env_file=None))._client.api_key == "sk-from-shell"


def test_missing_credentials_say_what_to_do() -> None:
    def no_credentials(**_kwargs: Any) -> None:
        raise TypeError("Could not resolve authentication method.")

    engine = Harness(respond_with(message())).engine
    engine._client.beta.messages.create = no_credentials  # type: ignore[method-assign]
    with pytest.raises(LLMConfigurationError, match="ANTHROPIC_API_KEY"):
        engine.classify(DOCUMENT)


def test_extraction_prompt_clears_the_cacheable_minimum() -> None:
    """Below 512 tokens the cache marker silently does nothing. Four characters per token."""
    assert len(EXTRACTION_SYSTEM) / 4 >= 512


def test_fallback_service_is_recorded() -> None:
    usage = {
        "input_tokens": 10,
        "output_tokens": 10,
        "iterations": [{"type": "fallback_message", "input_tokens": 10, "output_tokens": 10}],
    }
    payload = {**message(paystub_json()), "usage": usage, "model": "claude-sonnet-5"}
    result = Harness(respond_with(payload)).engine.extract(DOCUMENT, DocumentType.PAYSTUB)
    assert result.metrics.served_by_fallback is True
    assert result.metrics.cost_usd is None  # no price on file: unknown beats wrong


def test_brief_sends_no_document_block() -> None:
    brief = json.dumps({"summary": "s", "key_concerns": [], "verification_steps": []})
    harness = Harness(respond_with(message(brief)))
    harness.engine.write_brief({"risk_score": 50})
    assert [block["type"] for block in harness.body["messages"][0]["content"]] == ["text"]
    assert "<case_facts>" in harness.body["messages"][0]["content"][0]["text"]
