"""Runtime configuration, read from the environment or a local `.env` file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LOAN_TRIAGE_", env_file=".env", extra="ignore")

    # "demo" is an explicit choice, never a silent fallback: a production deployment that
    # loses its API key must fail loudly rather than start triaging with the offline parser.
    llm_mode: Literal["anthropic", "demo"] = "demo"
    # Read here so a key in `.env` works too: pydantic-settings does not export that file to the
    # process environment, which is the only place the SDK would look for it.
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    model: str = "claude-sonnet-5-5"
    classification_effort: Effort = "low"
    extraction_effort: Effort = "medium"
    brief_effort: Effort = "low"
    max_output_tokens: int = 16_000
    request_timeout_s: float = 180.0
    max_retries: int = 3

    # Same rule as `llm_mode`: the demo inbox is chosen, never fallen back to.
    mailbox_mode: Literal["demo", "gmail"] = "demo"
    # Scoped to one label so the app never searches the rest of the mailbox.
    gmail_query: str = "label:loan-applications has:attachment filename:pdf newer_than:30d"
    gmail_max_messages: int = 25
    gmail_client_secrets_path: Path = Path("secrets/gmail_client_secret.json")
    gmail_token_path: Path = Path("secrets/gmail_token.json")

    data_dir: Path = Path("data")
    policy_path: Path = Path("config/policy.yaml")
    max_upload_bytes: int = 20 * 1024 * 1024

    @property
    def database_path(self) -> Path:
        return self.data_dir / "loan_triage.sqlite"

    @property
    def checkpoint_path(self) -> Path:
        return self.data_dir / "checkpoints.sqlite"

    @property
    def documents_dir(self) -> Path:
        return self.data_dir / "documents"
