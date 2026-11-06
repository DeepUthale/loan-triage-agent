from pathlib import Path

import pytest
from typer.testing import CliRunner

from loan_triage.cli import app
from tests.conftest import ROOT

runner = CliRunner()


@pytest.fixture(autouse=True)
def environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOAN_TRIAGE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LOAN_TRIAGE_LLM_MODE", "demo")
    monkeypatch.setenv("LOAN_TRIAGE_POLICY_PATH", str(ROOT / "config" / "policy.yaml"))


def test_generate_then_run_a_folder(tmp_path: Path) -> None:
    samples = tmp_path / "samples"
    assert runner.invoke(app, ["generate", "--output", str(samples)]).exit_code == 0
    assert len(list((samples / "clean_prime-000").glob("*.pdf"))) == 5

    result = runner.invoke(app, ["run", str(samples / "clean_prime-000")])
    assert result.exit_code == 0
    assert "fast_tracked" in result.output


def test_run_rejects_an_empty_folder(tmp_path: Path) -> None:
    assert runner.invoke(app, ["run", str(tmp_path)]).exit_code != 0


def test_seed_then_verify_audit() -> None:
    assert "pending_review" in runner.invoke(app, ["seed"]).output
    result = runner.invoke(app, ["verify-audit"])
    assert result.exit_code == 0
    assert '"ok": true' in result.output


def test_eval_writes_a_report_and_names_the_engine(tmp_path: Path) -> None:
    output = tmp_path / "reports" / "eval.json"
    result = runner.invoke(app, ["eval", "--seeds", "1", "--output", str(output)])
    assert result.exit_code == 0
    assert "measures the pipeline, not Claude" in result.output
    assert '"unsafe_fast_tracks": 0' in output.read_text()


def test_inbox_dry_run_changes_nothing() -> None:
    result = runner.invoke(app, ["inbox", "--dry-run"])
    assert result.exit_code == 0
    assert "6 matching emails, 6 to triage" in result.output
    assert "0 to triage" not in result.output


def test_inbox_is_safe_to_run_on_a_schedule() -> None:
    first = runner.invoke(app, ["inbox"])
    assert "attached" in first.output
    assert "rejected" in first.output
    second = runner.invoke(app, ["inbox"])
    assert "6 matching emails, 0 to triage" in second.output


def test_inbox_explains_how_to_connect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOAN_TRIAGE_MAILBOX_MODE", "gmail")
    monkeypatch.setenv("LOAN_TRIAGE_GMAIL_TOKEN_PATH", str(tmp_path / "none.json"))
    result = runner.invoke(app, ["inbox"])
    assert result.exit_code == 1


def test_gmail_login_without_a_client_file_fails_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOAN_TRIAGE_GMAIL_CLIENT_SECRETS_PATH", str(tmp_path / "none.json"))
    assert runner.invoke(app, ["gmail-login"]).exit_code == 1


def test_gmail_logout_removes_the_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    token = tmp_path / "token.json"
    token.write_text("{}")
    monkeypatch.setenv("LOAN_TRIAGE_GMAIL_TOKEN_PATH", str(token))
    assert runner.invoke(app, ["gmail-logout"]).exit_code == 0
    assert not token.exists()
