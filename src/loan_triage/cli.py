"""Command line entry point."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from loan_triage.config import Settings
from loan_triage.domain.enums import CaseStatus
from loan_triage.evals.runner import run_evaluation
from loan_triage.inbox.base import MailboxError
from loan_triage.inbox.gmail import GmailAuth, connect_gmail
from loan_triage.inbox.importer import InboxImporter, clean_text, open_mailbox
from loan_triage.service import TriageService, Upload
from loan_triage.synth.generator import SCENARIOS, generate_file, write_file
from loan_triage.synth.portfolio import seed_portfolio

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Loan triage workflow tools.")


@app.command()
def generate(
    output: Annotated[Path, typer.Option(help="Where to write the files.")] = Path("data/samples"),
    seeds: Annotated[int, typer.Option(help="Variations of each scenario.")] = 1,
) -> None:
    """Write synthetic loan files to disk as PDFs."""
    today = datetime.now(UTC).date()
    for scenario in SCENARIOS:
        for seed in range(seeds):
            loan_file = generate_file(scenario, seed, today)
            write_file(loan_file, output / loan_file.case_key)
    typer.echo(f"Wrote {len(SCENARIOS) * seeds} loan files to {output}")


@app.command()
def seed() -> None:
    """Process a demo portfolio so the dashboard opens with data in it."""
    service = TriageService(Settings())
    try:
        for key, case in seed_portfolio(service):
            typer.echo(f"{case.case_id}  {key:<24} {case.status}")
    finally:
        service.close()


@app.command()
def run(
    directory: Annotated[Path, typer.Argument(help="Folder of PDFs making up one loan file.")],
) -> None:
    """Triage one folder of PDFs and print the result."""
    paths = sorted(directory.glob("*.pdf"))
    if not paths:
        raise typer.BadParameter(f"no PDFs in {directory}")
    service = TriageService(Settings())
    try:
        case = service.submit([Upload(path.name, path.read_bytes()) for path in paths])
    finally:
        service.close()
    typer.echo(f"Case       {case.case_id}")
    typer.echo(f"Status     {case.status}")
    typer.echo(f"Route      {case.route}")
    typer.echo(f"Risk       {case.risk_tier} ({case.risk_score})")
    typer.echo(f"Confidence {case.confidence}")
    if case.error:
        typer.echo(f"Error      {case.error}")
    for reason in case.state.get("routing", {}).get("reasons", []):
        typer.echo(f"  - {reason}")
    if case.status is CaseStatus.FAILED:
        raise typer.Exit(code=1)


@app.command("gmail-login")
def gmail_login() -> None:
    """Connect a Gmail account with read-only access. Opens a browser."""
    settings = Settings()
    auth = GmailAuth(settings.gmail_client_secrets_path, settings.gmail_token_path)
    try:
        auth.authorize()
    except MailboxError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Connected {connect_gmail(auth).address()}")


@app.command("gmail-logout")
def gmail_logout() -> None:
    """Delete the stored Gmail token."""
    settings = Settings()
    GmailAuth(settings.gmail_client_secrets_path, settings.gmail_token_path).disconnect()
    typer.echo("Disconnected. To revoke access fully, remove the app in your Google account.")


@app.command()
def inbox(
    dry_run: Annotated[bool, typer.Option(help="List what would be triaged, then stop.")] = False,
) -> None:
    """Triage new emails. Safe to schedule, because each email is only processed once."""
    service = TriageService(Settings())
    try:
        importer = InboxImporter(service, open_mailbox(service.settings))
        items = importer.preview()
        ready = [item for item in items if item.actionable]
        typer.echo(f"{len(items)} matching emails, {len(ready)} to triage")
        if dry_run:
            return
        for result in importer.run(item.message for item in ready):
            subject = clean_text(result.message.subject, 40)
            typer.echo(f"{result.outcome:<10} {result.case_id or '-':<20} {subject}")
    except MailboxError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    finally:
        service.close()


@app.command("eval")
def evaluate(
    seeds: Annotated[int, typer.Option(help="Variations of each scenario.")] = 3,
    output: Annotated[Path | None, typer.Option(help="Write the report as JSON.")] = None,
) -> None:
    """Score the pipeline against planted outcomes. Exits non-zero on any unsafe fast-track."""
    logging.disable(logging.ERROR)
    report = run_evaluation(Settings(), seeds=seeds)
    summary = report.as_dict()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, default=str))

    typer.echo(f"Engine              {report.engine}")
    if report.engine == "demo":
        typer.echo("                    Offline parser: measures the pipeline, not Claude.")
    typer.echo(f"Cases               {len(report.cases)}")
    typer.echo(f"Unsafe fast-tracks  {report.unsafe_fast_tracks}")
    typer.echo(f"Route accuracy      {report.route_accuracy:.1%}")
    typer.echo(f"Flag recall         {report.flag_recall:.1%}")
    typer.echo(f"Field accuracy      {report.field_accuracy:.1%}")
    if report.cost_usd:
        typer.echo(f"Cost                ${report.cost_usd:.2f}")
    for failure in summary["failures"]:
        typer.echo(f"  miss: {failure}")
    if report.unsafe_fast_tracks:
        raise typer.Exit(code=1)


@app.command("verify-audit")
def verify_audit() -> None:
    """Recompute the audit hash chain."""
    service = TriageService(Settings())
    try:
        report = service.audit.verify()
    finally:
        service.close()
    typer.echo(report.model_dump_json(indent=2))
    if not report.ok:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
