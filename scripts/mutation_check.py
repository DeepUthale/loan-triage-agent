"""Prove the test suite guards the safety-critical behaviour.

Coverage shows a line ran, not that a test would notice it breaking. Each mutation below disables
one safeguard, and the suite must then fail. A surviving mutation means that safeguard is
untested. Files are restored after every run; it takes a few minutes.

    uv run python scripts/mutation_check.py          # all of them
    uv run python scripts/mutation_check.py inbox    # names containing "inbox"
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SOURCE = ROOT / "src" / "loan_triage"

MUTATIONS = [
    (
        "routing: errors stop blocking",
        "rules/routing.py",
        "flag.severity.rank >= Severity.ERROR.rank and flag.code is not FlagCode.MISSING_DOCUMENT",
        "flag.severity.rank >= Severity.CRITICAL.rank and flag.code is not FlagCode.MISSING_DOCUMENT",
    ),
    (
        "routing: confidence gate off",
        "rules/routing.py",
        "if confidence < policy.quality.min_field_confidence:",
        "if False:",
    ),
    ("routing: reviewed not sticky", "rules/routing.py", "    if reviewed:\n", "    if False:\n"),
    (
        "routing: risk gate off",
        "rules/routing.py",
        "if risk.tier not in policy.routing.fast_track_tiers:",
        "if False:",
    ),
    (
        "grounding: quote never checked",
        "rules/grounding.py",
        "if needle in _squash(page):",
        "if True:",
    ),
    (
        "grounding: value-in-quote off",
        "rules/grounding.py",
        "if not field.quote.strip() or not _value_in_quote(kind, field.value, field.quote):",
        "if False:",
    ),
    (
        "grounding: ungrounded cap off",
        "rules/grounding.py",
        "confidence = min(model_confidence, quality.ungrounded_confidence_cap)",
        "confidence = model_confidence",
    ),
    (
        "grounding: merge always overwrites",
        "rules/grounding.py",
        "if current is None or (candidate.present, candidate.confidence) > (",
        "if True or (candidate.present, candidate.confidence) > (",
    ),
    (
        "risk: indeterminate off",
        "rules/risk.py",
        "    if missing:\n        tier = RiskTier.INDETERMINATE",
        "    if False:\n        tier = RiskTier.INDETERMINATE",
    ),
    (
        "risk: knockouts ignored",
        "rules/risk.py",
        "elif knockouts and tier in {RiskTier.LOW, RiskTier.MODERATE}:",
        "elif False:",
    ),
    (
        "income: uses higher source",
        "rules/income.py",
        "basis = min(known, key=lambda name: known[name])",
        "basis = max(known, key=lambda name: known[name])",
    ),
    (
        "graph: failure drops doc silently",
        "graph/nodes.py",
        '            return {"failures": {document_id: error.kind}}\n        self._log(\n            task["case_id"],\n            "extract",',
        '            return {}\n        self._log(\n            task["case_id"],\n            "extract",',
    ),
    (
        "graph: reextraction disabled",
        "graph/nodes.py",
        "retry = weak if passes < self.policy.quality.max_reextraction_passes else {}",
        "retry = {}",
    ),
    (
        "graph: reextraction unbounded",
        "graph/nodes.py",
        "retry = weak if passes < self.policy.quality.max_reextraction_passes else {}",
        "retry = weak if passes < 2 else {}",
    ),
    (
        "graph: log before interrupt",
        "graph/nodes.py",
        '        payload = interrupt({"kind": "review", "case_id": state["case_id"]})',
        '        self._log(state["case_id"], "review", "queued_for_underwriter", {})\n        payload = interrupt({"kind": "review", "case_id": state["case_id"]})',
    ),
    (
        "service: claim not atomic",
        "service.py",
        "if not self.cases.claim(case_id, CaseStatus.PENDING_REVIEW):",
        "if self.case(case_id).status is not CaseStatus.PENDING_REVIEW:",
    ),
    (
        "audit: redaction off",
        "storage/audit.py",
        '"detail": _canonical(_redact(detail or {})),',
        '"detail": _canonical(detail or {}),',
    ),
    (
        "audit: hash ignores content",
        "storage/audit.py",
        'return hashlib.sha256(f"{prev_hash}|{_canonical(body)}".encode()).hexdigest()',
        'return hashlib.sha256(f"{prev_hash}".encode()).hexdigest()',
    ),
    (
        "audit: no write lock",
        "storage/db.py",
        'connection.execute("BEGIN IMMEDIATE")',
        'connection.execute("BEGIN DEFERRED")',
    ),
    (
        "ssn: not masked in quote",
        "rules/grounding.py",
        "quote=mask_ssn(field.quote),",
        "quote=field.quote,",
    ),
    (
        "engine: refusal unchecked",
        "llm/anthropic_engine.py",
        '    if response.stop_reason == "refusal":',
        "    if False:",
    ),
    (
        "engine: truncation unchecked",
        "llm/anthropic_engine.py",
        '    if response.stop_reason == "max_tokens":',
        "    if False:",
    ),
    (
        "pdf: white text check off",
        "ingest/pdf.py",
        'or (_is_white(char.get("non_stroking_color")) and not _on_dark_fill(char, rects))',
        "or False",
    ),
    ("pdf: header band false positive", "ingest/pdf.py", "and not _on_dark_fill(char, rects)", ""),
    (
        "validate: stale check off",
        "rules/validation.py",
        "elif age > limits[document_type]:",
        "elif False:",
    ),
    (
        "ledger: every claim wins",
        "storage/mail.py",
        "return cursor.rowcount == 1\n\n    def finish",
        "return True\n\n    def finish",
    ),
    (
        "ledger: crashed claims never expire",
        "storage/mail.py",
        "(now - _STALE_CLAIM).isoformat())",
        '"")',
    ),
    (
        "inbox: replies processed before applications",
        "inbox/importer.py",
        "for message in sorted(messages, key=_by_arrival):",
        "for message in messages:",
    ),
    (
        "inbox: reply opens a new case",
        "inbox/importer.py",
        "        if case_id is not None:\n            return self._attach(message, source, uploads, case_id)\n",
        "",
    ),
    (
        "inbox: held reply is consumed",
        "inbox/importer.py",
        "still counts.\n        self._ledger.release(message.message_id)\n",
        "still counts.\n",
    ),
    (
        "inbox: other file types downloaded",
        "inbox/importer.py",
        "if not _looks_like_pdf(attachment):",
        "if False:",
    ),
    (
        "inbox: attachments not screened",
        "inbox/importer.py",
        "                self._service.check_upload(upload)\n",
        "                pass\n",
    ),
    (
        "inbox: failed triage consumes the email",
        "inbox/importer.py",
        "try again.\n            self._ledger.release(message.message_id)\n",
        "try again.\n",
    ),
    (
        "inbox: failed download consumes the email",
        "inbox/importer.py",
        "except MailboxError as error:\n            self._ledger.release(message.message_id)\n",
        "except MailboxError as error:\n",
    ),
    (
        "inbox: filenames not cleaned",
        "inbox/importer.py",
        "filename = clean_filename(attachment.filename)",
        "filename = attachment.filename",
    ),
    (
        "inbox: document age measured from today",
        "inbox/importer.py",
        "received_on=message.received_at.date(), ",
        "",
    ),
    (
        "gmail: asks for more than read access",
        "inbox/gmail.py",
        "auth/gmail.readonly",
        "auth/gmail.modify",
    ),
    (
        "gmail: token left readable by others",
        "inbox/gmail.py",
        "        self._token_path.chmod(0o600)\n",
        "",
    ),
    (
        "gmail: base64 padding not restored",
        "inbox/gmail.py",
        'data + "=" * (-len(data) % 4)',
        "data",
    ),
    (
        "gmail: nested attachments missed",
        "inbox/gmail.py",
        "yield from _walk(child)",
        "yield child",
    ),
    (
        "validate: integrity not critical",
        "rules/validation.py",
        "FlagCode.HIDDEN_TEXT_DETECTED: Severity.CRITICAL",
        "FlagCode.HIDDEN_TEXT_DETECTED: Severity.INFO",
    ),
]


def suite_fails() -> bool:
    command = ["uv", "run", "pytest", "tests", "-q", "-x", "--no-header", "-p", "no:cacheprovider"]
    return subprocess.run(command, cwd=ROOT, capture_output=True, timeout=300).returncode != 0


def main() -> int:
    wanted = sys.argv[1] if len(sys.argv) > 1 else ""
    selected = [item for item in MUTATIONS if wanted in item[0]]
    problems = []
    for name, relative_path, original, mutated in selected:
        path = SOURCE / relative_path
        source = path.read_text()
        if original not in source:
            print(f"STALE     {name}")
            problems.append(name)
            continue
        path.write_text(source.replace(original, mutated, 1))
        try:
            caught = suite_fails()
        finally:
            path.write_text(source)
        print(f"{'caught  ' if caught else 'SURVIVED'}  {name}", flush=True)
        if not caught:
            problems.append(name)
    print(f"{len(selected) - len(problems)}/{len(selected)} mutations caught")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
