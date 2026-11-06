# Loan Triage Agent

An agentic intake workflow for mortgage applications, built with LangGraph, Claude Sonnet, and Streamlit.

Claude reads the documents. Deterministic code checks them against each other and applies credit policy. Clean files move straight on to underwriting, while anything risky, uncertain, or suspicious stops at a human underwriter, and every decision lands in a tamper-evident audit log.

> **Portfolio project.** All documents are synthetic and the credit policy is illustrative. It is not a lending product and does not constitute compliance advice. See [Limitations](#limitations).

![Review queue](docs/images/queue.png)

## The core idea

**Claude reads, code decides.** The model transcribes what is printed. It never scores risk, picks a route, or approves anything.

Here is why that matters. This file has perfect finances: risk score 0, confidence 0.97. It is still marked urgent, because the paystub contains a planted instruction to approve it. No text in a document can change that outcome, because the model has no authority to change it.

![A prompt injection attempt escalated to an underwriter](docs/images/case_flags.png)

## How it works

```mermaid
flowchart TD
    A[Ingest PDFs<br/>hash, text, integrity scan] --> B[Classify<br/>parallel, per document]
    B --> C[Extract fields<br/>parallel, per document]
    C --> D{Quality gate<br/>quotes verified?}
    D -- weak fields, once --> E[Re-read weak fields] --> D
    D -- done --> F[Validate<br/>cross-document rules]
    F --> G[Score risk<br/>credit policy]
    G --> H{Route}
    H -- all gates pass --> K[Fast track]
    H -- documents missing --> I[Request documents<br/>pauses for upload] --> A
    H -- risky, unsure, or flagged --> J[Brief and queue<br/>pauses for underwriter]
    J -- correct a field --> F
    J -- need documents --> I
    J -- advance or decline --> L[Close case]
    K --> L

    classDef model fill:#e8f0fb,stroke:#2a78d6,color:#0b0b0b
    classDef human fill:#fdf1dc,stroke:#c98500,color:#0b0b0b
    class B,C,E model
    class I,J human
```

Blue steps call Claude and amber steps wait for a person. Everything else is deterministic Python. The brief in the review step is also written by Claude, and it explains a file without recommending an outcome.

| Stage | What happens |
|---|---|
| **Extraction** | Claude returns each field as printed, with a verbatim quote and a confidence score, through structured outputs. |
| **Verification** | Code confirms each quote exists in the PDF and contains the value. An invented quote is capped at 0.30 confidence however sure the model was. |
| **Validation** | 15 rules check completeness, identity, employer, income, arithmetic, document age, and integrity. |
| **Risk** | Eight policy factors produce a score, each with its adverse-action wording. A missing core metric means `indeterminate`, never `low`. |
| **Routing** | A file fast-tracks only if every gate passes. A machine can never decline. |
| **Review** | The underwriter sees evidence, can correct a field to re-run policy, and must give a rationale for every decision. |

### What it catches

| Category | Examples |
|---|---|
| Incomplete files | Missing documents, missing or unreadable fields |
| Inconsistency | Different name or SSN across documents, undeclared employer, stated income above documented income |
| Tampering | Paystub or bank statement that does not add up, implausible year-to-date pay |
| Staleness | Documents past their age limit, or dated after the file arrived |
| Underwriting attention | Large unsourced deposits, policy knockouts on DTI, LTV, credit, or public records |
| Attacks | Instructions aimed at the AI, white-on-white hidden text |
| Model errors | Quotes not found in the file, low confidence, refusals, truncated output |

The reasoning behind these choices, and what each one does not solve, is in [docs/design-decisions.md](docs/design-decisions.md).

## Running it

Requires [uv](https://docs.astral.sh/uv/).

**Without an API key.** This uses an offline label parser in place of Claude, which is clearly marked in the dashboard:

```bash
uv sync
uv run loan-triage seed                     # triage a sample book of files
uv run streamlit run app/streamlit_app.py   # open the dashboard
```

**With Claude:** do the same after adding your key, which makes real API calls:

```bash
cp .env.example .env      # then add your ANTHROPIC_API_KEY
```

The dashboard has a review queue, a case workspace with evidence and decisions, an intake page with uploads, an email inbox, and 22 ready-made scenarios, portfolio analytics, and an audit trail with chain verification.

| Command | Purpose |
|---|---|
| `uv run loan-triage run <folder>` | Triage one folder of PDFs |
| `uv run loan-triage generate` | Write synthetic loan files to `data/samples` |
| `uv run loan-triage inbox` | Triage new emails, safe to run on a schedule |
| `uv run loan-triage eval` | Score the pipeline against planted outcomes |
| `uv run loan-triage verify-audit` | Recompute the audit hash chain |

`make demo`, `make test`, and `make check` are shortcuts for the common tasks.

## Email intake

The **Email inbox** tab on New application pulls applications from Gmail. Each email's PDF attachments become one loan file, triaged exactly like an upload. When an applicant replies in the same thread with a missing document, it joins the waiting case instead of opening a new one.

![Email inbox](docs/images/inbox.png)

It starts on a synthetic demo inbox, so it works with no Google account. The demo covers a clean file, a defect, an injection attempt, a reply carrying a missing W-2, and an email with nothing usable.

**To connect a real Gmail account:**

1. In the [Google Cloud Console](https://console.cloud.google.com/), create a project, enable the Gmail API, and set up the OAuth consent screen, adding your address as a test user.
2. Create an OAuth client of type **Desktop app**, download its JSON, and save it as `secrets/gmail_client_secret.json`.
3. In Gmail, create the label `loan-applications` and apply it to the emails you want triaged.
4. Add `LOAN_TRIAGE_MAILBOX_MODE=gmail` to `.env`, then run `uv run loan-triage gmail-login`.

To test it safely, run `uv run loan-triage generate`, email yourself the PDFs from one folder in `data/samples`, and label that email. While the consent screen is in testing, Google expires the sign-in after about a week, so run `gmail-login` again when the tab asks.

| Safeguard | What it means |
|---|---|
| **Read-only access** | The `gmail.readonly` scope cannot send, change, or delete mail. |
| **One label** | Only emails matching `LOAN_TRIAGE_GMAIL_QUERY` are read, never the whole mailbox. |
| **Each email once** | A local ledger stops a schedule or a double click from triaging an email twice. |
| **Attachments only** | Subjects and message text are attacker-written and never reach Claude. |
| **Partial files survive** | A corrupt attachment is recorded on the case, and the good ones are still triaged. |
| **Nothing is lost** | A failed download or triage leaves the email available to try again. |

## Evaluation

`loan-triage eval` runs 22 scenarios, each planted with a known outcome, and reports:

- **Unsafe fast-tracks:** files that needed a person and did not get one. The build fails if this is ever above zero.
- **Route accuracy, flag recall, and field accuracy,** plus token usage and cost when running on Claude.

With the offline engine, over 66 files:

| Metric | Result |
|---|---|
| Unsafe fast-tracks | 0 |
| Route accuracy | 100% |
| Flag recall | 100% |
| Field accuracy | 99.9% |

**Read these as a regression gate, not a quality claim.** The same author wrote the scenarios and the rules, and the offline parser understands only these synthetic layouts. They show the pipeline behaves as designed, and they say nothing about Claude's extraction accuracy. To measure that, set `LOAN_TRIAGE_LLM_MODE=anthropic` and run the eval, which makes real API calls and costs money.

## Testing

```bash
uv run pytest                            # 326 tests
uv run python scripts/mutation_check.py  # do the tests actually catch regressions?
```

- **Property-based tests** check safety invariants over arbitrary inputs, such as "a critical flag always reaches a person."
- **The Claude and Gmail integrations** run through the real `anthropic` and `googleapiclient` libraries against fake transports, covering request shape, refusals, truncation, and every error class.
- **End-to-end tests** drive the real graph, checkpointer, and storage, with failures injected at each stage, plus concurrent reviewers and a process restart mid-review.
- **Dashboard tests** run each Streamlit page headlessly, including a hostile PDF through the full rendering path.
- **The mutation check** disables 41 safeguards one at a time and confirms the suite fails each time.

## Project layout

```
src/loan_triage/
  domain/       Enums, extraction schemas, models, value parsers
  ingest/       PDF reading, hidden-text and injection detection
  inbox/        Gmail and demo mailboxes, email-to-case importer
  llm/          Engine interface, Claude engine, demo engine, prompts
  rules/        Quote verification, validation, income, risk, routing, policy
  graph/        LangGraph state, nodes, and wiring
  storage/      SQLite, hash-chained audit log, case records
  synth/        Synthetic document generator and scenarios
  evals/        Evaluation harness
  service.py    Entry point shared by the dashboard, CLI, and evals
app/            Streamlit dashboard
config/         Credit policy
scripts/        Mutation check
```

## Limitations

- **Synthetic data only.** Real documents vary far more than these layouts. Do not upload real applicant documents or connect a mailbox that receives them: this project has no access control, encryption at rest, or retention policy.
- **Email identity.** The sender address is displayed but not verified, and it can be forged. Treat it as a label, not as proof of who applied.
- **One email, one file.** Documents sent across separate threads become separate cases.
- **Illustrative policy.** The thresholds are plausible, not validated against loan performance.
- **Evaluated offline.** A live Claude evaluation is the next step, and confidence thresholds should be tuned on its results.
- **Forgery.** Checks catch inconsistency. A forged file that is consistent throughout passes them.
- **Fair lending.** Protected characteristics are excluded, but that does not rule out proxy effects, which need testing on real outcomes.
- **Audit log.** It detects edits, but detecting deletion of the newest records needs an external anchor, which is not built.
- **Scale.** SQLite and a synchronous Streamlit process suit one reviewer on one machine. Reviewer identity is a text box where production would use SSO.

## Stack

Python 3.12, LangGraph, Claude Sonnet (`claude-sonnet-5-5`) via the Anthropic SDK, Gmail API, Pydantic, Streamlit, pdfplumber, ReportLab, SQLite, pytest, Hypothesis, Ruff, mypy, uv.
