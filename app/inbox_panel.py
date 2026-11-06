"""The Email inbox tab on the New application page.

Sender, subject, and filenames are written by whoever sent the email. On this tab they only ever
appear in dataframes and `st.text`, which show plain text, never in markdown.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from ui import STATUS_LABELS, open_case

from loan_triage.domain.enums import MailOutcome
from loan_triage.inbox.base import MailboxError, MailboxNotConnected
from loan_triage.inbox.importer import (
    ImportResult,
    InboxImporter,
    InboxItem,
    InboxState,
    clean_text,
    open_mailbox,
)
from loan_triage.service import TriageService

_STATE_LABELS = {
    InboxState.NEW: "New",
    InboxState.FOLLOW_UP: "Reply to a waiting case",
    InboxState.BLOCKED: "Held",
    InboxState.DONE: "Already processed",
}
_OUTCOME_LABELS = {
    MailOutcome.OPENED: "Case opened",
    MailOutcome.ATTACHED: "Added to case",
    MailOutcome.REJECTED: "Nothing usable",
    MailOutcome.SKIPPED: "Held",
    MailOutcome.DUPLICATE: "Already processed",
    MailOutcome.ERROR: "Failed",
}


def _render_setup(service: TriageService, problem: str) -> None:
    st.warning(problem, icon=":material/link_off:")
    st.markdown(
        f"""
**To connect Gmail**

1. In Google Cloud, enable the Gmail API and create an OAuth client of type **Desktop app**.
2. Save its JSON as `{service.settings.gmail_client_secrets_path}`.
3. In Gmail, create the label `loan-applications` and apply it to application emails.
4. Run `uv run loan-triage gmail-login`, then reload this page.

Access is read-only, and only emails matching this filter are read:
`{service.settings.gmail_query}`
"""
    )


def _refresh(importer: InboxImporter) -> None:
    try:
        st.session_state["inbox_items"] = importer.preview()
    except MailboxError as error:
        st.session_state.pop("inbox_items", None)
        st.error(str(error), icon=":material/error:")


def _inbox_frame(items: list[InboxItem]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Status": _STATE_LABELS[item.state],
                "From": clean_text(item.message.sender),
                "Subject": clean_text(item.message.subject),
                "Received": item.message.received_at.strftime("%b %d, %H:%M UTC"),
                "Files": len(item.message.attachments),
                "Case": item.case_id or "",
                "Note": item.note,
            }
            for item in items
        ]
    )


def _results_frame(results: list[ImportResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Subject": clean_text(result.message.subject),
                "Result": _OUTCOME_LABELS[result.outcome],
                "Case": result.case_id or "",
                "Status": STATUS_LABELS[result.case_status] if result.case_status else "",
                "Note": result.detail,
            }
            for result in results
        ]
    )


def _triage(importer: InboxImporter, items: list[InboxItem]) -> None:
    box = st.status(f"Triaging {len(items)} emails", expanded=True)
    with box:
        results = importer.run(
            (item.message for item in items),
            on_result=lambda result: st.text(
                f"{_OUTCOME_LABELS[result.outcome]}: {clean_text(result.message.subject, 60)}"
            ),
        )
    box.update(state="complete")
    st.session_state["inbox_results"] = results
    st.session_state["inbox_generation"] = st.session_state.get("inbox_generation", 0) + 1
    _refresh(importer)
    st.rerun()


def _render_results() -> None:
    results: list[ImportResult] = st.session_state.get("inbox_results", [])
    if not results:
        return
    st.subheader("Last run")
    generation = st.session_state.get("inbox_generation", 0)
    selection = st.dataframe(
        _results_frame(results),
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key=f"inbox_results_table_{generation}",
    )
    rows = selection["selection"]["rows"]
    if rows and results[rows[0]].case_id:
        open_case(str(results[rows[0]].case_id))
    st.caption("Select a row to open its case.")


def _render_items(importer: InboxImporter, items: list[InboxItem]) -> None:
    if not items:
        st.info("No emails match the filter.")
        return
    generation = st.session_state.get("inbox_generation", 0)
    selection = st.dataframe(
        _inbox_frame(items),
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="multi-row",
        key=f"inbox_table_{generation}",
        column_config={
            "Files": st.column_config.NumberColumn(width="small"),
            "Case": st.column_config.TextColumn(width="medium"),
        },
    )
    chosen = [items[row] for row in selection["selection"]["rows"] if items[row].actionable]
    ready = [item for item in items if item.actionable]

    left, right, _ = st.columns([1, 1, 2])
    run_chosen = left.button(
        f"Triage selected ({len(chosen)})",
        key="inbox_run_selected",
        disabled=not chosen,
        width="stretch",
    )
    run_ready = right.button(
        f"Triage all new ({len(ready)})",
        key="inbox_run_all",
        type="primary",
        disabled=not ready,
        width="stretch",
    )
    if run_chosen:
        _triage(importer, chosen)
    if run_ready:
        _triage(importer, ready)


def render_inbox(service: TriageService) -> None:
    try:
        mailbox = open_mailbox(service.settings)
    except MailboxNotConnected as error:
        _render_setup(service, str(error))
        return

    if mailbox.name == "demo":
        st.info(
            "**Demo inbox.** These are synthetic emails, so no Google account is needed. "
            "See the README to connect Gmail.",
            icon=":material/science:",
        )
    else:
        st.warning(
            "**Live mailbox.** Attachments are stored unencrypted on this machine and sent to the "
            "reading engine. This project has no access control, so use synthetic documents only.",
            icon=":material/shield:",
        )
    st.write("Each email's PDF attachments become one loan file. Replies join their existing case.")
    st.caption(
        f"Filter: `{service.settings.gmail_query}`. Access is read-only, and each email is "
        "only processed once. Only attachments are read: the message text is ignored."
    )

    importer = InboxImporter(service, mailbox)
    if st.button("Check inbox", key="inbox_check"):
        _refresh(importer)

    items: list[InboxItem] | None = st.session_state.get("inbox_items")
    if items is not None:
        _render_items(importer, items)
    _render_results()
