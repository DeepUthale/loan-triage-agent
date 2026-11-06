"""The underwriter's worklist, most urgent first."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import streamlit as st
from ui import PRIORITY_LABELS, get_service, open_case

from loan_triage.domain.enums import CaseStatus
from loan_triage.storage.cases import CaseRow
from loan_triage.synth.portfolio import seed_portfolio

service = get_service()


def waiting_time(received_at: str) -> str:
    elapsed = datetime.now(UTC) - datetime.fromisoformat(received_at)
    hours = int(elapsed.total_seconds() // 3600)
    if hours < 1:
        return f"{int(elapsed.total_seconds() // 60)} min"
    return f"{hours} h" if hours < 48 else f"{hours // 24} d"


def lead_reason(case: CaseRow) -> str:
    flags = case.state.get("flags", [])
    if flags:
        extra = f" +{len(flags) - 1} more" if len(flags) > 1 else ""
        return flags[0]["code"].replace("_", " ").capitalize() + extra
    return case.state.get("routing", {}).get("reasons", ["Failed"])[0]


def queue_table(cases: list[CaseRow]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Case": case.case_id,
                "Priority": f"{case.priority} {PRIORITY_LABELS.get(case.priority or 0, '')}",
                "Applicant": case.applicant or "Unidentified",
                "Loan": float(case.loan_amount) if case.loan_amount else None,
                "Risk": f"{(case.risk_tier or 'unknown').title()} ({case.risk_score})",
                "Confidence": case.confidence or 0.0,
                "Flags": case.flag_count,
                "Waiting": waiting_time(case.received_at),
                "Lead reason": lead_reason(case),
            }
            for case in cases
        ]
    )


st.title("Review queue")

everything = service.cases.list()
if not everything:
    st.info("No files yet. Load the sample book to see the workflow end to end.")
    if st.button("Load sample portfolio", type="primary"):
        with st.spinner("Triaging sample files"):
            seed_portfolio(service)
        st.rerun()
    st.stop()

pending = [case for case in everything if case.status is CaseStatus.PENDING_REVIEW]
waiting = [case for case in everything if case.status is CaseStatus.AWAITING_DOCUMENTS]
failed = [case for case in everything if case.status is CaseStatus.FAILED]
urgent = sum(case.priority == 1 for case in pending)
fast_tracked = sum(case.status is CaseStatus.FAST_TRACKED for case in everything)

columns = st.columns(4)
columns[0].metric("Pending review", len(pending), border=True)
columns[1].metric("Urgent", urgent, border=True, help="Critical integrity flags on the file.")
columns[2].metric("Awaiting documents", len(waiting), border=True)
columns[3].metric("Fast-tracked", fast_tracked, border=True)

for case in failed:
    st.error(f"{case.case_id} stopped with an error and needs attention.", icon=":material/error:")

if pending:
    st.subheader("Needs a decision")
    selection = st.dataframe(
        queue_table(pending),
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key="queue_table",
        column_config={
            "Loan": st.column_config.NumberColumn(format="dollar"),
            "Confidence": st.column_config.ProgressColumn(
                min_value=0, max_value=1, format="%.2f", width="small"
            ),
            "Flags": st.column_config.NumberColumn(width="small"),
            "Lead reason": st.column_config.TextColumn(width="large"),
        },
    )
    rows = selection["selection"]["rows"]
    if rows:
        open_case(pending[rows[0]].case_id)
    st.caption("Select a row to open the file.")
else:
    st.success("The queue is clear.", icon=":material/task_alt:")

if waiting:
    st.subheader("Waiting on the applicant")
    choice = st.dataframe(
        pd.DataFrame(
            [
                {
                    "Case": case.case_id,
                    "Applicant": case.applicant or "Unidentified",
                    "Requested": "; ".join(case.state.get("requested_documents", [])),
                }
                for case in waiting
            ]
        ),
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key="waiting_table",
    )
    if choice["selection"]["rows"]:
        open_case(waiting[choice["selection"]["rows"][0]].case_id)
