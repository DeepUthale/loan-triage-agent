"""Inspect the decision log and prove it has not been altered."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from ui import get_service

service = get_service()

st.title("Audit trail")
st.write(
    "Every agent and human decision, in order. Each event's hash covers its content and the "
    "previous hash, so changing any record breaks the chain from that point on."
)

left, right = st.columns([1, 3])
with left:
    verify = st.button("Verify chain", type="primary")
if verify:
    report = service.audit.verify()
    if report.ok:
        st.success(
            f"Chain intact across {report.events_checked} events.", icon=":material/verified:"
        )
    else:
        st.error(
            f"Chain broken at event {report.first_bad_seq}: {report.problem}.",
            icon=":material/gpp_bad:",
        )
st.caption(f"Head hash: `{service.audit.head_hash()}`")

events = service.audit.recent(1_000)
if not events:
    st.info("No events yet.")
    st.stop()

case_ids = sorted({event.case_id for event in events}, reverse=True)
chosen = st.selectbox("Filter by case", ["All cases", *case_ids])
if chosen != "All cases":
    events = [event for event in events if event.case_id == chosen]

selection = st.dataframe(
    pd.DataFrame(
        [
            {
                "Seq": event.seq,
                "When": event.recorded_at[:19].replace("T", " "),
                "Case": event.case_id,
                "Actor type": str(event.actor_type),
                "Actor": event.actor,
                "Stage": event.stage,
                "Action": event.action,
            }
            for event in events
        ]
    ),
    hide_index=True,
    width="stretch",
    height=420,
    on_select="rerun",
    selection_mode="single-row",
    key="audit_table",
)
rows = selection["selection"]["rows"]
if rows:
    event = events[rows[0]]
    st.subheader(f"Event {event.seq}")
    st.caption(f"Hash `{event.hash}`")
    st.json(event.detail)
