"""How the book is flowing through triage."""

from __future__ import annotations

from collections import Counter

import pandas as pd
import streamlit as st
from ui import ROUTE_LABELS, bars, get_service

from loan_triage.domain.enums import CaseStatus

service = get_service()

_TIER_ORDER = ["Low", "Moderate", "High", "Severe", "Indeterminate"]

st.title("Portfolio")

cases = service.cases.list()
if not cases:
    st.info("No files yet.")
    st.stop()

routed = [case for case in cases if case.route]
fast_tracked = sum(case.status is CaseStatus.FAST_TRACKED for case in cases)
straight_through = fast_tracked / len(cases)
scores = [case.confidence for case in cases if case.confidence is not None]
events = service.audit.recent(10_000)
calls = [event.detail["call"] for event in events if "call" in event.detail]
spend = sum(call["cost_usd"] or 0.0 for call in calls)
cached = sum(call["cache_read_tokens"] for call in calls)
prompted = sum(call["input_tokens"] for call in calls) + cached

columns = st.columns(4)
columns[0].metric("Files", len(cases), border=True)
columns[1].metric(
    "Straight-through",
    f"{straight_through:.0%}",
    help="Share of files that needed no person during triage.",
    border=True,
)
columns[2].metric(
    "Model spend",
    f"${spend:,.2f}",
    help="From the token counts on each audited call. Zero under the demo engine.",
    border=True,
)
columns[3].metric(
    "Prompt cache hit rate",
    f"{cached / prompted:.0%}" if prompted else "n/a",
    help="Share of input tokens served from cache.",
    border=True,
)

left, right = st.columns(2, gap="large")

route_counts = Counter(ROUTE_LABELS[case.route] for case in routed)
route_frame = pd.DataFrame(route_counts.items(), columns=["Route", "Files"])
with left:
    st.subheader("Where files went")
    st.altair_chart(bars(route_frame, "Route", "Files"), width="stretch")

tier_counts = Counter((case.risk_tier or "").title() for case in routed)
tier_frame = pd.DataFrame(
    [(tier, tier_counts[tier]) for tier in _TIER_ORDER if tier_counts[tier]],
    columns=["Tier", "Files"],
)
with right:
    st.subheader("Risk tier")
    st.altair_chart(bars(tier_frame, "Tier", "Files", order=_TIER_ORDER), width="stretch")

flag_counts = Counter(
    code.replace("_", " ").capitalize()
    for case in cases
    for code in {flag["code"] for flag in case.state.get("flags", [])}
)
flag_frame = pd.DataFrame(flag_counts.most_common(10), columns=["Flag", "Files"])
if not flag_frame.empty:
    st.subheader("Files by flag")
    st.altair_chart(bars(flag_frame, "Flag", "Files"), width="stretch")

with st.expander("View as tables"):
    st.dataframe(route_frame, hide_index=True)
    st.dataframe(tier_frame, hide_index=True)
    st.dataframe(flag_frame, hide_index=True)
