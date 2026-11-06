"""Open a new file, from uploads or from a synthetic scenario."""

from __future__ import annotations

import streamlit as st
from inbox_panel import render_inbox
from ui import get_service, open_case

from loan_triage.ingest.pdf import InvalidDocument
from loan_triage.service import ReviewRejected, Upload
from loan_triage.synth.generator import SCENARIOS
from loan_triage.synth.portfolio import submit_scenario

service = get_service()

st.title("New application")

sample, upload, email = st.tabs(["Sample scenario", "Upload documents", "Email inbox"])

with sample:
    st.write("Each scenario is a synthetic file with a known defect planted in it.")
    scenario = st.selectbox(
        "Scenario", SCENARIOS, format_func=lambda item: f"{item.key}: {item.description}"
    )
    st.caption(f"Expected route: {scenario.expected_route.value.replace('_', ' ')}")
    if st.button("Run triage", type="primary", key="run_sample"):
        with st.status("Reading documents and applying policy", expanded=False):
            case = submit_scenario(service, scenario)
        open_case(case.case_id)

with upload:
    st.write(
        "Add every document for one applicant: application, paystub, W-2, bank statement, and "
        "credit report. Use synthetic documents only."
    )
    files = st.file_uploader("PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Run triage", type="primary", key="run_upload", disabled=not files):
        uploads = [Upload(file.name, file.getvalue()) for file in files or []]
        try:
            with st.status("Reading documents and applying policy", expanded=False):
                case = service.submit(uploads)
        except (ReviewRejected, InvalidDocument) as error:
            st.error(str(error), icon=":material/block:")
        else:
            open_case(case.case_id)

with email:
    render_inbox(service)
