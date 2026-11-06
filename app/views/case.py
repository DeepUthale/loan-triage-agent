"""Everything an underwriter needs to reach a decision on one file."""

from __future__ import annotations

import pandas as pd
import pdfplumber
import streamlit as st
from ui import (
    PRIORITY_LABELS,
    PRIORITY_TONES,
    ROUTE_LABELS,
    SEVERITY_TONES,
    STATUS_LABELS,
    STATUS_TONES,
    TIER_TONES,
    bullet_list,
    esc,
    get_service,
    html,
    meter,
    money,
    percent,
    pill,
    pills,
)

from loan_triage.domain.enums import CaseStatus, ReviewAction
from loan_triage.domain.models import FieldCorrection, ReviewDecision
from loan_triage.ingest.pdf import InvalidDocument
from loan_triage.service import ReviewRejected, Upload
from loan_triage.storage.cases import CaseRow

service = get_service()

_ACTION_LABELS = {
    ReviewAction.ADVANCE: "Advance to underwriting",
    ReviewAction.REQUEST_DOCUMENTS: "Request documents",
    ReviewAction.DECLINE: "Decline",
}


@st.cache_data(show_spinner=False)
def page_image(path: str, page: int):
    with pdfplumber.open(path) as pdf:
        return pdf.pages[page].to_image(resolution=96).original.copy()


def factor(state: dict, key: str):
    factors = state.get("risk", {}).get("factors", [])
    return next((item["value"] for item in factors if item["key"] == key), None)


def pick_case() -> CaseRow | None:
    requested = st.session_state.get("case_id") or st.query_params.get("case")
    cases = service.cases.list()
    if not cases:
        st.info("No files yet. Start from the review queue or submit a new application.")
        return None
    ids = [case.case_id for case in cases]
    index = ids.index(requested) if requested in ids else 0
    label = {
        case.case_id: f"{case.case_id}  ·  {case.applicant or 'Unidentified'}" for case in cases
    }
    chosen = st.selectbox("File", ids, index=index, format_func=label.get)
    st.session_state["case_id"] = chosen
    st.query_params["case"] = chosen
    return service.case(chosen)


def render_header(case: CaseRow) -> None:
    state = case.state
    st.title(case.applicant or "Unidentified applicant")
    items = [pill(STATUS_LABELS[case.status], STATUS_TONES[case.status])]
    if case.priority:
        label = f"Priority: {PRIORITY_LABELS[case.priority]}"
        items.append(pill(label, PRIORITY_TONES[case.priority]))
    if case.risk_tier:
        items.append(pill(f"Risk: {case.risk_tier}", TIER_TONES[case.risk_tier]))
    if case.route:
        items.append(pill(f"Route: {ROUTE_LABELS[case.route]}"))
    pills(*items)

    columns = st.columns(6)
    columns[0].metric("Loan amount", money(case.loan_amount), border=True)
    columns[1].metric(
        "Risk score",
        case.risk_score if case.risk_score is not None else "n/a",
        help="0 to 100. Points from the policy factors.",
        border=True,
    )
    columns[2].metric("DTI", percent(factor(state, "dti")), border=True)
    columns[3].metric("LTV", percent(factor(state, "ltv")), border=True)
    columns[4].metric("Credit score", factor(state, "credit_score") or "n/a", border=True)
    columns[5].metric("Reserves (mo)", factor(state, "reserves_months") or "n/a", border=True)


def render_email_source(source: dict) -> None:
    if not source.get("message_id"):
        return
    st.subheader("Received by email")
    html(
        f'<p class="lt-note">From {esc(source["sender"])}<br>Subject: {esc(source["subject"])}</p>'
    )
    rejected = source.get("rejected_attachments", [])
    if rejected:
        st.warning(
            "Some attachments could not be used, so documents may look missing.",
            icon=":material/attach_file_off:",
        )
        bullet_list([f"{item['filename']}: {item['reason']}" for item in rejected])


def render_summary(case: CaseRow) -> None:
    state = case.state
    st.subheader("Why this file is here")
    bullet_list(state.get("routing", {}).get("reasons", []))

    render_email_source(state.get("source", {}))

    brief = state.get("brief")
    if brief:
        st.subheader("Brief")
        st.caption("Generated from the extracted facts. It explains; it does not recommend.")
        html(f'<p class="lt-note">{esc(brief["summary"])}</p>')
        if brief["key_concerns"]:
            st.markdown("**Look at first**")
            bullet_list(brief["key_concerns"])
        if brief["verification_steps"]:
            st.markdown("**Checks that would resolve this**")
            bullet_list(brief["verification_steps"])

    income = state.get("income", {})
    if income:
        st.subheader("Income")
        columns = st.columns(4)
        columns[0].metric("Stated /mo", money(income.get("stated_monthly")))
        columns[1].metric("Paystub /mo", money(income.get("paystub_monthly")))
        columns[2].metric("W-2 /mo", money(income.get("w2_monthly")))
        columns[3].metric(
            "Verified /mo",
            money(income.get("verified_monthly")),
            help="The lower documented figure. Risk is scored on this, never on stated income.",
        )


def render_flags(case: CaseRow) -> None:
    flags = case.state.get("flags", [])
    if not flags:
        st.success("No validation flags.", icon=":material/check_circle:")
        return
    rows = []
    for item in flags:
        badge = pill(item["severity"], SEVERITY_TONES[item["severity"]])
        # A message can quote the document, so it goes through `esc` like everything else.
        rows.append(
            f'<div class="lt-flag"><div>{badge}</div><div>{esc(item["message"])}'
            f'<div class="lt-flag-code">{esc(item["code"])}</div></div></div>'
        )
    html("".join(rows))


def render_fields(case: CaseRow) -> None:
    extractions = case.state.get("extractions", {})
    if not extractions:
        st.info("Nothing was extracted from this file.")
        return

    threshold = service.policy.quality.min_field_confidence
    for document_id, extraction in extractions.items():
        label = extraction["document_type"].replace("_", " ").title()
        scores = [f["confidence"] for f in extraction["fields"].values() if f["required"]]
        with st.expander(f"{label}  ·  weakest field {min(scores, default=0):.2f}"):
            frame = pd.DataFrame(
                [
                    {
                        "Field": name,
                        "Value": field["value"] if field["value"] is not None else field["raw"],
                        "Source quote": field["quote"],
                        "Confidence": field["confidence"],
                        "In file": {True: "Verified", False: "Not found", None: "Unverifiable"}[
                            field["grounded"]
                        ]
                        if field["present"]
                        else "Absent",
                        "Page": field["page"] or None,
                        "Corrected by": field["corrected_by"],
                    }
                    for name, field in extraction["fields"].items()
                ]
            )
            st.dataframe(
                frame,
                hide_index=True,
                width="stretch",
                key=f"fields_{document_id}",
                column_config={
                    "Confidence": st.column_config.ProgressColumn(
                        min_value=0, max_value=1, format="%.2f", width="small"
                    ),
                    "Page": st.column_config.NumberColumn(width="small"),
                    "Field": st.column_config.TextColumn(width="small"),
                    "Source quote": st.column_config.TextColumn(width="medium"),
                },
            )
            st.caption(f"Fields below {threshold:.2f} keep a file out of the fast track.")


def render_correction(case: CaseRow) -> None:
    extractions = case.state["extractions"]
    st.markdown("**Correct a field**")
    st.caption(
        "Risk and routing re-run on the corrected value. The original stays in the audit log."
    )
    documents = list(extractions)
    document_id = st.selectbox(
        "Document",
        documents,
        format_func=lambda key: extractions[key]["document_type"].replace("_", " ").title(),
        key=f"correct_doc_{case.case_id}",
    )
    field = st.selectbox(
        "Field", list(extractions[document_id]["fields"]), key=f"correct_field_{case.case_id}"
    )
    value = st.text_input("Correct value", key=f"correct_value_{case.case_id}")
    reason = st.text_input("Reason", key=f"correct_reason_{case.case_id}")
    if st.button("Apply correction", key=f"correct_apply_{case.case_id}"):
        submit(
            case.case_id,
            ReviewDecision(
                action=ReviewAction.CORRECT,
                reviewer=st.session_state.get("reviewer", ""),
                rationale=reason,
                corrections=[FieldCorrection(document_id=document_id, field=field, value=value)],
            ),
        )


def render_risk(case: CaseRow) -> None:
    risk = case.state.get("risk")
    if not risk:
        st.info("Risk was not scored.")
        return
    if risk["knockouts"]:
        st.warning("Policy knockouts: " + ", ".join(risk["knockouts"]), icon=":material/warning:")
    if risk["missing_core_factors"]:
        st.warning(
            "Could not compute: " + ", ".join(risk["missing_core_factors"]),
            icon=":material/help:",
        )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Factor": item["label"],
                    "Value": item["value"],
                    "Points": item["points"],
                    "Code": item["reason_code"],
                    "Adverse-action wording": item["adverse_action_reason"],
                }
                for item in risk["factors"]
            ]
        ),
        hide_index=True,
        width="stretch",
        column_config={"Points": st.column_config.NumberColumn(width="small")},
    )


def render_documents(case: CaseRow) -> None:
    documents = case.state.get("documents", {})
    classifications = case.state.get("classifications", {})
    if not documents:
        st.info("No documents on file.")
        return
    chosen = st.selectbox(
        "Document",
        list(documents),
        format_func=lambda key: documents[key]["filename"],
        key=f"preview_{case.case_id}",
    )
    record = documents[chosen]
    classification = classifications.get(chosen, {})
    left, right = st.columns([3, 2])
    with left:
        st.image(page_image(record["path"], 0), width="stretch")
    with right:
        st.text(f"Type: {classification.get('document_type', 'unknown')}")
        st.text(f"Pages: {record['page_count']}")
        st.text(f"Text layer: {'yes' if record['has_text_layer'] else 'no'}")
        st.text(f"SHA-256: {record['sha256'][:24]}...")
        if classification.get("rationale"):
            st.text(f"Classifier note: {classification['rationale']}")
        for finding in record["findings"]:
            st.error(f"{finding['detail']} (page {finding['page']})", icon=":material/gpp_bad:")
            st.text(finding["excerpt"])


def render_audit(case: CaseRow) -> None:
    events = service.audit.for_case(case.case_id)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "When": event.recorded_at[:19].replace("T", " "),
                    "Actor": f"{event.actor_type}: {event.actor}",
                    "Stage": event.stage,
                    "Action": event.action,
                    "Hash": event.hash[:12],
                }
                for event in events
            ]
        ),
        hide_index=True,
        width="stretch",
    )


def submit(case_id: str, decision: ReviewDecision) -> None:
    try:
        with st.spinner("Recording decision"):
            service.decide(case_id, decision)
    except ReviewRejected as rejection:
        st.error(str(rejection), icon=":material/block:")
        return
    st.toast("Decision recorded.")
    st.rerun()


def render_decision(case: CaseRow) -> None:
    state = case.state
    st.subheader("Decision")
    html(
        meter(
            "Weakest required field",
            case.confidence or 0.0,
            service.policy.quality.min_field_confidence,
        )
    )

    action = st.radio(
        "Action",
        list(_ACTION_LABELS),
        format_func=_ACTION_LABELS.get,
        key=f"action_{case.case_id}",
    )
    reason_codes: list[str] = []
    requested: list[str] = []
    if action is ReviewAction.DECLINE:
        reasons = service.policy.adverse_action_reasons
        reason_codes = st.multiselect(
            "Adverse-action reasons",
            list(reasons),
            format_func=lambda code: f"{code}: {reasons[code]}",
            key=f"reasons_{case.case_id}",
        )
    if action is ReviewAction.REQUEST_DOCUMENTS:
        note = st.text_area(
            "Documents needed, one per line", key=f"requested_{case.case_id}", height=90
        )
        requested = [line.strip() for line in note.splitlines() if line.strip()]

    rationale = st.text_area("Rationale", key=f"rationale_{case.case_id}", height=110)

    critical = [flag for flag in state.get("flags", []) if flag["severity"] == "critical"]
    if critical and action is ReviewAction.ADVANCE:
        st.warning(
            "This file has critical integrity flags. Advancing records an override under your "
            "name.",
            icon=":material/gpp_maybe:",
        )

    if st.button("Record decision", type="primary", width="stretch"):
        submit(
            case.case_id,
            ReviewDecision(
                action=action,
                reviewer=st.session_state.get("reviewer", ""),
                rationale=rationale,
                reason_codes=reason_codes,
                requested_documents=requested,
            ),
        )

    if state.get("extractions"):
        st.divider()
        render_correction(case)


def render_awaiting(case: CaseRow) -> None:
    st.subheader("Waiting on documents")
    bullet_list(case.state.get("requested_documents", []))
    files = st.file_uploader(
        "Add documents", type="pdf", accept_multiple_files=True, key=f"more_{case.case_id}"
    )
    if st.button("Resume triage", type="primary", disabled=not files):
        uploads = [Upload(file.name, file.getvalue()) for file in files or []]
        try:
            with st.spinner("Reading the new documents"):
                service.add_documents(case.case_id, uploads)
        except (ReviewRejected, InvalidDocument) as error:
            st.error(str(error), icon=":material/block:")
        else:
            st.rerun()


def render_outcome(case: CaseRow) -> None:
    st.subheader("Outcome")
    outcome = case.state.get("outcome")
    if case.error:
        st.error("The pipeline stopped before this file could be routed.", icon=":material/error:")
        st.text(case.error)
    if outcome:
        st.text(f"Decided by: {outcome['decided_by']}")
        st.text(outcome["rationale"])
        reasons = service.policy.adverse_action_reasons
        for code in outcome["reason_codes"]:
            st.text(f"{code}: {reasons.get(code, '')}")


case = pick_case()
if case is None:
    st.stop()

render_header(case)
main, side = st.columns([7, 3], gap="large")
with main:
    flag_total = len(case.state.get("flags", []))
    tabs = st.tabs(
        ["Summary", f"Flags ({flag_total})", "Extracted fields", "Risk", "Documents", "Audit"]
    )
    with tabs[0]:
        render_summary(case)
    with tabs[1]:
        render_flags(case)
    with tabs[2]:
        render_fields(case)
    with tabs[3]:
        render_risk(case)
    with tabs[4]:
        render_documents(case)
    with tabs[5]:
        render_audit(case)

with side:
    match case.status:
        case CaseStatus.PENDING_REVIEW:
            render_decision(case)
        case CaseStatus.AWAITING_DOCUMENTS:
            render_awaiting(case)
        case _:
            render_outcome(case)
