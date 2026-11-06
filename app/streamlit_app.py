"""Underwriter console."""

from __future__ import annotations

import streamlit as st
from ui import get_service, install_styles

st.set_page_config(page_title="Loan Triage", page_icon=":material/account_balance:", layout="wide")
install_styles()

pages = [
    st.Page("views/queue.py", title="Review queue", icon=":material/inbox:", default=True),
    st.Page("views/case.py", title="Case workspace", icon=":material/folder_open:"),
    st.Page("views/intake.py", title="New application", icon=":material/upload_file:"),
    st.Page("views/portfolio.py", title="Portfolio", icon=":material/monitoring:"),
    st.Page("views/audit.py", title="Audit trail", icon=":material/verified_user:"),
]
navigation = st.navigation(pages)

service = get_service()
with st.sidebar:
    st.text_input(
        "Reviewer",
        key="reviewer",
        placeholder="Your name",
        help="Stamped on every decision. A production deployment would take this from SSO.",
    )
    if service.engine.name == "demo":
        st.warning(
            "**Offline demo engine.** A label parser is reading the documents, not Claude. "
            "See the README to connect Claude.",
            icon=":material/science:",
        )
    else:
        st.success(f"Reading with `{service.settings.model}`", icon=":material/check_circle:")
    st.caption(f"Credit policy {service.policy.version}. All data shown is synthetic.")

navigation.run()
