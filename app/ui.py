"""Shared dashboard pieces.

Anything that came out of an applicant's document is hostile input: flag messages and field
values quote it. `esc` must wrap every such string before it reaches markdown or HTML, or a
crafted PDF could plant a link or a tracking image on the underwriter's screen.
"""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from loan_triage.config import Settings
from loan_triage.service import TriageService

SERIES = "#2a78d6"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#6f6e69"
RULE = "#e1e0d9"
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"

PRIORITY_LABELS = {1: "Urgent", 2: "High", 3: "Normal", 4: "Low"}
STATUS_LABELS = {
    "processing": "Processing",
    "pending_review": "Pending review",
    "awaiting_documents": "Awaiting documents",
    "fast_tracked": "Fast-tracked",
    "advanced": "Advanced to underwriting",
    "declined": "Declined",
    "failed": "Failed",
}
ROUTE_LABELS = {
    "fast_track": "Fast track",
    "underwriter_review": "Underwriter review",
    "request_documents": "Request documents",
}

# A status colour never carries meaning alone: each pill pairs it with a glyph and a label.
_TONES = {
    "good": (GOOD, "●"),
    "warning": (WARNING, "▲"),
    "serious": (SERIOUS, "◆"),
    "critical": (CRITICAL, "■"),
    "neutral": ("#c3c2b7", "○"),
}
TIER_TONES = {
    "low": "good",
    "moderate": "warning",
    "high": "serious",
    "severe": "critical",
    "indeterminate": "critical",
}
SEVERITY_TONES = {
    "info": "neutral",
    "warning": "warning",
    "error": "serious",
    "critical": "critical",
}
STATUS_TONES = {
    "pending_review": "warning",
    "awaiting_documents": "neutral",
    "processing": "neutral",
    "fast_tracked": "good",
    "advanced": "good",
    "declined": "serious",
    "failed": "critical",
}
PRIORITY_TONES = {1: "critical", 2: "serious", 3: "warning", 4: "neutral"}

_CSS = """
<style>
  .block-container { padding-top: 2.4rem; max-width: 1280px; }
  h1, h2, h3 { letter-spacing: -0.012em; }
  [data-testid="stMetric"] { background: #fff; }
  [data-testid="stMetricValue"] { font-size: 1.65rem; font-variant-numeric: tabular-nums; }
  .lt-eyebrow { color: #6f6e69; font-size: .78rem; letter-spacing: .03em; margin-bottom: .1rem; }
  .lt-pill {
    display: inline-flex; align-items: center; gap: .36rem; padding: .14rem .56rem;
    border: 1px solid #e1e0d9; border-radius: 999px; background: #fff;
    font-size: .8rem; font-weight: 550; color: #0b0b0b; white-space: nowrap;
  }
  .lt-pills { display: flex; flex-wrap: wrap; gap: .4rem; margin: .3rem 0 .9rem; }
  .lt-reasons { margin: 0; padding-left: 1.1rem; }
  .lt-reasons li { margin: .22rem 0; line-height: 1.42; }
  .lt-flag {
    display: grid; grid-template-columns: 7.2rem 1fr; gap: .8rem; align-items: start;
    padding: .62rem 0; border-bottom: 1px solid #e1e0d9;
  }
  .lt-flag:last-child { border-bottom: 0; }
  .lt-flag-code { color: #6f6e69; font-size: .76rem; font-family: ui-monospace, monospace; }
  .lt-meter { margin: .15rem 0 .85rem; }
  .lt-meter-row { display: flex; justify-content: space-between; font-size: .84rem; }
  .lt-meter-track {
    position: relative; height: 8px; border-radius: 4px; background: #e8f0fb; margin-top: .3rem;
  }
  .lt-meter-fill { height: 100%; border-radius: 4px; background: #2a78d6; }
  .lt-meter-tick {
    position: absolute; top: -3px; bottom: -3px; width: 2px; background: #52514e;
  }
  .lt-note { color: #52514e; font-size: .84rem; line-height: 1.45; }
</style>
"""


@st.cache_resource
def get_service() -> TriageService:
    return TriageService(Settings())


def install_styles() -> None:
    st.html(_CSS)


def esc(value: Any) -> str:
    """Encode everything but plain text, which neutralizes both HTML and markdown syntax."""
    safe = " .,:;'/$%-"
    return "".join(
        char if char.isalnum() or char in safe else f"&#{ord(char)};"
        for char in " ".join(str(value).split())
    )


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def pill(label: str, tone: str = "neutral") -> str:
    color, glyph = _TONES[tone]
    return (
        f'<span class="lt-pill"><span style="color:{color}" aria-hidden="true">{glyph}</span>'
        f"{esc(label)}</span>"
    )


def pills(*items: str) -> None:
    html(f'<div class="lt-pills">{"".join(items)}</div>')


def meter(label: str, value: float, threshold: float | None = None) -> str:
    percent = max(0.0, min(value, 1.0)) * 100
    tick = (
        f'<div class="lt-meter-tick" style="left:{threshold * 100:.1f}%"'
        f' title="Bar: {threshold:.2f}"></div>'
        if threshold is not None
        else ""
    )
    return (
        f'<div class="lt-meter"><div class="lt-meter-row"><span>{esc(label)}</span>'
        f"<strong>{value:.2f}</strong></div>"
        f'<div class="lt-meter-track" role="meter" aria-valuenow="{value:.2f}" aria-valuemin="0"'
        f' aria-valuemax="1" aria-label="{esc(label)}">'
        f'<div class="lt-meter-fill" style="width:{percent:.1f}%"></div>{tick}</div></div>'
    )


def bullet_list(items: list[str]) -> None:
    rows = "".join(f"<li>{esc(item)}</li>" for item in items)
    html(f'<ul class="lt-reasons">{rows}</ul>')


def money(value: Any) -> str:
    return "n/a" if value in (None, "None") else f"${float(value):,.0f}"


def percent(value: Any) -> str:
    return "n/a" if value in (None, "None") else f"{float(value):.1%}"


def open_case(case_id: str) -> None:
    st.session_state["case_id"] = case_id
    st.switch_page("views/case.py")


def bars(
    data: pd.DataFrame, category: str, value: str, *, order: list[str] | None = None
) -> alt.LayerChart:
    """Horizontal bars on one baseline, values printed directly, a tooltip on every mark."""
    height = max(len(data) * 34 + 8, 80)
    axis = alt.Axis(
        title=None,
        domain=False,
        ticks=False,
        labelLimit=240,
        labelColor=INK_SECONDARY,
        labelPadding=8,
    )
    base = alt.Chart(data).encode(
        y=alt.Y(f"{category}:N", sort=order or "-x", axis=axis),
        x=alt.X(f"{value}:Q", axis=None, scale=alt.Scale(domainMin=0, nice=False)),
        tooltip=[alt.Tooltip(f"{category}:N"), alt.Tooltip(f"{value}:Q")],
    )
    columns = base.mark_bar(color=SERIES, cornerRadiusEnd=4, height=18)
    labels = base.mark_text(align="left", dx=6, color=INK, fontWeight=600).encode(
        text=alt.Text(f"{value}:Q")
    )
    return (
        (columns + labels)
        .properties(height=height, padding={"right": 28})
        .configure_view(strokeWidth=0)
        .configure(background="transparent", font="sans-serif")
    )
