from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitAPIException

from .data import build_pipeline_status, project_version
from .metrics import status_pill

logger = logging.getLogger(__name__)


NAVIGATION = [
    ("Overview", "dashboard/app.py"),
    ("Customer Segmentation", "dashboard/pages/2_Customer_Segmentation.py"),
    ("Recommendations", "dashboard/pages/3_Recommendations.py"),
    ("New Customer", "dashboard/pages/4_New_Customer.py"),
    ("Business Analytics", "dashboard/pages/5_Business_Analytics.py"),
]


def render_sidebar(active_page: str) -> None:
    status = build_pipeline_status()
    st.sidebar.markdown(
        f"""
        <div class="sidebar-title">Hyper-Personalized Loyalty Engine</div>
        <div class="sidebar-subtitle">Enterprise banking analytics dashboard</div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f"{status_pill('Pipeline', status['pipeline_status'])}<br>"
        f"{status_pill('Model', status['model_status'])}<br>"
        f"{status_pill('Artifacts', status['artifacts_status'])}",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

    st.sidebar.markdown("### Navigation")
    for label, target in NAVIGATION:
        if label == active_page:
            st.sidebar.markdown(
                f"<div class='status-pill ready' style='display:block;text-align:left;margin-bottom:0.35rem;'>{label}</div>",
                unsafe_allow_html=True,
            )
        else:
            try:
                st.sidebar.page_link(target, label=label)
            except StreamlitAPIException as exc:
                logger.warning("Could not render navigation link for %s: %s", target, exc)
                st.sidebar.markdown(f"- {label}")

    st.sidebar.markdown(
        f"""
        <div style="margin-top:1rem;padding:0.9rem;border-radius:18px;background:rgba(29,102,209,0.08);border:1px solid rgba(29,102,209,0.12);">
            <div style="color:var(--muted);font-size:0.76rem;text-transform:uppercase;letter-spacing:0.12em;font-weight:700;">Project Version</div>
            <div style="color:var(--navy);font-size:1rem;font-weight:800;">{project_version()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

