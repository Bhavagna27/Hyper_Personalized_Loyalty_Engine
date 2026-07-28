from __future__ import annotations

from datetime import datetime

import streamlit as st

from .data import generated_timestamp, project_version


def configure_page(page_title: str) -> None:
    st.set_page_config(
        page_title=page_title,
        page_icon=":material/account_balance:",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def apply_theme() -> None:
    st.markdown(
        """
        <style>
            :root {
                --navy: #061a33;
                --navy-2: #0f2d57;
                --blue: #1d66d1;
                --blue-2: #2f86ff;
                --sky: #eef5fd;
                --border: rgba(18, 56, 104, 0.12);
                --text: #0b1f3d;
                --muted: #58708f;
                --card: rgba(255, 255, 255, 0.92);
            }

            html, body, [class*="css"]  {
                font-family: "Segoe UI", "Inter", -apple-system, BlinkMacSystemFont, sans-serif;
            }

            .stApp {
                background:
                    radial-gradient(circle at top right, rgba(47, 134, 255, 0.14), transparent 28%),
                    radial-gradient(circle at bottom left, rgba(29, 102, 209, 0.10), transparent 24%),
                    linear-gradient(180deg, #f9fbfe 0%, #edf4fc 100%);
                color: var(--text);
            }

            [data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer {
                visibility: hidden;
                height: 0;
            }

            div[data-testid="stSidebarNav"] {
                display: none;
            }

            .page-shell {
                padding-top: 0.25rem;
            }

            .hero {
                background: linear-gradient(135deg, rgba(6, 26, 51, 0.97), rgba(29, 102, 209, 0.95));
                color: white;
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 24px;
                padding: 1.4rem 1.5rem;
                box-shadow: 0 18px 48px rgba(6, 26, 51, 0.18);
                margin-bottom: 1rem;
            }

            .hero h1 {
                margin: 0;
                font-size: 2.1rem;
                font-weight: 760;
                letter-spacing: -0.02em;
            }

            .hero p {
                margin: 0.5rem 0 0;
                color: rgba(255,255,255,0.88);
                font-size: 0.98rem;
            }

            .section-card {
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 22px;
                padding: 1rem 1.05rem;
                box-shadow: 0 12px 30px rgba(9, 33, 65, 0.06);
            }

            .kpi-card {
                background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(240,246,255,0.95));
                border: 1px solid rgba(28, 89, 169, 0.12);
                border-radius: 18px;
                padding: 1rem 1rem 0.9rem;
                box-shadow: 0 10px 24px rgba(7, 26, 51, 0.07);
                min-height: 120px;
            }

            .kpi-label {
                color: var(--muted);
                font-size: 0.76rem;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                font-weight: 700;
                margin-bottom: 0.35rem;
            }

            .kpi-value {
                color: var(--text);
                font-size: 1.9rem;
                font-weight: 800;
                line-height: 1.0;
            }

            .kpi-detail {
                color: var(--muted);
                font-size: 0.9rem;
                margin-top: 0.35rem;
            }

            .status-pill {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                border-radius: 999px;
                padding: 0.35rem 0.7rem;
                font-size: 0.8rem;
                font-weight: 700;
                border: 1px solid rgba(29, 102, 209, 0.16);
                background: rgba(29, 102, 209, 0.08);
                color: var(--navy-2);
            }

            .status-pill.ready {
                background: rgba(16, 185, 129, 0.10);
                border-color: rgba(16, 185, 129, 0.20);
                color: #0f7b5c;
            }

            .status-pill.partial {
                background: rgba(245, 158, 11, 0.10);
                border-color: rgba(245, 158, 11, 0.20);
                color: #9a6700;
            }

            .status-pill.missing {
                background: rgba(239, 68, 68, 0.10);
                border-color: rgba(239, 68, 68, 0.20);
                color: #b42318;
            }

            .sidebar-title {
                font-size: 1.08rem;
                font-weight: 800;
                color: var(--navy);
                margin-bottom: 0.2rem;
            }

            .sidebar-subtitle {
                color: var(--muted);
                font-size: 0.86rem;
                margin-bottom: 1rem;
            }

            .footer-bar {
                margin-top: 1.25rem;
                padding: 0.85rem 1rem;
                border-radius: 16px;
                background: rgba(255,255,255,0.72);
                border: 1px solid var(--border);
                color: var(--muted);
                font-size: 0.86rem;
            }

            .subtle-note {
                color: var(--muted);
                font-size: 0.88rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_heading(title: str, subtitle: str | None = None) -> None:
    st.markdown(
        f"""
        <div style="margin: 1rem 0 0.65rem;">
            <div style="font-size:1.25rem;font-weight:800;color:var(--navy);">{title}</div>
            {"<div class='subtle-note'>" + subtitle + "</div>" if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        f"""
        <div class="footer-bar">
            <strong>Project Version:</strong> {project_version()} &nbsp;|&nbsp;
            <strong>Generated Time:</strong> {generated_timestamp()}
        </div>
        """,
        unsafe_allow_html=True,
    )
