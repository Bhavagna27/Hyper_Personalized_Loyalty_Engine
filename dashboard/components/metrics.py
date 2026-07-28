from __future__ import annotations

from typing import Any

import streamlit as st


def _format_value(value: Any, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.0f}{suffix}"
        return f"{value:,.2f}{suffix}"
    if isinstance(value, int):
        return f"{value:,}{suffix}"
    return f"{value}{suffix}"


def render_kpi_cards(kpis: list[dict[str, Any]]) -> None:
    """Render a row of polished KPI cards."""
    if not kpis:
        return

    cols = st.columns(len(kpis))
    for col, item in zip(cols, kpis):
        label = item.get("label", "")
        value = _format_value(item.get("value"), str(item.get("suffix", "")))
        detail = item.get("detail", "")
        trend = item.get("trend")
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-detail">{detail}</div>
                    {f"<div class='kpi-detail'>{trend}</div>" if trend else ""}
                </div>
                """,
                unsafe_allow_html=True,
            )


def status_pill(label: str, value: str) -> str:
    css_class = "ready" if value.lower() == "ready" else "partial" if value.lower() == "partial" else "missing"
    return f"<span class='status-pill {css_class}'>{label}: {value}</span>"

