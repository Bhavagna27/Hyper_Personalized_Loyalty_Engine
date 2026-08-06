from __future__ import annotations

import ast
from typing import Any

import pandas as pd
import streamlit as st

from .data import extract_reward_names, extract_scores


def _safe_parse(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return ast.literal_eval(text)
        except Exception:
            return text
    return value


def render_table_section(
    title: str,
    df: pd.DataFrame,
    *,
    subtitle: str | None = None,
    columns: list[str] | None = None,
    max_rows: int = 10,
    height: int = 360,
) -> None:
    subtitle_html = f"<div class=\"subtle-note\">{subtitle}</div>" if subtitle else ""
    st.markdown(
        f"<div class='section-card'><div style='font-size:1.05rem;font-weight:800;color:var(--navy);margin-bottom:0.25rem;'>{title}</div>"
        f"{subtitle_html}</div>",
        unsafe_allow_html=True,
    )
    if df.empty:
        st.info("No data available for this section yet.")
        return
    display_df = df.copy()
    if columns:
        existing = [column for column in columns if column in display_df.columns]
        display_df = display_df[existing]
    if len(display_df) > max_rows:
        display_df = display_df.head(max_rows)
    st.dataframe(display_df, width="stretch", height=height)


def flatten_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    preview = df.copy()
    preview["Top_3"] = preview["Top_3_Rewards"].apply(
        lambda value: ", ".join(extract_reward_names(value)[:3])
    )
    preview["Top_Score"] = preview["Scores"].apply(
        lambda value: extract_scores(value)[0] if extract_scores(value) else None
    )
    columns = [column for column in [
        "Customer_ID",
        "Cluster",
        "Persona",
        "Top_3",
        "Top_Score",
        "Business_Insight",
        "Expected_Impact",
        "Customer_Message",
    ] if column in preview.columns]
    return preview[columns]


def flatten_new_customer_predictions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    preview = df.copy()
    preview["Recommended_Rewards"] = preview["recommended_rewards"].apply(
        lambda value: ", ".join(extract_reward_names(value)[:3])
    ) if "recommended_rewards" in preview.columns else ""
    columns = [column for column in [
        "customer_id",
        "predicted_cluster",
        "persona",
        "similarity_score",
        "nearest_cluster_distance",
        "confidence_level",
        "Recommended_Rewards",
        "business_insight",
        "expected_impact",
    ] if column in preview.columns]
    return preview[columns]


def summary_records_table(records: list[dict[str, Any]], title: str, subtitle: str | None = None) -> None:
    df = pd.DataFrame(records)
    render_table_section(title, df, subtitle=subtitle, max_rows=20, height=420)
