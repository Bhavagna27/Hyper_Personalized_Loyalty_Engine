from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.charts import (
    architecture_diagram,
    cluster_comparison_figure,
    cluster_sizes_figure,
    dataset_summary_figure,
    feature_summary_figure,
    model_metrics_figure,
    persona_distribution_figure,
    pipeline_flow_figure,
    recommendation_scores_figure,
    reward_catalog_figure,
    similarity_figure,
    top_rewards_figure,
    validation_issues_figure,
)
from dashboard.components.data import (
    build_kpis,
    build_pipeline_status,
    generated_timestamp,
    load_ai_recommendations,
    load_cluster_profiles,
    load_cluster_statistics,
    load_cluster_summary,
    load_customer_features,
    load_feature_metadata,
    load_llm_recommendations,
    load_model_evaluation,
    load_new_customer_predictions,
    load_recommendations,
    load_reward_catalog,
    load_scored_customers,
    load_validation_report,
    extract_reward_names,
    summarize_model_evaluation,
    summarize_validation_report,
)
from dashboard.components.metrics import render_kpi_cards
from dashboard.components.tables import (
    flatten_new_customer_predictions,
    flatten_recommendations,
    render_table_section,
)
from dashboard.components.theme import apply_theme, render_footer, section_heading


def _page_hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_standard_kpis() -> None:
    kpis = build_kpis()
    render_kpi_cards(
        [
            {
                "label": "Total Customers",
                "value": kpis.get("total_customers"),
                "detail": "Customer records represented in the engineered pipeline.",
            },
            {
                "label": "Number of Clusters",
                "value": kpis.get("num_clusters"),
                "detail": "Operational personas currently active in the model.",
            },
            {
                "label": "Recommendations Generated",
                "value": kpis.get("recommendations_generated"),
                "detail": "Customer-level recommendation rows exported.",
            },
            {
                "label": "Average Similarity Score",
                "value": kpis.get("average_similarity_score"),
                "detail": "Average new-customer fit score from the latest output.",
            },
            {
                "label": "Average Recommendation Score",
                "value": kpis.get("average_recommendation_score"),
                "detail": "Average top-ranked recommendation score across customers.",
            },
        ]
    )


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _download_csv_button(label: str, df: pd.DataFrame, filename: str, *, key: str | None = None, disabled: bool = False) -> None:
    st.download_button(
        label,
        data=_df_to_csv_bytes(df) if not df.empty else b"",
        file_name=filename,
        mime="text/csv",
        width="stretch",
        disabled=disabled or df.empty,
        key=key,
    )


def _chart_with_spinner(message: str, figure: go.Figure) -> None:
    with st.spinner(message):
        st.plotly_chart(figure, width="stretch")


def _status_strip(items: list[tuple[str, str]]) -> None:
    chips = "".join(
        f"<span class='status-pill {value.lower()}' style='margin-right:0.45rem;margin-bottom:0.35rem;'>{label}: {value}</span>"
        for label, value in items
    )
    st.markdown(f"<div style='margin:0.2rem 0 1rem;'>{chips}</div>", unsafe_allow_html=True)


def _figure_download_section(title: str, subtitle: str, df: pd.DataFrame, filename: str, *, key: str) -> None:
    st.markdown(
        f"""
        <div class="section-card">
            <div style="font-size:1.05rem;font-weight:800;color:var(--navy);margin-bottom:0.25rem;">{title}</div>
            <div class="subtle-note">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _download_csv_button("Download CSV", df, filename, key=key)


def _confusion_matrix_figure(matrix: list[list[Any]], labels: list[str], title: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=matrix,
                x=labels,
                y=labels,
                colorscale=["#EEF5FD", "#8AB8FF", "#1D66D1", "#0B1F3A"],
                hovertemplate="Predicted %{x}<br>Actual %{y}<br>Count %{z}<extra></extra>",
                showscale=False,
            )
        ]
    )
    fig.update_layout(
        template="plotly_white",
        height=340,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text=title, x=0.02, xanchor="left"),
        font=dict(family="Segoe UI, Inter, Arial, sans-serif", color="#0b1f3d"),
    )
    return fig


def render_overview_page() -> None:
    apply_theme()
    from dashboard.components.sidebar import render_sidebar

    render_sidebar("Overview")
    _page_hero(
        "Enterprise Banking Loyalty Dashboard",
        "A production dashboard for customer segmentation, recommendation performance, and model governance.",
    )
    validation_df = summarize_validation_report()
    validation_report = load_validation_report()
    feature_metadata = load_feature_metadata()
    cluster_summary = load_cluster_summary()
    recommendations = load_recommendations()
    cluster_profiles = load_cluster_profiles()
    model_eval_df = summarize_model_evaluation()
    model_eval_payload = load_model_evaluation()
    pipeline_status = build_pipeline_status()

    _render_standard_kpis()
    _status_strip(
        [
            ("Pipeline", pipeline_status["pipeline_status"]),
            ("Model", pipeline_status["model_status"]),
            ("Artifacts", pipeline_status["artifacts_status"]),
        ]
    )

    executive_tab, governance_tab, commercial_tab = st.tabs(
        ["Executive View", "Governance View", "Commercial View"]
    )

    with executive_tab:
        left, right = st.columns([1.15, 0.85], gap="large")
        with left:
            section_heading(
                "Program Architecture",
                "How the pipeline connects ingestion, feature engineering, segmentation, and recommendation outputs.",
            )
            _chart_with_spinner("Loading architecture diagram...", architecture_diagram())
            _chart_with_spinner("Loading pipeline flow...", pipeline_flow_figure())
        with right:
            section_heading(
                "Validation Snapshot",
                "Inspect sheet-level quality controls from the latest validation run.",
            )
            if validation_report.get("sheets"):
                sheet_names = list(validation_report["sheets"].keys())
                selected_sheet = st.selectbox("Select a validation sheet", sheet_names)
                selected_sheet_payload = validation_report["sheets"].get(selected_sheet, {})
                sheet_cards = st.columns(3)
                sheet_metrics = [
                    ("Rows", selected_sheet_payload.get("total_rows", 0)),
                    ("Columns", selected_sheet_payload.get("total_columns", 0)),
                    ("Issues", len(selected_sheet_payload.get("invalid_records", []))),
                ]
                for col, (label, value) in zip(sheet_cards, sheet_metrics):
                    with col:
                        st.markdown(
                            f"""
                            <div class="kpi-card">
                                <div class="kpi-label">{label}</div>
                                <div class="kpi-value" style="font-size:1.5rem;">{value}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                sheet_columns = pd.DataFrame(selected_sheet_payload.get("column_stats", []))
                render_table_section(
                    f"{selected_sheet} Column Stats",
                    sheet_columns,
                    max_rows=8,
                    height=260,
                )
            else:
                st.info("No validation report was found for this environment.")

            st.markdown("### Downloads")
            _download_csv_button("Download validation summary", validation_df, "validation_summary.csv", key="overview-validation-summary")
            feature_groups = feature_metadata.get("feature_groups", {}) if isinstance(feature_metadata, dict) else {}
            feature_rows = pd.DataFrame(
                [
                    {"Feature Group": group.replace("_", " ").title(), "Count": len(features)}
                    for group, features in feature_groups.items()
                ]
            )
            _download_csv_button("Download feature groups", feature_rows, "feature_groups.csv", key="overview-feature-groups")

    with governance_tab:
        left, right = st.columns([1.0, 1.0], gap="large")
        with left:
            section_heading("Dataset Summary", "Workbook-scale and engineered-data counts from the generated artifacts.")
            dataset_summary = (
                validation_df[["Sheet", "Rows", "Columns", "Missing Values", "Duplicate Rows", "Issues"]]
                if not validation_df.empty
                else pd.DataFrame()
            )
            _chart_with_spinner("Loading dataset summary...", dataset_summary_figure(validation_df))
            render_table_section("Validation Snapshot", dataset_summary, max_rows=10, height=280)
            _download_csv_button("Download validation snapshot", dataset_summary, "validation_snapshot.csv", key="overview-validation-snapshot")
        with right:
            section_heading("Feature Summary", "Feature groups used by the current pipeline.")
            _chart_with_spinner("Loading feature summary...", feature_summary_figure(feature_metadata))
            feature_groups = feature_metadata.get("feature_groups", {}) if isinstance(feature_metadata, dict) else {}
            feature_rows = pd.DataFrame(
                [
                    {"Feature Group": group.replace("_", " ").title(), "Count": len(features)}
                    for group, features in feature_groups.items()
                ]
            )
            render_table_section("Feature Group Counts", feature_rows, max_rows=10, height=280)
            _download_csv_button("Download feature groups", feature_rows, "feature_groups.csv", key="overview-feature-groups-governance")

        st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)
        section_heading("Execution Statistics", "Model performance and validation issues for the latest run.")
        stats_left, stats_right = st.columns([1.0, 1.0], gap="large")
        with stats_left:
            _chart_with_spinner("Loading model metrics...", model_metrics_figure(model_eval_df))
            render_table_section("Model Evaluation", model_eval_df, max_rows=10, height=250)
            _download_csv_button("Download model metrics", model_eval_df, "model_evaluation.csv", key="overview-model-metrics")
        with stats_right:
            _chart_with_spinner("Loading validation issues...", validation_issues_figure(validation_df))
            issues_rows = (
                validation_df[["Sheet", "Issues", "Missing Values", "Duplicate Rows"]]
                if not validation_df.empty
                else pd.DataFrame()
            )
            render_table_section("Execution Issues Snapshot", issues_rows, max_rows=10, height=250)

    with commercial_tab:
        left, right = st.columns([1.0, 1.0], gap="large")
        with left:
            section_heading("Cluster Summary", "Segment sizes and profile shapes from the latest clustering artifact.")
            _chart_with_spinner("Loading cluster distribution...", cluster_sizes_figure(cluster_summary))
            cluster_preview = (
                cluster_profiles[[
                    "cluster_id",
                    "persona",
                    "cluster_size",
                    "average_spend",
                    "customer_engagement_score",
                    "reward_utilization_pct",
                ]]
                if not cluster_profiles.empty
                else pd.DataFrame()
            )
            render_table_section("Cluster Profiles", cluster_preview, max_rows=10, height=280)
            _download_csv_button("Download cluster profiles", cluster_preview, "cluster_profiles.csv", key="overview-cluster-profiles")
        with right:
            section_heading("Recommendation Summary", "Current recommendation distribution across personas and reward names.")
            _chart_with_spinner("Loading persona distribution...", persona_distribution_figure(recommendations))
            _chart_with_spinner("Loading top reward demand...", top_rewards_figure(recommendations))
            recommendation_preview = flatten_recommendations(recommendations)
            render_table_section(
                "Recommendation Preview",
                recommendation_preview,
                max_rows=10,
                height=280,
            )
            _download_csv_button(
                "Download recommendation preview",
                recommendation_preview,
                "recommendation_preview.csv",
                key="overview-recommendations",
            )

    if model_eval_payload:
        st.caption("The overview page is connected to the latest production artifacts and can be exported directly from each section.")

    render_footer()


def render_customer_segmentation_page() -> None:
    apply_theme()
    from dashboard.components.sidebar import render_sidebar

    render_sidebar("Customer Segmentation")
    _page_hero(
        "Customer Segmentation",
        "Cluster composition, persona assignment, and centroid-level business interpretation.",
    )
    _render_standard_kpis()

    cluster_summary = load_cluster_summary()
    cluster_profiles = load_cluster_profiles()
    cluster_statistics = load_cluster_statistics()

    section_heading("Cluster Overview", "Cluster size and persona distribution from the latest segmentation artifact.")
    left, right = st.columns([1.0, 1.0], gap="large")
    with left:
        st.plotly_chart(cluster_sizes_figure(cluster_summary), width="stretch")
    with right:
        cluster_profile_rows = cluster_profiles[[
            "cluster_id",
            "persona",
            "cluster_size",
            "percentage_of_customers",
            "business_summary",
        ]] if not cluster_profiles.empty else pd.DataFrame()
        render_table_section("Cluster Profiles", cluster_profile_rows, max_rows=20, height=360)

    selected_cluster = None
    if not cluster_profiles.empty and "cluster_id" in cluster_profiles.columns:
        selected_cluster = int(cluster_profiles["cluster_id"].iloc[0])
        if len(cluster_profiles) > 1:
            selected_cluster = st.selectbox(
                "Select a cluster for detailed inspection",
                options=cluster_profiles["cluster_id"].astype(int).tolist(),
                format_func=lambda value: f"Cluster {value}",
                index=0,
            )

    section_heading("Centroid Comparison", "Normalized feature profiles across the active clusters.")
    st.plotly_chart(cluster_comparison_figure(cluster_statistics, selected_cluster), width="stretch")

    if not cluster_profiles.empty and "cluster_id" in cluster_profiles.columns:
        selected_row = cluster_profiles[cluster_profiles["cluster_id"].astype(int) == int(selected_cluster)].iloc[0]
        insight_cols = st.columns(3)
        insight_items = [
            ("Persona", selected_row.get("persona", "N/A")),
            ("Cluster Size", f"{int(selected_row.get('cluster_size', 0)):,}"),
            ("Avg Spend", f"{float(selected_row.get('average_spend', 0.0)):,.2f}"),
        ]
        for col, (label, value) in zip(insight_cols, insight_items):
            with col:
                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-label">{label}</div>
                        <div class="kpi-value" style="font-size:1.45rem;">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    render_footer()


def render_recommendations_page() -> None:
    apply_theme()
    from dashboard.components.sidebar import render_sidebar

    render_sidebar("Recommendations")
    _page_hero(
        "Recommendations",
        "Customer-level recommendation exports with score ranking, persona alignment, and grounded messages.",
    )
    recommendations = load_recommendations()
    reward_catalog = load_reward_catalog()
    recommendation_preview = flatten_recommendations(recommendations)
    if not recommendation_preview.empty:
        recommendation_preview["Top_Score"] = pd.to_numeric(recommendation_preview["Top_Score"], errors="coerce")

    _render_standard_kpis()

    kpi_cards = [
        {
            "label": "Recommendation Rows",
            "value": len(recommendations),
            "detail": "Customer-level recommendation outputs currently available.",
        },
        {
            "label": "Personas",
            "value": recommendation_preview["Persona"].nunique() if not recommendation_preview.empty and "Persona" in recommendation_preview.columns else None,
            "detail": "Distinct personas represented in the current export.",
        },
        {
            "label": "Avg Top Score",
            "value": recommendation_preview["Top_Score"].mean() if not recommendation_preview.empty and "Top_Score" in recommendation_preview.columns else None,
            "detail": "Mean score of the highest-ranked reward per customer.",
        },
        {
            "label": "Reward Catalog Rows",
            "value": len(reward_catalog),
            "detail": "Approved rewards available to the ranking engine.",
        },
    ]
    render_kpi_cards(kpi_cards)

    filters_left, filters_right, filters_third = st.columns([1.15, 0.95, 0.9], gap="large")
    with filters_left:
        persona_options = (
            sorted(recommendation_preview["Persona"].dropna().unique().tolist())
            if not recommendation_preview.empty and "Persona" in recommendation_preview.columns
            else []
        )
        selected_personas = st.multiselect(
            "Filter by persona",
            options=persona_options,
            default=persona_options,
        )
    with filters_right:
        min_score = float(recommendation_preview["Top_Score"].min()) if not recommendation_preview.empty and recommendation_preview["Top_Score"].notna().any() else 0.0
        max_score = float(recommendation_preview["Top_Score"].max()) if not recommendation_preview.empty and recommendation_preview["Top_Score"].notna().any() else 100.0
        score_floor = st.slider("Minimum top score", 0.0, max(100.0, max_score), min(100.0, max(min_score, 0.0)), 0.5)
    with filters_third:
        preview_limit = st.slider("Preview rows", 5, 30, 15, 5)

    filtered_preview = recommendation_preview.copy()
    if not filtered_preview.empty:
        if selected_personas:
            filtered_preview = filtered_preview[filtered_preview["Persona"].isin(selected_personas)]
        filtered_preview = filtered_preview[
            filtered_preview["Top_Score"].fillna(0.0) >= float(score_floor)
        ]
        filtered_preview = filtered_preview.sort_values("Top_Score", ascending=False).head(preview_limit)

    filtered_recommendations = recommendations.loc[filtered_preview.index] if not filtered_preview.empty else recommendations.iloc[0:0].copy()

    filtered_cards = st.columns(4)
    filtered_metrics = [
        ("Filtered Rows", len(filtered_preview)),
        ("Avg Top Score", filtered_preview["Top_Score"].mean() if not filtered_preview.empty else None),
        ("Unique Personas", filtered_preview["Persona"].nunique() if not filtered_preview.empty and "Persona" in filtered_preview.columns else None),
        ("Top Reward", extract_reward_names(filtered_recommendations.iloc[0]["Top_3_Rewards"])[0] if not filtered_recommendations.empty else "N/A"),
    ]
    for col, (label, value) in zip(filtered_cards, filtered_metrics):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value" style="font-size:1.45rem;">{value if value is not None else 'N/A'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    tab_quality, tab_preview, tab_catalog = st.tabs(["Recommendation Quality", "Filtered Preview", "Reward Catalog"])

    with tab_quality:
        left, right = st.columns([1.0, 1.0], gap="large")
        with left:
            section_heading(
                "Persona Distribution",
                "Customer and persona mix for the filtered recommendation set.",
            )
            _chart_with_spinner("Loading persona distribution...", persona_distribution_figure(filtered_recommendations))
        with right:
            section_heading(
                "Score Distribution",
                "Distribution of top-ranked recommendation scores across the current filter.",
            )
            _chart_with_spinner("Loading recommendation scores...", recommendation_scores_figure(filtered_recommendations))

        section_heading("Top Reward Demand", "Rewards most frequently appearing across the top-3 recommendations.")
        _chart_with_spinner("Loading top reward demand...", top_rewards_figure(filtered_recommendations))

    with tab_preview:
        left, right = st.columns([1.1, 0.9], gap="large")
        with left:
            render_table_section(
                "Filtered Recommendation Preview",
                filtered_preview,
                subtitle="Filtered by persona and score threshold.",
                max_rows=preview_limit,
                height=420,
            )
            _download_csv_button(
                "Download filtered recommendations",
                filtered_preview,
                "filtered_recommendations.csv",
                key="recommendations-filtered-download",
            )
        with right:
            with st.expander("Recommendation summary", expanded=True):
                st.markdown(
                    """
                    <div class="section-card">
                        <div style="font-size:1.05rem;font-weight:800;color:var(--navy);margin-bottom:0.35rem;">Filters in effect</div>
                        <div class="subtle-note">The table and charts reflect the selected personas and score floor.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(f"Rows retained: {len(filtered_preview)}")
                st.caption(f"Selected personas: {', '.join(selected_personas) if selected_personas else 'All'}")
            st.caption(f"Minimum score: {score_floor:.1f}")
            if not filtered_preview.empty:
                top_reward_series = filtered_recommendations["Top_3_Rewards"].apply(extract_reward_names).explode().dropna()
                top_reward_rows = (
                    top_reward_series.value_counts().rename_axis("Reward").reset_index(name="Count")
                    if not top_reward_series.empty
                    else pd.DataFrame()
                )
                render_table_section(
                    "Top Reward Mentions",
                    top_reward_rows,
                    max_rows=10,
                    height=240,
                )

    with tab_catalog:
        section_heading("Reward Catalog", "The approved reward library used by the ranking engine.")
        left, right = st.columns([1.0, 1.0], gap="large")
        with left:
            _chart_with_spinner("Loading reward catalog...", reward_catalog_figure(reward_catalog))
        with right:
            render_table_section("Reward Catalog", reward_catalog, max_rows=20, height=360)
            _download_csv_button("Download reward catalog", reward_catalog, "reward_catalog.csv", key="recommendations-catalog-download")

    render_footer()


def render_new_customer_page() -> None:
    apply_theme()
    from dashboard.components.sidebar import render_sidebar

    render_sidebar("New Customer")
    _page_hero(
        "New Customer Prediction",
        "Cluster assignment, similarity scoring, and reward guidance for unseen customer profiles.",
    )
    new_predictions = load_new_customer_predictions()
    preview_df = flatten_new_customer_predictions(new_predictions)
    if not preview_df.empty:
        preview_df["similarity_score"] = pd.to_numeric(preview_df["similarity_score"], errors="coerce")
        preview_df["nearest_cluster_distance"] = pd.to_numeric(preview_df["nearest_cluster_distance"], errors="coerce")

    _render_standard_kpis()

    kpis = [
        {
            "label": "New Customers",
            "value": len(new_predictions),
            "detail": "Unseen customers ready for scoring and routing.",
        },
        {
            "label": "Avg Similarity",
            "value": preview_df["similarity_score"].mean() if not preview_df.empty else None,
            "detail": "Mean fit score across the loaded predictions.",
        },
        {
            "label": "Avg Distance",
            "value": preview_df["nearest_cluster_distance"].mean() if not preview_df.empty else None,
            "detail": "Average distance to the nearest cluster centroid.",
        },
        {
            "label": "Personas",
            "value": preview_df["persona"].nunique() if not preview_df.empty and "persona" in preview_df.columns else None,
            "detail": "Distinct personas represented in the prediction file.",
        },
    ]
    render_kpi_cards(kpis)

    cluster_options = (
        sorted(preview_df["predicted_cluster"].dropna().astype(int).unique().tolist())
        if not preview_df.empty and "predicted_cluster" in preview_df.columns
        else []
    )
    filter_left, filter_right, filter_third = st.columns([1.1, 0.9, 0.8], gap="large")
    with filter_left:
        selected_cluster = st.selectbox(
            "Focus cluster",
            options=["All"] + cluster_options,
            format_func=lambda value: "All clusters" if value == "All" else f"Cluster {value}",
        )
    with filter_right:
        min_similarity = float(preview_df["similarity_score"].min()) if not preview_df.empty else 0.0
        max_similarity = float(preview_df["similarity_score"].max()) if not preview_df.empty else 100.0
        similarity_floor = st.slider("Minimum similarity", 0.0, max(100.0, max_similarity), min(100.0, max(min_similarity, 0.0)), 0.5)
    with filter_third:
        preview_limit = st.slider("Preview rows", 5, 30, 15, 5)

    filtered_preview = preview_df.copy()
    if not filtered_preview.empty:
        if selected_cluster != "All" and "predicted_cluster" in filtered_preview.columns:
            filtered_preview = filtered_preview[filtered_preview["predicted_cluster"].astype(int) == int(selected_cluster)]
        filtered_preview = filtered_preview[filtered_preview["similarity_score"].fillna(0.0) >= float(similarity_floor)]
        filtered_preview = filtered_preview.sort_values("similarity_score", ascending=False).head(preview_limit)

    filtered_predictions = new_predictions.loc[filtered_preview.index] if not filtered_preview.empty else new_predictions.iloc[0:0].copy()

    summary_cards = st.columns(4)
    summary_values = [
        ("Filtered Rows", len(filtered_preview)),
        ("Best Fit", filtered_preview["similarity_score"].max() if not filtered_preview.empty else None),
        ("Cluster", selected_cluster if selected_cluster != "All" else "All"),
        ("Churn Risk", filtered_preview["confidence_level"].mode().iloc[0] if not filtered_preview.empty and "confidence_level" in filtered_preview.columns and not filtered_preview["confidence_level"].mode().empty else "N/A"),
    ]
    for col, (label, value) in zip(summary_cards, summary_values):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value" style="font-size:1.45rem;">{value if value is not None else 'N/A'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    section_heading("Prediction Confidence", "Average similarity score and cluster allocation for the filtered prediction set.")
    left, right = st.columns([1.0, 1.0], gap="large")
    with left:
        _chart_with_spinner("Loading similarity gauge...", similarity_figure(filtered_predictions if not filtered_predictions.empty else new_predictions))
    with right:
        if not filtered_preview.empty and "predicted_cluster" in filtered_preview.columns:
            cluster_counts = (
                filtered_preview["predicted_cluster"].astype(int).value_counts().sort_index().reset_index()
            )
            cluster_counts.columns = ["predicted_cluster", "count"]
            bar_fig = go.Figure(
                data=[
                    go.Bar(
                        x=[f"Cluster {int(value)}" for value in cluster_counts["predicted_cluster"]],
                        y=cluster_counts["count"],
                        marker=dict(color="#1D66D1", line=dict(color="white", width=1)),
                        hovertemplate="%{x}<br>%{y:,} customers<extra></extra>",
                    )
                ]
            )
            bar_fig.update_layout(
                template="plotly_white",
                height=330,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title=dict(text="Predicted Cluster Distribution", x=0.02, xanchor="left"),
                xaxis_title="Predicted Cluster",
                yaxis_title="Customers",
            )
            _chart_with_spinner("Loading cluster distribution...", bar_fig)
        else:
            st.info("No cluster distribution is available for the current filter.")

    with st.expander("Prediction Explorer", expanded=True):
        left, right = st.columns([1.1, 0.9], gap="large")
        with left:
            render_table_section(
                "New Customer Predictions",
                filtered_preview,
                subtitle="Loaded from new_customer_predictions.csv and filtered in the UI.",
                max_rows=preview_limit,
                height=420,
            )
            _download_csv_button(
                "Download filtered predictions",
                filtered_preview,
                "filtered_new_customer_predictions.csv",
                key="new-customer-download",
            )
        with right:
            if not filtered_preview.empty and "customer_id" in filtered_preview.columns:
                customer_options = filtered_preview["customer_id"].astype(str).tolist()
                selected_customer = st.selectbox("Inspect prediction", customer_options)
                row = filtered_predictions[filtered_predictions["customer_id"].astype(str) == str(selected_customer)].iloc[0]
                cols = st.columns(3)
                detail_values = [
                    ("Customer ID", row.get("customer_id", "N/A")),
                    ("Similarity", f"{float(row.get('similarity_score', 0.0)):.2f}"),
                    ("Distance", f"{float(row.get('nearest_cluster_distance', 0.0)):.4f}"),
                ]
                for col, (label, value) in zip(cols, detail_values):
                    with col:
                        st.markdown(
                            f"""
                            <div class="kpi-card">
                                <div class="kpi-label">{label}</div>
                                <div class="kpi-value" style="font-size:1.35rem;">{value}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                st.markdown(
                    f"""
                    <div class="section-card" style="margin-top:0.9rem;">
                        <div style="font-size:1.05rem;font-weight:800;color:var(--navy);margin-bottom:0.35rem;">Business Insight</div>
                        <div>{row.get('business_insight', '')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div class="section-card" style="margin-top:0.75rem;">
                        <div style="font-size:1.05rem;font-weight:800;color:var(--navy);margin-bottom:0.35rem;">Expected Impact</div>
                        <div>{row.get('expected_impact', '')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    render_footer()


def render_business_analytics_page() -> None:
    apply_theme()
    from dashboard.components.sidebar import render_sidebar

    render_sidebar("Business Analytics")
    _page_hero(
        "Business Analytics",
        "Validation outcomes, model quality, and catalog-level operating metrics for governance review.",
    )
    validation_df = summarize_validation_report()
    validation_report = load_validation_report()
    model_eval_df = summarize_model_evaluation()
    model_eval_payload = load_model_evaluation()
    reward_catalog = load_reward_catalog()
    llm_recommendations = load_llm_recommendations()
    scored_customers = load_scored_customers()
    ai_recommendations = load_ai_recommendations()
    _render_standard_kpis()

    kpi_cards = [
        {
            "label": "Validation Sheets",
            "value": len(validation_df),
            "detail": "Sheets reviewed in the latest validation run.",
        },
        {
            "label": "Evaluated Models",
            "value": len(model_eval_df),
            "detail": "Model families with current holdout metrics.",
        },
        {
            "label": "Reward Rows",
            "value": len(reward_catalog),
            "detail": "Approved catalog entries available to the engine.",
        },
        {
            "label": "Operational Artifacts",
            "value": len([
                item
                for item in [scored_customers, ai_recommendations, llm_recommendations]
                if not item.empty
            ]),
            "detail": "Production exports currently present in the workspace.",
        },
    ]
    render_kpi_cards(kpi_cards)

    governance_tab, operations_tab = st.tabs(["Governance View", "Operations View"])

    with governance_tab:
        left, right = st.columns([1.0, 1.0], gap="large")
        with left:
            section_heading("Validation Health", "Sheet-level issues surfaced by the latest validation run.")
            _chart_with_spinner("Loading validation health...", validation_issues_figure(validation_df))
            render_table_section("Validation Summary", validation_df, max_rows=10, height=280)
            _download_csv_button("Download validation summary", validation_df, "validation_summary.csv", key="analytics-validation-download")
            if validation_report.get("sheets"):
                st.markdown("### Validation Drilldown")
                sheet_name = st.selectbox("Select a sheet", list(validation_report["sheets"].keys()))
                sheet_payload = validation_report["sheets"].get(sheet_name, {})
                drilldown_cards = st.columns(3)
                drilldown_values = [
                    ("Rows", sheet_payload.get("total_rows", 0)),
                    ("Columns", sheet_payload.get("total_columns", 0)),
                    ("Issues", len(sheet_payload.get("invalid_records", []))),
                ]
                for col, (label, value) in zip(drilldown_cards, drilldown_values):
                    with col:
                        st.markdown(
                            f"""
                            <div class="kpi-card">
                                <div class="kpi-label">{label}</div>
                                <div class="kpi-value" style="font-size:1.45rem;">{value}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                render_table_section(
                    f"{sheet_name} Column Stats",
                    pd.DataFrame(sheet_payload.get("column_stats", [])),
                    max_rows=10,
                    height=250,
                )
        with right:
            section_heading("Model Performance", "Out-of-sample metrics from the supervised model evaluation.")
            model_options = list(model_eval_payload.keys())
            selected_model_key = st.selectbox(
                "Select model",
                options=model_options,
                format_func=lambda value: value.replace("_", " ").title(),
            ) if model_options else None
            selected_model_name = selected_model_key.replace("_", " ").title() if selected_model_key else None
            selected_row = (
                model_eval_df[model_eval_df["Model"] == selected_model_name].iloc[0]
                if selected_model_name and not model_eval_df.empty and not model_eval_df[model_eval_df["Model"] == selected_model_name].empty
                else None
            )
            if selected_row is not None:
                metric_cards = st.columns(4)
                metric_values = [
                    ("Accuracy", selected_row.get("Accuracy")),
                    ("Precision", selected_row.get("Precision")),
                    ("Recall", selected_row.get("Recall")),
                    ("F1", selected_row.get("F1")),
                ]
                for col, (label, value) in zip(metric_cards, metric_values):
                    with col:
                        st.markdown(
                            f"""
                            <div class="kpi-card">
                                <div class="kpi-label">{label}</div>
                                <div class="kpi-value" style="font-size:1.45rem;">{value if value is not None else 'N/A'}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            _chart_with_spinner("Loading model metrics...", model_metrics_figure(model_eval_df))
            if selected_model_key and isinstance(model_eval_payload.get(selected_model_key), dict):
                matrix = model_eval_payload[selected_model_key].get("confusion_matrix", [])
                matrix_labels = [f"Class {index}" for index in range(len(matrix))] if matrix else ["Class 0", "Class 1"]
                _chart_with_spinner(
                    "Loading confusion matrix...",
                    _confusion_matrix_figure(matrix, matrix_labels, f"{selected_model_key.replace('_', ' ').title()} Confusion Matrix"),
                )
            render_table_section("Model Metrics", model_eval_df, max_rows=10, height=280)
            _download_csv_button("Download model metrics", model_eval_df, "model_evaluation.csv", key="analytics-model-download")

    with operations_tab:
        lower_left, lower_right = st.columns([1.0, 1.0], gap="large")
        with lower_left:
            section_heading("Reward Library", "Category composition of the current reward catalog.")
            _chart_with_spinner("Loading reward catalog...", reward_catalog_figure(reward_catalog))
            render_table_section("Reward Catalog", reward_catalog, max_rows=15, height=300)
            _download_csv_button("Download reward catalog", reward_catalog, "reward_catalog.csv", key="analytics-reward-download")
        with lower_right:
            section_heading("Operational Outputs", "Recommendation outputs and supporting artifacts produced by the pipeline.")
            operational_rows = pd.DataFrame(
                [
                    {"Artifact": "scored_customers.csv", "Rows": len(scored_customers), "Columns": len(scored_customers.columns)},
                    {"Artifact": "AI_Recommendations.csv", "Rows": len(ai_recommendations), "Columns": len(ai_recommendations.columns)},
                    {"Artifact": "llm_recommendations.csv", "Rows": len(llm_recommendations), "Columns": len(llm_recommendations.columns)},
                ]
            )
            render_table_section("Artifact Inventory", operational_rows, max_rows=10, height=300)
            _download_csv_button("Download artifact inventory", operational_rows, "artifact_inventory.csv", key="analytics-artifact-download")
            with st.expander("Operational snapshot", expanded=True):
                st.caption("These counts reflect the currently loaded production exports in the workspace.")
                st.caption(f"Scored customers: {len(scored_customers):,}")
                st.caption(f"AI recommendations: {len(ai_recommendations):,}")
                st.caption(f"LLM recommendations: {len(llm_recommendations):,}")

    render_footer()
