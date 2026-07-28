from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.components.charts import cluster_comparison_figure
from dashboard.components.data import load_cluster_profiles, load_cluster_statistics, load_cluster_summary
from dashboard.components.metrics import render_kpi_cards
from dashboard.components.tables import render_table_section
from dashboard.components.theme import apply_theme, configure_page, render_footer, section_heading


PALETTE = [
    "#0B1F3A",
    "#1D66D1",
    "#2F86FF",
    "#57A0FF",
    "#8AB8FF",
    "#BFD4FF",
]


def _inject_page_styles() -> None:
    st.markdown(
        """
        <style>
            .seg-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
                gap: 0.9rem;
            }

            .seg-card {
                background: rgba(255, 255, 255, 0.92);
                border: 1px solid rgba(18, 56, 104, 0.12);
                border-radius: 18px;
                padding: 1rem 1rem 0.9rem;
                box-shadow: 0 10px 24px rgba(7, 26, 51, 0.07);
            }

            .seg-card-title {
                color: var(--navy);
                font-size: 1rem;
                font-weight: 800;
                margin-bottom: 0.2rem;
            }

            .seg-card-subtitle {
                color: var(--muted);
                font-size: 0.86rem;
                margin-bottom: 0.7rem;
            }

            .seg-chip {
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                padding: 0.35rem 0.65rem;
                border-radius: 999px;
                border: 1px solid rgba(29, 102, 209, 0.15);
                background: rgba(29, 102, 209, 0.08);
                color: var(--navy-2);
                font-size: 0.8rem;
                font-weight: 700;
                margin: 0 0.35rem 0.35rem 0;
            }

            .persona-summary {
                background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(240,246,255,0.96));
                border: 1px solid rgba(28, 89, 169, 0.12);
                border-radius: 20px;
                padding: 1rem 1rem 0.95rem;
                box-shadow: 0 10px 24px rgba(7, 26, 51, 0.07);
                height: 100%;
            }

            .persona-summary h4 {
                margin: 0 0 0.35rem;
                color: var(--navy);
                font-size: 1.05rem;
            }

            .persona-summary p {
                margin: 0;
                color: var(--text);
                line-height: 1.55;
            }

            .cluster-note {
                color: var(--muted);
                font-size: 0.9rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _cluster_summary_frame() -> pd.DataFrame:
    profiles = load_cluster_profiles()
    stats = load_cluster_statistics()

    if profiles.empty and stats.empty:
        return pd.DataFrame()

    if not profiles.empty and not stats.empty:
        merged = profiles.merge(stats, on="cluster_id", how="outer", suffixes=("_profile", "_stats"))
        if "persona" not in merged.columns and "persona_profile" in merged.columns:
            merged["persona"] = merged["persona_profile"].fillna(merged.get("persona_stats"))
        if "persona_explanation" not in merged.columns and "persona_explanation_profile" in merged.columns:
            merged["persona_explanation"] = merged["persona_explanation_profile"].fillna(
                merged.get("persona_explanation_stats")
            )
        if "cluster_size" not in merged.columns and "cluster_size_profile" in merged.columns:
            merged["cluster_size"] = pd.to_numeric(merged["cluster_size_profile"], errors="coerce").fillna(
                pd.to_numeric(merged.get("cluster_size_stats"), errors="coerce")
            )
        return merged.sort_values("cluster_id").reset_index(drop=True)

    if not profiles.empty:
        return profiles.sort_values("cluster_id").reset_index(drop=True)

    return stats.sort_values("cluster_id").reset_index(drop=True)


def _distribution_source(summary: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    if not frame.empty and {"cluster_id", "persona", "cluster_size"}.issubset(frame.columns):
        return frame[["cluster_id", "persona", "cluster_size"]].drop_duplicates("cluster_id").sort_values("cluster_id")

    cluster_sizes = summary.get("cluster_sizes", {}) if isinstance(summary, dict) else {}
    personas = summary.get("personas", {}) if isinstance(summary, dict) else {}
    rows = [
        {
            "cluster_id": int(cluster_id),
            "persona": personas.get(str(cluster_id), f"Cluster {cluster_id}"),
            "cluster_size": int(size),
        }
        for cluster_id, size in cluster_sizes.items()
    ]
    return pd.DataFrame(rows).sort_values("cluster_id") if rows else pd.DataFrame()


def _numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number]).copy()
    if "cluster_id" in numeric.columns:
        numeric = numeric.drop(columns=["cluster_id"])
    return numeric


def _pca_projection(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    numeric = _numeric_features(frame)
    if numeric.empty:
        return pd.DataFrame()

    numeric = numeric.fillna(0.0)
    centered = numeric - numeric.mean(axis=0)
    scaled = centered / centered.std(axis=0, ddof=0).replace(0, 1)
    values = scaled.to_numpy(dtype=float)

    if values.shape[1] == 1:
        coords = np.column_stack([values[:, 0], np.zeros(values.shape[0])])
    else:
        _, _, vh = np.linalg.svd(values, full_matrices=False)
        coords = values @ vh[:2].T
        if coords.shape[1] == 1:
            coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0])])

    projected = frame[["cluster_id"]].copy()
    projected["pc1"] = coords[:, 0]
    projected["pc2"] = coords[:, 1] if coords.shape[1] > 1 else 0.0
    if "persona" in frame.columns:
        projected["persona"] = frame["persona"].astype(str)
    if "cluster_size" in frame.columns:
        projected["cluster_size"] = pd.to_numeric(frame["cluster_size"], errors="coerce").fillna(0).astype(float)
    return projected


def _animated_bar_figure(data: pd.DataFrame, selected_cluster: int | None) -> go.Figure:
    if data.empty:
        return go.Figure()

    labels = [f"Cluster {int(value)}" for value in data["cluster_id"]]
    values = data["cluster_size"].astype(float).tolist()
    colors = [
        "#1D66D1" if int(cluster_id) != selected_cluster else "#0B1F3A"
        for cluster_id in data["cluster_id"]
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=[0 for _ in values],
                marker=dict(color=colors, line=dict(color="white", width=1)),
                hovertemplate="%{x}<br>%{y:,} customers<extra></extra>",
            )
        ]
    )

    frames = []
    for index in range(len(values)):
        reveal = [values[i] if i <= index else 0 for i in range(len(values))]
        frames.append(
            go.Frame(
                data=[go.Bar(x=labels, y=reveal, marker=dict(color=colors, line=dict(color="white", width=1)))],
                name=str(index),
            )
        )

    fig.frames = frames
    fig.update_layout(
        template="plotly_white",
        height=390,
        margin=dict(l=20, r=20, t=55, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, Inter, Arial, sans-serif", color="#0b1f3d"),
        title=dict(text="Cluster Distribution", x=0.02, xanchor="left"),
        xaxis_title="Cluster",
        yaxis_title="Customers",
        transition=dict(duration=700, easing="cubic-in-out"),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.01,
                y=1.18,
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=550, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=500, easing="cubic-in-out"),
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))],
                    ),
                ],
            )
        ],
    )
    return fig


def _animated_pca_figure(projected: pd.DataFrame, selected_cluster: int | None) -> go.Figure:
    if projected.empty:
        return go.Figure()

    frames = []
    ordered_clusters = projected.sort_values("cluster_id")["cluster_id"].astype(int).tolist()
    for cluster_id in ordered_clusters:
        focus = projected["cluster_id"].astype(int) == cluster_id
        colors = ["#0B1F3A" if value else "rgba(91, 110, 136, 0.35)" for value in focus]
        sizes = [20 if int(value) == cluster_id else 11 for value in projected["cluster_id"]]
        if selected_cluster is not None:
            sizes = [24 if int(value) == selected_cluster else (18 if int(value) == cluster_id else 11) for value in projected["cluster_id"]]
            colors = [
                "#0B1F3A" if int(value) == selected_cluster else ("#1D66D1" if int(value) == cluster_id else "rgba(91, 110, 136, 0.32)")
                for value in projected["cluster_id"]
            ]
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=projected["pc1"],
                        y=projected["pc2"],
                        mode="markers+text",
                        text=[f"Cluster {int(value)}" if int(value) == cluster_id else "" for value in projected["cluster_id"]],
                        textposition="top center",
                        marker=dict(
                            size=sizes,
                            color=colors,
                            line=dict(color="white", width=1.5),
                        ),
                    )
                ],
                name=str(cluster_id),
            )
        )

    base_sizes = [24 if selected_cluster is not None and int(value) == selected_cluster else 12 for value in projected["cluster_id"]]
    base_colors = [
        "#0B1F3A" if selected_cluster is not None and int(value) == selected_cluster else "#1D66D1"
        for value in projected["cluster_id"]
    ]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=projected["pc1"],
                y=projected["pc2"],
                mode="markers+text",
                text=[f"Cluster {int(value)}" for value in projected["cluster_id"]],
                textposition="top center",
                hovertemplate=(
                    "Cluster %{customdata[0]}<br>"
                    "Persona %{customdata[1]}<br>"
                    "PC1 %{x:.2f}<br>"
                    "PC2 %{y:.2f}<extra></extra>"
                ),
                customdata=np.stack(
                    [
                        projected["cluster_id"].astype(int),
                        projected.get("persona", pd.Series(["N/A"] * len(projected))).astype(str),
                    ],
                    axis=-1,
                ),
                marker=dict(
                    size=base_sizes,
                    color=base_colors,
                    line=dict(color="white", width=1.5),
                ),
            )
        ],
        frames=frames,
    )
    fig.update_layout(
        template="plotly_white",
        height=430,
        margin=dict(l=20, r=20, t=55, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, Inter, Arial, sans-serif", color="#0b1f3d"),
        title=dict(text="PCA Scatter Plot", x=0.02, xanchor="left"),
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        transition=dict(duration=700, easing="cubic-in-out"),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.01,
                y=1.16,
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=650, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=550, easing="cubic-in-out"),
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))],
                    ),
                ],
            )
        ],
    )
    return fig


def _render_cluster_detail_cards(row: pd.Series) -> None:
    metric_rows = [
        {
            "label": "Average Spend",
            "value": row.get("average_spend"),
            "detail": "Cluster-level total spend across the latest outputs.",
        },
        {
            "label": "Frequency",
            "value": row.get("purchase_frequency"),
            "detail": "Average purchases per period for the cluster.",
        },
        {
            "label": "Reward Utilization",
            "value": row.get("reward_utilization_pct"),
            "suffix": "%",
            "detail": "Observed reward usage across the cluster.",
        },
        {
            "label": "Engagement",
            "value": row.get("customer_engagement_score"),
            "detail": "Composite engagement score from the current pipeline.",
        },
    ]
    render_kpi_cards(metric_rows)

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            f"""
            <div class="persona-summary">
                <h4>Business Summary</h4>
                <p>{row.get('business_summary', 'N/A')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"""
            <div class="persona-summary">
                <h4>Persona Context</h4>
                <p>{row.get('persona_explanation', 'N/A')}</p>
                <div style="margin-top:0.75rem;">
                    <span class="seg-chip">Favorite Category: {row.get('favorite_category', 'N/A')}</span>
                    <span class="seg-chip">Favorite Brand: {row.get('favorite_brand', 'N/A')}</span>
                    <span class="seg-chip">Churn Risk: {row.get('churn_risk_dominant', 'N/A')}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_persona_cards(frame: pd.DataFrame, selected_cluster: int) -> None:
    st.markdown(
        "<div class='seg-grid'>"
        + "".join(
            f"""
            <div class="seg-card">
                <div class="seg-card-title">Cluster {int(row['cluster_id'])} - {row.get('persona', 'N/A')}</div>
                <div class="seg-card-subtitle">{row.get('cluster_size', 0):,} customers</div>
                <div class="cluster-note">{row.get('persona_explanation', 'N/A')}</div>
                <div style="margin-top:0.7rem;">
                    <span class="seg-chip">Spend {row.get('average_spend', 'N/A')}</span>
                    <span class="seg-chip">Engagement {row.get('customer_engagement_score', 'N/A')}</span>
                    <span class="seg-chip">Risk {row.get('churn_risk_dominant', 'N/A')}</span>
                </div>
            </div>
            """
            for _, row in frame.iterrows()
        )
        + "</div>",
        unsafe_allow_html=True,
    )


def _cluster_profiles_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "cluster_id",
        "persona",
        "cluster_size",
        "percentage_of_customers",
        "average_spend",
        "purchase_frequency",
        "average_order_value",
        "favorite_category",
        "favorite_brand",
        "reward_utilization_pct",
        "customer_engagement_score",
        "churn_risk_dominant",
        "business_summary",
    ]
    existing = [column for column in columns if column in frame.columns]
    return frame[existing].copy() if existing else pd.DataFrame()


def _cluster_statistics_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "cluster_id",
        "cluster_size",
        "recency_mean",
        "frequency_mean",
        "monetary_mean",
        "reward_utilization_pct_mean",
        "customer_engagement_score_mean",
        "online_purchase_ratio_mean",
        "category_diversity_mean",
        "average_days_between_purchases_mean",
        "customer_ltv_mean",
        "persona",
    ]
    existing = [column for column in columns if column in frame.columns]
    return frame[existing].copy() if existing else pd.DataFrame()


def render_customer_segmentation_page() -> None:
    apply_theme()
    _inject_page_styles()

    summary = load_cluster_summary()
    frame = _cluster_summary_frame()
    distribution = _distribution_source(summary, frame)
    projected = _pca_projection(frame)

    from dashboard.components.sidebar import render_sidebar

    render_sidebar("Customer Segmentation")

    st.markdown(
        """
        <div class="hero">
            <h1>Customer Segmentation</h1>
            <p>Cluster distribution, centroids, persona summaries, and operational statistics for the loyalty engine.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_customers = int(sum(summary.get("cluster_sizes", {}).values())) if isinstance(summary, dict) and summary.get("cluster_sizes") else (int(frame["cluster_size"].sum()) if not frame.empty and "cluster_size" in frame.columns else None)
    kpis = [
        {
            "label": "Clusters",
            "value": summary.get("optimal_k") if isinstance(summary, dict) else None,
            "detail": "Active cluster count from the latest segmentation run.",
        },
        {
            "label": "Customers",
            "value": total_customers,
            "detail": "Population represented by the cluster artifacts.",
        },
        {
            "label": "Top Persona",
            "value": frame["persona"].iloc[0] if not frame.empty and "persona" in frame.columns else None,
            "detail": "Leading business persona from the current output set.",
        },
        {
            "label": "Avg Engagement",
            "value": float(frame["customer_engagement_score"].mean()) if not frame.empty and "customer_engagement_score" in frame.columns else None,
            "detail": "Average cluster engagement across personas.",
        },
    ]
    render_kpi_cards(kpis)

    if frame.empty or distribution.empty:
        st.info("No cluster artifacts were found for the segmentation page yet.")
        render_footer()
        return

    cluster_options = distribution["cluster_id"].astype(int).tolist()
    persona_lookup = distribution.set_index("cluster_id")["persona"].to_dict()
    selected_cluster = st.selectbox(
        "Select a cluster to inspect",
        options=cluster_options,
        format_func=lambda cluster_id: f"Cluster {cluster_id} - {persona_lookup.get(cluster_id, 'N/A')}",
    )

    selected_row = frame[frame["cluster_id"].astype(int) == int(selected_cluster)]
    selected_row = selected_row.iloc[0] if not selected_row.empty else frame.iloc[0]

    section_heading("Cluster Distribution", "Interactive pies and animated bars showing how the customer base is split across personas.")
    left, right = st.columns(2, gap="large")
    with left:
        pie_colors = [PALETTE[index % len(PALETTE)] for index in range(len(distribution))]
        pulls = [0.12 if int(cluster_id) == int(selected_cluster) else 0.0 for cluster_id in distribution["cluster_id"]]
        pie_fig = go.Figure(
            data=[
                go.Pie(
                    labels=[f"Cluster {int(value)}" for value in distribution["cluster_id"]],
                    values=distribution["cluster_size"].astype(float),
                    hole=0.45,
                    sort=False,
                    marker=dict(colors=pie_colors, line=dict(color="white", width=2)),
                    pull=pulls,
                    textinfo="label+percent",
                    hovertemplate="%{label}<br>%{value:,} customers<br>%{percent}<extra></extra>",
                )
            ]
        )
        pie_fig.update_layout(
            template="plotly_white",
            height=390,
            margin=dict(l=20, r=20, t=55, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Segoe UI, Inter, Arial, sans-serif", color="#0b1f3d"),
            title=dict(text="Interactive Pie Chart", x=0.02, xanchor="left"),
            transition=dict(duration=650, easing="cubic-in-out"),
        )
        st.plotly_chart(pie_fig, width="stretch")
    with right:
        st.plotly_chart(_animated_bar_figure(distribution, int(selected_cluster)), width="stretch")

    section_heading("PCA Scatter Plot", "A compact view of the cluster centroids projected into two principal components.")
    st.plotly_chart(_animated_pca_figure(projected, int(selected_cluster)), width="stretch")

    section_heading("Cluster Centroids", "The radar comparison below shows centroid behavior across the active feature set.")
    st.plotly_chart(cluster_comparison_figure(load_cluster_statistics(), int(selected_cluster)), width="stretch")

    st.markdown(
        f"""
        <div class="seg-card" style="margin-top:0.25rem;">
            <div class="seg-card-title">Selected Cluster Deep Dive</div>
            <div class="seg-card-subtitle">Cluster {int(selected_row.get('cluster_id', selected_cluster))} - {selected_row.get('persona', 'N/A')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_cluster_detail_cards(selected_row)

    with st.expander("Business Personas", expanded=True):
        _render_persona_cards(frame, int(selected_cluster))

    section_heading("Cluster Profiles Table", "Business-facing profile details for each persona in the latest run.")
    render_table_section(
        "Cluster Profiles",
        _cluster_profiles_table(frame),
        max_rows=20,
        height=360,
    )

    section_heading("Cluster Statistics", "Centroid-level statistics used by the segmentation model and dashboard charts.")
    render_table_section(
        "Cluster Statistics",
        _cluster_statistics_table(load_cluster_statistics()),
        max_rows=20,
        height=360,
    )

    render_footer()


configure_page("Customer Segmentation | Hyper-Personalized Loyalty Engine")
render_customer_segmentation_page()
