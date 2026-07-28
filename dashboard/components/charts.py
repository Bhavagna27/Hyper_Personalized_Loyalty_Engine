from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .data import extract_reward_names, extract_scores


BLUE_PALETTE = [
    "#0B1F3A",
    "#1D66D1",
    "#2F86FF",
    "#57A0FF",
    "#8AB8FF",
    "#BFD4FF",
]


def _blank_figure(message: str = "No data available", height: int = 360) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16, color="#5b6e88"),
    )
    fig.update_layout(
        height=height,
        template="plotly_white",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def _layout(fig: go.Figure, title: str | None = None, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, Inter, Arial, sans-serif", color="#0b1f3d"),
        title=dict(text=title, x=0.02, xanchor="left") if title else None,
    )
    return fig


def architecture_diagram() -> go.Figure:
    labels = [
        "Workbook",
        "Ingestion",
        "Validation",
        "Cleaned Data",
        "Feature Engineering",
        "Segmentation",
        "Recommendation Engine",
        "Dashboard Outputs",
    ]
    source = [0, 1, 1, 3, 4, 5, 6]
    target = [1, 2, 3, 4, 5, 6, 7]
    values = [1, 1, 1, 1, 1, 1, 1]
    colors = [
        "rgba(29,102,209,0.35)",
        "rgba(47,134,255,0.35)",
        "rgba(87,160,255,0.35)",
        "rgba(146,187,255,0.35)",
        "rgba(191,212,255,0.35)",
        "rgba(11,31,58,0.35)",
        "rgba(22,107,206,0.35)",
    ]
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=22,
                    thickness=18,
                    line=dict(color="rgba(11,31,58,0.14)", width=1),
                    label=labels,
                    color=[
                        "#0B1F3A",
                        "#1D66D1",
                        "#2F86FF",
                        "#57A0FF",
                        "#6CA8FF",
                        "#88BAFF",
                        "#A2C8FF",
                        "#C4D8FF",
                    ],
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=values,
                    color=colors,
                ),
            )
        ]
    )
    return _layout(fig, "Project Architecture", 430)


def pipeline_flow_figure() -> go.Figure:
    stages = [
        "Transactions",
        "Customer Profiles",
        "Engineered Features",
        "Clusters",
        "Recommendations",
        "New Customers",
    ]
    values = [1410, 200, 200, 3, 200, 1]
    fig = go.Figure(
        data=[
            go.Scatter(
                x=stages,
                y=values,
                mode="lines+markers+text",
                text=[f"{value:,}" for value in values],
                textposition="top center",
                line=dict(color="#1D66D1", width=4),
                marker=dict(size=13, color="#0B1F3A", line=dict(color="white", width=2)),
                hovertemplate="%{x}<br>%{y:,} records<extra></extra>",
            )
        ]
    )
    fig.update_yaxes(title_text="Volume / Count", gridcolor="rgba(12,31,58,0.08)")
    fig.update_xaxes(title_text="")
    return _layout(fig, "Pipeline Flow", 360)


def dataset_summary_figure(summary_df: pd.DataFrame) -> go.Figure:
    if summary_df.empty:
        return _blank_figure()
    fig = go.Figure(
        data=[
            go.Bar(
                x=summary_df["Sheet"],
                y=summary_df["Rows"],
                marker_color=["#1D66D1", "#2F86FF", "#8AB8FF"],
                hovertemplate="%{x}<br>%{y:,} rows<extra></extra>",
            )
        ]
    )
    fig.update_yaxes(title_text="Rows")
    return _layout(fig, "Dataset Rows by Sheet", 360)


def feature_summary_figure(feature_metadata: dict[str, Any]) -> go.Figure:
    feature_groups = feature_metadata.get("feature_groups", {}) if isinstance(feature_metadata, dict) else {}
    if not feature_groups:
        return _blank_figure()
    labels = []
    values = []
    for group_name, features in feature_groups.items():
        labels.append(group_name.replace("_", " ").title())
        values.append(len(features))
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=["#0B1F3A", "#1D66D1", "#2F86FF", "#57A0FF", "#88BAFF", "#BFD4FF"],
                hovertemplate="%{x}<br>%{y} features<extra></extra>",
            )
        ]
    )
    fig.update_yaxes(title_text="Feature Count")
    fig.update_xaxes(tickangle=-15)
    return _layout(fig, "Feature Groups", 360)


def cluster_sizes_figure(cluster_summary: dict[str, Any]) -> go.Figure:
    cluster_sizes = cluster_summary.get("cluster_sizes", {}) if isinstance(cluster_summary, dict) else {}
    if not cluster_sizes:
        return _blank_figure()
    items = sorted(cluster_sizes.items(), key=lambda item: int(item[0]))
    clusters = [f"Cluster {cluster}" for cluster, _ in items]
    values = [value for _, value in items]
    colors = ["#0B1F3A", "#1D66D1", "#2F86FF", "#57A0FF", "#8AB8FF"]
    fig = go.Figure(
        data=[
            go.Bar(
                x=clusters,
                y=values,
                marker_color=colors[: len(values)],
                hovertemplate="%{x}<br>%{y:,} customers<extra></extra>",
            )
        ]
    )
    fig.update_yaxes(title_text="Customers")
    return _layout(fig, "Cluster Size Distribution", 360)


def cluster_comparison_figure(cluster_statistics: pd.DataFrame, selected_cluster: int | None = None) -> go.Figure:
    if cluster_statistics.empty:
        return _blank_figure()
    numeric_columns = [
        "recency_mean",
        "frequency_mean",
        "monetary_mean",
        "reward_utilization_pct_mean",
        "customer_engagement_score_mean",
        "online_purchase_ratio_mean",
        "category_diversity_mean",
        "average_days_between_purchases_mean",
    ]
    available = [column for column in numeric_columns if column in cluster_statistics.columns]
    if not available:
        return _blank_figure()
    normalized = cluster_statistics[["cluster_id"] + available].copy()
    for column in available:
        max_value = float(normalized[column].max()) or 1.0
        normalized[column] = normalized[column] / max_value * 100.0
    fig = go.Figure()
    theta_labels = [column.replace("_mean", "").replace("_", " ").title() for column in available]
    for _, row in normalized.iterrows():
        cluster_id = int(row["cluster_id"])
        fig.add_trace(
            go.Scatterpolar(
                r=[float(row[column]) for column in available],
                theta=theta_labels,
                fill="toself" if selected_cluster is None or cluster_id == selected_cluster else "none",
                name=f"Cluster {cluster_id}",
                line=dict(width=2),
            )
        )
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True)
    return _layout(fig, "Cluster Feature Profile", 420)


def persona_distribution_figure(recommendations: pd.DataFrame) -> go.Figure:
    if recommendations.empty or "Persona" not in recommendations.columns:
        return _blank_figure()
    counts = recommendations["Persona"].value_counts().sort_values(ascending=False)
    fig = go.Figure(
        data=[
            go.Bar(
                x=counts.index.tolist(),
                y=counts.values.tolist(),
                marker_color=["#0B1F3A", "#1D66D1", "#2F86FF", "#57A0FF", "#88BAFF"][: len(counts)],
                hovertemplate="%{x}<br>%{y:,} customers<extra></extra>",
            )
        ]
    )
    fig.update_yaxes(title_text="Customers")
    fig.update_xaxes(tickangle=-15)
    return _layout(fig, "Persona Distribution", 360)


def top_rewards_figure(recommendations: pd.DataFrame) -> go.Figure:
    if recommendations.empty or "Top_3_Rewards" not in recommendations.columns:
        return _blank_figure()
    reward_names: list[str] = []
    for value in recommendations["Top_3_Rewards"]:
        reward_names.extend(extract_reward_names(value)[:3])
    if not reward_names:
        return _blank_figure()
    counts = pd.Series(reward_names).value_counts().sort_values(ascending=False).head(10)
    fig = go.Figure(
        data=[
            go.Bar(
                x=counts.values[::-1],
                y=counts.index[::-1],
                orientation="h",
                marker_color="#1D66D1",
                hovertemplate="%{y}<br>%{x:,} mentions<extra></extra>",
            )
        ]
    )
    fig.update_xaxes(title_text="Mentions")
    return _layout(fig, "Top Reward Mentions", 420)


def recommendation_scores_figure(recommendations: pd.DataFrame) -> go.Figure:
    if recommendations.empty or "Scores" not in recommendations.columns:
        return _blank_figure()
    top_scores: list[float] = []
    for value in recommendations["Scores"]:
        scores = extract_scores(value)
        if scores:
            top_scores.append(scores[0])
    if not top_scores:
        return _blank_figure()
    fig = go.Figure(
        data=[
            go.Histogram(
                x=top_scores,
                nbinsx=18,
                marker_color="#2F86FF",
                opacity=0.9,
            )
        ]
    )
    fig.update_xaxes(title_text="Top Recommendation Score")
    fig.update_yaxes(title_text="Customers")
    return _layout(fig, "Recommendation Score Distribution", 360)


def similarity_figure(predictions: pd.DataFrame) -> go.Figure:
    if predictions.empty or "similarity_score" not in predictions.columns:
        return _blank_figure()
    values = pd.to_numeric(predictions["similarity_score"], errors="coerce").dropna().tolist()
    if not values:
        return _blank_figure()
    fig = go.Figure(
        data=[
            go.Indicator(
                mode="gauge+number",
                value=float(np.mean(values)),
                number={"suffix": "", "font": {"size": 44}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1D66D1"},
                    "steps": [
                        {"range": [0, 40], "color": "#eff4fb"},
                        {"range": [40, 70], "color": "#cfe0fb"},
                        {"range": [70, 100], "color": "#8ab8ff"},
                    ],
                },
                title={"text": "Average Similarity Score"},
            )
        ]
    )
    return _layout(fig, height=330)


def model_metrics_figure(model_eval_df: pd.DataFrame) -> go.Figure:
    if model_eval_df.empty:
        return _blank_figure()
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    fig = go.Figure()
    for index, row in model_eval_df.iterrows():
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=[row.get(metric.lower(), None) for metric in metrics],
                name=str(row.get("Model", f"Model {index + 1}")),
            )
        )
    fig.update_yaxes(range=[0, 1.05], title_text="Score")
    fig.update_layout(barmode="group")
    return _layout(fig, "Model Performance", 360)


def validation_issues_figure(validation_df: pd.DataFrame) -> go.Figure:
    if validation_df.empty:
        return _blank_figure()
    fig = go.Figure(
        data=[
            go.Bar(
                x=validation_df["Sheet"],
                y=validation_df["Issues"],
                marker_color=["#0B1F3A", "#1D66D1", "#2F86FF"],
                hovertemplate="%{x}<br>%{y} issue(s)<extra></extra>",
            )
        ]
    )
    fig.update_yaxes(title_text="Issues")
    return _layout(fig, "Validation Issues by Sheet", 360)


def reward_catalog_figure(reward_catalog: pd.DataFrame) -> go.Figure:
    if reward_catalog.empty:
        return _blank_figure()
    if "category" not in reward_catalog.columns:
        return _blank_figure()
    counts = reward_catalog["category"].value_counts().sort_values(ascending=False)
    fig = go.Figure(
        data=[
            go.Bar(
                x=counts.index.tolist(),
                y=counts.values.tolist(),
                marker_color="#1D66D1",
            )
        ]
    )
    fig.update_yaxes(title_text="Rewards")
    fig.update_xaxes(title_text="", tickangle=-15)
    return _layout(fig, "Reward Catalog by Category", 360)

