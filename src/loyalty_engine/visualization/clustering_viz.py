"""Clustering Visualizations module for Customer Segmentation.

This module generates and saves the 5 required visualization plots:
1. `elbow_curve.png` & `silhouette_scores.png` (Optimal K evaluation)
2. `pca_2d_clusters.png` (2D PCA projection of customer clusters)
3. `pca_explained_variance.png` (Scree plot of PCA components)
4. `cluster_centroids.png` (Heatmap comparing normalized cluster centroids)
5. `pairwise_cluster_comparison.png` (Pairwise boxplots/bar charts across key features)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from loyalty_engine.io.persistence import ensure_dir

logger = logging.getLogger(__name__)


# Set publication-quality plot aesthetics
sns.set_theme(style="whitegrid", palette="tab10")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


# ---------------------------------------------------------------------------
# 1. Optimal K Evaluation Plots (Elbow & Silhouette)
# ---------------------------------------------------------------------------

def plot_elbow_and_silhouette(
    k_range: list[int],
    inertias: dict[int, float],
    silhouette_scores: dict[int, float],
    optimal_k: int,
    output_dir: Path,
) -> None:
    """Generate and save Elbow Curve and Silhouette Score evaluation plots."""
    ensure_dir(output_dir)

    ks = list(k_range)
    inertia_vals = [inertias[k] for k in ks]
    sil_vals = [silhouette_scores[k] for k in ks]

    # Plot 1: Elbow Curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, inertia_vals, "o-", color="#1f77b4", linewidth=2, markersize=8)
    ax.axvline(x=optimal_k, color="#d62728", linestyle="--", label=f"Optimal K = {optimal_k}")
    ax.set_title("Elbow Method — Inertia vs. Number of Clusters (K)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Clusters (K)", fontsize=11)
    ax.set_ylabel("Inertia (Sum of Squared Distances)", fontsize=11)
    ax.set_xticks(ks)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "elbow_curve.png", dpi=200)
    plt.close(fig)

    # Plot 2: Silhouette Scores
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, sil_vals, "s-", color="#2ca02c", linewidth=2, markersize=8)
    ax.axvline(x=optimal_k, color="#d62728", linestyle="--", label=f"Optimal K = {optimal_k}")
    ax.set_title("Silhouette Score vs. Number of Clusters (K)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Clusters (K)", fontsize=11)
    ax.set_ylabel("Silhouette Score", fontsize=11)
    ax.set_xticks(ks)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "silhouette_scores.png", dpi=200)
    plt.close(fig)

    logger.info("Saved elbow_curve.png and silhouette_scores.png to %s", output_dir)


# ---------------------------------------------------------------------------
# 2. PCA 2D Cluster Scatter Plot
# ---------------------------------------------------------------------------

def plot_pca_2d_clusters(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    centroids_scaled: np.ndarray,
    personas: dict[int, str],
    output_dir: Path,
) -> None:
    """Generate and save 2D PCA projection scatter plot of customer clusters."""
    ensure_dir(output_dir)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    centroids_pca = pca.transform(centroids_scaled)

    var1, var2 = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(10, 7))

    unique_labels = np.unique(labels)
    colors = sns.color_palette("tab10", n_colors=len(unique_labels))

    for idx, cid in enumerate(unique_labels):
        mask = labels == cid
        persona = personas.get(cid, f"Cluster {cid}")
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=[colors[idx]],
            label=f"Cluster {cid}: {persona}",
            alpha=0.7,
            edgecolor="none",
            s=50,
        )

    # Plot centroids
    ax.scatter(
        centroids_pca[:, 0],
        centroids_pca[:, 1],
        c="black",
        marker="X",
        s=200,
        linewidths=2,
        edgecolor="white",
        zorder=10,
        label="Centroids",
    )

    ax.set_title(
        f"2D PCA Projection of Customer Clusters (Total Variance Explained: {var1+var2:.1f}%)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(f"Principal Component 1 ({var1:.1f}% Variance)", fontsize=11)
    ax.set_ylabel(f"Principal Component 2 ({var2:.1f}% Variance)", fontsize=11)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_2d_clusters.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved pca_2d_clusters.png to %s", output_dir)


# ---------------------------------------------------------------------------
# 3. PCA Explained Variance Plot
# ---------------------------------------------------------------------------

def plot_pca_explained_variance(
    X_scaled: np.ndarray,
    output_dir: Path,
) -> None:
    """Generate and save PCA Scree plot showing individual & cumulative explained variance."""
    ensure_dir(output_dir)

    n_features = X_scaled.shape[1]
    pca = PCA(n_components=n_features, random_state=42)
    pca.fit(X_scaled)

    exp_var = pca.explained_variance_ratio_ * 100
    cum_var = np.cumsum(exp_var)

    fig, ax1 = plt.subplots(figsize=(9, 5))

    components = [f"PC{i+1}" for i in range(n_features)]
    bars = ax1.bar(components, exp_var, color="#4c72b0", alpha=0.8, label="Individual Variance (%)")
    ax1.set_xlabel("Principal Components", fontsize=11)
    ax1.set_ylabel("Individual Variance Explained (%)", fontsize=11, color="#4c72b0")
    ax1.tick_params(axis="y", labelcolor="#4c72b0")

    ax2 = ax1.twinx()
    ax2.plot(components, cum_var, "r-o", linewidth=2, label="Cumulative Variance (%)")
    ax2.set_ylabel("Cumulative Variance Explained (%)", fontsize=11, color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(0, 105)

    ax1.set_title("PCA Explained Variance Scree Plot", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_explained_variance.png", dpi=200)
    plt.close(fig)

    logger.info("Saved pca_explained_variance.png to %s", output_dir)


# ---------------------------------------------------------------------------
# 4. Cluster Centroids Heatmap
# ---------------------------------------------------------------------------

def plot_cluster_centroids_heatmap(
    centroids_scaled: np.ndarray,
    feature_names: list[str],
    personas: dict[int, str],
    output_dir: Path,
) -> None:
    """Generate and save a heatmap of normalized cluster centroids across features."""
    ensure_dir(output_dir)

    cluster_ids = list(range(len(centroids_scaled)))
    yticklabels = [f"C{cid}: {personas.get(cid, '')}" for cid in cluster_ids]

    df_centroids = pd.DataFrame(centroids_scaled, index=yticklabels, columns=feature_names)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        df_centroids,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        cbar_kws={"label": "Standardized Feature Value (Z-Score)"},
        ax=ax,
        linewidths=0.5,
    )

    ax.set_title("Normalized Cluster Centroids (Z-Scores across Features)", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "cluster_centroids.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved cluster_centroids.png to %s", output_dir)


# ---------------------------------------------------------------------------
# 5. Pairwise Cluster Comparison Plot
# ---------------------------------------------------------------------------

def plot_pairwise_cluster_comparison(
    customer_features: pd.DataFrame,
    personas: dict[int, str],
    output_dir: Path,
) -> None:
    """Generate and save key pairwise feature comparison boxplots across clusters."""
    ensure_dir(output_dir)

    df = customer_features.copy()
    df["persona_label"] = df["cluster_id"].map(lambda cid: f"C{cid}: {personas.get(cid, '')}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    key_features = [
        ("monetary", "Total Monetary Spend ($)"),
        ("customer_engagement_score", "Customer Engagement Score (0-100)"),
        ("reward_utilization_pct", "Reward Utilization (%)"),
        ("online_purchase_ratio", "Online Purchase Ratio"),
    ]

    for idx, (feat, title) in enumerate(key_features):
        ax = axes[idx // 2, idx % 2]
        sns.boxplot(
            data=df,
            x="persona_label",
            y=feat,
            ax=ax,
            palette="tab10",
            showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"},
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=25)

    fig.suptitle("Pairwise Feature Comparison Across Customer Clusters", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "pairwise_cluster_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved pairwise_cluster_comparison.png to %s", output_dir)
