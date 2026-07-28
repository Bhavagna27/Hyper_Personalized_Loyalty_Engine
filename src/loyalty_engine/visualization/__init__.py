"""Visualization sub-package for the Hyper-Personalized Loyalty Engine.

Public API
----------
plot_customer_overview : callable
    Generate EDA overview distribution plots.
plot_elbow_and_silhouette : callable
    Generate Elbow curve and Silhouette score evaluation plots.
plot_pca_2d_clusters : callable
    Generate 2D PCA cluster projection scatter plot.
plot_pca_explained_variance : callable
    Generate PCA explained variance scree plot.
plot_cluster_centroids_heatmap : callable
    Generate normalized cluster centroids heatmap.
plot_pairwise_cluster_comparison : callable
    Generate pairwise boxplot comparisons across clusters.
"""

from .clustering_viz import (
    plot_cluster_centroids_heatmap,
    plot_elbow_and_silhouette,
    plot_pairwise_cluster_comparison,
    plot_pca_2d_clusters,
    plot_pca_explained_variance,
)
from .eda import plot_customer_overview

__all__ = [
    "plot_customer_overview",
    "plot_elbow_and_silhouette",
    "plot_pca_2d_clusters",
    "plot_pca_explained_variance",
    "plot_cluster_centroids_heatmap",
    "plot_pairwise_cluster_comparison",
]
