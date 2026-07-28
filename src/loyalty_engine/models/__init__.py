"""Models sub-package for the Hyper-Personalized Loyalty Engine.

Public API
----------
ModelBundle : dataclass
    Supervised model bundle container.
SegmentationModelBundle : dataclass
    Customer segmentation model bundle container.
evaluate_optimal_k : callable
    Evaluate K=2 to 10 via Elbow & Silhouette score.
train_kmeans_segmentation : callable
    Train KMeans clustering model and profile clusters.
predict_customer_segment : callable
    Assign new customer(s) to clusters with distance metrics and personas.
save_bundle / load_bundle : callable
    Joblib persistence helpers.
train_customer_models : callable
    Train supervised classification models.
"""

from .artifacts import ModelBundle, load_bundle, save_bundle
from .segmentation import (
    SEGMENTATION_FEATURES,
    SegmentationModelBundle,
    evaluate_optimal_k,
    generate_cluster_persona,
    predict_customer_segment,
    train_kmeans_segmentation,
)
from .trainers import train_customer_models

__all__ = [
    "ModelBundle",
    "SegmentationModelBundle",
    "SEGMENTATION_FEATURES",
    "evaluate_optimal_k",
    "train_kmeans_segmentation",
    "predict_customer_segment",
    "generate_cluster_persona",
    "load_bundle",
    "save_bundle",
    "train_customer_models",
]
