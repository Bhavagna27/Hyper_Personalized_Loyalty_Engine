"""Features sub-package for the Hyper-Personalized Loyalty Engine.

Public API
----------
FeatureEngineeringPipeline : class
    Orchestrates end-to-end feature generation, pipeline fitting, and artifact export.
engineer_all_features : callable
    Engineers all 6 feature groups into a single customer-level DataFrame.
build_feature_preprocessing_pipeline : callable
    Builds the reusable sklearn ColumnTransformer for ML preprocessing.
create_feature_metadata : callable
    Generates JSON feature metadata.
build_customer_features : callable
    Legacy helper maintained for backward compatibility.
build_training_table : callable
    Legacy helper maintained for backward compatibility.
"""

from .engineering import (
    BUSINESS_INSIGHT_FEATURES,
    CUSTOMER_VALUE_FEATURES,
    DERIVED_FEATURES,
    ENGAGEMENT_FEATURES,
    KMEANS_FEATURES,
    REWARD_FEATURES,
    RFM_FEATURES,
    SHOPPING_BEHAVIOR_FEATURES,
    FeatureEngineeringPipeline,
    FeatureEngineeringPipelineResult,
    build_customer_features,
    build_feature_preprocessing_pipeline,
    build_training_table,
    create_feature_metadata,
    engineer_all_features,
)

__all__ = [
    "FeatureEngineeringPipeline",
    "FeatureEngineeringPipelineResult",
    "engineer_all_features",
    "build_feature_preprocessing_pipeline",
    "create_feature_metadata",
    "build_customer_features",
    "build_training_table",
    "RFM_FEATURES",
    "REWARD_FEATURES",
    "ENGAGEMENT_FEATURES",
    "SHOPPING_BEHAVIOR_FEATURES",
    "CUSTOMER_VALUE_FEATURES",
    "DERIVED_FEATURES",
    "KMEANS_FEATURES",
    "BUSINESS_INSIGHT_FEATURES",
]
