"""Unit tests for :mod:`loyalty_engine.features.engineering`."""
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from loyalty_engine.features.engineering import (
    CUSTOMER_VALUE_FEATURES,
    DERIVED_FEATURES,
    KMEANS_FEATURES,
    RFM_FEATURES,
    _mode_or_default,
    _shannon_entropy,
    build_feature_preprocessing_pipeline,
    create_feature_metadata,
    engineer_all_features,
    engineer_transaction_aggregates,
)


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------


def test_shannon_entropy_zero_for_single_category():
    assert _shannon_entropy(pd.Series(["a", "a", "a"])) == 0.0


def test_shannon_entropy_max_for_uniform_distribution():
    # Two equally likely categories -> normalized entropy of 1.0.
    result = _shannon_entropy(pd.Series(["a", "b", "a", "b"]))
    assert result == pytest.approx(1.0, abs=1e-6)


def test_shannon_entropy_between_zero_and_one_for_skewed():
    result = _shannon_entropy(pd.Series(["a", "a", "a", "b"]))
    assert 0.0 < result < 1.0


def test_mode_or_default_returns_most_common():
    assert _mode_or_default(pd.Series(["x", "y", "x"])) == "x"


def test_mode_or_default_returns_default_for_empty():
    assert _mode_or_default(pd.Series([np.nan, np.nan]), default="Unknown") == "Unknown"


# --------------------------------------------------------------------------
# engineer_transaction_aggregates
# --------------------------------------------------------------------------


def _transactions(customer_id="C1", n=3) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="7D")
    return pd.DataFrame(
        {
            "Customer_ID": [customer_id] * n,
            "Transaction_Date": dates,
            "Purchase_Amount": [100.0, 200.0, 300.0][:n],
            "Reward_Points_Earned": [10.0, 20.0, 30.0][:n],
            "Reward_Points_Redeemed": [1.0, 2.0, 3.0][:n],
            "Reward_Points_Available": [50.0, 60.0, 70.0][:n],
            "Coupon_Used": [1, 0, 1][:n],
            "Website_Visits": [2, 3, 4][:n],
            "Session_Duration_Min": [5.0, 6.0, 7.0][:n],
            "Wishlist_Added": [0, 1, 0][:n],
            "Cart_Abandoned": [1, 0, 0][:n],
            "Email_Clicked": [1, 0, 1][:n],
            "Push_Notification_Clicked": [0, 1, 0][:n],
            "App_Opened": [1, 1, 0][:n],
            "Store_Channel": ["Online", "In-Store", "Online"][:n],
            "Product_Category": ["Travel", "Dining", "Travel"][:n],
            "Brand": ["BrandA", "BrandB", "BrandA"][:n],
        }
    )


def test_engineer_transaction_aggregates_empty_returns_customer_id_frame():
    out = engineer_transaction_aggregates(pd.DataFrame())
    assert list(out.columns) == ["Customer_ID"]
    assert out.empty


def test_engineer_transaction_aggregates_computes_expected_metrics():
    out = engineer_transaction_aggregates(_transactions(n=3))

    assert len(out) == 1
    row = out.iloc[0]
    assert row["Customer_ID"] == "C1"
    assert row["tx_frequency"] == 3
    assert row["tx_total_spend"] == 600.0
    assert row["tx_coupon_count"] == 2.0
    assert row["tx_online_cnt"] == 2.0
    assert row["tx_offline_cnt"] == 1.0
    assert row["favorite_category"] == "Travel"
    assert row["favorite_brand"] == "BrandA"
    assert row["category_diversity"] == 2
    assert row["brand_diversity"] == 2


def test_engineer_transaction_aggregates_single_transaction_defaults():
    out = engineer_transaction_aggregates(_transactions(n=1))
    row = out.iloc[0]
    # Single-transaction customers get default inter-purchase / seasonality.
    assert row["average_days_between_purchases"] == 180.0
    assert row["seasonality_score"] == 0.0


# --------------------------------------------------------------------------
# engineer_all_features
# --------------------------------------------------------------------------


def _profile() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Customer_ID": ["C1", "C2"],
            "Membership_Tier": ["Gold", "Bronze"],
            "Customer_Since": ["2020-01-01", "2022-01-01"],
            "Age": [35, 28],
            "Gender": ["Male", "Female"],
            "City": ["NYC", "LA"],
            "Occupation": ["Eng", "Doc"],
            "Income_Level": ["High", "Medium"],
            "Total_Transactions_6M": [3, 1],
            "Total_Spend_6M": [600.0, 50.0],
            "Average_Order_Value": [200.0, 50.0],
            "Preferred_Brand": ["BrandA", "BrandC"],
            "Reward_Utilization": [0.5, 0.1],
            "Reward_Expiry_Risk": [0.1, 0.2],
            "Purchase_Frequency": [0.5, 0.2],
            "Loyalty_Engagement": ["High", "Low"],
            "Days_Since_Last_Purchase": [5, 90],
            "Upgrade_Readiness": ["High", "Low"],
            "Customer_Health": ["Healthy", "Critical"],
            "Churn_Risk": ["Low", "High"],
        }
    )


def test_engineer_all_features_produces_all_feature_groups():
    profile = _profile()
    transactions = _transactions(customer_id="C1", n=3)

    features = engineer_all_features(profile, transactions)

    assert len(features) == 2
    for col in RFM_FEATURES + CUSTOMER_VALUE_FEATURES + DERIVED_FEATURES:
        assert col in features.columns, f"missing feature column: {col}"

    # No temporary tx_* helper columns should leak into the output.
    assert not [c for c in features.columns if c.startswith("tx_")]


def test_engineer_all_features_flags_and_bounds():
    features = engineer_all_features(_profile(), _transactions("C1", 3)).set_index(
        "Customer_ID"
    )

    # C1 is a Gold member with recent activity; C2 is Bronze, dormant (90 days).
    assert features.loc["C1", "premium_customer_flag"] == 1
    assert features.loc["C2", "premium_customer_flag"] == 0
    assert features.loc["C2", "dormancy_flag"] == 1

    # Scores are clipped to their documented ranges.
    assert 0.0 <= features.loc["C1", "loyalty_score"] <= 100.0
    assert 0.0 <= features.loc["C1", "reward_utilization_pct"] <= 100.0
    assert 0.0 <= features.loc["C1", "customer_engagement_score"] <= 100.0


def test_engineer_all_features_uses_profile_fallbacks_for_customers_without_tx():
    # Only C1 has transactions; C2 has none, so its features fall back to the
    # profile-derived monetary / frequency values and default categories.
    features = engineer_all_features(_profile(), _transactions("C1", 3)).set_index(
        "Customer_ID"
    )
    assert features.loc["C2", "monetary"] == 50.0
    assert features.loc["C2", "frequency"] == 1.0
    assert features.loc["C2", "favorite_category"] == "General"


# --------------------------------------------------------------------------
# preprocessing pipeline & metadata
# --------------------------------------------------------------------------


def test_build_feature_preprocessing_pipeline_structure():
    pipeline = build_feature_preprocessing_pipeline()
    assert isinstance(pipeline, ColumnTransformer)
    transformer_names = [name for name, _, _ in pipeline.transformers]
    assert transformer_names == ["num", "cat"]


def test_build_feature_preprocessing_pipeline_fits_and_transforms():
    features = engineer_all_features(_profile(), _transactions("C1", 3))
    pipeline = build_feature_preprocessing_pipeline()
    transformed = pipeline.fit_transform(features)
    assert transformed.shape[0] == len(features)


def test_create_feature_metadata_contents():
    features = engineer_all_features(_profile(), _transactions("C1", 3))
    metadata = create_feature_metadata(features)

    assert metadata["total_customers"] == len(features)
    assert metadata["total_features"] == len(features.columns)
    assert metadata["kmeans_ml_features"] == KMEANS_FEATURES
    assert set(metadata["feature_groups"]) == {
        "rfm_features",
        "reward_features",
        "engagement_features",
        "shopping_behavior_features",
        "customer_value_features",
        "derived_features",
    }
    # Business-insight features are filtered to those present in the frame.
    assert all(c in features.columns for c in metadata["business_insight_features"])
