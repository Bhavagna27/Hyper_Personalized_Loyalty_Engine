"""Feature engineering pipeline for the Hyper-Personalized Loyalty Engine.

This module consumes cleaned datasets from the ingestion layer and constructs
a single, unified, customer-level feature table where each row represents one
customer.

It generates 6 feature groups:
1. RFM Features (Recency, Frequency, Monetary, Average Order Value, Total Spend, Purchase Frequency, Days Since Last Purchase)
2. Reward Features (Reward Points Earned/Redeemed, Reward Utilization %, Coupon Usage %, Reward Redemption Rate, Average Reward Per Transaction)
3. Engagement Features (Website Visits, Session Duration, Wishlist Added, Cart Abandoned, Email Click Rate, Push Notification Response, App Usage Score, Customer Engagement Score)
4. Shopping Behavior (Favorite Category, Favorite Brand, Top Merchant, Online/Offline Purchase Ratio, Weekend Shopping Ratio, Average Monthly Spend, Seasonality Score)
5. Customer Value (Customer Lifetime Value, Estimated Annual Spend, Profitability Score, Customer Health Score, Upgrade Readiness Score, Churn Risk Score, Loyalty Score)
6. Derived Features (Spending Diversity Score, Category Diversity, Brand Diversity, Average Days Between Purchases, High Value Customer Flag, Dormancy Flag, Premium Customer Flag)

It also constructs and saves a reusable sklearn preprocessing pipeline
(`feature_pipeline.joblib`) and feature metadata (`feature_metadata.json`).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from loyalty_engine.config import PATHS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature Categorization Lists
# ---------------------------------------------------------------------------

RFM_FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "average_order_value",
    "total_spend",
    "purchase_frequency",
    "days_since_last_purchase",
]

REWARD_FEATURES = [
    "reward_points_earned",
    "reward_points_redeemed",
    "reward_utilization_pct",
    "coupon_usage_pct",
    "reward_redemption_rate",
    "average_reward_per_transaction",
]

ENGAGEMENT_FEATURES = [
    "website_visits",
    "session_duration",
    "wishlist_added",
    "cart_abandoned",
    "email_click_rate",
    "push_notification_response",
    "app_usage_score",
    "customer_engagement_score",
]

SHOPPING_BEHAVIOR_FEATURES = [
    "favorite_category",
    "favorite_brand",
    "top_merchant",
    "online_purchase_ratio",
    "offline_purchase_ratio",
    "weekend_shopping_ratio",
    "average_monthly_spend",
    "seasonality_score",
]

CUSTOMER_VALUE_FEATURES = [
    "customer_lifetime_value",
    "estimated_annual_spend",
    "profitability_score",
    "customer_health_score",
    "upgrade_readiness_score",
    "churn_risk_score",
    "loyalty_score",
]

DERIVED_FEATURES = [
    "spending_diversity_score",
    "category_diversity",
    "brand_diversity",
    "average_days_between_purchases",
    "high_value_customer_flag",
    "dormancy_flag",
    "premium_customer_flag",
]

KMEANS_FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "average_order_value",
    "purchase_frequency",
    "reward_utilization_pct",
    "coupon_usage_pct",
    "reward_redemption_rate",
    "customer_engagement_score",
    "app_usage_score",
    "online_purchase_ratio",
    "weekend_shopping_ratio",
    "customer_lifetime_value",
    "spending_diversity_score",
    "category_diversity",
    "brand_diversity",
    "average_days_between_purchases",
]

BUSINESS_INSIGHT_FEATURES = [
    "Customer_ID",
    "Membership_Tier",
    "Gender",
    "City",
    "Occupation",
    "Income_Level",
    "Customer_Since",
    "Age",
    "favorite_category",
    "favorite_brand",
    "top_merchant",
    "Customer_Health",
    "Churn_Risk",
    "Upgrade_Readiness",
    "Loyalty_Engagement",
    "high_value_customer_flag",
    "dormancy_flag",
    "premium_customer_flag",
    "total_spend",
    "estimated_annual_spend",
    "profitability_score",
    "customer_health_score",
    "upgrade_readiness_score",
    "churn_risk_score",
    "loyalty_score",
]


# ---------------------------------------------------------------------------
# Helper Calculation Functions
# ---------------------------------------------------------------------------

def _shannon_entropy(series: pd.Series) -> float:
    """Calculate normalized Shannon entropy for diversity metrics."""
    counts = series.value_counts(dropna=True)
    if len(counts) <= 1:
        return 0.0
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-9))
    max_entropy = np.log2(len(counts))
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def _mode_or_default(series: pd.Series, default: str = "Unknown") -> str:
    """Return the mode of a series, or default if empty/null."""
    non_null = series.dropna()
    if non_null.empty:
        return default
    mode_vals = non_null.mode()
    return str(mode_vals.iloc[0]) if not mode_vals.empty else default


# ---------------------------------------------------------------------------
# Individual Feature Group Transformers
# ---------------------------------------------------------------------------

def engineer_transaction_aggregates(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate row-level transaction logs into customer-level raw metrics.

    Parameters
    ----------
    transactions:
        Cleaned transaction history DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by Customer_ID with aggregated transaction metrics.
    """
    if transactions.empty:
        return pd.DataFrame(columns=["Customer_ID"])

    tx = transactions.copy()
    tx["Transaction_Date"] = pd.to_datetime(tx["Transaction_Date"], errors="coerce")
    ref_date = tx["Transaction_Date"].max() + pd.Timedelta(days=1)

    records: list[dict[str, Any]] = []

    for cid, group in tx.groupby("Customer_ID"):
        group_sorted = group.sort_values("Transaction_Date")
        tx_count = len(group)
        max_date = group_sorted["Transaction_Date"].max()
        recency = (ref_date - max_date).days if pd.notna(max_date) else 90

        # Purchase amounts
        tot_spend = float(group["Purchase_Amount"].sum(skipna=True))
        avg_aov = float(group["Purchase_Amount"].mean(skipna=True)) if tx_count > 0 else 0.0

        # Reward metrics
        rw_earned = float(group["Reward_Points_Earned"].sum(skipna=True))
        rw_redeemed = float(group["Reward_Points_Redeemed"].sum(skipna=True))
        rw_avail = float(group["Reward_Points_Available"].max(skipna=True))
        coupon_used_cnt = float((group["Coupon_Used"] == 1).sum())

        # Engagement metrics
        web_visits = float(group["Website_Visits"].sum(skipna=True))
        sess_dur = float(group["Session_Duration_Min"].mean(skipna=True)) if tx_count > 0 else 0.0
        wishlist_sum = float(group["Wishlist_Added"].sum(skipna=True))
        cart_abandon_sum = float(group["Cart_Abandoned"].sum(skipna=True))
        email_clicked_cnt = float((group["Email_Clicked"] == 1).sum())
        push_clicked_cnt = float((group["Push_Notification_Clicked"] == 1).sum())
        app_opened_sum = float(group["App_Opened"].sum(skipna=True))

        # Shopping Channels
        online_cnt = float(group["Store_Channel"].isin(["Online", "App", "Mobile App"]).sum())
        offline_cnt = float(group["Store_Channel"].isin(["Offline", "In-Store"]).sum())
        weekend_cnt = float((group_sorted["Transaction_Date"].dt.dayofweek >= 5).sum())

        # Modes & Diversities
        fav_cat = _mode_or_default(group["Product_Category"])
        fav_brand = _mode_or_default(group["Brand"])
        top_chan = _mode_or_default(group["Store_Channel"])
        cat_div = int(group["Product_Category"].nunique(dropna=True))
        brand_div = int(group["Brand"].nunique(dropna=True))
        spend_div = _shannon_entropy(group["Product_Category"])

        # Inter-purchase interval
        if tx_count > 1:
            date_diffs = group_sorted["Transaction_Date"].diff().dt.days.dropna()
            avg_days_between = float(date_diffs.mean()) if not date_diffs.empty else 30.0
        else:
            avg_days_between = 180.0

        # Monthly spend variation (seasonality)
        if tx_count > 1:
            monthly_spends = group.groupby(group["Transaction_Date"].dt.to_period("M"))["Purchase_Amount"].sum()
            mean_m = monthly_spends.mean()
            seasonality = float(monthly_spends.std() / mean_m) if mean_m > 0 and len(monthly_spends) > 1 else 0.0
        else:
            seasonality = 0.0

        records.append({
            "Customer_ID": cid,
            "tx_recency": recency,
            "tx_frequency": tx_count,
            "tx_total_spend": tot_spend,
            "tx_aov": avg_aov,
            "tx_rw_earned": rw_earned,
            "tx_rw_redeemed": rw_redeemed,
            "tx_rw_avail": rw_avail,
            "tx_coupon_count": coupon_used_cnt,
            "tx_web_visits": web_visits,
            "tx_sess_duration": sess_dur,
            "tx_wishlist": wishlist_sum,
            "tx_cart_abandon": cart_abandon_sum,
            "tx_email_clicks": email_clicked_cnt,
            "tx_push_clicks": push_clicked_cnt,
            "tx_app_opened": app_opened_sum,
            "tx_online_cnt": online_cnt,
            "tx_offline_cnt": offline_cnt,
            "tx_weekend_cnt": weekend_cnt,
            "favorite_category": fav_cat,
            "favorite_brand": fav_brand,
            "top_merchant": top_chan,
            "category_diversity": cat_div,
            "brand_diversity": brand_div,
            "spending_diversity_score": spend_div,
            "average_days_between_purchases": avg_days_between,
            "seasonality_score": seasonality,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Primary Feature Calculation Engine
# ---------------------------------------------------------------------------

def engineer_all_features(
    profile: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Build the complete, customer-level feature dataset.

    Parameters
    ----------
    profile:
        Cleaned Customer_Loyalty_Profile DataFrame.
    transactions:
        Cleaned Transaction_History DataFrame.

    Returns
    -------
    pd.DataFrame
        Complete feature table with one row per customer.
    """
    logger.info("Starting feature engineering for %d profile rows…", len(profile))
    df = profile.copy()

    # 1. Compute transaction-level aggregations
    tx_agg = engineer_transaction_aggregates(transactions)
    if not tx_agg.empty:
        df = df.merge(tx_agg, on="Customer_ID", how="left")

    # 2. RFM Features
    df["recency"] = df["tx_recency"].fillna(df["Days_Since_Last_Purchase"]).fillna(60.0)
    df["frequency"] = df["tx_frequency"].fillna(df["Total_Transactions_6M"]).fillna(1.0)
    df["monetary"] = df["tx_total_spend"].fillna(df["Total_Spend_6M"]).fillna(0.0)
    df["total_spend"] = df["monetary"]

    df["average_order_value"] = np.where(
        df["frequency"] > 0,
        df["monetary"] / df["frequency"],
        df["Average_Order_Value"].fillna(0.0),
    )
    df["average_order_value"] = df["average_order_value"].fillna(df["Average_Order_Value"]).fillna(0.0)

    df["purchase_frequency"] = df["frequency"] / 6.0
    df["days_since_last_purchase"] = df["recency"]

    # 3. Reward Features
    df["reward_points_earned"] = df["tx_rw_earned"].fillna(df["monetary"] * 0.01)
    df["reward_points_redeemed"] = df["tx_rw_redeemed"].fillna(0.0)

    df["reward_utilization_pct"] = np.where(
        df["reward_points_earned"] > 0,
        (df["reward_points_redeemed"] / (df["reward_points_earned"] + 1e-5)) * 100,
        df["Reward_Utilization"].fillna(0.0) * 100,
    )
    df["reward_utilization_pct"] = np.clip(df["reward_utilization_pct"].fillna(0.0), 0.0, 100.0)

    df["coupon_usage_pct"] = np.where(
        df["frequency"] > 0,
        (df["tx_coupon_count"].fillna(0.0) / df["frequency"]) * 100,
        0.0,
    )
    df["coupon_usage_pct"] = np.clip(df["coupon_usage_pct"].fillna(0.0), 0.0, 100.0)

    df["reward_redemption_rate"] = np.where(
        (df["reward_points_earned"] + df["tx_rw_avail"].fillna(0.0)) > 0,
        df["reward_points_redeemed"] / (df["reward_points_earned"] + df["tx_rw_avail"].fillna(0.0) + 1e-5),
        df["Reward_Utilization"].fillna(0.0),
    )
    df["reward_redemption_rate"] = np.clip(df["reward_redemption_rate"].fillna(0.0), 0.0, 1.0)

    df["average_reward_per_transaction"] = np.where(
        df["frequency"] > 0,
        df["reward_points_earned"] / df["frequency"],
        0.0,
    )

    # 4. Engagement Features
    df["website_visits"] = df["tx_web_visits"].fillna(0.0)
    df["session_duration"] = df["tx_sess_duration"].fillna(0.0)
    df["wishlist_added"] = df["tx_wishlist"].fillna(0.0)
    df["cart_abandoned"] = df["tx_cart_abandon"].fillna(0.0)

    df["email_click_rate"] = np.where(
        df["frequency"] > 0,
        df["tx_email_clicks"].fillna(0.0) / df["frequency"],
        0.0,
    )
    df["push_notification_response"] = np.where(
        df["frequency"] > 0,
        df["tx_push_clicks"].fillna(0.0) / df["frequency"],
        0.0,
    )

    app_opens = df["tx_app_opened"].fillna(0.0)
    df["app_usage_score"] = np.clip(
        app_opens * 10.0 + df["website_visits"] * 2.0 + df["session_duration"] * 1.0,
        0.0,
        100.0,
    )

    # Engagement composite score
    engagement_rank = df["Loyalty_Engagement"].map({"High": 90.0, "Medium": 60.0, "Low": 30.0}).fillna(50.0)
    digital_score = (
        df["app_usage_score"] * 0.4
        + np.clip(df["website_visits"] * 5.0, 0.0, 100.0) * 0.3
        + df["email_click_rate"] * 100.0 * 0.15
        + df["push_notification_response"] * 100.0 * 0.15
    )
    df["customer_engagement_score"] = np.clip(digital_score * 0.6 + engagement_rank * 0.4, 0.0, 100.0)

    # 5. Shopping Behavior Features
    df["favorite_category"] = df["favorite_category"].fillna("General")
    df["favorite_brand"] = df["favorite_brand"].fillna(df["Preferred_Brand"]).fillna("Unknown")
    df["top_merchant"] = df["top_merchant"].fillna("Online")

    df["online_purchase_ratio"] = np.where(
        df["frequency"] > 0,
        df["tx_online_cnt"].fillna(0.0) / df["frequency"],
        0.5,
    )
    df["offline_purchase_ratio"] = np.where(
        df["frequency"] > 0,
        df["tx_offline_cnt"].fillna(0.0) / df["frequency"],
        0.5,
    )
    df["weekend_shopping_ratio"] = np.where(
        df["frequency"] > 0,
        df["tx_weekend_cnt"].fillna(0.0) / df["frequency"],
        0.25,
    )

    df["average_monthly_spend"] = df["total_spend"] / 6.0
    df["seasonality_score"] = df["seasonality_score"].fillna(0.0)

    # 6. Customer Value Features
    df["Customer_Since"] = pd.to_datetime(df["Customer_Since"], errors="coerce")
    ref_date = pd.Timestamp.now()
    tenure_years = np.clip(((ref_date - df["Customer_Since"]).dt.days / 365.25).fillna(2.0), 0.5, 20.0)

    df["customer_lifetime_value"] = round(df["average_order_value"] * df["purchase_frequency"] * 12.0 * tenure_years, 2)
    df["estimated_annual_spend"] = round(df["average_monthly_spend"] * 12.0, 2)
    df["profitability_score"] = round((df["estimated_annual_spend"] * 0.15) - (df["reward_points_redeemed"] * 0.01), 2)

    # Numerical mappings for Health, Upgrade, Churn
    health_map = {"Healthy": 100.0, "Stable": 70.0, "Needs Attention": 40.0, "Critical": 10.0}
    df["customer_health_score"] = df["Customer_Health"].map(health_map).fillna(50.0)

    upgrade_map = {"High": 100.0, "Medium": 60.0, "Low": 30.0, "No": 0.0}
    df["upgrade_readiness_score"] = df["Upgrade_Readiness"].map(upgrade_map).fillna(0.0)

    churn_map = {"High": 90.0, "Medium": 50.0, "Low": 10.0}
    df["churn_risk_score"] = df["Churn_Risk"].map(churn_map).fillna(20.0)

    tier_rank = df["Membership_Tier"].map({"Bronze": 25.0, "Silver": 50.0, "Gold": 75.0, "Platinum": 100.0}).fillna(25.0)
    df["loyalty_score"] = np.clip(
        tier_rank * 0.3
        + df["customer_engagement_score"] * 0.3
        + (100.0 - df["churn_risk_score"]) * 0.2
        + df["reward_utilization_pct"] * 0.2,
        0.0,
        100.0,
    )

    # 7. Derived Features
    df["spending_diversity_score"] = df["spending_diversity_score"].fillna(0.0)
    df["category_diversity"] = df["category_diversity"].fillna(1.0).astype(int)
    df["brand_diversity"] = df["brand_diversity"].fillna(1.0).astype(int)
    df["average_days_between_purchases"] = df["average_days_between_purchases"].fillna(30.0)

    p75_spend = df["total_spend"].quantile(0.75) if not df.empty else 100000.0
    df["high_value_customer_flag"] = (df["total_spend"] >= p75_spend).astype(int)
    df["dormancy_flag"] = (df["recency"] > 60.0).astype(int)
    df["premium_customer_flag"] = df["Membership_Tier"].isin(["Gold", "Platinum"]).astype(int)

    # Clean intermediate temporary helper columns if present
    temp_cols = [c for c in df.columns if c.startswith("tx_")]
    if temp_cols:
        df = df.drop(columns=temp_cols)

    logger.info("Successfully engineered %d features across %d rows.", len(df.columns), len(df))
    return df


# ---------------------------------------------------------------------------
# Sklearn Preprocessing Pipeline Builder
# ---------------------------------------------------------------------------

def build_feature_preprocessing_pipeline(
    ml_numerical_features: list[str] = KMEANS_FEATURES,
    ml_categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Create a reusable sklearn ColumnTransformer for ML preprocessing.

    Parameters
    ----------
    ml_numerical_features:
        List of numerical feature names to impute and scale.
    ml_categorical_features:
        Optional list of categorical feature names to impute and one-hot encode.

    Returns
    -------
    ColumnTransformer
        Fitted or unfitted sklearn transformer object.
    """
    if ml_categorical_features is None:
        ml_categorical_features = ["Membership_Tier", "Gender", "Income_Level", "favorite_category"]

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, ml_numerical_features),
            ("cat", cat_pipeline, ml_categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


# ---------------------------------------------------------------------------
# Metadata Export Helper
# ---------------------------------------------------------------------------

def create_feature_metadata(
    df: pd.DataFrame,
    ml_numerical_features: list[str] = KMEANS_FEATURES,
    ml_categorical_features: list[str] | None = None,
) -> dict[str, Any]:
    """Generate structured JSON metadata for the feature engineering pipeline.

    Parameters
    ----------
    df:
        Engineered feature DataFrame.
    ml_numerical_features:
        List of numerical ML features.
    ml_categorical_features:
        List of categorical ML features.

    Returns
    -------
    dict[str, Any]
        Metadata dict ready for serialization.
    """
    if ml_categorical_features is None:
        ml_categorical_features = ["Membership_Tier", "Gender", "Income_Level", "favorite_category"]

    all_cols = list(df.columns)
    metadata = {
        "total_customers": len(df),
        "total_features": len(all_cols),
        "feature_columns": all_cols,
        "feature_groups": {
            "rfm_features": RFM_FEATURES,
            "reward_features": REWARD_FEATURES,
            "engagement_features": ENGAGEMENT_FEATURES,
            "shopping_behavior_features": SHOPPING_BEHAVIOR_FEATURES,
            "customer_value_features": CUSTOMER_VALUE_FEATURES,
            "derived_features": DERIVED_FEATURES,
        },
        "kmeans_ml_features": ml_numerical_features,
        "business_insight_features": [c for c in BUSINESS_INSIGHT_FEATURES if c in all_cols],
        "ml_preprocessing_pipeline": {
            "numerical_features": ml_numerical_features,
            "categorical_features": ml_categorical_features,
            "scaling_method": "StandardScaler",
            "categorical_encoding": "OneHotEncoder(handle_unknown='ignore')",
            "imputation_strategy": {"numerical": "median", "categorical": "most_frequent"},
        },
    }
    return metadata


# ---------------------------------------------------------------------------
# Backward Compatibility Wrappers
# ---------------------------------------------------------------------------

def build_customer_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Legacy helper maintained for backward compatibility."""
    return engineer_transaction_aggregates(transactions)


def build_training_table(profile: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Legacy helper maintained for backward compatibility."""
    return engineer_all_features(profile, transactions)


# ---------------------------------------------------------------------------
# High-Level Feature Engineering Pipeline Class
# ---------------------------------------------------------------------------

class FeatureEngineeringPipelineResult(NamedTuple):
    """Result container returned by :meth:`FeatureEngineeringPipeline.run`."""

    customer_features: pd.DataFrame
    pipeline: ColumnTransformer
    metadata: dict[str, Any]
    saved_paths: list[Path]


@dataclass
class FeatureEngineeringPipeline:
    """Orchestrates feature engineering, preprocessing pipeline fitting, and artifact export.

    Parameters
    ----------
    output_dir:
        Directory for saving ``customer_features.csv``. Defaults to ``data/processed/``.
    artifacts_dir:
        Directory for saving ``feature_pipeline.joblib`` and ``feature_metadata.json``.
    """

    output_dir: Path = field(default_factory=lambda: PATHS.processed_dir)
    artifacts_dir: Path = field(default_factory=lambda: PATHS.artifacts_dir)

    def run(
        self,
        profile: pd.DataFrame,
        transactions: pd.DataFrame,
        *,
        save: bool = True,
    ) -> FeatureEngineeringPipelineResult:
        """Execute the feature engineering pipeline.

        Parameters
        ----------
        profile:
            Cleaned Customer_Loyalty_Profile DataFrame.
        transactions:
            Cleaned Transaction_History DataFrame.
        save:
            Whether to persist CSV, joblib, and JSON outputs to disk.

        Returns
        -------
        FeatureEngineeringPipelineResult
        """
        logger.info("Executing FeatureEngineeringPipeline…")

        # 1. Engineer full customer-level features
        customer_features = engineer_all_features(profile, transactions)

        # 2. Build & fit sklearn preprocessing pipeline
        pipeline = build_feature_preprocessing_pipeline()
        pipeline.fit(customer_features)
        logger.info("Fitted sklearn preprocessing pipeline.")

        # 3. Create metadata
        metadata = create_feature_metadata(customer_features)

        saved_paths: list[Path] = []
        if save:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)

            # Save customer_features.csv
            features_csv_path = self.output_dir / "customer_features.csv"
            customer_features.to_csv(features_csv_path, index=False)
            saved_paths.append(features_csv_path)
            logger.info("Saved customer features to %s (%d rows)", features_csv_path, len(customer_features))

            # Save feature_pipeline.joblib
            pipeline_joblib_path = self.artifacts_dir / "feature_pipeline.joblib"
            joblib.dump(pipeline, pipeline_joblib_path)
            saved_paths.append(pipeline_joblib_path)
            logger.info("Saved feature preprocessing pipeline to %s", pipeline_joblib_path)

            # Save feature_metadata.json
            metadata_json_path = self.artifacts_dir / "feature_metadata.json"
            with open(metadata_json_path, "w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2, default=str)
            saved_paths.append(metadata_json_path)
            logger.info("Saved feature metadata to %s", metadata_json_path)

        return FeatureEngineeringPipelineResult(
            customer_features=customer_features,
            pipeline=pipeline,
            metadata=metadata,
            saved_paths=saved_paths,
        )
