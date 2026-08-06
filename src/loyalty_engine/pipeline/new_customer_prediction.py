from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from loyalty_engine.config import PATHS
from loyalty_engine.features import build_training_table
from loyalty_engine.io import ExcelDatasetLoader
from loyalty_engine.models import SEGMENTATION_FEATURES, predict_customer_segment
from loyalty_engine.preprocessing import clean_customer_profile, clean_transaction_history
from loyalty_engine.recommendations import RecommendationEngine
from loyalty_engine.security import load_artifact
from loyalty_engine.validation import validate_workbook

logger = logging.getLogger(__name__)


class NewCustomerPredictor:
    """Reusable prediction entry point for brand-new customers.

    The class reuses the existing feature engineering pipeline, clustering
    artifacts, and recommendation engine without retraining.
    """

    def __init__(
        self,
        feature_pipeline_path: Path | None = None,
        kmeans_model_path: Path | None = None,
        cluster_profiles_path: Path | None = None,
        cluster_statistics_path: Path | None = None,
    ) -> None:
        self.feature_pipeline_path = feature_pipeline_path or PATHS.feature_pipeline_path
        self.kmeans_model_path = kmeans_model_path or PATHS.kmeans_model_path
        self.cluster_profiles_path = cluster_profiles_path or PATHS.cluster_profiles_path
        self.cluster_statistics_path = cluster_statistics_path or PATHS.cluster_statistics_path

        self.feature_pipeline = self._load_feature_pipeline()
        self.feature_metadata = self._load_feature_metadata()
        self.segmentation_bundle = self._load_segmentation_bundle()
        self.cluster_profiles = self._load_cluster_profiles()
        self.cluster_statistics = self._load_cluster_statistics()
        self.distance_statistics = self._load_or_compute_distance_statistics()

    def predict_customer(self, profile: pd.DataFrame | dict[str, Any] | Any) -> dict[str, Any]:
        """Predict cluster/persona/rewards for a single new customer profile."""
        customer_frame = self._coerce_to_profile_frame(profile)
        if customer_frame.empty:
            return {
                "customer_id": None,
                "predicted_cluster": None,
                "persona": None,
                "similarity_score": None,
                "nearest_cluster_distance": None,
                "recommended_rewards": [],
                "business_insight": "",
                "expected_impact": "",
            }

        features = self._engineer_features(customer_frame)
        cluster_prediction = predict_customer_segment(features[SEGMENTATION_FEATURES], self.segmentation_bundle)
        if isinstance(cluster_prediction, dict):
            cluster_prediction_df = pd.DataFrame([cluster_prediction])
        else:
            cluster_prediction_df = cluster_prediction.copy()

        cluster_row = cluster_prediction_df.iloc[0]
        customer_id = str(customer_frame.iloc[0].get("Customer_ID", "unknown"))
        cluster_id = int(cluster_row.get("cluster_id", -1))
        persona = str(cluster_row.get("persona") or "Unknown")
        distance = float(cluster_row.get("distance_to_centroid", 0.0))
        similarity_score = self._similarity_from_distance(distance)
        confidence_level = self._confidence_from_similarity(similarity_score)

        recommendation_engine = RecommendationEngine(
            bundle=None,
            recommendation_bank=pd.DataFrame(),
            customer_features=features,
            segmentation_bundle=self.segmentation_bundle,
            cluster_profiles=self.cluster_profiles,
            cluster_statistics=self.cluster_statistics,
        )
        payload = recommendation_engine._build_recommendation_payload(
            {
                "Customer_ID": customer_id,
                "Cluster": str(cluster_id),
                "Persona": persona,
                "cluster_id": str(cluster_id),
            }
        )

        return {
            "customer_id": customer_id,
            "predicted_cluster": cluster_id,
            "persona": persona,
            "distance_to_centroid": round(distance, 4),
            "similarity_score": round(similarity_score, 2),
            "confidence_level": confidence_level,
            "nearest_cluster_distance": round(distance, 4),
            "recommended_rewards": payload.get("Top_3_Rewards", []),
            "business_insight": payload.get("Business_Insight", ""),
            "expected_impact": payload.get("Expected_Impact", ""),
        }

    def predict_dataframe(self, df: pd.DataFrame, output_path: str | Path | None = None) -> pd.DataFrame:
        """Predict cluster/persona/rewards for a batch of new customers from a DataFrame."""
        rows = []
        for _, row in df.iterrows():
            rows.append(self.predict_customer(row.to_dict()))
        predictions = pd.DataFrame(rows)
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            predictions.to_csv(output_path, index=False)
        return predictions

    def predict_excel(self, path: str | Path, output_path: str | Path | None = None) -> pd.DataFrame:
        """Predict cluster/persona/rewards for a batch of new customers from an Excel file."""
        workbook = ExcelDatasetLoader(path)
        frames = workbook.load_all()
        validate_workbook(frames)
        if "Customer_Loyalty_Profile" in frames:
            profile_df = clean_customer_profile(frames["Customer_Loyalty_Profile"])
            transactions_df = clean_transaction_history(frames.get("Transaction_History", pd.DataFrame(columns=["Customer_ID"])))
            feature_table = build_training_table(profile_df, transactions_df)
            final_output = output_path or PATHS.root / "new_customer_predictions.csv"
            return self.predict_dataframe(feature_table, output_path=final_output)
        default_frame = pd.DataFrame(frames.get("Sheet1", []))
        final_output = output_path or PATHS.root / "new_customer_predictions.csv"
        return self.predict_dataframe(default_frame, output_path=final_output)

    def _coerce_to_profile_frame(self, profile: pd.DataFrame | dict[str, Any] | Any) -> pd.DataFrame:
        if isinstance(profile, dict):
            frame = pd.DataFrame([profile])
        elif isinstance(profile, pd.DataFrame):
            frame = profile.copy()
        else:
            frame = pd.DataFrame([profile])

        if "Customer_ID" not in frame.columns and "customer_id" in frame.columns:
            frame = frame.rename(columns={"customer_id": "Customer_ID"})
        if "Customer_ID" not in frame.columns:
            frame["Customer_ID"] = ["new_customer"]
        return frame

    def _engineer_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.feature_pipeline is None:
            logger.warning("Feature pipeline artifact is unavailable; using the existing feature engineering path only.")

        if "Customer_ID" in frame.columns:
            frame = frame.copy()
            frame["Customer_ID"] = frame["Customer_ID"].astype(str)

        if self.feature_metadata and "feature_columns" in self.feature_metadata:
            expected_columns = [col for col in self.feature_metadata["feature_columns"] if col in frame.columns]
            frame = frame[expected_columns].copy() if expected_columns else frame.copy()

        if "Transaction_Date" in frame.columns and "Purchase_Amount" in frame.columns:
            transaction_row = frame.iloc[0].copy()
            transactions = pd.DataFrame(
                [
                    {
                        "Customer_ID": transaction_row.get("Customer_ID", "new_customer"),
                        "Transaction_Date": transaction_row.get("Transaction_Date", pd.Timestamp.today().strftime("%Y-%m-%d")),
                        "Purchase_Amount": transaction_row.get("Purchase_Amount", 0.0),
                        "Reward_Points_Earned": transaction_row.get("Reward_Points_Earned", 0.0),
                        "Reward_Points_Redeemed": transaction_row.get("Reward_Points_Redeemed", 0.0),
                        "Reward_Points_Available": transaction_row.get("Reward_Points_Available", 0.0),
                        "Reward_Points_Expired": transaction_row.get("Reward_Points_Expired", 0.0),
                        "Coupon_Used": transaction_row.get("Coupon_Used", 0),
                        "Product_Viewed": transaction_row.get("Product_Viewed", 0),
                        "Wishlist_Added": transaction_row.get("Wishlist_Added", 0),
                        "Cart_Abandoned": transaction_row.get("Cart_Abandoned", 0),
                        "Email_Clicked": transaction_row.get("Email_Clicked", 0),
                        "Push_Notification_Clicked": transaction_row.get("Push_Notification_Clicked", 0),
                        "App_Opened": transaction_row.get("App_Opened", 0),
                        "Website_Visits": transaction_row.get("Website_Visits", 0),
                        "Session_Duration_Min": transaction_row.get("Session_Duration_Min", 0.0),
                        "Product_Category": transaction_row.get("Product_Category", "General"),
                        "Brand": transaction_row.get("Brand", "Unknown"),
                        "Store_Channel": transaction_row.get("Store_Channel", "Online"),
                    }
                ]
            )
            profile = frame.copy()
            profile = profile.drop(
                columns=[
                    "Purchase_Amount",
                    "Transaction_Date",
                    "Reward_Points_Earned",
                    "Reward_Points_Redeemed",
                    "Reward_Points_Available",
                    "Reward_Points_Expired",
                    "Coupon_Used",
                    "Product_Category",
                    "Brand",
                    "Store_Channel",
                    "Product_Viewed",
                    "Wishlist_Added",
                    "Cart_Abandoned",
                    "Email_Clicked",
                    "Push_Notification_Clicked",
                    "App_Opened",
                    "Website_Visits",
                    "Session_Duration_Min",
                ],
                errors="ignore",
            )
            features = build_training_table(profile, transactions)
            return features

        profile = clean_customer_profile(frame)
        transaction_frame = pd.DataFrame(
            [{
                "Customer_ID": profile.iloc[0].get("Customer_ID", "new_customer"),
                "Transaction_Date": pd.Timestamp.today().strftime("%Y-%m-%d"),
                "Purchase_Amount": 0.0,
                "Reward_Points_Earned": 0.0,
                "Reward_Points_Redeemed": 0.0,
                "Reward_Points_Available": 0.0,
                "Reward_Points_Expired": 0.0,
                "Coupon_Used": 0,
                "Product_Viewed": 0,
                "Wishlist_Added": 0,
                "Cart_Abandoned": 0,
                "Email_Clicked": 0,
                "Push_Notification_Clicked": 0,
                "App_Opened": 0,
                "Website_Visits": 0,
                "Session_Duration_Min": 0.0,
                "Product_Category": "General",
                "Brand": "Unknown",
                "Store_Channel": "Online",
            }]
        )
        features = build_training_table(profile, transaction_frame)
        return features

    def _load_feature_pipeline(self) -> Any | None:
        if not self.feature_pipeline_path.exists():
            return None
        return load_artifact(self.feature_pipeline_path)

    def _load_feature_metadata(self) -> dict[str, Any] | None:
        metadata_path = PATHS.feature_metadata_path
        if not metadata_path.exists():
            return None
        with open(metadata_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _load_segmentation_bundle(self) -> Any:
        if not self.kmeans_model_path.exists():
            raise FileNotFoundError(f"Segmentation model artifact not found at {self.kmeans_model_path}")
        return load_artifact(self.kmeans_model_path)

    def _load_cluster_profiles(self) -> pd.DataFrame | None:
        if not self.cluster_profiles_path.exists():
            return None
        return pd.read_csv(self.cluster_profiles_path)

    def _load_cluster_statistics(self) -> pd.DataFrame | None:
        if not self.cluster_statistics_path.exists():
            return None
        return pd.read_csv(self.cluster_statistics_path)

    def _load_or_compute_distance_statistics(self) -> dict[str, float]:
        stats_path = self.cluster_statistics_path.parent / "cluster_distance_stats.json"
        if stats_path.exists():
            with open(stats_path, "r", encoding="utf-8") as handle:
                return json.load(handle)

        training_features_path = PATHS.customer_features_path
        if not training_features_path.exists():
            return {"min_distance": 0.0, "max_distance": 1.0, "average_distance": 0.0}

        training_features = pd.read_csv(training_features_path)
        available_features = [feature for feature in SEGMENTATION_FEATURES if feature in training_features.columns]
        if len(available_features) < 1:
            return {"min_distance": 0.0, "max_distance": 1.0, "average_distance": 0.0}

        scaled = self.segmentation_bundle.scaler.transform(training_features[available_features])
        all_distances = self.segmentation_bundle.kmeans_model.transform(scaled)
        centroid_distances = np.min(all_distances, axis=1)
        stats = {
            "min_distance": float(np.min(centroid_distances)),
            "max_distance": float(np.max(centroid_distances)),
            "average_distance": float(np.mean(centroid_distances)),
        }

        with open(stats_path, "w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)

        if self.cluster_statistics is not None:
            updated_statistics = self.cluster_statistics.copy()
            for key, value in stats.items():
                updated_statistics[key] = value
            updated_statistics.to_csv(self.cluster_statistics_path, index=False)
            self.cluster_statistics = updated_statistics

        return stats

    def _similarity_from_distance(self, distance: float) -> float:
        min_distance = float(self.distance_statistics.get("min_distance", 0.0))
        max_distance = float(self.distance_statistics.get("max_distance", 1.0))
        if max_distance <= min_distance:
            return 100.0
        normalized_distance = (distance - min_distance) / (max_distance - min_distance)
        normalized_distance = max(0.0, min(1.0, normalized_distance))
        return max(0.0, min(100.0, (1.0 - normalized_distance) * 100.0))

    def _confidence_from_similarity(self, similarity: float) -> str:
        if similarity >= 90:
            return "Excellent Match"
        if similarity >= 75:
            return "Strong Match"
        if similarity >= 60:
            return "Moderate Match"
        if similarity >= 40:
            return "Weak Match"
        return "Outlier"


def predict_customer(profile: pd.DataFrame | dict[str, Any] | Any) -> dict[str, Any]:
    return NewCustomerPredictor().predict_customer(profile)


def predict_dataframe(df: pd.DataFrame, output_path: str | Path | None = None) -> pd.DataFrame:
    return NewCustomerPredictor().predict_dataframe(df, output_path=output_path)


def predict_excel(path: str | Path, output_path: str | Path | None = None) -> pd.DataFrame:
    return NewCustomerPredictor().predict_excel(path, output_path=output_path)
