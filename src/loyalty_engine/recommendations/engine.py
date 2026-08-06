from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from loyalty_engine.config import PATHS
from loyalty_engine.io.persistence import ensure_dir, load_joblib, read_csv, write_csv
from loyalty_engine.models.artifacts import ModelBundle
from loyalty_engine.models.segmentation import SEGMENTATION_FEATURES, predict_customer_segment

logger = logging.getLogger(__name__)


@dataclass
class RecommendationEngine:
    bundle: ModelBundle | None
    recommendation_bank: pd.DataFrame
    customer_features: pd.DataFrame | None = None
    segmentation_bundle: Any | None = None
    cluster_profiles: pd.DataFrame | None = None
    cluster_statistics: pd.DataFrame | None = None
    model_path: Path | None = None
    cluster_profiles_path: Path | None = None
    cluster_statistics_path: Path | None = None
    reward_catalog_: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.customer_features is None:
            self.customer_features = self._load_customer_features()
        if self.segmentation_bundle is None:
            self.segmentation_bundle = self._load_segmentation_bundle()
        if self.cluster_profiles is None:
            self.cluster_profiles = self._load_cluster_profiles()
        if self.cluster_statistics is None:
            self.cluster_statistics = self._load_cluster_statistics()

    def predict_customer_states(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.bundle is not None and hasattr(self.bundle, "churn_model") and hasattr(self.bundle, "health_model"):
            churn_pred = self.bundle.churn_model.predict(features[self.bundle.feature_columns])
            health_pred = self.bundle.health_model.predict(features[self.bundle.feature_columns])
            output = features[["Customer_ID"]].copy()
            output["predicted_churn_risk"] = churn_pred
            output["predicted_customer_health"] = health_pred
            return output

        output = features[["Customer_ID"]].copy()
        output["predicted_churn_risk"] = ["Low"] * len(output)
        output["predicted_customer_health"] = ["Stable"] * len(output)
        return output

    def generate_recommendations(self, scored_customers: pd.DataFrame) -> pd.DataFrame:
        recommendations = scored_customers.copy()
        context = self._prepare_customer_context(recommendations)
        recommendations["recommended_action"] = recommendations.apply(
            self._rule_based_action, axis=1
        )
        recommendations["customer_message"] = recommendations["recommended_action"].map(
            self._message_from_bank
        )

        hybrid_rows: list[dict[str, Any]] = []
        for _, row in context.iterrows():
            hybrid_rows.append(self._build_recommendation_payload(row))

        hybrid_df = pd.DataFrame(hybrid_rows)
        if not hybrid_df.empty:
            for col in ["Top_3_Rewards", "Scores", "Business_Insight", "Expected_Impact"]:
                if col in hybrid_df.columns:
                    recommendations[col] = hybrid_df[col].values
            if "Cluster" in hybrid_df.columns:
                recommendations["Cluster"] = hybrid_df["Cluster"].values
            if "Persona" in hybrid_df.columns:
                recommendations["Persona"] = hybrid_df["Persona"].values

        return recommendations

    def _rule_based_action(self, row: pd.Series) -> str:
        churn = str(row.get("predicted_churn_risk", "")).lower()
        health = str(row.get("predicted_customer_health", "")).lower()

        if "high" in churn or "critical" in health:
            return "retention_offer"
        if "medium" in churn or "needs" in health:
            return "loyalty_nudge"
        return "premium_upgrade"

    def _message_from_bank(self, action: str) -> str:
        bank = self.recommendation_bank.copy()
        if bank.empty:
            return {
                "retention_offer": "We are prioritizing a tailored retention offer to protect engagement.",
                "loyalty_nudge": "We are nudging the customer with a personalized loyalty experience.",
                "premium_upgrade": "We are positioning an upgrade path aligned to spend and engagement.",
            }.get(action, "A personalized action has been selected for this customer.")

        action_text = action.replace("_", " ").lower()
        for column in ["Business_Issue", "Recommended_Action", "Customer_Insight"]:
            if column not in bank.columns:
                continue
            matches = bank[bank[column].astype(str).str.lower().str.contains(action_text, na=False)]
            if not matches.empty:
                if "Customer_Message" in matches.columns:
                    return matches.iloc[0]["Customer_Message"]
                return matches.iloc[0].to_dict()
        return {
            "retention_offer": "We are prioritizing a tailored retention offer to protect engagement.",
            "loyalty_nudge": "We are nudging the customer with a personalized loyalty experience.",
            "premium_upgrade": "We are positioning an upgrade path aligned to spend and engagement.",
        }.get(action, "A personalized action has been selected for this customer.")

    def recommend_existing_customer(self, customer_id: Any) -> dict[str, Any]:
        context = self._prepare_customer_context(self.customer_features)
        match = context[context["Customer_ID"].astype(str) == str(customer_id)]
        if match.empty:
            return {"customer_id": str(customer_id), "recommendations": []}
        return self._build_recommendation_payload(match.iloc[0])

    def recommend_customer(self, profile_dataframe: pd.DataFrame) -> dict[str, Any]:
        context = self._prepare_customer_context(profile_dataframe)
        return {
            "customers": [self._build_recommendation_payload(row) for _, row in context.iterrows()]
        }

    def recommend_cluster(self, cluster_id: int) -> dict[str, Any]:
        context = self._prepare_customer_context(self.customer_features)
        cluster_rows = context[context["cluster_id"] == int(cluster_id)]
        if cluster_rows.empty:
            return {"cluster_id": int(cluster_id), "recommendations": []}
        representative = cluster_rows.iloc[0]
        payload = self._build_recommendation_payload(representative)
        payload["cluster_size"] = int(len(cluster_rows))
        return payload

    def populate_ai_recommendations(self, output_path: Path | None = None) -> pd.DataFrame:
        context = self._prepare_customer_context(self.customer_features)
        rows: list[dict[str, Any]] = []
        for _, row in context.iterrows():
            payload = self._build_recommendation_payload(row)
            payload["Customer_Message"] = self._build_customer_message(payload)
            rows.append(payload)

        recommendations_df = pd.DataFrame(rows)
        recommendations_df = self._recommendation_export_frame(recommendations_df)
        if output_path is None:
            output_path = PATHS.processed_dir / "AI_Recommendations.csv"
        write_csv(recommendations_df, output_path)
        return recommendations_df

    def save_outputs(self, output_dir: Path | None = None) -> dict[str, Path]:
        if output_dir is None:
            output_dir = PATHS.root / "outputs" / "recommendations"
        ensure_dir(output_dir)
        catalog_path = output_dir / "reward_catalog.csv"
        recommendations_path = output_dir / "recommendations.csv"
        ai_recommendations_path = PATHS.processed_dir / "AI_Recommendations.csv"

        write_csv(self._reward_catalog(), catalog_path)
        recommendations_df = self.populate_ai_recommendations(output_path=ai_recommendations_path)
        write_csv(recommendations_df, recommendations_path)
        return {
            "reward_catalog": catalog_path,
            "recommendations": recommendations_path,
            "ai_recommendations": ai_recommendations_path,
        }

    def _prepare_customer_context(self, profiles: pd.DataFrame | None) -> pd.DataFrame:
        if profiles is None or profiles.empty:
            base = self.customer_features.copy() if self.customer_features is not None else pd.DataFrame()
        else:
            base = profiles.copy()

        if base.empty:
            return pd.DataFrame()

        if "Customer_ID" not in base.columns and self.customer_features is not None and "Customer_ID" in self.customer_features.columns:
            base = base.copy()
            base["Customer_ID"] = self.customer_features.iloc[: len(base)]["Customer_ID"].values

        if "Customer_ID" not in base.columns:
            base["Customer_ID"] = [f"cust_{idx}" for idx in range(len(base))]

        if self.customer_features is not None and "Customer_ID" in self.customer_features.columns and "Customer_ID" in base.columns:
            merged = base.merge(
                self.customer_features[[col for col in self.customer_features.columns if col in base.columns or col in {"Customer_ID", "Membership_Tier", "favorite_category", "favorite_brand", "customer_lifetime_value", "monetary", "purchase_frequency", "reward_utilization_pct", "customer_engagement_score", "online_purchase_ratio", "category_diversity", "customer_lifetime_value", "Churn_Risk", "frequency", "recency"}]],
                on="Customer_ID",
                how="left",
                suffixes=("", "_feature"),
            )
            merged = merged.loc[:, ~merged.columns.duplicated()]
            base = merged

        if self.segmentation_bundle is not None and (
            "cluster_id" not in base.columns or "persona" not in base.columns
        ):
            required_columns = [col for col in SEGMENTATION_FEATURES if col in base.columns]
            if len(required_columns) == len(SEGMENTATION_FEATURES):
                predicted = predict_customer_segment(base[required_columns], self.segmentation_bundle)
                if isinstance(predicted, dict):
                    predicted_df = pd.DataFrame([predicted])
                else:
                    predicted_df = predicted.copy()
                for col in ["cluster_id", "distance_to_centroid", "nearest_centroid", "persona"]:
                    if col in predicted_df.columns:
                        base[col] = predicted_df[col].values
                if "cluster_statistics" in predicted_df.columns:
                    base["cluster_statistics"] = predicted_df["cluster_statistics"].values

        if self.cluster_profiles is not None and "cluster_id" in base.columns:
            cluster_map = self.cluster_profiles[["cluster_id", "persona"]].dropna().drop_duplicates("cluster_id")
            if not cluster_map.empty:
                base["Persona"] = base["cluster_id"].map(cluster_map.set_index("cluster_id")["persona"])
            else:
                base["Persona"] = base["persona"].fillna("Unknown")
            base["Cluster"] = base["cluster_id"].astype(int)
        else:
            base["Persona"] = base.get("persona", pd.Series(["Unknown"] * len(base))).fillna("Unknown")
            base["Cluster"] = base.get("cluster_id", pd.Series([-1] * len(base))).astype(int)

        return base

    def _build_recommendation_payload(self, row: pd.Series | dict[str, Any]) -> dict[str, Any]:
        if isinstance(row, dict):
            payload = row
        else:
            payload = row.to_dict()

        customer_id = payload.get("Customer_ID", "unknown")
        persona = str(payload.get("Persona") or payload.get("persona") or "Unknown")
        cluster_id = payload.get("Cluster")
        if cluster_id is None:
            cluster_id = payload.get("cluster_id")
        cluster_id = int(cluster_id) if cluster_id is not None else -1

        customer_metrics = self._coerce_customer_metrics(payload)
        eligible_rewards = self._filter_rewards(customer_metrics)
        scored_rewards = self._score_rewards(customer_metrics, eligible_rewards)
        top_rewards = scored_rewards.head(3).copy()

        reasons = []
        business_benefits = []
        customer_benefits = []
        for _, reward in top_rewards.iterrows():
            reasons.append(reward["reason"])
            business_benefits.append(reward["business_benefit"])
            customer_benefits.append(reward["customer_benefit"])

        business_insight = self._build_business_insight(customer_metrics, top_rewards)
        return {
            "Customer_ID": customer_id,
            "Cluster": cluster_id,
            "Persona": persona,
            "Top_3_Rewards": top_rewards[["reward_name", "score", "reason", "business_benefit", "customer_benefit"]].to_dict(orient="records"),
            "Scores": [round(float(score), 2) for score in top_rewards["score"]],
            "Business_Insight": business_insight["customer_insight"],
            "Expected_Impact": business_insight["expected_impact"],
            "recommended_action": self._rule_based_action(pd.Series(payload)),
            "customer_message": self._message_from_bank(self._rule_based_action(pd.Series(payload))),
        }

    def _recommendation_export_frame(self, recommendations_df: pd.DataFrame) -> pd.DataFrame:
        export_columns = [
            "Customer_ID",
            "Cluster",
            "Persona",
            "Top_3_Rewards",
            "Scores",
            "Business_Insight",
            "Expected_Impact",
            "Customer_Message",
        ]
        export_df = recommendations_df.copy()
        for column in export_columns:
            if column not in export_df.columns:
                export_df[column] = ""
        return export_df[export_columns]

    def _build_customer_message(self, row: pd.Series | dict[str, Any]) -> str:
        if isinstance(row, pd.Series):
            payload = row.to_dict()
        else:
            payload = dict(row)

        persona = str(payload.get("Persona") or payload.get("persona") or "Customer").strip()
        cluster_value = payload.get("Cluster")
        if cluster_value is None:
            cluster_value = payload.get("cluster_id")

        reward_names = self._extract_reward_names(payload.get("Top_3_Rewards"))
        reward_text = ", ".join(reward_names[:3]) if reward_names else "the recommended rewards"
        cluster_text = ""
        if cluster_value is not None and not pd.isna(cluster_value):
            try:
                cluster_text = f" in cluster {int(cluster_value)}"
            except (TypeError, ValueError):
                cluster_text = f" in cluster {cluster_value}"

        business_insight = str(
            payload.get("Business_Insight")
            or payload.get("Business Insight")
            or ""
        ).strip()
        expected_impact = str(
            payload.get("Expected_Impact")
            or payload.get("Expected Impact")
            or ""
        ).strip()

        parts: list[str] = [f"{persona}{cluster_text}: prioritize {reward_text}."]
        if business_insight:
            parts.append(business_insight.rstrip("."))
        if expected_impact:
            parts.append(f"Expected impact: {expected_impact.rstrip('.')}")

        message = " ".join(part for part in parts if part).strip()
        if not message.endswith("."):
            message += "."
        return message

    def _extract_reward_names(self, raw_rewards: Any) -> list[str]:
        if raw_rewards is None or (isinstance(raw_rewards, float) and pd.isna(raw_rewards)):
            return []

        parsed_rewards: Any = raw_rewards
        if isinstance(raw_rewards, str):
            text = raw_rewards.strip()
            if not text:
                return []
            try:
                parsed_rewards = json.loads(text)
            except json.JSONDecodeError:
                return [text]

        if isinstance(parsed_rewards, dict):
            reward_name = parsed_rewards.get("reward_name") or parsed_rewards.get("name") or parsed_rewards.get("reward")
            return [str(reward_name)] if reward_name else []

        if isinstance(parsed_rewards, list):
            names: list[str] = []
            for item in parsed_rewards:
                if isinstance(item, dict):
                    reward_name = item.get("reward_name") or item.get("name") or item.get("reward")
                    if reward_name:
                        names.append(str(reward_name))
                elif item is not None:
                    text = str(item).strip()
                    if text:
                        names.append(text)
            return names

        text = str(parsed_rewards).strip()
        return [text] if text else []

    def _coerce_customer_metrics(self, payload: dict[str, Any]) -> dict[str, Any]:
        metrics = dict(payload)
        metrics.setdefault("membership_tier", str(metrics.get("Membership_Tier") or "Bronze"))
        metrics.setdefault("reward_utilization", float(metrics.get("reward_utilization_pct") or metrics.get("reward_utilization") or 0.0))
        metrics.setdefault("engagement", float(metrics.get("customer_engagement_score") or metrics.get("customer_engagement") or 0.0))
        metrics.setdefault("online_ratio", float(metrics.get("online_purchase_ratio") or 0.0))
        metrics.setdefault("purchase_frequency", float(metrics.get("purchase_frequency") or 0.0))
        metrics.setdefault("customer_lifetime_value", float(metrics.get("customer_lifetime_value") or metrics.get("average_customer_ltv") or 0.0))
        metrics.setdefault("favorite_category", str(metrics.get("favorite_category") or "General"))
        metrics.setdefault("favorite_brand", str(metrics.get("favorite_brand") or "Unknown"))
        metrics.setdefault("churn_risk", str(metrics.get("Churn_Risk") or metrics.get("churn_risk") or "Low"))
        metrics.setdefault("persona", str(metrics.get("Persona") or metrics.get("persona") or "Unknown"))
        cluster_value = metrics.get("Cluster")
        if cluster_value is None:
            cluster_value = metrics.get("cluster_id")
        if cluster_value is None or pd.isna(cluster_value):
            cluster_id = -1
        else:
            cluster_id = int(cluster_value)
        metrics["cluster_id"] = cluster_id
        metrics.setdefault("spend", float(metrics.get("monetary") or metrics.get("average_spend") or 0.0))
        metrics.setdefault("diversity", float(metrics.get("category_diversity") or 0.0))
        metrics.setdefault("loyalty_score", float(metrics.get("loyalty_score") or 50.0))
        return metrics

    def _filter_rewards(self, customer_metrics: dict[str, Any]) -> pd.DataFrame:
        catalog = self._reward_catalog()
        eligible: list[dict[str, Any]] = []
        persona = str(customer_metrics.get("persona") or "Unknown")
        membership = str(customer_metrics.get("membership_tier") or "Bronze").lower()
        spend = float(customer_metrics.get("spend") or 0.0)
        reward_util = float(customer_metrics.get("reward_utilization") or 0.0)
        engagement = float(customer_metrics.get("engagement") or 0.0)
        online_ratio = float(customer_metrics.get("online_ratio") or 0.0)
        purchase_frequency = float(customer_metrics.get("purchase_frequency") or 0.0)
        ltv = float(customer_metrics.get("customer_lifetime_value") or 0.0)
        churn = str(customer_metrics.get("churn_risk") or "Low").lower()
        favorite_category = str(customer_metrics.get("favorite_category") or "General").lower()
        cluster_id = int(customer_metrics.get("cluster_id") or -1)

        for _, reward in catalog.iterrows():
            tags = [x.lower() for x in str(reward["eligibility_tags"]).split(",") if x]
            text = " ".join([reward["reward_name"].lower(), reward["category"].lower(), reward["description"].lower()])
            allowed = True
            if persona in {"Premium Traveler", "Luxury Lifestyle"} and reward["business_priority"] < 5 and reward["category"] in {"Travel Cashback", "Air Miles", "Airport Lounge Access", "Hotel Discounts", "Lifestyle Offers"}:
                allowed = True
            if persona == "Digital Explorer" and any(tag in text for tag in ["digital", "shopping", "electronics", "dining"]):
                allowed = True
            if persona == "Value Shopper" and any(tag in text for tag in ["cashback", "points", "voucher", "fuel", "dining"]):
                allowed = True
            if persona == "Dormant Customer" and any(tag in text for tag in ["cashback", "points", "travel", "lifestyle"]):
                allowed = True
            if persona == "Loyal Cashback User" and any(tag in text for tag in ["cashback", "points", "fuel", "voucher"]):
                allowed = True

            if any(tag in {"premium", "travel", "hotel", "electronics", "lounge"} for tag in tags) and membership not in {"gold", "platinum"} and spend < 300000:
                allowed = False

            if "digital" in tags and online_ratio < 0.4 and persona != "Digital Explorer":
                allowed = False

            if "cashback" in tags and reward_util < 20 and persona not in {"Value Shopper", "Loyal Cashback User"}:
                allowed = False

            if "points" in tags and reward_util < 15 and persona not in {"Loyal Cashback User", "Value Shopper"}:
                allowed = False

            if favorite_category and favorite_category in {"audio", "electronics", "travel", "dining", "shopping"}:
                if any(tag in text for tag in [favorite_category, "shopping", "dining", "travel", "electronics"]):
                    pass
                elif persona in {"Dormant Customer"}:
                    allowed = True

            if cluster_id == 1 and reward["business_priority"] >= 8:
                allowed = True
            if cluster_id == 2 and reward["category"] in {"Cashback", "Reward Points", "Fuel Cashback", "Dining Offers"}:
                allowed = True
            if cluster_id == 0 and reward["category"] in {"Shopping Vouchers", "Dining Offers", "Electronics Discounts"}:
                allowed = True

            if churn in {"high", "critical"} and reward["business_priority"] < 4:
                allowed = False

            if allowed:
                eligible.append(reward.to_dict())

        return pd.DataFrame(eligible)

    def _score_rewards(self, customer_metrics: dict[str, Any], eligible_rewards: pd.DataFrame) -> pd.DataFrame:
        """Score eligible rewards using a weighted hybrid formula.

        The formula combines persona fit, category alignment, reward-utilization fit,
        engagement level, membership tier fit, business priority, expected ROI,
        loyalty score, and cluster context. The resulting score is normalized to 0-100.
        """
        if eligible_rewards.empty:
            return pd.DataFrame(columns=["reward_name", "score", "reason", "business_benefit", "customer_benefit"])

        persona = str(customer_metrics.get("persona") or "Unknown")
        membership = str(customer_metrics.get("membership_tier") or "Bronze").lower()
        reward_util = float(customer_metrics.get("reward_utilization") or 0.0)
        engagement = float(customer_metrics.get("engagement") or 0.0)
        online_ratio = float(customer_metrics.get("online_ratio") or 0.0)
        purchase_frequency = float(customer_metrics.get("purchase_frequency") or 0.0)
        ltv = float(customer_metrics.get("customer_lifetime_value") or 0.0)
        spend = float(customer_metrics.get("spend") or 0.0)
        loyalty_score = float(customer_metrics.get("loyalty_score") or 50.0)
        favorite_category = str(customer_metrics.get("favorite_category") or "General").lower()
        cluster_id = int(customer_metrics.get("cluster_id") or -1)

        scored_rows: list[dict[str, Any]] = []
        for _, reward in eligible_rewards.iterrows():
            reward_name = reward["reward_name"]
            category = reward["category"]
            tags = [x.lower() for x in str(reward["eligibility_tags"]).split(",") if x]
            text = " ".join([reward_name.lower(), category.lower(), reward["description"].lower()])

            persona_match = self._persona_score(persona, reward, text)
            category_match = 1.0 if favorite_category and favorite_category in text else 0.6
            reward_util_factor = min(1.0, reward_util / 100.0)
            engagement_factor = min(1.0, engagement / 100.0)
            membership_factor = 1.0 if membership in {"gold", "platinum"} and any(tag in text for tag in ["travel", "premium", "electronics", "lounge"]) else 0.7
            business_priority_factor = min(1.0, reward["business_priority"] / 10.0)
            roi_factor = min(1.0, float(reward["expected_roi"]) / 0.2)
            loyalty_factor = min(1.0, loyalty_score / 100.0)
            cluster_factor = 1.0 if cluster_id == 1 and any(tag in text for tag in ["travel", "lifestyle", "premium", "electronics"]) else 0.8 if cluster_id == 2 and any(tag in text for tag in ["cashback", "points", "voucher", "fuel"]) else 0.75

            score = (
                0.22 * persona_match
                + 0.15 * category_match
                + 0.12 * reward_util_factor
                + 0.10 * engagement_factor
                + 0.10 * membership_factor
                + 0.12 * business_priority_factor
                + 0.10 * roi_factor
                + 0.09 * loyalty_factor
                + 0.10 * cluster_factor
            ) * 100.0

            score = max(0.0, min(100.0, round(float(score), 2)))
            reason = self._reward_reason(persona, reward, customer_metrics)
            business_benefit = self._business_benefit(reward, customer_metrics)
            customer_benefit = self._customer_benefit(reward, customer_metrics)
            scored_rows.append(
                {
                    "reward_id": reward["reward_id"],
                    "reward_name": reward["reward_name"],
                    "category": reward["category"],
                    "score": score,
                    "reason": reason,
                    "business_benefit": business_benefit,
                    "customer_benefit": customer_benefit,
                }
            )

        scored_df = pd.DataFrame(scored_rows)
        return scored_df.sort_values(["score", "reward_name"], ascending=[False, True]).reset_index(drop=True)

    def _persona_score(self, persona: str, reward: pd.Series, text: str) -> float:
        persona = persona.lower()
        if persona in {"premium traveler", "luxury lifestyle"}:
            return 1.0 if any(term in text for term in ["travel", "hotel", "lounge", "premium", "electronics", "lifestyle"]) else 0.55
        if persona == "digital explorer":
            return 1.0 if any(term in text for term in ["digital", "shopping", "electronics", "dining", "cashback"]) else 0.6
        if persona == "value shopper":
            return 1.0 if any(term in text for term in ["cashback", "points", "voucher", "fuel", "dining"]) else 0.6
        if persona == "dormant customer":
            return 1.0 if any(term in text for term in ["cashback", "points", "travel", "lifestyle"]) else 0.5
        if persona == "loyal cashback user":
            return 1.0 if any(term in text for term in ["cashback", "points", "voucher", "fuel"]) else 0.6
        return 0.7

    def _reward_reason(self, persona: str, reward: pd.Series, customer_metrics: dict[str, Any]) -> str:
        reward_name = reward["reward_name"]
        persona = str(persona or "Unknown")
        favorite_category = str(customer_metrics.get("favorite_category") or "General")
        return (
            f"{persona} customers respond well to {reward_name.lower()} because the offer is aligned to their {favorite_category.lower()} preference and reward behavior."
        )

    def _business_benefit(self, reward: pd.Series, customer_metrics: dict[str, Any]) -> str:
        churn = str(customer_metrics.get("churn_risk") or "Low")
        if churn.lower() in {"high", "critical"}:
            return "Protects retention by strengthening engagement and reducing churn exposure."
        return "Improves loyalty performance and encourages repeat purchases."

    def _customer_benefit(self, reward: pd.Series, customer_metrics: dict[str, Any]) -> str:
        reward_name = reward["reward_name"]
        return f"Helps the customer receive more value from their purchases through {reward_name.lower()}."

    def _build_business_insight(self, customer_metrics: dict[str, Any], top_rewards: pd.DataFrame) -> dict[str, Any]:
        persona = str(customer_metrics.get("persona") or "Unknown")
        cluster_value = customer_metrics.get("cluster_id")
        cluster_id = int(cluster_value) if cluster_value is not None and not pd.isna(cluster_value) else -1
        churn = str(customer_metrics.get("churn_risk") or "Low")
        spend = float(customer_metrics.get("spend") or 0.0)
        top_names = ", ".join(top_rewards["reward_name"].tolist()) if not top_rewards.empty else "value-based offers"
        return {
            "customer_insight": f"{persona} customers in cluster {cluster_id} show a strong fit for {top_names} based on spend, engagement, and reward behavior.",
            "business_opportunity": f"Use {top_names} to lift conversion and deepen loyalty for this segment.",
            "retention_strategy": f"Pair {top_names} with targeted nudges to sustain activity for {churn.lower()} churn risk customers.",
            "upsell_opportunity": f"Prioritize premium or lifestyle rewards when spend exceeds {spend:,.0f} and engagement is strong.",
            "expected_impact": "Moderate uplift in engagement and repeat purchase intent with a clear ROI path.",
        }

    def _reward_catalog(self) -> pd.DataFrame:
        if self.reward_catalog_ is not None:
            return self.reward_catalog_.copy()

        catalog = pd.DataFrame(
            [
                {
                    "reward_id": "R001",
                    "reward_name": "Cashback",
                    "category": "Cashback",
                    "description": "Direct cashback on eligible purchases.",
                    "eligibility_tags": "cashback,value,retention",
                    "business_priority": 8,
                    "cost_score": 4,
                    "expected_roi": 0.16,
                },
                {
                    "reward_id": "R002",
                    "reward_name": "Reward Points",
                    "category": "Reward Points",
                    "description": "Bonus loyalty points for repeat purchases.",
                    "eligibility_tags": "points,loyalty,value",
                    "business_priority": 7,
                    "cost_score": 5,
                    "expected_roi": 0.14,
                },
                {
                    "reward_id": "R003",
                    "reward_name": "Air Miles",
                    "category": "Air Miles",
                    "description": "Travel-oriented miles for frequent travelers.",
                    "eligibility_tags": "travel,premium,loyalty",
                    "business_priority": 6,
                    "cost_score": 6,
                    "expected_roi": 0.12,
                },
                {
                    "reward_id": "R004",
                    "reward_name": "Airport Lounge Access",
                    "category": "Travel",
                    "description": "Priority lounge access for high-value travelers.",
                    "eligibility_tags": "travel,premium,lounge",
                    "business_priority": 9,
                    "cost_score": 7,
                    "expected_roi": 0.18,
                },
                {
                    "reward_id": "R005",
                    "reward_name": "Fuel Cashback",
                    "category": "Fuel Cashback",
                    "description": "Fuel-specific cashback for frequent drivers.",
                    "eligibility_tags": "cashback,fuel,value",
                    "business_priority": 5,
                    "cost_score": 4,
                    "expected_roi": 0.13,
                },
                {
                    "reward_id": "R006",
                    "reward_name": "Dining Offers",
                    "category": "Dining",
                    "description": "Dining-related promotions and discounts.",
                    "eligibility_tags": "dining,engagement,shopping",
                    "business_priority": 6,
                    "cost_score": 3,
                    "expected_roi": 0.11,
                },
                {
                    "reward_id": "R007",
                    "reward_name": "Shopping Vouchers",
                    "category": "Shopping",
                    "description": "Vouchers for retail and online shopping.",
                    "eligibility_tags": "shopping,retention,value",
                    "business_priority": 7,
                    "cost_score": 4,
                    "expected_roi": 0.15,
                },
                {
                    "reward_id": "R008",
                    "reward_name": "Lifestyle Offers",
                    "category": "Lifestyle",
                    "description": "Curated lifestyle rewards for premium customers.",
                    "eligibility_tags": "lifestyle,premium,engagement",
                    "business_priority": 8,
                    "cost_score": 6,
                    "expected_roi": 0.17,
                },
                {
                    "reward_id": "R009",
                    "reward_name": "Travel Cashback",
                    "category": "Travel Cashback",
                    "description": "Travel-related cashback for high-intent travelers.",
                    "eligibility_tags": "travel,cashback,premium",
                    "business_priority": 7,
                    "cost_score": 5,
                    "expected_roi": 0.15,
                },
                {
                    "reward_id": "R010",
                    "reward_name": "Hotel Discounts",
                    "category": "Hotel",
                    "description": "Discounted hotel stays for premium travelers.",
                    "eligibility_tags": "travel,premium,hotel",
                    "business_priority": 8,
                    "cost_score": 6,
                    "expected_roi": 0.17,
                },
                {
                    "reward_id": "R011",
                    "reward_name": "Electronics Discounts",
                    "category": "Electronics",
                    "description": "Discounts on electronics and smart devices.",
                    "eligibility_tags": "electronics,digital,premium",
                    "business_priority": 7,
                    "cost_score": 5,
                    "expected_roi": 0.16,
                },
            ]
        )
        self.reward_catalog_ = catalog
        return catalog.copy()

    def _load_customer_features(self) -> pd.DataFrame:
        return read_csv(PATHS.customer_features_path, default=pd.DataFrame())

    def _load_segmentation_bundle(self) -> Any | None:
        path = self.model_path or PATHS.kmeans_model_path
        return load_joblib(path)

    def _load_cluster_profiles(self) -> pd.DataFrame | None:
        path = self.cluster_profiles_path or PATHS.cluster_profiles_path
        return read_csv(path)

    def _load_cluster_statistics(self) -> pd.DataFrame | None:
        path = self.cluster_statistics_path or PATHS.cluster_statistics_path
        return read_csv(path)
