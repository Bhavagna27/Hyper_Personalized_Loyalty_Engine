"""Customer Segmentation module using K-Means clustering.

This module determines optimal K (evaluating K=2 to 10 via Elbow method and
Silhouette score), fits a K-Means model, profiles clusters, dynamically assigns
personas based on cluster statistics, and provides reusable prediction functions.

Features used for clustering (10 continuous numerical features):
- recency
- frequency
- monetary
- reward_utilization_pct
- customer_engagement_score
- online_purchase_ratio
- spending_diversity_score
- category_diversity
- average_days_between_purchases
- weekend_shopping_ratio
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


SEGMENTATION_FEATURES: list[str] = [
    "recency",
    "frequency",
    "monetary",
    "reward_utilization_pct",
    "customer_engagement_score",
    "online_purchase_ratio",
    "spending_diversity_score",
    "category_diversity",
    "average_days_between_purchases",
    "weekend_shopping_ratio",
]


# ---------------------------------------------------------------------------
# Optimal K Evaluation Container
# ---------------------------------------------------------------------------

class KEvaluationResult(NamedTuple):
    """Container holding K-evaluation metrics across ranges of K."""

    k_range: list[int]
    inertias: dict[int, float]
    silhouette_scores: dict[int, float]
    optimal_k: int
    justification: str


def evaluate_optimal_k(
    df: pd.DataFrame,
    features: list[str] = SEGMENTATION_FEATURES,
    k_min: int = 2,
    k_max: int = 8,
    random_state: int = 42,
) -> KEvaluationResult:
    """Evaluate K from k_min to k_max using Elbow (Inertia) and Silhouette Score.

    Parameters
    ----------
    df:
        Engineered customer feature table.
    features:
        List of numerical feature names to use.
    k_min:
        Minimum K to test (default: 2).
    k_max:
        Maximum K to test (default: 10).
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    KEvaluationResult
        Dicts of inertias, silhouette_scores, optimal K, and textual justification.
    """
    X_raw = df[features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    k_range = list(range(k_min, k_max + 1))
    inertias: dict[int, float] = {}
    sil_scores: dict[int, float] = {}
    interpretability_scores: dict[int, float] = {}

    logger.info("Evaluating KMeans for K in range [%d, %d]…", k_min, k_max)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = km.fit_predict(X_scaled)
        inertia = float(km.inertia_)
        sil = float(silhouette_score(X_scaled, labels))

        population_stats = {
            "average_spend": float(df["monetary"].mean()),
            "purchase_frequency": float(df["purchase_frequency"].mean()),
            "recency": float(df["recency"].mean()),
            "online_purchase_ratio": float(df["online_purchase_ratio"].mean()),
            "reward_utilization": float(df["reward_utilization_pct"].mean()),
            "customer_engagement": float(df["customer_engagement_score"].mean()),
            "category_diversity": float(df["category_diversity"].mean()),
            "average_ltv": float(df["customer_lifetime_value"].mean()),
        }
        profile_rows, _, _, _ = _profile_clusters_for_evaluation(df, labels, k, population_stats)
        interpretability_scores[k] = _calculate_business_interpretability(profile_rows)

        inertias[k] = inertia
        sil_scores[k] = sil
        logger.debug("K=%d | Inertia=%.2f | Silhouette=%.4f | Interpretability=%.4f", k, inertia, sil, interpretability_scores[k])

    best_sil = max(sil_scores.values())
    top_candidates = [k for k in k_range if abs(sil_scores[k] - best_sil) < 0.05]

    if len(top_candidates) > 1:
        best_k = max(top_candidates, key=lambda k: (interpretability_scores[k], sil_scores[k], -inertias[k]))
    else:
        best_k = max(sil_scores, key=sil_scores.get)  # type: ignore[arg-type]

    best_sil = sil_scores[best_k]

    justification = (
        f"K={best_k} offered the strongest balance of clustering quality and business interpretability. "
        f"Its Silhouette Score of {best_sil:.4f} was within 0.05 of the best observed values, and "
        f"its persona-based separation scored {interpretability_scores[best_k]:.4f} across spend, frequency, reward behavior, engagement, online activity, diversity, and CLV."
    )

    logger.info("Optimal K selected: %d (%s)", best_k, justification)

    return KEvaluationResult(
        k_range=k_range,
        inertias=inertias,
        silhouette_scores=sil_scores,
        optimal_k=best_k,
        justification=justification,
    )


# ---------------------------------------------------------------------------
# Persona Assignment Engine (Data-Driven)
# ---------------------------------------------------------------------------

def _mode_str(series: pd.Series, default: str = "Unknown") -> str:
    """Return top mode of a pandas Series as string."""
    non_null = series.dropna()
    if non_null.empty:
        return default
    m = non_null.mode()
    return str(m.iloc[0]) if not m.empty else default


def _relative_ratio(value: float, baseline: float) -> float:
    """Return a ratio relative to the population baseline, guarding against zero denominators."""
    if baseline in (None, 0):
        return 1.0 if value > 0 else 0.0
    return float(value / baseline)


def _churn_safety_score(churn_risk: str | None) -> float:
    """Map churn risk labels to a normalized safety score where higher is better."""
    risk = str(churn_risk or "").strip().lower()
    if risk in {"low", "healthy"}:
        return 1.0
    if risk in {"medium", "stable"}:
        return 0.6
    return 0.2


def generate_cluster_persona(
    cluster_stats: dict[str, Any],
    pop_stats: dict[str, Any],
    used_personas: set[str] | None = None,
) -> tuple[str, str, str]:
    """Dynamically assign a descriptive persona name, business summary, and rationale from cluster stats.

    Parameters
    ----------
    cluster_stats:
        Mean/mode stats for the target cluster.
    pop_stats:
        Mean/mode stats across the whole customer population.

    Returns
    -------
    tuple[str, str]
        (Persona Title, Business Summary Text)
    """
    spend = cluster_stats.get("average_spend", 0.0)
    pop_spend = pop_stats.get("average_spend", 1.0)

    recency = cluster_stats.get("recency", 0.0)
    pop_recency = pop_stats.get("recency", 1.0)

    online_ratio = cluster_stats.get("online_purchase_ratio", 0.5)
    pop_online = pop_stats.get("online_purchase_ratio", 0.5)

    reward_util = cluster_stats.get("reward_utilization", 0.0)
    pop_reward_util = pop_stats.get("reward_utilization", 1.0)

    engagement = cluster_stats.get("customer_engagement", 50.0)
    pop_engagement = pop_stats.get("customer_engagement", 50.0)

    diversity = cluster_stats.get("category_diversity", 1.0)
    pop_diversity = pop_stats.get("category_diversity", 1.0)

    purchase_frequency = cluster_stats.get("purchase_frequency", 0.0)
    pop_purchase_frequency = pop_stats.get("purchase_frequency", 0.0)

    ltv = cluster_stats.get("average_ltv", 0.0)
    pop_ltv = pop_stats.get("average_ltv", 1.0)

    dominant_tier = str(cluster_stats.get("dominant_membership_tier", "")).strip()
    dominant_churn = str(cluster_stats.get("dominant_churn_risk", "")).strip()

    spend_score = _relative_ratio(spend, pop_spend)
    frequency_score = _relative_ratio(purchase_frequency, pop_purchase_frequency)
    reward_score = _relative_ratio(reward_util, pop_reward_util)
    engagement_score = _relative_ratio(engagement, pop_engagement)
    online_score = _relative_ratio(online_ratio, pop_online)
    diversity_score = _relative_ratio(diversity, pop_diversity)
    ltv_score = _relative_ratio(ltv, pop_ltv)
    recency_score = _relative_ratio(recency, pop_recency)
    churn_safety = _churn_safety_score(dominant_churn)

    persona_scores = {
        "Premium Traveler": 0.35 * spend_score + 0.2 * engagement_score + 0.2 * ltv_score + 0.1 * frequency_score + 0.1 * diversity_score + 0.05 * churn_safety,
        "Luxury Lifestyle": 0.4 * spend_score + 0.25 * ltv_score + 0.15 * engagement_score + 0.1 * diversity_score + 0.05 * reward_score + 0.05 * churn_safety,
        "Digital Explorer": 0.35 * online_score + 0.25 * engagement_score + 0.2 * frequency_score + 0.1 * diversity_score + 0.1 * spend_score,
        "Value Shopper": 0.35 * reward_score + 0.25 * frequency_score + 0.2 * spend_score + 0.1 * engagement_score + 0.1 * diversity_score,
        "Dormant Customer": 0.35 * recency_score + 0.25 * max(0.0, 1.0 - engagement_score) + 0.2 * max(0.0, 1.0 - frequency_score) + 0.2 * max(0.0, 1.0 - churn_safety),
        "Loyal Cashback User": 0.35 * reward_score + 0.25 * frequency_score + 0.2 * spend_score + 0.1 * online_score + 0.1 * engagement_score,
    }

    ranked_personas = sorted(persona_scores.items(), key=lambda item: item[1], reverse=True)
    used_personas = used_personas or set()

    chosen_persona = None
    for persona_name, _ in ranked_personas:
        if persona_name not in used_personas:
            chosen_persona = persona_name
            break

    if chosen_persona is None:
        chosen_persona = "Balanced Growth Segment"

    if chosen_persona == "Premium Traveler":
        summary = (
            f"This cluster combines above-average spend (${spend:,.2f}), strong engagement ({engagement:.1f}/100), and elevated CLV (${ltv:,.2f}) with low churn risk. "
            f"It looks like a premium, loyalty-ready customer segment centered on {cluster_stats.get('favorite_category')} and {cluster_stats.get('favorite_brand')}"
        )
    elif chosen_persona == "Luxury Lifestyle":
        summary = (
            f"This cluster is the highest-value group on spend and CLV, with strong diversification and premium-tier membership patterns. "
            f"Its high engagement ({engagement:.1f}/100) and low churn risk make it a strong fit for curated, high-end offers."
        )
    elif chosen_persona == "Digital Explorer":
        summary = (
            f"This cluster shows a digital-first pattern with {online_ratio * 100:.1f}% of purchases made online or through the app and strong engagement ({engagement:.1f}/100). "
            f"Its behavior suggests a customer who responds well to digital journeys and app-led promotions."
        )
    elif chosen_persona == "Value Shopper":
        summary = (
            f"This cluster is driven by reward redemption and shopping frequency rather than raw spend alone. "
            f"With reward utilization at {reward_util:.1f}% and moderate-to-high purchase frequency, it is well-suited to value-focused incentives."
        )
    elif chosen_persona == "Dormant Customer":
        summary = (
            f"This cluster is characterized by slower recent activity, lower engagement ({engagement:.1f}/100), and higher churn risk. "
            f"Its recency profile indicates it needs reactivation offers and tailored retention messaging."
        )
    elif chosen_persona == "Loyal Cashback User":
        summary = (
            f"This cluster demonstrates sustained shopping behavior with frequent purchases and strong reward utilization ({reward_util:.1f}%). "
            f"It is primed for cash-back, loyalty acceleration, and repeat-purchase campaigns."
        )
    else:
        summary = (
            f"This cluster sits near the middle of the customer base on spend, engagement, and reward behavior. "
            f"It offers a steady opportunity for expansion and retention campaigns."
        )

    explanation = (
        f"Assigned as {chosen_persona} because the cluster is above the population baseline on "
        f"spend ({spend_score:.2f}x), frequency ({frequency_score:.2f}x), reward utilization ({reward_score:.2f}x), engagement ({engagement_score:.2f}x), "
        f"online ratio ({online_score:.2f}x), diversity ({diversity_score:.2f}x), and CLV ({ltv_score:.2f}x), while also showing {dominant_churn} churn risk and {dominant_tier} membership."
    )

    return chosen_persona, summary, explanation


def _calculate_business_interpretability(profile_rows: list[dict[str, Any]]) -> float:
    """Score how interpretable a segmentation result is from a business perspective."""
    if not profile_rows:
        return 0.0

    stats_df = pd.DataFrame(profile_rows)
    feature_cols = [
        "average_spend",
        "purchase_frequency",
        "reward_utilization_pct",
        "customer_engagement_score",
        "online_purchase_ratio",
        "category_diversity",
        "average_customer_ltv",
    ]
    if not set(feature_cols).issubset(stats_df.columns):
        return 0.0

    standardized = stats_df[feature_cols].astype(float)
    pop_means = standardized.mean()
    pop_scales = standardized.std().replace(0, 1.0)
    separation = np.mean(np.abs(standardized - pop_means).div(pop_scales, axis=1).mean(axis=1))
    persona_diversity = len({row.get("persona", "") for row in profile_rows}) / max(1, len(profile_rows))
    return float(0.7 * separation + 0.3 * persona_diversity)


def _profile_clusters_for_evaluation(
    df: pd.DataFrame,
    labels: np.ndarray,
    n_clusters: int,
    pop_stats: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, str], dict[int, str]]:
    """Generate cluster profile rows for evaluation and training without changing the public API."""
    profile_records: list[dict[str, Any]] = []
    stats_records: list[dict[str, Any]] = []
    personas_dict: dict[int, str] = {}
    explanations_dict: dict[int, str] = {}
    used_personas: set[str] = set()

    for cid in range(n_clusters):
        c_df = df.loc[labels == cid]
        size = len(c_df)

        avg_spend = float(c_df["monetary"].mean())
        purchase_freq = float(c_df["purchase_frequency"].mean())
        aov = float(c_df["average_order_value"].mean())
        recency = float(c_df["recency"].mean())
        fav_cat = _mode_str(c_df["favorite_category"])
        fav_brand = _mode_str(c_df["favorite_brand"])
        reward_util = float(c_df["reward_utilization_pct"].mean())
        engagement = float(c_df["customer_engagement_score"].mean())
        online_ratio = float(c_df["online_purchase_ratio"].mean())
        cat_diversity = float(c_df["category_diversity"].mean())
        avg_ltv = float(c_df["customer_lifetime_value"].mean())

        top_tier = _mode_str(c_df["Membership_Tier"])
        tier_dist = c_df["Membership_Tier"].value_counts(normalize=True).to_dict()
        tier_dist_str = ", ".join([f"{k}: {v * 100:.1f}%" for k, v in tier_dist.items()])

        top_churn = _mode_str(c_df["Churn_Risk"])
        churn_dist = c_df["Churn_Risk"].value_counts(normalize=True).to_dict()
        churn_dist_str = ", ".join([f"{k}: {v * 100:.1f}%" for k, v in churn_dist.items()])

        c_stat_summary = {
            "average_spend": avg_spend,
            "purchase_frequency": purchase_freq,
            "average_order_value": aov,
            "recency": recency,
            "favorite_category": fav_cat,
            "favorite_brand": fav_brand,
            "reward_utilization": reward_util,
            "customer_engagement": engagement,
            "online_purchase_ratio": online_ratio,
            "category_diversity": cat_diversity,
            "average_ltv": avg_ltv,
            "dominant_membership_tier": top_tier,
            "dominant_churn_risk": top_churn,
        }

        persona, summary, explanation = generate_cluster_persona(c_stat_summary, pop_stats, used_personas=used_personas)
        used_personas.add(persona)
        personas_dict[cid] = persona
        explanations_dict[cid] = explanation

        profile_records.append({
            "cluster_id": cid,
            "persona": persona,
            "persona_explanation": explanation,
            "cluster_size": size,
            "percentage_of_customers": round((size / len(df)) * 100, 2),
            "average_spend": round(avg_spend, 2),
            "purchase_frequency": round(purchase_freq, 2),
            "average_order_value": round(aov, 2),
            "favorite_category": fav_cat,
            "favorite_brand": fav_brand,
            "reward_utilization_pct": round(reward_util, 2),
            "customer_engagement_score": round(engagement, 2),
            "membership_tier_dominant": top_tier,
            "membership_tier_distribution": tier_dist_str,
            "churn_risk_dominant": top_churn,
            "churn_risk_distribution": churn_dist_str,
            "online_purchase_ratio": round(online_ratio, 4),
            "category_diversity": round(cat_diversity, 2),
            "average_customer_ltv": round(avg_ltv, 2),
            "business_summary": summary,
        })

        stats_records.append({
            "cluster_id": cid,
            "cluster_size": size,
            "recency_mean": round(recency, 2),
            "frequency_mean": round(float(c_df["frequency"].mean()), 2),
            "monetary_mean": round(avg_spend, 2),
            "reward_utilization_pct_mean": round(reward_util, 2),
            "customer_engagement_score_mean": round(engagement, 2),
            "online_purchase_ratio_mean": round(online_ratio, 4),
            "spending_diversity_score_mean": round(float(c_df["spending_diversity_score"].mean()), 4),
            "category_diversity_mean": round(cat_diversity, 2),
            "average_days_between_purchases_mean": round(float(c_df["average_days_between_purchases"].mean()), 2),
            "weekend_shopping_ratio_mean": round(float(c_df["weekend_shopping_ratio"].mean()), 4),
            "customer_ltv_mean": round(avg_ltv, 2),
            "persona": persona,
            "persona_explanation": explanation,
        })

    return profile_records, stats_records, personas_dict, explanations_dict


# ---------------------------------------------------------------------------
# Model Bundle Dataclass
# ---------------------------------------------------------------------------

@dataclass
class SegmentationModelBundle:
    """Artifact bundle holding fitted KMeans model, scaler, and cluster profiles.

    Attributes
    ----------
    kmeans_model:
        Fitted sklearn KMeans model instance.
    scaler:
        Fitted StandardScaler for the 10 segmentation features.
    feature_names:
        List of feature column names used for clustering.
    optimal_k:
        Selected cluster count.
    inertia:
        Final model inertia.
    silhouette_score:
        Final overall silhouette score.
    centroids_scaled:
        Centroid array in scaled space (shape: [K, 10]).
    centroids_raw:
        DataFrame of centroids in unscaled/raw feature units.
    cluster_sizes:
        Mapping of cluster_id -> number of assigned customers.
    cluster_profiles:
        DataFrame containing complete cluster summary metrics and personas.
    cluster_statistics:
        DataFrame containing numerical cluster averages.
    personas:
        Mapping of cluster_id -> assigned persona name.
    """

    kmeans_model: KMeans
    scaler: StandardScaler
    feature_names: list[str]
    optimal_k: int
    inertia: float
    silhouette_score: float
    centroids_scaled: np.ndarray
    centroids_raw: pd.DataFrame
    cluster_sizes: dict[int, int]
    cluster_profiles: pd.DataFrame
    cluster_statistics: pd.DataFrame
    personas: dict[int, str]


# ---------------------------------------------------------------------------
# Core Trainer & Profiler
# ---------------------------------------------------------------------------

def train_kmeans_segmentation(
    customer_features: pd.DataFrame,
    n_clusters: int | None = None,
    features: list[str] = SEGMENTATION_FEATURES,
    random_state: int = 42,
    n_init: int = 20,
) -> tuple[SegmentationModelBundle, pd.DataFrame]:
    """Train the final KMeans clustering model and generate comprehensive cluster profiles.

    Parameters
    ----------
    customer_features:
        Engineered customer feature table.
    n_clusters:
        Number of clusters K. If None, automatically evaluated via `evaluate_optimal_k`.
    features:
        List of feature names to use.
    random_state:
        Seed for reproducibility (default: 42).
    n_init:
        Number of initializations (default: 20).

    Returns
    -------
    tuple[SegmentationModelBundle, pd.DataFrame]
        (Model Bundle, Customer DataFrame with appended 'cluster_id' and 'persona' columns).
    """
    df = customer_features.copy()

    # 1. Evaluate optimal K if not explicitly provided
    if n_clusters is None:
        k_res = evaluate_optimal_k(df, features=features, random_state=random_state)
        n_clusters = k_res.optimal_k

    logger.info("Training final KMeans model with K=%d, random_state=%d, n_init=%d…", n_clusters, random_state, n_init)

    # 2. Fit StandardScaler and KMeans
    X_raw = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = kmeans.fit_predict(X_scaled)

    df["cluster_id"] = labels

    overall_inertia = float(kmeans.inertia_)
    overall_silhouette = float(silhouette_score(X_scaled, labels))

    logger.info("Model fit complete. Inertia: %.2f | Silhouette Score: %.4f", overall_inertia, overall_silhouette)

    # 3. Compute cluster centroids in raw units
    centroids_scaled = kmeans.cluster_centers_
    centroids_raw_arr = scaler.inverse_transform(centroids_scaled)
    centroids_raw_df = pd.DataFrame(centroids_raw_arr, columns=features)
    centroids_raw_df.index.name = "cluster_id"

    # 4. Compute cluster sizes
    sizes_series = pd.Series(labels).value_counts().sort_index()
    cluster_sizes = {int(k): int(v) for k, v in sizes_series.items()}

    # 5. Compute Population Baseline Stats for Persona Assignment
    pop_stats = {
        "average_spend": float(df["monetary"].mean()),
        "purchase_frequency": float(df["purchase_frequency"].mean()),
        "recency": float(df["recency"].mean()),
        "online_purchase_ratio": float(df["online_purchase_ratio"].mean()),
        "reward_utilization": float(df["reward_utilization_pct"].mean()),
        "customer_engagement": float(df["customer_engagement_score"].mean()),
        "category_diversity": float(df["category_diversity"].mean()),
        "average_ltv": float(df["customer_lifetime_value"].mean()),
    }

    # 6. Profile Every Cluster
    profile_records, stats_records, personas_dict, _ = _profile_clusters_for_evaluation(
        df=df,
        labels=df["cluster_id"].to_numpy(),
        n_clusters=n_clusters,
        pop_stats=pop_stats,
    )

    df["persona"] = df["cluster_id"].map(personas_dict)

    cluster_profiles_df = pd.DataFrame(profile_records)
    cluster_statistics_df = pd.DataFrame(stats_records)

    bundle = SegmentationModelBundle(
        kmeans_model=kmeans,
        scaler=scaler,
        feature_names=features,
        optimal_k=n_clusters,
        inertia=overall_inertia,
        silhouette_score=overall_silhouette,
        centroids_scaled=centroids_scaled,
        centroids_raw=centroids_raw_df,
        cluster_sizes=cluster_sizes,
        cluster_profiles=cluster_profiles_df,
        cluster_statistics=cluster_statistics_df,
        personas=personas_dict,
    )

    return bundle, df


# ---------------------------------------------------------------------------
# Reusable Inference Prediction Function (STEP 8)
# ---------------------------------------------------------------------------

def predict_customer_segment(
    customer_input: pd.DataFrame | dict[str, Any],
    bundle: SegmentationModelBundle,
) -> pd.DataFrame | dict[str, Any]:
    """Assign a cluster ID, calculate centroid distance, and attach persona/stats for new customer(s).

    Parameters
    ----------
    customer_input:
        Engineered customer feature table (DataFrame or single dict/Series).
    bundle:
        Fitted SegmentationModelBundle instance.

    Returns
    -------
    pd.DataFrame | dict[str, Any]
        Original input augmented with:
        - `cluster_id`
        - `distance_to_centroid`
        - `nearest_centroid`
        - `persona`
        - `cluster_statistics`
    """
    is_single = isinstance(customer_input, dict)
    if is_single:
        input_df = pd.DataFrame([customer_input])
    else:
        input_df = customer_input.copy()

    # Extract 10 segmentation features
    X_raw = input_df[bundle.feature_names].copy()
    X_scaled = bundle.scaler.transform(X_raw)

    # 1. Predict cluster IDs
    cluster_ids = bundle.kmeans_model.predict(X_scaled)

    # 2. Compute Euclidean distance to each cluster centroid in scaled space
    # transform() returns distance to all centroids shape: [N, K]
    all_distances = bundle.kmeans_model.transform(X_scaled)

    min_distances = np.min(all_distances, axis=1)

    # 3. Nearest centroid coordinates
    nearest_centroids_scaled = bundle.centroids_scaled[cluster_ids]

    results_df = input_df.copy()
    results_df["cluster_id"] = cluster_ids
    results_df["distance_to_centroid"] = np.round(min_distances, 4)
    results_df["nearest_centroid"] = [list(c) for c in nearest_centroids_scaled]
    results_df["persona"] = [bundle.personas.get(cid, "Unknown Persona") for cid in cluster_ids]

    # Attach cluster statistics dict
    stats_lookup = bundle.cluster_statistics.set_index("cluster_id").to_dict(orient="index")
    results_df["cluster_statistics"] = [stats_lookup.get(cid, {}) for cid in cluster_ids]

    if is_single:
        return results_df.iloc[0].to_dict()
    return results_df
