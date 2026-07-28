from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
NEW_CUSTOMER_PREDICTIONS_PATH = PROJECT_ROOT / "new_customer_predictions.csv"
LLM_RECOMMENDATIONS_PATH = PROJECT_ROOT / "llm_recommendations.csv"
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"


@dataclass(frozen=True)
class PathGroup:
    """Convenience bundle for paths used in the dashboard."""

    customer_features: Path = PROCESSED_DIR / "customer_features.csv"
    validation_report: Path = REPORTS_DIR / "validation_report.json"
    model_evaluation: Path = OUTPUTS_DIR / "models" / "model_evaluation.json"
    cluster_summary_artifact: Path = ARTIFACTS_DIR / "cluster_summary.json"
    cluster_summary_output: Path = OUTPUTS_DIR / "clustering" / "cluster_summary.json"
    cluster_profiles: Path = OUTPUTS_DIR / "clustering" / "cluster_profiles.csv"
    cluster_statistics: Path = OUTPUTS_DIR / "clustering" / "cluster_statistics.csv"
    recommendations: Path = OUTPUTS_DIR / "recommendations" / "recommendations.csv"
    reward_catalog: Path = OUTPUTS_DIR / "recommendations" / "reward_catalog.csv"
    ai_recommendations: Path = PROCESSED_DIR / "AI_Recommendations.csv"
    scored_customers: Path = ARTIFACTS_DIR / "scored_customers.csv"
    feature_metadata: Path = ARTIFACTS_DIR / "feature_metadata.json"
    model_bundle: Path = ARTIFACTS_DIR / "model_bundle.joblib"
    feature_pipeline: Path = ARTIFACTS_DIR / "feature_pipeline.joblib"
    kmeans_model: Path = ARTIFACTS_DIR / "kmeans_model.joblib"
    new_customer_predictions: Path = NEW_CUSTOMER_PREDICTIONS_PATH
    llm_recommendations: Path = LLM_RECOMMENDATIONS_PATH


PATHS = PathGroup()


def _mtime_signature(path: Path) -> int:
    return path.stat().st_mtime_ns if path.exists() else 0


@st.cache_data(show_spinner=False)
def _read_csv_cached(path_str: str, signature: int) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _read_json_cached(path_str: str, signature: int) -> Any:
    path = Path(path_str)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file with cache invalidation based on file mtime."""
    return _read_csv_cached(str(path), _mtime_signature(path)).copy()


def load_json(path: Path) -> Any:
    """Load a JSON file with cache invalidation based on file mtime."""
    payload = _read_json_cached(str(path), _mtime_signature(path))
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(payload, list):
        return list(payload)
    return payload


def load_first_existing_csv(paths: list[Path]) -> tuple[pd.DataFrame, Path | None]:
    """Return the first existing CSV from a list of candidate paths."""
    for path in paths:
        if path.exists():
            return load_csv(path), path
    return pd.DataFrame(), None


def load_first_existing_json(paths: list[Path]) -> tuple[Any, Path | None]:
    """Return the first existing JSON artifact from a list of candidate paths."""
    for path in paths:
        if path.exists():
            return load_json(path), path
    return {}, None


def load_feature_metadata() -> dict[str, Any]:
    metadata = load_json(PATHS.feature_metadata)
    return metadata if isinstance(metadata, dict) else {}


def load_validation_report() -> dict[str, Any]:
    report = load_json(PATHS.validation_report)
    return report if isinstance(report, dict) else {}


def load_model_evaluation() -> dict[str, Any]:
    evaluation = load_json(PATHS.model_evaluation)
    return evaluation if isinstance(evaluation, dict) else {}


def load_cluster_summary() -> dict[str, Any]:
    summary, _ = load_first_existing_json(
        [PATHS.cluster_summary_artifact, PATHS.cluster_summary_output]
    )
    return summary if isinstance(summary, dict) else {}


def load_customer_features() -> pd.DataFrame:
    return load_csv(PATHS.customer_features)


def load_cluster_profiles() -> pd.DataFrame:
    return load_csv(PATHS.cluster_profiles)


def load_cluster_statistics() -> pd.DataFrame:
    return load_csv(PATHS.cluster_statistics)


def load_recommendations() -> pd.DataFrame:
    return load_csv(PATHS.recommendations)


def load_reward_catalog() -> pd.DataFrame:
    return load_csv(PATHS.reward_catalog)


def load_ai_recommendations() -> pd.DataFrame:
    return load_csv(PATHS.ai_recommendations)


def load_scored_customers() -> pd.DataFrame:
    return load_csv(PATHS.scored_customers)


def load_new_customer_predictions() -> pd.DataFrame:
    return load_csv(PATHS.new_customer_predictions)


def load_llm_recommendations() -> pd.DataFrame:
    return load_csv(PATHS.llm_recommendations)


def _parse_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return ast.literal_eval(text)
        except Exception:
            try:
                return json.loads(text)
            except Exception:
                return text
    return value


def extract_reward_names(raw_value: Any) -> list[str]:
    parsed = _parse_value(raw_value)
    if parsed is None:
        return []
    if isinstance(parsed, dict):
        name = parsed.get("reward_name") or parsed.get("name") or parsed.get("reward")
        return [str(name)] if name else []
    if isinstance(parsed, list):
        reward_names: list[str] = []
        for item in parsed:
            if isinstance(item, dict):
                name = item.get("reward_name") or item.get("name") or item.get("reward")
                if name:
                    reward_names.append(str(name))
            else:
                text = str(item).strip()
                if text:
                    reward_names.append(text)
        return reward_names
    text = str(parsed).strip()
    return [text] if text else []


def extract_scores(raw_value: Any) -> list[float]:
    parsed = _parse_value(raw_value)
    if parsed is None:
        return []
    if isinstance(parsed, list):
        scores: list[float] = []
        for item in parsed:
            try:
                scores.append(float(item))
            except Exception:
                continue
        return scores
    try:
        return [float(parsed)]
    except Exception:
        return []


def project_version(default: str = "0.1.0") -> str:
    if not PYPROJECT_PATH.exists():
        return default
    try:
        for line in PYPROJECT_PATH.read_text(encoding="utf-8").splitlines():
            if line.startswith("version = "):
                return line.split("=", 1)[1].strip().strip('"')
    except Exception:
        return default
    return default


def generated_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")


def _recommendation_average_score(df: pd.DataFrame) -> float | None:
    if df.empty or "Scores" not in df.columns:
        return None
    top_scores: list[float] = []
    for value in df["Scores"]:
        scores = extract_scores(value)
        if scores:
            top_scores.append(scores[0])
    if not top_scores:
        return None
    return float(sum(top_scores) / len(top_scores))


def build_kpis() -> dict[str, float | int | None]:
    cluster_summary = load_cluster_summary()
    recommendations = load_recommendations()
    new_customer_predictions = load_new_customer_predictions()
    feature_metadata = load_feature_metadata()

    total_customers = feature_metadata.get("total_customers")
    if total_customers is None and recommendations is not None and not recommendations.empty:
        total_customers = int(len(recommendations))

    cluster_sizes = cluster_summary.get("cluster_sizes", {}) if isinstance(cluster_summary, dict) else {}
    num_clusters = cluster_summary.get("optimal_k") if isinstance(cluster_summary, dict) else None
    if num_clusters is None and isinstance(cluster_sizes, dict):
        num_clusters = len(cluster_sizes)

    recommendations_generated = int(len(recommendations)) if not recommendations.empty else 0
    average_similarity_score = None
    if not new_customer_predictions.empty and "similarity_score" in new_customer_predictions.columns:
        average_similarity_score = float(new_customer_predictions["similarity_score"].mean())

    average_recommendation_score = _recommendation_average_score(recommendations)

    return {
        "total_customers": int(total_customers) if total_customers is not None else None,
        "num_clusters": int(num_clusters) if num_clusters is not None else None,
        "recommendations_generated": recommendations_generated,
        "average_similarity_score": average_similarity_score,
        "average_recommendation_score": average_recommendation_score,
    }


def build_pipeline_status() -> dict[str, str]:
    """Return high-level readiness states for the sidebar."""
    pipeline_ready = all(
        [
            not load_customer_features().empty,
            not load_recommendations().empty,
            bool(load_validation_report()),
            bool(load_cluster_summary()),
        ]
    )
    model_ready = PATHS.model_bundle.exists() and PATHS.model_evaluation.exists()
    artifacts_ready = all(
        [
            PATHS.cluster_summary_artifact.exists() or PATHS.cluster_summary_output.exists(),
            PATHS.feature_metadata.exists(),
            PATHS.feature_pipeline.exists(),
            PATHS.kmeans_model.exists(),
        ]
    )

    def label(is_ready: bool) -> str:
        return "Ready" if is_ready else "Partial"

    return {
        "pipeline_status": "Ready" if pipeline_ready else "Partial",
        "model_status": "Ready" if model_ready else "Partial",
        "artifacts_status": "Ready" if artifacts_ready else "Partial",
    }


def summarize_validation_report() -> pd.DataFrame:
    report = load_validation_report()
    sheets = report.get("sheets", {}) if isinstance(report, dict) else {}
    rows: list[dict[str, Any]] = []
    for sheet_name, payload in sheets.items():
        invalid_records = payload.get("invalid_records", []) if isinstance(payload, dict) else []
        rows.append(
            {
                "Sheet": sheet_name,
                "Rows": int(payload.get("total_rows", 0)) if isinstance(payload, dict) else 0,
                "Columns": int(payload.get("total_columns", 0)) if isinstance(payload, dict) else 0,
                "Missing Values": int(payload.get("total_missing", 0)) if isinstance(payload, dict) else 0,
                "Duplicate Rows": int(payload.get("duplicate_rows", 0)) if isinstance(payload, dict) else 0,
                "Issues": len(invalid_records),
            }
        )
    return pd.DataFrame(rows)


def summarize_model_evaluation() -> pd.DataFrame:
    evaluation = load_model_evaluation()
    rows: list[dict[str, Any]] = []
    for model_name, payload in evaluation.items():
        if not isinstance(payload, dict):
            continue
        rows.append(
            {
                "Model": model_name.replace("_", " ").title(),
                "Accuracy": payload.get("accuracy"),
                "Precision": payload.get("precision"),
                "Recall": payload.get("recall"),
                "F1": payload.get("f1"),
                "Test Size": payload.get("test_size"),
                "Random State": payload.get("random_state"),
            }
        )
    return pd.DataFrame(rows)

