from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path = Path(__file__).resolve().parents[2]

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def logs_dir(self) -> Path:
        return self.reports_dir / "logs"

    @property
    def customer_features_path(self) -> Path:
        return self.processed_dir / "customer_features.csv"

    @property
    def feature_pipeline_path(self) -> Path:
        return self.artifacts_dir / "feature_pipeline.joblib"

    @property
    def feature_metadata_path(self) -> Path:
        return self.artifacts_dir / "feature_metadata.json"

    @property
    def clustering_outputs_dir(self) -> Path:
        return self.root / "outputs" / "clustering"

    @property
    def kmeans_model_path(self) -> Path:
        return self.artifacts_dir / "kmeans_model.joblib"

    @property
    def cluster_profiles_path(self) -> Path:
        return self.clustering_outputs_dir / "cluster_profiles.csv"

    @property
    def cluster_statistics_path(self) -> Path:
        return self.clustering_outputs_dir / "cluster_statistics.csv"

    @property
    def cluster_summary_path(self) -> Path:
        return self.artifacts_dir / "cluster_summary.json"


@dataclass(frozen=True)
class ModelConfig:
    random_state: int = 42
    test_size: float = 0.2
    target_columns: tuple[str, ...] = ("Churn_Risk", "Customer_Health")


PATHS = ProjectPaths()
MODEL_CONFIG = ModelConfig()

