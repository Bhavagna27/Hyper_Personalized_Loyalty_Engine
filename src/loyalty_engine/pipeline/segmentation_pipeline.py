"""Customer Segmentation Pipeline orchestrator.

Executes end-to-end customer segmentation:
1. Ingestion & Validation
2. Feature Engineering
3. K-Evaluation (K=2 to 10 via Elbow & Silhouette)
4. Model Training (KMeans, random_state=42, n_init=20)
5. Visualization Generation (5 plots saved under outputs/clustering/)
6. Cluster Profiling & Dynamic Persona Assignment
7. Artifact Saving (kmeans_model.joblib, cluster_profiles.csv, cluster_statistics.csv, cluster_summary.json)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd

from loyalty_engine.config import PATHS
from loyalty_engine.features import FeatureEngineeringPipeline
from loyalty_engine.io import IngestionPipeline, dump_joblib, ensure_dir, write_csv, write_json
from loyalty_engine.models.segmentation import (
    SEGMENTATION_FEATURES,
    KEvaluationResult,
    SegmentationModelBundle,
    evaluate_optimal_k,
    train_kmeans_segmentation,
)
from loyalty_engine.visualization.clustering_viz import (
    plot_cluster_centroids_heatmap,
    plot_elbow_and_silhouette,
    plot_pairwise_cluster_comparison,
    plot_pca_2d_clusters,
    plot_pca_explained_variance,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result NamedTuple
# ---------------------------------------------------------------------------

class SegmentationPipelineResult(NamedTuple):
    """Result container returned by :meth:`SegmentationPipeline.run`."""

    bundle: SegmentationModelBundle
    customer_features: pd.DataFrame
    k_evaluation: KEvaluationResult
    saved_paths: list[Path]


# ---------------------------------------------------------------------------
# Segmentation Pipeline Orchestrator Class
# ---------------------------------------------------------------------------

@dataclass
class SegmentationPipeline:
    """Orchestrates end-to-end K-Means customer segmentation workflow.

    Parameters
    ----------
    input_path:
        Path to source .xlsx workbook.
    output_dir:
        Directory for plot figures and CSV outputs (default: outputs/clustering/).
    artifacts_dir:
        Directory for joblib models and JSON summaries (default: artifacts/).
    """

    input_path: Path
    output_dir: Path = field(default_factory=lambda: PATHS.clustering_outputs_dir)
    artifacts_dir: Path = field(default_factory=lambda: PATHS.artifacts_dir)

    def run(
        self,
        n_clusters: int | None = None,
        save: bool = True,
    ) -> SegmentationPipelineResult:
        """Execute the segmentation pipeline.

        Parameters
        ----------
        n_clusters:
            Optional fixed K. If None, K is evaluated and selected automatically.
        save:
            Whether to write plots and artifacts to disk.

        Returns
        -------
        SegmentationPipelineResult
        """
        logger.info("=" * 60)
        logger.info("Starting Customer Segmentation Pipeline: %s", self.input_path)
        logger.info("=" * 60)

        # 1. Load, Validate, Clean Data
        ingest_res = IngestionPipeline(workbook_path=self.input_path).run(save=save)
        profile = ingest_res.cleaned_frames["Customer_Loyalty_Profile"]
        transactions = ingest_res.cleaned_frames["Transaction_History"]

        # 2. Engineer Customer Features
        fe_res = FeatureEngineeringPipeline(
            output_dir=PATHS.processed_dir,
            artifacts_dir=self.artifacts_dir,
        ).run(profile, transactions, save=save)
        customer_df = fe_res.customer_features

        # 3. Evaluate Optimal K
        k_eval = evaluate_optimal_k(customer_df, features=SEGMENTATION_FEATURES)
        selected_k = n_clusters if n_clusters is not None else k_eval.optimal_k

        # 4. Train Final KMeans Model and Profile Clusters
        bundle, segmented_df = train_kmeans_segmentation(
            customer_features=customer_df,
            n_clusters=selected_k,
            features=SEGMENTATION_FEATURES,
            random_state=42,
            n_init=20,
        )

        saved_paths: list[Path] = []

        if save:
            ensure_dir(self.output_dir)
            ensure_dir(self.artifacts_dir)

            # 5. Generate Visualizations (Outputs under outputs/clustering/)
            plot_elbow_and_silhouette(
                k_range=k_eval.k_range,
                inertias=k_eval.inertias,
                silhouette_scores=k_eval.silhouette_scores,
                optimal_k=selected_k,
                output_dir=self.output_dir,
            )
            saved_paths.extend([
                self.output_dir / "elbow_curve.png",
                self.output_dir / "silhouette_scores.png",
            ])

            X_raw = segmented_df[SEGMENTATION_FEATURES]
            X_scaled = bundle.scaler.transform(X_raw)

            plot_pca_2d_clusters(
                X_scaled=X_scaled,
                labels=segmented_df["cluster_id"].values,
                centroids_scaled=bundle.centroids_scaled,
                personas=bundle.personas,
                output_dir=self.output_dir,
            )
            saved_paths.append(self.output_dir / "pca_2d_clusters.png")

            plot_pca_explained_variance(
                X_scaled=X_scaled,
                output_dir=self.output_dir,
            )
            saved_paths.append(self.output_dir / "pca_explained_variance.png")

            plot_cluster_centroids_heatmap(
                centroids_scaled=bundle.centroids_scaled,
                feature_names=SEGMENTATION_FEATURES,
                personas=bundle.personas,
                output_dir=self.output_dir,
            )
            saved_paths.append(self.output_dir / "cluster_centroids.png")

            plot_pairwise_cluster_comparison(
                customer_features=segmented_df,
                personas=bundle.personas,
                output_dir=self.output_dir,
            )
            saved_paths.append(self.output_dir / "pairwise_cluster_comparison.png")

            # 6. Save Artifacts (kmeans_model.joblib, cluster_profiles.csv, cluster_statistics.csv, cluster_summary.json)
            # Save model joblib
            model_joblib_path = self.artifacts_dir / "kmeans_model.joblib"
            dump_joblib(bundle, model_joblib_path)
            saved_paths.append(model_joblib_path)

            # Copy model joblib to outputs/clustering for easy access
            outputs_model_path = self.output_dir / "kmeans_model.joblib"
            dump_joblib(bundle, outputs_model_path)
            saved_paths.append(outputs_model_path)

            # Save cluster_profiles.csv
            profiles_csv_path = self.output_dir / "cluster_profiles.csv"
            write_csv(bundle.cluster_profiles, profiles_csv_path)
            saved_paths.append(profiles_csv_path)

            # Save cluster_statistics.csv
            stats_csv_path = self.output_dir / "cluster_statistics.csv"
            write_csv(bundle.cluster_statistics, stats_csv_path)
            saved_paths.append(stats_csv_path)

            # Save cluster_summary.json
            summary_dict = {
                "optimal_k": selected_k,
                "justification": k_eval.justification,
                "overall_inertia": bundle.inertia,
                "overall_silhouette_score": bundle.silhouette_score,
                "cluster_sizes": bundle.cluster_sizes,
                "personas": bundle.personas,
                "k_evaluation": {
                    "inertias": {str(k): v for k, v in k_eval.inertias.items()},
                    "silhouette_scores": {str(k): v for k, v in k_eval.silhouette_scores.items()},
                },
                "cluster_profiles": bundle.cluster_profiles.to_dict(orient="records"),
            }

            summary_json_path = self.artifacts_dir / "cluster_summary.json"
            write_json(summary_dict, summary_json_path)
            saved_paths.append(summary_json_path)

            # Also save JSON in outputs/clustering/
            outputs_json_path = self.output_dir / "cluster_summary.json"
            write_json(summary_dict, outputs_json_path)
            saved_paths.append(outputs_json_path)

            logger.info("Saved %d segmentation artifacts and plots.", len(saved_paths))

        return SegmentationPipelineResult(
            bundle=bundle,
            customer_features=segmented_df,
            k_evaluation=k_eval,
            saved_paths=saved_paths,
        )


# ---------------------------------------------------------------------------
# High-Level Function Wrapper
# ---------------------------------------------------------------------------

def run_segmentation_pipeline(
    input_path: Path,
    output_dir: Path = PATHS.clustering_outputs_dir,
    artifacts_dir: Path = PATHS.artifacts_dir,
) -> SegmentationPipelineResult:
    """Run the complete customer segmentation pipeline.

    Parameters
    ----------
    input_path:
        Path to source .xlsx dataset.
    output_dir:
        Directory for plot outputs and profiles CSVs.
    artifacts_dir:
        Directory for saved model bundle and JSON summary.

    Returns
    -------
    SegmentationPipelineResult
    """
    pipeline = SegmentationPipeline(input_path=input_path, output_dir=output_dir, artifacts_dir=artifacts_dir)
    return pipeline.run()
