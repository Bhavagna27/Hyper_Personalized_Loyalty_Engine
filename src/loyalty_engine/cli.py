"""Command-line interface for the Hyper-Personalized Loyalty Engine.

Sub-commands
------------
train
    Train churn and health models on a workbook.
score
    Score a workbook using a saved model bundle.
eda
    Generate EDA visualisations from a workbook.
ingest
    Run the full ingestion pipeline: load → validate → clean → save.
features
    Run the feature engineering pipeline: generate all 6 feature groups, fit sklearn pipeline, and export artifacts.
segment
    Run the customer segmentation pipeline: evaluate optimal K, train KMeans, profile clusters, assign personas, and generate plots.

Usage
-----
::

    # Ingest and validate data
    python -m loyalty_engine.cli ingest --input data/workbook.xlsx

    # Feature engineering
    python -m loyalty_engine.cli features --input data/workbook.xlsx

    # Customer segmentation
    python -m loyalty_engine.cli segment --input data/workbook.xlsx

    # Train models
    python -m loyalty_engine.cli train --input data/workbook.xlsx

    # Score / recommend
    python -m loyalty_engine.cli score --input data/workbook.xlsx

    # EDA visualisations
    python -m loyalty_engine.cli eda --input data/workbook.xlsx
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from loyalty_engine.config import PATHS
from loyalty_engine.features import FeatureEngineeringPipeline
from loyalty_engine.io import ExcelDatasetLoader, IngestionPipeline, configure_logging, write_csv
from loyalty_engine.pipeline import (
    SegmentationPipeline,
    run_training_pipeline,
    score_workbook,
)
from loyalty_engine.preprocessing import clean_customer_profile
from loyalty_engine.visualization import plot_customer_overview

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="loyalty-engine",
        description="Hyper-Personalized Credit Card Loyalty Recommendation Engine",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- train --------------------------------------------------------------
    train_parser = subparsers.add_parser(
        "train",
        help="Train churn and health models on a workbook.",
    )
    train_parser.add_argument("--input", type=Path, required=True, help="Path to .xlsx workbook")
    train_parser.add_argument(
        "--artifact",
        type=Path,
        default=PATHS.artifacts_dir / "model_bundle.joblib",
        help="Destination path for the saved model bundle.",
    )

    # -- score --------------------------------------------------------------
    score_parser = subparsers.add_parser(
        "score",
        help="Score a workbook using a saved model bundle.",
    )
    score_parser.add_argument("--input", type=Path, required=True)
    score_parser.add_argument(
        "--artifact",
        type=Path,
        default=PATHS.artifacts_dir / "model_bundle.joblib",
    )
    score_parser.add_argument(
        "--output",
        type=Path,
        default=PATHS.artifacts_dir / "scored_customers.csv",
    )

    # -- eda ----------------------------------------------------------------
    eda_parser = subparsers.add_parser(
        "eda",
        help="Generate EDA visualisations.",
    )
    eda_parser.add_argument("--input", type=Path, required=True)
    eda_parser.add_argument("--output-dir", type=Path, default=PATHS.reports_dir)

    # -- ingest -------------------------------------------------------------
    ingest_parser = subparsers.add_parser(
        "ingest",
        help=(
            "Load, validate, clean, and save all worksheets. "
            "Produces cleaned CSVs in --output-dir and a JSON report in --report-dir."
        ),
    )
    ingest_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to source .xlsx workbook.",
    )
    ingest_parser.add_argument(
        "--output-dir",
        type=Path,
        default=PATHS.processed_dir,
        help="Directory for cleaned CSV outputs (default: data/processed/).",
    )
    ingest_parser.add_argument(
        "--report-dir",
        type=Path,
        default=PATHS.reports_dir,
        help="Directory for the JSON validation report (default: reports/).",
    )
    ingest_parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path to write pipeline logs to a file.",
    )
    ingest_parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Validate and clean without writing any files to disk.",
    )
    ingest_parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )

    # -- features -----------------------------------------------------------
    features_parser = subparsers.add_parser(
        "features",
        help=(
            "Run feature engineering pipeline. Consumes cleaned datasets and produces "
            "customer_features.csv, feature_pipeline.joblib, and feature_metadata.json."
        ),
    )
    features_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to source .xlsx workbook.",
    )
    features_parser.add_argument(
        "--output-dir",
        type=Path,
        default=PATHS.processed_dir,
        help="Directory for customer_features.csv output (default: data/processed/).",
    )
    features_parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=PATHS.artifacts_dir,
        help="Directory for feature_pipeline.joblib and metadata (default: artifacts/).",
    )
    features_parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )

    # -- segment (NEW) ------------------------------------------------------
    segment_parser = subparsers.add_parser(
        "segment",
        help=(
            "Run Customer Segmentation pipeline. Evaluates optimal K (K=2..10), "
            "trains KMeans, profiles clusters, assigns personas, and exports plots to outputs/clustering/."
        ),
    )
    segment_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to source .xlsx workbook.",
    )
    segment_parser.add_argument(
        "--output-dir",
        type=Path,
        default=PATHS.clustering_outputs_dir,
        help="Directory for plots and cluster CSVs (default: outputs/clustering/).",
    )
    segment_parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=PATHS.artifacts_dir,
        help="Directory for kmeans_model.joblib and JSON summary (default: artifacts/).",
    )
    segment_parser.add_argument(
        "--clusters",
        type=int,
        default=None,
        help="Optional fixed number of clusters K. If omitted, K is selected automatically.",
    )
    segment_parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )

    return parser


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------


def _handle_train(args: argparse.Namespace) -> None:
    reports = run_training_pipeline(args.input, args.artifact)
    print(reports["churn_report"])
    print(reports["health_report"])


def _handle_score(args: argparse.Namespace) -> None:
    scored = score_workbook(args.input, args.artifact)
    write_csv(scored, args.output)
    print(f"Saved scored recommendations to {args.output}")


def _handle_eda(args: argparse.Namespace) -> None:
    frames = ExcelDatasetLoader(args.input).load_all()
    profile = clean_customer_profile(frames["Customer_Loyalty_Profile"])
    plot_customer_overview(profile, args.output_dir)
    print(f"Saved EDA outputs to {args.output_dir}")


def _handle_ingest(args: argparse.Namespace) -> None:
    """Run the IngestionPipeline and print a summary report."""
    log_level = logging.DEBUG if args.verbose else logging.INFO
    configure_logging(level=log_level, log_file=args.log_file)

    pipeline = IngestionPipeline(
        workbook_path=args.input,
        processed_dir=args.output_dir,
        report_dir=args.report_dir,
    )

    try:
        result = pipeline.run(save=not args.no_save, log_file=args.log_file)
    except ValueError as exc:
        logger.error("Structural validation failed: %s", exc)
        sys.exit(1)

    print(result.report.summary())

    if not args.no_save:
        print("\nFiles written:")
        for path in result.saved_paths:
            print(f"  {path}")


def _handle_features(args: argparse.Namespace) -> None:
    """Run feature engineering pipeline and export artifacts."""
    log_level = logging.DEBUG if args.verbose else logging.INFO
    configure_logging(level=log_level)

    logger.info("Starting Ingestion & Validation step prior to Feature Engineering…")
    ingest_result = IngestionPipeline(workbook_path=args.input).run(save=True)

    fe_pipeline = FeatureEngineeringPipeline(
        output_dir=args.output_dir,
        artifacts_dir=args.artifacts_dir,
    )
    result = fe_pipeline.run(
        profile=ingest_result.cleaned_frames["Customer_Loyalty_Profile"],
        transactions=ingest_result.cleaned_frames["Transaction_History"],
        save=True,
    )

    print("=" * 60)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 60)
    print(f"Total Customer Records: {len(result.customer_features)}")
    print(f"Total Feature Columns : {len(result.customer_features.columns)}")
    print("\nSaved Artifacts:")
    for path in result.saved_paths:
        print(f"  • {path}")


def _handle_segment(args: argparse.Namespace) -> None:
    """Run Customer Segmentation pipeline and export artifacts & plots."""
    log_level = logging.DEBUG if args.verbose else logging.INFO
    configure_logging(level=log_level)

    pipeline = SegmentationPipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        artifacts_dir=args.artifacts_dir,
    )
    result = pipeline.run(n_clusters=args.clusters, save=True)

    print("=" * 60)
    print("CUSTOMER SEGMENTATION SUMMARY")
    print("=" * 60)
    print(f"Selected Optimal K       : {result.bundle.optimal_k}")
    print(f"Optimal K Justification  : {result.k_evaluation.justification}")
    print(f"Model Inertia            : {result.bundle.inertia:.2f}")
    print(f"Overall Silhouette Score : {result.bundle.silhouette_score:.4f}")
    print("\nCluster Size & Persona Breakdown:")
    for cid, persona in result.bundle.personas.items():
        size = result.bundle.cluster_sizes[cid]
        pct = (size / len(result.customer_features)) * 100
        print(f"  Cluster {cid}: {persona:<38} ({size:3d} customers | {pct:.1f}%)")

    print("\nGenerated Artifacts & Visualizations:")
    for path in result.saved_paths:
        print(f"  • {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and dispatch to the appropriate sub-command handler."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        _handle_train(args)
    elif args.command == "score":
        _handle_score(args)
    elif args.command == "eda":
        _handle_eda(args)
    elif args.command == "ingest":
        _handle_ingest(args)
    elif args.command == "features":
        _handle_features(args)
    elif args.command == "segment":
        _handle_segment(args)


if __name__ == "__main__":
    main()
