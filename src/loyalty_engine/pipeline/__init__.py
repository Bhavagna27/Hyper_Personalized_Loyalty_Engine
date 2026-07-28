"""Pipeline sub-package for the Hyper-Personalized Loyalty Engine.

Public API
----------
run_training_pipeline : callable
    Train supervised classification models for churn and health.
score_workbook : callable
    Generate customer recommendations.
run_segmentation_pipeline : callable
    Train K-Means customer segmentation, profile clusters, assign personas, and export plots.
SegmentationPipeline : class
    Orchestrator class for customer segmentation.
"""

from .infer import score_workbook
from .segmentation_pipeline import SegmentationPipeline, run_segmentation_pipeline
from .train import run_training_pipeline

__all__ = [
    "run_training_pipeline",
    "score_workbook",
    "run_segmentation_pipeline",
    "SegmentationPipeline",
]
