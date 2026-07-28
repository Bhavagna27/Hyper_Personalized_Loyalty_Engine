"""IO sub-package for the loyalty engine.

Public API
----------
ExcelDatasetLoader : class
    Loads named or auto-discovered worksheets from an ``.xlsx`` workbook.
IngestionPipeline : class
    Orchestrates the full ingest → validate → clean → save workflow.
IngestionResult : NamedTuple
    Immutable result returned by :meth:`IngestionPipeline.run`.
configure_logging : callable
    Set up structured logging for the application.
"""

from .excel_loader import ExcelDatasetLoader
from .ingestion import IngestionPipeline, IngestionResult, configure_logging

__all__ = [
    "ExcelDatasetLoader",
    "IngestionPipeline",
    "IngestionResult",
    "configure_logging",
]
