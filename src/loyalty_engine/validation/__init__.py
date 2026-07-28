"""Validation sub-package for the loyalty engine.

Public API
----------
REQUIRED_COLUMNS : dict
    Required column names per worksheet.
REQUIRED_SHEETS : tuple
    Names of the three expected worksheets.
validate_workbook : callable
    Structural + data-quality validator; returns a :class:`ValidationReport`.
ValidationReport : dataclass
    Aggregated report for all worksheets.
SheetReport : dataclass
    Per-worksheet report.
build_validation_report : callable
    Low-level builder for a single sheet's :class:`SheetReport`.
build_full_report : callable
    Batch builder for all sheets.
"""

from .report import (
    SheetReport,
    ValidationReport,
    build_full_report,
    build_validation_report,
)
from .schema import REQUIRED_COLUMNS, REQUIRED_SHEETS, validate_workbook

__all__ = [
    "REQUIRED_COLUMNS",
    "REQUIRED_SHEETS",
    "validate_workbook",
    "ValidationReport",
    "SheetReport",
    "build_validation_report",
    "build_full_report",
]
