"""Validation report dataclasses and builder.

This module defines the typed output of the validation layer.  It is kept
separate from :mod:`~loyalty_engine.validation.rules` so callers can import
just the data structures without pulling in rule logic.

Example
-------
>>> import pandas as pd
>>> from loyalty_engine.validation.report import build_validation_report
>>> df = pd.DataFrame({"Customer_ID": [1, 2, 2], "Amount": [10.0, -5.0, 20.0]})
>>> report = build_validation_report(
...     "demo",
...     df,
...     monetary_columns=["Amount"],
...     date_columns=[],
...     categorical_constraints={},
...     id_column="Customer_ID",
... )
>>> report.total_rows
3
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from loyalty_engine.validation.rules import (
    Finding,
    check_duplicate_customer_ids,
    check_duplicate_rows,
    check_dtype_mismatches,
    check_empty_columns,
    check_invalid_categoricals,
    check_invalid_dates,
    check_invalid_monetary,
    check_missing_values,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column-level stats
# ---------------------------------------------------------------------------


@dataclass
class ColumnStats:
    """Per-column summary statistics for the validation report.

    Attributes
    ----------
    name:
        Column name.
    dtype:
        String representation of the pandas dtype.
    missing_count:
        Number of null values.
    missing_pct:
        Percentage of null values (0–100).
    is_empty:
        ``True`` when *every* value is null.
    unique_count:
        Number of distinct non-null values.
    """

    name: str
    dtype: str
    missing_count: int
    missing_pct: float
    is_empty: bool
    unique_count: int


def _build_column_stats(df: pd.DataFrame) -> list[ColumnStats]:
    """Build a :class:`ColumnStats` for every column in *df*."""
    stats: list[ColumnStats] = []
    total = len(df)
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        pct = round(n_missing / total * 100, 2) if total else 0.0
        stats.append(
            ColumnStats(
                name=col,
                dtype=str(df[col].dtype),
                missing_count=n_missing,
                missing_pct=pct,
                is_empty=(n_missing == total),
                unique_count=int(df[col].nunique(dropna=True)),
            )
        )
    return stats


# ---------------------------------------------------------------------------
# Sheet-level report
# ---------------------------------------------------------------------------


@dataclass
class SheetReport:
    """Complete validation report for a single worksheet.

    Attributes
    ----------
    sheet_name:
        Name of the worksheet.
    total_rows:
        Number of rows in the raw (uncleaned) DataFrame.
    total_columns:
        Number of columns in the raw DataFrame.
    column_stats:
        Per-column statistics.
    invalid_records:
        List of :class:`Finding` dicts produced by the rule functions.
    suggested_fixes:
        Human-readable remediation hints derived from the findings.
    """

    sheet_name: str
    total_rows: int
    total_columns: int
    column_stats: list[ColumnStats] = field(default_factory=list)
    invalid_records: list[Finding] = field(default_factory=list)
    suggested_fixes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def duplicate_rows(self) -> int:
        """Number of fully-duplicate rows identified during validation."""
        for f in self.invalid_records:
            if f.get("rule") == "duplicate_rows":
                return int(f["duplicate_count"])
        return 0

    @property
    def total_missing(self) -> int:
        """Total count of missing values across all columns."""
        return sum(cs.missing_count for cs in self.column_stats)

    def is_clean(self) -> bool:
        """Return ``True`` when no issues were detected."""
        return len(self.invalid_records) == 0

    def summary_lines(self) -> list[str]:
        """Return a short list of human-readable summary lines."""
        lines = [
            f"Sheet: {self.sheet_name}",
            f"  Rows: {self.total_rows} | Columns: {self.total_columns}",
            f"  Total missing values: {self.total_missing}",
            f"  Duplicate rows: {self.duplicate_rows}",
            f"  Issues found: {len(self.invalid_records)}",
        ]
        if self.suggested_fixes:
            lines.append("  Suggested fixes:")
            for fix in self.suggested_fixes:
                lines.append(f"    • {fix}")
        return lines

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (JSON-safe)."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Workbook-level report
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Aggregated validation report for all worksheets in a workbook.

    Attributes
    ----------
    sheet_reports:
        Mapping from sheet name to its :class:`SheetReport`.
    """

    sheet_reports: dict[str, SheetReport] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a multi-line human-readable summary of all sheets."""
        lines: list[str] = ["=" * 60, "VALIDATION REPORT SUMMARY", "=" * 60]
        for report in self.sheet_reports.values():
            lines.extend(report.summary_lines())
            lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire report to a plain dict."""
        return {
            "sheets": {
                name: r.to_dict() for name, r in self.sheet_reports.items()
            }
        }

    def save_json(self, path: Path) -> None:
        """Write the report as a JSON file to *path*.

        Parent directories are created automatically.

        Parameters
        ----------
        path:
            Destination file path (should end in ``.json``).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)
        logger.info("Validation report saved to %s", path)


# ---------------------------------------------------------------------------
# Suggested-fix generator
# ---------------------------------------------------------------------------

_FIX_TEMPLATES: dict[str, str] = {
    "missing_values": "Fill or drop missing values in column '{column}'.",
    "duplicate_rows": "Remove {duplicate_count} fully-duplicate rows.",
    "empty_column": "Drop or investigate entirely empty column '{column}'.",
    "invalid_date": "Parse/repair {invalid_count} invalid date(s) in '{column}'.",
    "invalid_monetary": (
        "Fix {negative_count} negative and {non_numeric_count} non-numeric "
        "value(s) in monetary column '{column}'."
    ),
    "duplicate_customer_ids": (
        "Resolve {duplicate_count} rows with duplicated '{id_column}' values."
    ),
    "invalid_categorical": (
        "Replace {invalid_count} out-of-vocabulary value(s) in '{column}' "
        "with one of: {valid_values}."
    ),
    "dtype_mismatch": (
        "Convert column '{column}' from dtype '{actual_dtype}' to kind "
        "'{expected_kind}'."
    ),
}


def _generate_fixes(findings: list[Finding]) -> list[str]:
    """Translate *findings* into human-readable remediation hints."""
    fixes: list[str] = []
    for f in findings:
        template = _FIX_TEMPLATES.get(f.get("rule", ""))
        if template:
            try:
                fixes.append(template.format(**f))
            except KeyError:
                fixes.append(f"Investigate rule '{f.get('rule')}' finding.")
    return fixes


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_validation_report(
    sheet_name: str,
    df: pd.DataFrame,
    *,
    date_columns: list[str] | None = None,
    monetary_columns: list[str] | None = None,
    categorical_constraints: dict[str, list[str]] | None = None,
    expected_dtypes: dict[str, str] | None = None,
    id_column: str = "Customer_ID",
) -> SheetReport:
    """Run all validation rules on *df* and return a :class:`SheetReport`.

    Parameters
    ----------
    sheet_name:
        Human-readable label for the worksheet.
    df:
        The raw (uncleaned) DataFrame.
    date_columns:
        Columns expected to contain date values.
    monetary_columns:
        Columns expected to contain non-negative numeric values.
    categorical_constraints:
        Mapping of ``column → valid_value_set``.
    expected_dtypes:
        Mapping of ``column → dtype_kind`` (e.g. ``'f'``, ``'M'``).
    id_column:
        Name of the customer-ID column to check for duplicates.

    Returns
    -------
    SheetReport
    """
    logger.info("Validating sheet '%s' (%d rows, %d cols)…", sheet_name, len(df), len(df.columns))

    findings: list[Finding] = []
    findings.extend(check_missing_values(df))
    findings.extend(check_duplicate_rows(df))
    findings.extend(check_empty_columns(df))

    if date_columns:
        findings.extend(check_invalid_dates(df, date_columns))
    if monetary_columns:
        findings.extend(check_invalid_monetary(df, monetary_columns))
    if id_column in df.columns:
        findings.extend(check_duplicate_customer_ids(df, id_column))
    if categorical_constraints:
        findings.extend(check_invalid_categoricals(df, categorical_constraints))
    if expected_dtypes:
        findings.extend(check_dtype_mismatches(df, expected_dtypes))

    column_stats = _build_column_stats(df)
    suggested_fixes = _generate_fixes(findings)

    report = SheetReport(
        sheet_name=sheet_name,
        total_rows=len(df),
        total_columns=len(df.columns),
        column_stats=column_stats,
        invalid_records=findings,
        suggested_fixes=suggested_fixes,
    )

    status = "CLEAN" if report.is_clean() else f"{len(findings)} issue(s)"
    logger.info("Sheet '%s' validated — %s.", sheet_name, status)
    return report


def build_full_report(
    frames: dict[str, pd.DataFrame],
    sheet_configs: dict[str, dict[str, Any]],
) -> ValidationReport:
    """Build a :class:`ValidationReport` for every sheet in *frames*.

    Parameters
    ----------
    frames:
        Dict of sheet name → raw DataFrame.
    sheet_configs:
        Dict of sheet name → keyword arguments forwarded to
        :func:`build_validation_report` (e.g. ``date_columns``,
        ``monetary_columns``, etc.).

    Returns
    -------
    ValidationReport
    """
    sheet_reports: dict[str, SheetReport] = {}
    for sheet_name, df in frames.items():
        cfg = sheet_configs.get(sheet_name, {})
        sheet_reports[sheet_name] = build_validation_report(sheet_name, df, **cfg)
    return ValidationReport(sheet_reports=sheet_reports)
