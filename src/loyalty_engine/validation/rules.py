"""Individual, unit-testable validation rule functions.

Each function accepts a :class:`pandas.DataFrame` (plus optional parameters)
and returns a **list of finding dicts**.  Callers can aggregate these findings
into a :class:`~loyalty_engine.validation.report.SheetReport`.

All functions are pure with respect to the DataFrame — they never mutate it.

Example
-------
>>> import pandas as pd
>>> from loyalty_engine.validation.rules import check_missing_values
>>> df = pd.DataFrame({"a": [1, None], "b": ["x", "y"]})
>>> check_missing_values(df)
[{'rule': 'missing_values', 'column': 'a', 'missing_count': 1, 'missing_pct': 50.0}]
"""
from __future__ import annotations

import logging
from typing import Any, Collection

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Finding = dict[str, Any]


# ---------------------------------------------------------------------------
# Rule: missing values
# ---------------------------------------------------------------------------


def check_missing_values(df: pd.DataFrame) -> list[Finding]:
    """Return one finding per column that contains at least one null.

    Parameters
    ----------
    df:
        The DataFrame to inspect.

    Returns
    -------
    list[Finding]
        Each finding contains ``rule``, ``column``, ``missing_count``,
        ``missing_pct``.
    """
    findings: list[Finding] = []
    total = len(df)
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        if n_missing > 0:
            pct = round(n_missing / total * 100, 2) if total else 0.0
            findings.append(
                {
                    "rule": "missing_values",
                    "column": col,
                    "missing_count": n_missing,
                    "missing_pct": pct,
                }
            )
            logger.debug("Column '%s': %d missing (%.1f%%)", col, n_missing, pct)
    return findings


# ---------------------------------------------------------------------------
# Rule: duplicate rows
# ---------------------------------------------------------------------------


def check_duplicate_rows(df: pd.DataFrame) -> list[Finding]:
    """Return a finding if fully-duplicate rows are present.

    Parameters
    ----------
    df:
        The DataFrame to inspect.

    Returns
    -------
    list[Finding]
        Zero or one finding with ``rule``, ``duplicate_count``,
        ``duplicate_indices`` (first 20 indices for brevity).
    """
    dupes = df[df.duplicated(keep="first")]
    if dupes.empty:
        return []
    indices = list(dupes.index[:20])
    finding: Finding = {
        "rule": "duplicate_rows",
        "duplicate_count": len(dupes),
        "duplicate_indices_sample": indices,
    }
    logger.warning("Found %d fully-duplicate rows.", len(dupes))
    return [finding]


# ---------------------------------------------------------------------------
# Rule: empty columns
# ---------------------------------------------------------------------------


def check_empty_columns(df: pd.DataFrame) -> list[Finding]:
    """Return one finding per column where every value is null.

    Parameters
    ----------
    df:
        The DataFrame to inspect.

    Returns
    -------
    list[Finding]
        Each finding contains ``rule`` and ``column``.
    """
    findings: list[Finding] = []
    for col in df.columns:
        if df[col].isna().all():
            findings.append({"rule": "empty_column", "column": col})
            logger.warning("Column '%s' is entirely empty.", col)
    return findings


# ---------------------------------------------------------------------------
# Rule: invalid dates
# ---------------------------------------------------------------------------


def check_invalid_dates(
    df: pd.DataFrame, date_columns: Collection[str]
) -> list[Finding]:
    """Return one finding per date column containing unparseable values.

    Only columns that are already present in *df* are inspected.

    Parameters
    ----------
    df:
        The DataFrame to inspect.
    date_columns:
        Column names that are expected to contain parseable date values.

    Returns
    -------
    list[Finding]
        Each finding contains ``rule``, ``column``, ``invalid_count``,
        ``invalid_samples`` (up to 5 raw values).
    """
    findings: list[Finding] = []
    for col in date_columns:
        if col not in df.columns:
            continue
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            invalid_mask = series.isna()
        else:
            coerced = pd.to_datetime(series, errors="coerce")
            # Positions that were not originally null but became null after coerce
            invalid_mask = coerced.isna() & series.notna()
        n_invalid = int(invalid_mask.sum())
        if n_invalid > 0:
            samples = series[invalid_mask].dropna().unique()[:5].tolist()
            findings.append(
                {
                    "rule": "invalid_date",
                    "column": col,
                    "invalid_count": n_invalid,
                    "invalid_samples": samples,
                }
            )
            logger.warning("Column '%s': %d invalid date values.", col, n_invalid)
    return findings


# ---------------------------------------------------------------------------
# Rule: invalid monetary values
# ---------------------------------------------------------------------------


def check_invalid_monetary(
    df: pd.DataFrame, monetary_columns: Collection[str]
) -> list[Finding]:
    """Return one finding per monetary column with negative or non-numeric values.

    Parameters
    ----------
    df:
        The DataFrame to inspect.
    monetary_columns:
        Column names that are expected to contain non-negative numeric values.

    Returns
    -------
    list[Finding]
        Each finding contains ``rule``, ``column``, ``negative_count``,
        ``non_numeric_count``, ``samples``.
    """
    findings: list[Finding] = []
    for col in monetary_columns:
        if col not in df.columns:
            continue
        series = df[col]
        numeric = pd.to_numeric(series, errors="coerce")
        n_non_numeric = int(series.notna().sum()) - int(numeric.notna().sum())
        n_negative = int((numeric < 0).sum())
        if n_non_numeric > 0 or n_negative > 0:
            bad_mask = (numeric < 0) | (series.notna() & numeric.isna())
            samples = series[bad_mask].unique()[:5].tolist()
            findings.append(
                {
                    "rule": "invalid_monetary",
                    "column": col,
                    "negative_count": n_negative,
                    "non_numeric_count": n_non_numeric,
                    "samples": samples,
                }
            )
            logger.warning(
                "Column '%s': %d negative, %d non-numeric monetary values.",
                col,
                n_negative,
                n_non_numeric,
            )
    return findings


# ---------------------------------------------------------------------------
# Rule: duplicate Customer IDs
# ---------------------------------------------------------------------------


def check_duplicate_customer_ids(
    df: pd.DataFrame, id_column: str = "Customer_ID"
) -> list[Finding]:
    """Return a finding if *id_column* contains duplicated values.

    Parameters
    ----------
    df:
        The DataFrame to inspect.
    id_column:
        Name of the primary-key column.

    Returns
    -------
    list[Finding]
        Zero or one finding with ``rule``, ``id_column``, ``duplicate_count``,
        ``duplicate_ids_sample``.
    """
    if id_column not in df.columns:
        return []
    dupes = df[df.duplicated(subset=[id_column], keep=False)][id_column]
    if dupes.empty:
        return []
    unique_duped = dupes.unique()[:20].tolist()
    finding: Finding = {
        "rule": "duplicate_customer_ids",
        "id_column": id_column,
        "duplicate_count": int(dupes.shape[0]),
        "duplicate_ids_sample": unique_duped,
    }
    logger.warning(
        "Column '%s': %d rows with duplicated IDs.", id_column, int(dupes.shape[0])
    )
    return [finding]


# ---------------------------------------------------------------------------
# Rule: invalid categorical values
# ---------------------------------------------------------------------------


def check_invalid_categoricals(
    df: pd.DataFrame,
    categorical_constraints: dict[str, Collection[str]],
) -> list[Finding]:
    """Return one finding per categorical column containing out-of-vocabulary values.

    Parameters
    ----------
    df:
        The DataFrame to inspect.
    categorical_constraints:
        Mapping of ``column_name → valid_values``.

    Returns
    -------
    list[Finding]
        Each finding contains ``rule``, ``column``, ``invalid_count``,
        ``invalid_values_sample``.
    """
    findings: list[Finding] = []
    for col, valid_set in categorical_constraints.items():
        if col not in df.columns:
            continue
        actual = df[col].dropna().unique()
        invalid = [v for v in actual if v not in valid_set]
        if invalid:
            invalid_mask = df[col].isin(invalid)
            findings.append(
                {
                    "rule": "invalid_categorical",
                    "column": col,
                    "invalid_count": int(invalid_mask.sum()),
                    "invalid_values_sample": invalid[:10],
                    "valid_values": sorted(valid_set),
                }
            )
            logger.warning(
                "Column '%s': %d invalid categorical values: %s",
                col,
                int(invalid_mask.sum()),
                invalid[:5],
            )
    return findings


# ---------------------------------------------------------------------------
# Rule: dtype mismatches
# ---------------------------------------------------------------------------


def check_dtype_mismatches(
    df: pd.DataFrame,
    expected_dtypes: dict[str, str],
) -> list[Finding]:
    """Return one finding per column whose pandas dtype does not match the expectation.

    Parameters
    ----------
    df:
        The DataFrame to inspect.
    expected_dtypes:
        Mapping of ``column_name → expected_dtype_kind``.
        Kind is a single character: ``'i'`` (int), ``'f'`` (float),
        ``'O'`` (object/str), ``'M'`` (datetime), ``'b'`` (bool).

    Returns
    -------
    list[Finding]
        Each finding contains ``rule``, ``column``, ``expected_kind``,
        ``actual_dtype``.
    """
    findings: list[Finding] = []
    for col, expected_kind in expected_dtypes.items():
        if col not in df.columns:
            continue
        actual_kind = df[col].dtype.kind
        if actual_kind != expected_kind:
            findings.append(
                {
                    "rule": "dtype_mismatch",
                    "column": col,
                    "expected_kind": expected_kind,
                    "actual_dtype": str(df[col].dtype),
                }
            )
            logger.debug(
                "Column '%s': expected dtype kind '%s', got '%s'.",
                col,
                expected_kind,
                actual_kind,
            )
    return findings
