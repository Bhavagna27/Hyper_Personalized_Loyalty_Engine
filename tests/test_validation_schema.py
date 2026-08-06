"""Unit tests for :mod:`loyalty_engine.validation.schema`."""
import pandas as pd
import pytest

from loyalty_engine.validation import schema
from loyalty_engine.validation.report import ValidationReport
from loyalty_engine.validation.schema import (
    REQUIRED_COLUMNS,
    REQUIRED_SHEETS,
    validate_workbook,
)


def _valid_frames() -> dict[str, pd.DataFrame]:
    """Build a minimal, structurally-valid set of frames (one row per sheet)."""
    frames: dict[str, pd.DataFrame] = {}
    for sheet, cols in REQUIRED_COLUMNS.items():
        frames[sheet] = pd.DataFrame({col: ["x"] for col in cols})
    return frames


def test_validate_workbook_returns_report_for_valid_frames():
    report = validate_workbook(_valid_frames())
    assert isinstance(report, ValidationReport)
    assert set(report.sheet_reports) == set(REQUIRED_SHEETS)


def test_validate_workbook_raises_on_missing_sheet():
    frames = _valid_frames()
    del frames["AI_Recommendations"]
    with pytest.raises(ValueError, match="Missing required sheets"):
        validate_workbook(frames)


def test_validate_workbook_raises_on_missing_column():
    frames = _valid_frames()
    frames["Transaction_History"] = frames["Transaction_History"].drop(
        columns=["Customer_ID"]
    )
    with pytest.raises(ValueError, match="missing columns"):
        validate_workbook(frames)


def test_sheet_configs_cover_all_required_sheets():
    configs = schema._sheet_configs()
    assert set(configs) == set(REQUIRED_SHEETS)
    for cfg in configs.values():
        assert cfg["id_column"] == "Customer_ID"
        assert "date_columns" in cfg
        assert "monetary_columns" in cfg
        assert "categorical_constraints" in cfg


def test_schema_constants_are_internally_consistent():
    # Every sheet declared in REQUIRED_SHEETS has a column spec, and every
    # per-sheet mapping is keyed by the same set of sheets.
    assert set(REQUIRED_COLUMNS) == set(REQUIRED_SHEETS)
    assert set(schema.DATE_COLUMNS) == set(REQUIRED_SHEETS)
    assert set(schema.MONETARY_COLUMNS) == set(REQUIRED_SHEETS)
    assert set(schema.NUMERIC_COLUMNS) == set(REQUIRED_SHEETS)
    assert set(schema.CATEGORICAL_VALUES) == set(REQUIRED_SHEETS)
