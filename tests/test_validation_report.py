"""Unit tests for :mod:`loyalty_engine.validation.report`."""
import json

import pandas as pd

from loyalty_engine.validation.report import (
    ColumnStats,
    SheetReport,
    ValidationReport,
    build_full_report,
    build_validation_report,
)


# --------------------------------------------------------------------------
# SheetReport convenience properties
# --------------------------------------------------------------------------


def _sheet_report_with(findings, column_stats=None):
    return SheetReport(
        sheet_name="s",
        total_rows=3,
        total_columns=2,
        column_stats=column_stats or [],
        invalid_records=findings,
    )


def test_sheet_report_duplicate_rows_property():
    report = _sheet_report_with([{"rule": "duplicate_rows", "duplicate_count": 4}])
    assert report.duplicate_rows == 4


def test_sheet_report_duplicate_rows_defaults_to_zero():
    assert _sheet_report_with([]).duplicate_rows == 0


def test_sheet_report_total_missing_sums_column_stats():
    stats = [
        ColumnStats("a", "float64", 2, 50.0, False, 1),
        ColumnStats("b", "object", 1, 25.0, False, 3),
    ]
    report = _sheet_report_with([], column_stats=stats)
    assert report.total_missing == 3


def test_sheet_report_is_clean():
    assert _sheet_report_with([]).is_clean() is True
    assert _sheet_report_with([{"rule": "missing_values"}]).is_clean() is False


def test_sheet_report_summary_lines_include_fixes():
    report = SheetReport(
        sheet_name="Demo",
        total_rows=10,
        total_columns=3,
        invalid_records=[{"rule": "duplicate_rows", "duplicate_count": 1}],
        suggested_fixes=["Remove 1 fully-duplicate rows."],
    )
    lines = report.summary_lines()
    assert "Sheet: Demo" in lines[0]
    assert any("Suggested fixes:" in ln for ln in lines)
    assert any("Remove 1 fully-duplicate rows." in ln for ln in lines)


# --------------------------------------------------------------------------
# build_validation_report
# --------------------------------------------------------------------------


def test_build_validation_report_aggregates_findings_and_fixes():
    df = pd.DataFrame(
        {
            "Customer_ID": [1, 2, 2],
            "Amount": [10.0, -5.0, 20.0],
            "Since": ["2024-01-01", "bad", "2024-02-02"],
            "Tier": ["Gold", "Diamond", "Gold"],
        }
    )

    report = build_validation_report(
        "demo",
        df,
        date_columns=["Since"],
        monetary_columns=["Amount"],
        categorical_constraints={"Tier": ["Gold", "Silver"]},
        id_column="Customer_ID",
    )

    assert report.total_rows == 3
    assert report.total_columns == 4
    rules_found = {f["rule"] for f in report.invalid_records}
    assert {
        "invalid_monetary",
        "invalid_date",
        "duplicate_customer_ids",
        "invalid_categorical",
    } <= rules_found
    # Each finding with a template produces a suggested fix.
    assert len(report.suggested_fixes) >= 4
    assert not report.is_clean()


def test_build_validation_report_clean_dataset():
    df = pd.DataFrame({"Customer_ID": [1, 2], "Amount": [10.0, 20.0]})
    report = build_validation_report("clean", df, monetary_columns=["Amount"])
    assert report.is_clean()
    assert report.suggested_fixes == []


def test_build_validation_report_column_stats_capture_dtype_and_uniques():
    df = pd.DataFrame({"a": [1, 1, None]})
    report = build_validation_report("s", df, id_column="a")
    stats = {cs.name: cs for cs in report.column_stats}
    assert stats["a"].missing_count == 1
    assert stats["a"].unique_count == 1
    assert stats["a"].is_empty is False


def test_build_validation_report_dtype_mismatch_fix_generated():
    df = pd.DataFrame({"n": ["1", "2"]})
    report = build_validation_report("s", df, expected_dtypes={"n": "f"})
    assert any("Convert column 'n'" in fix for fix in report.suggested_fixes)


# --------------------------------------------------------------------------
# ValidationReport serialization
# --------------------------------------------------------------------------


def test_validation_report_summary_and_to_dict_roundtrip():
    df = pd.DataFrame({"Customer_ID": [1, 2], "Amount": [10.0, 20.0]})
    full = build_full_report(
        {"demo": df},
        {"demo": {"monetary_columns": ["Amount"], "id_column": "Customer_ID"}},
    )

    assert isinstance(full, ValidationReport)
    assert "VALIDATION REPORT SUMMARY" in full.summary()
    as_dict = full.to_dict()
    assert "demo" in as_dict["sheets"]
    assert as_dict["sheets"]["demo"]["total_rows"] == 2


def test_validation_report_save_json_writes_file(tmp_path):
    df = pd.DataFrame({"Customer_ID": [1, 2]})
    full = build_full_report({"demo": df}, {"demo": {}})

    out = tmp_path / "nested" / "report.json"
    full.save_json(out)

    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded["sheets"]["demo"]["total_rows"] == 2
