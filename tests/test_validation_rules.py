"""Unit tests for :mod:`loyalty_engine.validation.rules`.

These rule functions are pure (never mutate their input) and each returns a
list of finding dicts. The tests cover both the "issue present" and "clean"
branches for every rule.
"""
import pandas as pd

from loyalty_engine.validation.rules import (
    check_duplicate_customer_ids,
    check_duplicate_rows,
    check_dtype_mismatches,
    check_empty_columns,
    check_invalid_categoricals,
    check_invalid_dates,
    check_invalid_monetary,
    check_missing_values,
)


# --------------------------------------------------------------------------
# check_missing_values
# --------------------------------------------------------------------------


def test_check_missing_values_reports_per_column_counts_and_pct():
    df = pd.DataFrame({"a": [1, None, None, 4], "b": ["x", "y", "z", "w"]})

    findings = check_missing_values(df)

    assert findings == [
        {
            "rule": "missing_values",
            "column": "a",
            "missing_count": 2,
            "missing_pct": 50.0,
        }
    ]


def test_check_missing_values_returns_empty_when_no_nulls():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert check_missing_values(df) == []


def test_check_missing_values_does_not_mutate_input():
    df = pd.DataFrame({"a": [1, None]})
    before = df.copy()
    check_missing_values(df)
    pd.testing.assert_frame_equal(df, before)


# --------------------------------------------------------------------------
# check_duplicate_rows
# --------------------------------------------------------------------------


def test_check_duplicate_rows_counts_and_samples_indices():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})

    findings = check_duplicate_rows(df)

    assert len(findings) == 1
    finding = findings[0]
    assert finding["rule"] == "duplicate_rows"
    assert finding["duplicate_count"] == 1
    assert finding["duplicate_indices_sample"] == [1]


def test_check_duplicate_rows_returns_empty_when_unique():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert check_duplicate_rows(df) == []


# --------------------------------------------------------------------------
# check_empty_columns
# --------------------------------------------------------------------------


def test_check_empty_columns_flags_all_null_columns_only():
    df = pd.DataFrame({"empty": [None, None], "partial": [1, None], "full": [1, 2]})

    findings = check_empty_columns(df)

    assert findings == [{"rule": "empty_column", "column": "empty"}]


# --------------------------------------------------------------------------
# check_invalid_dates
# --------------------------------------------------------------------------


def test_check_invalid_dates_detects_unparseable_strings():
    df = pd.DataFrame({"d": ["2024-01-01", "not-a-date", "2024-03-15"]})

    findings = check_invalid_dates(df, ["d"])

    assert len(findings) == 1
    assert findings[0]["rule"] == "invalid_date"
    assert findings[0]["column"] == "d"
    assert findings[0]["invalid_count"] == 1
    assert findings[0]["invalid_samples"] == ["not-a-date"]


def test_check_invalid_dates_uses_na_mask_for_datetime_dtype():
    df = pd.DataFrame({"d": pd.to_datetime(["2024-01-01", None])})

    findings = check_invalid_dates(df, ["d"])

    assert findings[0]["invalid_count"] == 1


def test_check_invalid_dates_skips_missing_columns_and_valid_dates():
    df = pd.DataFrame({"d": ["2024-01-01", "2024-02-02"]})
    assert check_invalid_dates(df, ["d", "not_present"]) == []


# --------------------------------------------------------------------------
# check_invalid_monetary
# --------------------------------------------------------------------------


def test_check_invalid_monetary_detects_negative_and_non_numeric():
    df = pd.DataFrame({"amount": [10.0, -5.0, "oops", 20.0]})

    findings = check_invalid_monetary(df, ["amount"])

    assert len(findings) == 1
    finding = findings[0]
    assert finding["rule"] == "invalid_monetary"
    assert finding["negative_count"] == 1
    assert finding["non_numeric_count"] == 1
    assert -5.0 in finding["samples"]
    assert "oops" in finding["samples"]


def test_check_invalid_monetary_clean_when_all_non_negative_numeric():
    df = pd.DataFrame({"amount": [0.0, 10.0, 20.5]})
    assert check_invalid_monetary(df, ["amount"]) == []


# --------------------------------------------------------------------------
# check_duplicate_customer_ids
# --------------------------------------------------------------------------


def test_check_duplicate_customer_ids_reports_duplicated_ids():
    df = pd.DataFrame({"Customer_ID": ["A", "B", "A", "C"]})

    findings = check_duplicate_customer_ids(df)

    assert len(findings) == 1
    finding = findings[0]
    assert finding["rule"] == "duplicate_customer_ids"
    assert finding["id_column"] == "Customer_ID"
    # Both "A" rows are counted (keep=False).
    assert finding["duplicate_count"] == 2
    assert finding["duplicate_ids_sample"] == ["A"]


def test_check_duplicate_customer_ids_missing_column_returns_empty():
    df = pd.DataFrame({"other": [1, 2]})
    assert check_duplicate_customer_ids(df) == []


def test_check_duplicate_customer_ids_custom_column():
    df = pd.DataFrame({"uid": [1, 1, 2]})
    findings = check_duplicate_customer_ids(df, id_column="uid")
    assert findings[0]["id_column"] == "uid"


# --------------------------------------------------------------------------
# check_invalid_categoricals
# --------------------------------------------------------------------------


def test_check_invalid_categoricals_reports_out_of_vocab_values():
    df = pd.DataFrame({"Tier": ["Gold", "Diamond", "Gold", "Wood"]})

    findings = check_invalid_categoricals(df, {"Tier": ["Gold", "Silver", "Bronze"]})

    assert len(findings) == 1
    finding = findings[0]
    assert finding["rule"] == "invalid_categorical"
    assert finding["invalid_count"] == 2
    assert set(finding["invalid_values_sample"]) == {"Diamond", "Wood"}
    assert finding["valid_values"] == ["Bronze", "Gold", "Silver"]


def test_check_invalid_categoricals_clean_when_all_in_vocab():
    df = pd.DataFrame({"Tier": ["Gold", "Silver"]})
    assert check_invalid_categoricals(df, {"Tier": ["Gold", "Silver"]}) == []


def test_check_invalid_categoricals_ignores_nulls_and_missing_columns():
    df = pd.DataFrame({"Tier": ["Gold", None]})
    assert check_invalid_categoricals(df, {"Tier": ["Gold"], "Missing": ["x"]}) == []


# --------------------------------------------------------------------------
# check_dtype_mismatches
# --------------------------------------------------------------------------


def test_check_dtype_mismatches_flags_wrong_kind():
    df = pd.DataFrame({"n": ["1", "2"], "f": [1.0, 2.0]})

    findings = check_dtype_mismatches(df, {"n": "f", "f": "f"})

    assert len(findings) == 1
    finding = findings[0]
    assert finding["column"] == "n"
    assert finding["expected_kind"] == "f"
    assert finding["actual_dtype"] == "object"


def test_check_dtype_mismatches_clean_and_skips_missing_columns():
    df = pd.DataFrame({"f": [1.0, 2.0]})
    assert check_dtype_mismatches(df, {"f": "f", "absent": "i"}) == []
