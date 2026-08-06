"""Unit tests for :mod:`loyalty_engine.preprocessing.cleaning`."""
import numpy as np
import pandas as pd

from loyalty_engine.preprocessing.cleaning import (
    _drop_duplicates,
    _fill_numeric_with_median,
    _to_bool,
    _trim_string_columns,
    clean_customer_profile,
    clean_recommendation_bank,
    clean_transaction_history,
)


# --------------------------------------------------------------------------
# Private helpers
# --------------------------------------------------------------------------


def test_to_bool_maps_yes_no_and_leaves_unknown():
    series = pd.Series(["Yes", "No", "maybe"])
    result = _to_bool(series)
    assert result.tolist() == [1, 0, "maybe"]


def test_trim_string_columns_strips_whitespace_only_on_strings():
    df = pd.DataFrame({"s": ["  a ", "b  "], "n": [1, 2]})
    _trim_string_columns(df)
    assert df["s"].tolist() == ["a", "b"]
    assert df["n"].tolist() == [1, 2]


def test_drop_duplicates_removes_full_duplicate_rows():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    out = _drop_duplicates(df, "sheet")
    assert len(out) == 2


def test_fill_numeric_with_median_fills_when_coverage_sufficient():
    df = pd.DataFrame({"v": [10.0, 20.0, 30.0, np.nan]})
    _fill_numeric_with_median(df, ["v"])
    # median of [10, 20, 30] == 20
    assert df["v"].iloc[3] == 20.0
    assert df["v"].isna().sum() == 0


def test_fill_numeric_with_median_skips_when_coverage_too_low():
    df = pd.DataFrame({"v": [10.0, np.nan, np.nan, np.nan]})
    _fill_numeric_with_median(df, ["v"], min_coverage=0.5)
    # Only 25% coverage -> not filled.
    assert df["v"].isna().sum() == 3


def test_fill_numeric_with_median_ignores_missing_and_complete_columns():
    df = pd.DataFrame({"v": [1.0, 2.0]})
    _fill_numeric_with_median(df, ["v", "absent"])
    assert df["v"].tolist() == [1.0, 2.0]


# --------------------------------------------------------------------------
# clean_transaction_history
# --------------------------------------------------------------------------


def _raw_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Transaction_Date": ["2024-01-01 ", " 2024-03-15"],
            "Purchase_Amount": ["10.5", None],
            "Age": [30, 30],
            "Quantity": [1, 2],
            "Reward_Points_Earned": [100, 120],
            "Reward_Points_Redeemed": [0, 10],
            "Reward_Points_Available": [100, 110],
            "Reward_Points_Expired": [0, 0],
            "Product_Viewed": [1, 1],
            "Wishlist_Added": [0, 1],
            "Cart_Abandoned": [0, 0],
            "Email_Clicked": [1, 0],
            "Push_Notification_Clicked": [0, 1],
            "App_Opened": [1, 1],
            "Website_Visits": [3, 4],
            "Session_Duration_Min": [10.0, 12.0],
            "Customer_Since": ["2020-01-01", "2020-01-01"],
            "Coupon_Used": ["Yes", "No"],
        }
    )


def test_clean_transaction_history_parses_dates_and_coupon():
    cleaned = clean_transaction_history(_raw_transactions())

    assert cleaned.dtypes["Transaction_Date"].kind == "M"
    assert cleaned.dtypes["Customer_Since"].kind == "M"
    assert cleaned["Coupon_Used"].tolist() == [1, 0]
    # Purchase_Amount coerced to numeric and median-filled (single value 10.5).
    assert cleaned["Purchase_Amount"].isna().sum() == 0
    assert cleaned["Purchase_Amount"].iloc[1] == 10.5


def test_clean_transaction_history_does_not_mutate_input():
    raw = _raw_transactions()
    before = raw.copy()
    clean_transaction_history(raw)
    pd.testing.assert_frame_equal(raw, before)


# --------------------------------------------------------------------------
# clean_customer_profile
# --------------------------------------------------------------------------


def test_clean_customer_profile_coerces_numeric_and_fills_median():
    raw = pd.DataFrame(
        {
            "Customer_Since": ["2019-05-01", "2020-06-01"],
            "Age": ["30", "40"],
            "Total_Transactions_6M": [5, 7],
            "Total_Spend_6M": [500.0, None],
            "Average_Order_Value": [100.0, 120.0],
            "Reward_Utilization": [0.5, 0.6],
            "Reward_Expiry_Risk": [0.1, 0.2],
            "Purchase_Frequency": [1.0, 2.0],
            "Days_Since_Last_Purchase": [3, 9],
        }
    )

    cleaned = clean_customer_profile(raw)

    assert cleaned.dtypes["Customer_Since"].kind == "M"
    assert cleaned["Age"].tolist() == [30, 40]
    # Total_Spend_6M: one null of two rows -> 50% coverage -> filled with median.
    assert cleaned["Total_Spend_6M"].isna().sum() == 0


# --------------------------------------------------------------------------
# clean_recommendation_bank
# --------------------------------------------------------------------------


def test_clean_recommendation_bank_casts_string_and_fills_blanks():
    raw = pd.DataFrame(
        {
            "Customer_ID": [" C1 ", "C2"],
            "Recommended_Action": ["offer", None],
        }
    )

    cleaned = clean_recommendation_bank(raw)

    assert cleaned["Customer_ID"].tolist() == ["C1", "C2"]
    assert cleaned["Recommended_Action"].tolist() == ["offer", ""]
    assert all(str(dt) == "string" for dt in cleaned.dtypes)
