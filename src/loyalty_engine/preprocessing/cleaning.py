"""Safe, idempotent cleaning functions for each worksheet.

All three public functions follow the same contract:

* Accept a raw :class:`pandas.DataFrame`.
* Return a **new** DataFrame (the original is never mutated).
* Apply only *safe* transformations:
  - Trim whitespace on string columns.
  - Drop fully-duplicate rows.
  - Coerce columns to their expected dtypes.
  - Fill *numeric* columns with their column median only where a value is
    missing *and* the column has sufficient coverage (≥ 50 % non-null).

Example
-------
>>> import pandas as pd
>>> from loyalty_engine.preprocessing import clean_transaction_history
>>> raw = pd.DataFrame({
...     "Transaction_Date": ["2024-01-01 ", " 2024-03-15"],
...     "Purchase_Amount": ["10.5", None],
...     "Age": [30, 30],
...     "Quantity": [1, 1],
...     "Reward_Points_Earned": [100, 100],
...     "Reward_Points_Redeemed": [0, 0],
...     "Reward_Points_Available": [100, 100],
...     "Reward_Points_Expired": [0, 0],
...     "Product_Viewed": [1, 1],
...     "Wishlist_Added": [0, 0],
...     "Cart_Abandoned": [0, 0],
...     "Email_Clicked": [1, 1],
...     "Push_Notification_Clicked": [0, 0],
...     "App_Opened": [1, 1],
...     "Website_Visits": [3, 3],
...     "Session_Duration_Min": [10.0, 10.0],
...     "Customer_Since": ["2020-01-01", "2020-01-01"],
...     "Coupon_Used": ["Yes", "No"],
... })
>>> cleaned = clean_transaction_history(raw)
>>> cleaned.dtypes["Transaction_Date"]
dtype('<M8[ns]')
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_bool(series: pd.Series) -> pd.Series:
    """Map Yes/No/True/False/1/0 → 1/0, leaving unrecognised values as-is."""
    return series.map(
        {
            "Yes": 1,
            "No": 0,
            True: 1,
            False: 0,
            1: 1,
            0: 0,
        }
    ).fillna(series)


def _trim_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from every object/string column.

    Parameters
    ----------
    df:
        DataFrame to trim (mutated in place and returned for chaining).

    Returns
    -------
    pd.DataFrame
    """
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df[col] = df[col].str.strip()
    if len(str_cols):
        logger.debug("Trimmed whitespace from %d string column(s).", len(str_cols))
    return df


def _drop_duplicates(df: pd.DataFrame, sheet_name: str = "") -> pd.DataFrame:
    """Drop fully-duplicate rows and log the result.

    Parameters
    ----------
    df:
        DataFrame to deduplicate.
    sheet_name:
        Optional label used in log messages.

    Returns
    -------
    pd.DataFrame
        DataFrame with duplicates removed.
    """
    before = len(df)
    df = df.drop_duplicates(keep="first")
    removed = before - len(df)
    if removed:
        logger.warning(
            "[%s] Removed %d duplicate row(s). %d row(s) remain.",
            sheet_name or "sheet",
            removed,
            len(df),
        )
    else:
        logger.debug("[%s] No duplicate rows found.", sheet_name or "sheet")
    return df


def _fill_numeric_with_median(
    df: pd.DataFrame,
    numeric_cols: list[str],
    min_coverage: float = 0.5,
) -> pd.DataFrame:
    """Fill missing values in numeric columns with the column median.

    Only fills columns whose non-null ratio meets *min_coverage* to avoid
    propagating a median from too few observations.

    Parameters
    ----------
    df:
        DataFrame to fill (mutated in place and returned for chaining).
    numeric_cols:
        Column names to consider.
    min_coverage:
        Minimum fraction of non-null values required before filling.
        Default is 0.5 (50 %).

    Returns
    -------
    pd.DataFrame
    """
    total = len(df)
    for col in numeric_cols:
        if col not in df.columns:
            continue
        n_null = int(df[col].isna().sum())
        if n_null == 0:
            continue
        coverage = 1 - n_null / total if total else 0.0
        if coverage >= min_coverage:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.debug(
                "Filled %d missing value(s) in '%s' with median %.4f.",
                n_null,
                col,
                median_val,
            )
        else:
            logger.warning(
                "Skipped median-fill for '%s' — coverage %.1f%% < %.0f%%.",
                col,
                coverage * 100,
                min_coverage * 100,
            )
    return df


# ---------------------------------------------------------------------------
# Public cleaning functions
# ---------------------------------------------------------------------------


def clean_transaction_history(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the ``Transaction_History`` worksheet.

    Transformations applied (in order):

    1. Trim whitespace on all string columns.
    2. Drop fully-duplicate rows.
    3. Parse ``Transaction_Date`` and ``Customer_Since`` as datetime.
    4. Coerce numeric columns to float/int.
    5. Normalise ``Coupon_Used`` to 0/1.
    6. Fill missing numeric values with column median (≥ 50 % coverage).

    Parameters
    ----------
    df:
        Raw ``Transaction_History`` DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned copy; the input is never mutated.
    """
    cleaned = df.copy()
    _trim_string_columns(cleaned)
    cleaned = _drop_duplicates(cleaned, "Transaction_History")

    cleaned["Transaction_Date"] = pd.to_datetime(
        cleaned["Transaction_Date"], errors="coerce"
    )
    cleaned["Customer_Since"] = pd.to_datetime(
        cleaned["Customer_Since"], errors="coerce"
    )

    numeric_cols = [
        "Age",
        "Quantity",
        "Purchase_Amount",
        "Reward_Points_Earned",
        "Reward_Points_Redeemed",
        "Reward_Points_Available",
        "Reward_Points_Expired",
        "Product_Viewed",
        "Wishlist_Added",
        "Cart_Abandoned",
        "Email_Clicked",
        "Push_Notification_Clicked",
        "App_Opened",
        "Website_Visits",
        "Session_Duration_Min",
    ]
    for col in numeric_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    if "Coupon_Used" in cleaned.columns:
        cleaned["Coupon_Used"] = _to_bool(cleaned["Coupon_Used"])

    _fill_numeric_with_median(cleaned, numeric_cols)

    logger.info(
        "clean_transaction_history: %d rows ready.", len(cleaned)
    )
    return cleaned


def clean_customer_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the ``Customer_Loyalty_Profile`` worksheet.

    Transformations applied (in order):

    1. Trim whitespace on all string columns.
    2. Drop fully-duplicate rows.
    3. Parse ``Customer_Since`` as datetime.
    4. Coerce numeric columns to float.
    5. Fill missing numeric values with column median (≥ 50 % coverage).

    Parameters
    ----------
    df:
        Raw ``Customer_Loyalty_Profile`` DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned copy; the input is never mutated.
    """
    cleaned = df.copy()
    _trim_string_columns(cleaned)
    cleaned = _drop_duplicates(cleaned, "Customer_Loyalty_Profile")

    cleaned["Customer_Since"] = pd.to_datetime(
        cleaned["Customer_Since"], errors="coerce"
    )

    numeric_cols = [
        "Age",
        "Total_Transactions_6M",
        "Total_Spend_6M",
        "Average_Order_Value",
        "Reward_Utilization",
        "Reward_Expiry_Risk",
        "Purchase_Frequency",
        "Days_Since_Last_Purchase",
    ]
    for col in numeric_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    _fill_numeric_with_median(cleaned, numeric_cols)

    logger.info(
        "clean_customer_profile: %d rows ready.", len(cleaned)
    )
    return cleaned


def clean_recommendation_bank(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the ``AI_Recommendations`` worksheet.

    Transformations applied (in order):

    1. Trim whitespace on all string columns.
    2. Drop fully-duplicate rows.
    3. Cast every column to the ``string`` dtype and fill nulls with ``""``.

    Parameters
    ----------
    df:
        Raw ``AI_Recommendations`` DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned copy; the input is never mutated.
    """
    cleaned = df.copy()
    _trim_string_columns(cleaned)
    cleaned = _drop_duplicates(cleaned, "AI_Recommendations")

    for col in cleaned.columns:
        cleaned[col] = cleaned[col].astype("string").fillna("")

    logger.info(
        "clean_recommendation_bank: %d rows ready.", len(cleaned)
    )
    return cleaned
