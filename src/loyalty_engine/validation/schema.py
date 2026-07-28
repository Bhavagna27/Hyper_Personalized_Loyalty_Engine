"""Schema definitions and workbook-level validation for the loyalty engine.

This module provides:

* ``REQUIRED_SHEETS`` — ordered tuple of expected sheet names.
* ``REQUIRED_COLUMNS`` — required column names per sheet.
* ``DATE_COLUMNS`` — columns expected to hold date values per sheet.
* ``MONETARY_COLUMNS`` — columns expected to hold non-negative numeric values.
* ``NUMERIC_COLUMNS`` — all columns that must be numeric per sheet.
* ``CATEGORICAL_VALUES`` — mapping of column → valid value set per sheet.
* :func:`validate_workbook` — raises on structural failures and returns a
  :class:`~loyalty_engine.validation.report.ValidationReport`.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

import pandas as pd

from loyalty_engine.validation.report import (
    ValidationReport,
    build_full_report,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sheet & column presence
# ---------------------------------------------------------------------------

REQUIRED_SHEETS = (
    "Transaction_History",
    "Customer_Loyalty_Profile",
    "AI_Recommendations",
)

REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "Transaction_History": (
        "Customer_ID",
        "Transaction_ID",
        "Transaction_Date",
        "Membership_Tier",
        "Customer_Since",
        "Age",
        "Gender",
        "City",
        "Occupation",
        "Income_Level",
        "Store_Channel",
        "Brand",
        "Product_Category",
        "Product_Name",
        "Quantity",
        "Purchase_Amount",
        "Coupon_Used",
        "Reward_Points_Earned",
        "Reward_Points_Redeemed",
        "Reward_Points_Available",
        "Reward_Points_Expired",
        "Reward_Source",
        "Product_Viewed",
        "Wishlist_Added",
        "Cart_Abandoned",
        "Email_Clicked",
        "Push_Notification_Clicked",
        "App_Opened",
        "Website_Visits",
        "Session_Duration_Min",
    ),
    "Customer_Loyalty_Profile": (
        "Customer_ID",
        "Membership_Tier",
        "Customer_Since",
        "Age",
        "Gender",
        "City",
        "Occupation",
        "Income_Level",
        "Total_Transactions_6M",
        "Total_Spend_6M",
        "Average_Order_Value",
        "Preferred_Brand",
        "Reward_Utilization",
        "Reward_Expiry_Risk",
        "Purchase_Frequency",
        "Loyalty_Engagement",
        "Days_Since_Last_Purchase",
        "Upgrade_Readiness",
        "Customer_Health",
        "Churn_Risk",
    ),
    "AI_Recommendations": (
        "Customer_ID",
        "Customer_Insight",
        "Business_Issue",
        "Recommended_Action",
        "Expected_Impact",
        "Customer_Message",
    ),
}


# ---------------------------------------------------------------------------
# Date columns per sheet
# ---------------------------------------------------------------------------

DATE_COLUMNS: dict[str, list[str]] = {
    "Transaction_History": ["Transaction_Date", "Customer_Since"],
    "Customer_Loyalty_Profile": ["Customer_Since"],
    "AI_Recommendations": [],
}


# ---------------------------------------------------------------------------
# Monetary columns per sheet (must be non-negative numeric)
# ---------------------------------------------------------------------------

MONETARY_COLUMNS: dict[str, list[str]] = {
    "Transaction_History": [
        "Purchase_Amount",
        "Reward_Points_Earned",
        "Reward_Points_Redeemed",
        "Reward_Points_Available",
        "Reward_Points_Expired",
    ],
    "Customer_Loyalty_Profile": [
        "Total_Spend_6M",
        "Average_Order_Value",
    ],
    "AI_Recommendations": [],
}


# ---------------------------------------------------------------------------
# Numeric columns per sheet
# ---------------------------------------------------------------------------

NUMERIC_COLUMNS: dict[str, list[str]] = {
    "Transaction_History": [
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
    ],
    "Customer_Loyalty_Profile": [
        "Age",
        "Total_Transactions_6M",
        "Total_Spend_6M",
        "Average_Order_Value",
        "Reward_Utilization",
        "Reward_Expiry_Risk",
        "Purchase_Frequency",
        "Days_Since_Last_Purchase",
    ],
    "AI_Recommendations": [],
}


# ---------------------------------------------------------------------------
# Valid categorical values per sheet
# ---------------------------------------------------------------------------

CATEGORICAL_VALUES: dict[str, dict[str, list[str]]] = {
    "Transaction_History": {
        "Membership_Tier": ["Bronze", "Silver", "Gold", "Platinum"],
        "Gender": ["Male", "Female", "Other"],
        "Income_Level": ["Low", "Medium", "High"],
        "Store_Channel": ["Online", "In-Store", "Mobile App"],
        "Coupon_Used": ["Yes", "No"],
    },
    "Customer_Loyalty_Profile": {
        "Membership_Tier": ["Bronze", "Silver", "Gold", "Platinum"],
        "Gender": ["Male", "Female", "Other"],
        "Income_Level": ["Low", "Medium", "High"],
        "Upgrade_Readiness": ["Low", "Medium", "High"],
        "Customer_Health": ["At Risk", "Stable", "Healthy"],
        "Churn_Risk": ["Low", "Medium", "High"],
    },
    "AI_Recommendations": {},
}


# ---------------------------------------------------------------------------
# Sheet-level validation configs (passed to build_full_report)
# ---------------------------------------------------------------------------

def _sheet_configs() -> dict[str, dict[str, Any]]:
    """Return per-sheet keyword-argument dicts for :func:`build_full_report`."""
    return {
        "Transaction_History": {
            "date_columns": DATE_COLUMNS["Transaction_History"],
            "monetary_columns": MONETARY_COLUMNS["Transaction_History"],
            "categorical_constraints": CATEGORICAL_VALUES["Transaction_History"],
            "id_column": "Customer_ID",
        },
        "Customer_Loyalty_Profile": {
            "date_columns": DATE_COLUMNS["Customer_Loyalty_Profile"],
            "monetary_columns": MONETARY_COLUMNS["Customer_Loyalty_Profile"],
            "categorical_constraints": CATEGORICAL_VALUES["Customer_Loyalty_Profile"],
            "id_column": "Customer_ID",
        },
        "AI_Recommendations": {
            "date_columns": DATE_COLUMNS["AI_Recommendations"],
            "monetary_columns": MONETARY_COLUMNS["AI_Recommendations"],
            "categorical_constraints": CATEGORICAL_VALUES["AI_Recommendations"],
            "id_column": "Customer_ID",
        },
    }


# ---------------------------------------------------------------------------
# Public validator
# ---------------------------------------------------------------------------


def validate_workbook(
    frames: Mapping[str, pd.DataFrame],
) -> ValidationReport:
    """Validate structural integrity and data quality of all sheets.

    **Structural checks** (raises :exc:`ValueError` on failure):

    * All required sheets are present.
    * All required columns are present per sheet.

    **Data-quality checks** (reported, never raised):

    * Missing values
    * Duplicate rows
    * Empty columns
    * Invalid dates
    * Invalid monetary values
    * Duplicate Customer IDs
    * Invalid categorical values

    Parameters
    ----------
    frames:
        Dict of sheet name → raw DataFrame, as returned by
        :meth:`~loyalty_engine.io.ExcelDatasetLoader.load_all`.

    Returns
    -------
    ValidationReport
        Full report — callers can ignore this for backward-compatibility.

    Raises
    ------
    ValueError
        When required sheets or columns are absent.
    """
    # --- structural checks ------------------------------------------------
    missing_sheets = [s for s in REQUIRED_SHEETS if s not in frames]
    if missing_sheets:
        raise ValueError(f"Missing required sheets: {missing_sheets}")

    for sheet_name, required_cols in REQUIRED_COLUMNS.items():
        if sheet_name not in frames:
            continue
        actual_cols = set(frames[sheet_name].columns)
        missing_cols = [c for c in required_cols if c not in actual_cols]
        if missing_cols:
            raise ValueError(
                f"Sheet '{sheet_name}' is missing columns: {missing_cols}"
            )

    logger.info("Structural validation passed for all required sheets.")

    # --- data-quality checks ----------------------------------------------
    report = build_full_report(dict(frames), _sheet_configs())
    logger.info("Data-quality validation complete.\n%s", report.summary())
    return report
