"""Excel workbook loader for the loyalty engine.

This module provides :class:`ExcelDatasetLoader`, which can load individual
named sheets or **auto-discover every sheet** in a workbook.

Example
-------
>>> from pathlib import Path
>>> from loyalty_engine.io import ExcelDatasetLoader
>>> loader = ExcelDatasetLoader(Path("data/Consultant_Loyalty_Dataset_200_Customers.xlsx"))
>>> frames = loader.load_all()               # load the three required sheets
>>> all_frames = loader.load_all_sheets()    # auto-discover every sheet
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


DEFAULT_SHEETS = (
    "Transaction_History",
    "Customer_Loyalty_Profile",
    "AI_Recommendations",
)


@dataclass
class ExcelDatasetLoader:
    """Load the source workbook into typed pandas DataFrames.

    Parameters
    ----------
    path:
        Absolute or relative path to the ``.xlsx`` workbook.

    Attributes
    ----------
    path:
        The resolved workbook path.
    """

    path: Path

    # ------------------------------------------------------------------
    # Single-sheet loader
    # ------------------------------------------------------------------

    def load_sheet(self, sheet_name: str) -> pd.DataFrame:
        """Load a single worksheet by name.

        Parameters
        ----------
        sheet_name:
            Exact name of the worksheet tab.

        Returns
        -------
        pd.DataFrame
            Raw DataFrame — no type coercions applied.

        Raises
        ------
        ValueError
            If *sheet_name* is not found in the workbook.
        """
        logger.debug("Loading sheet '%s' from %s …", sheet_name, self.path)
        df = pd.read_excel(self.path, sheet_name=sheet_name, engine="openpyxl")
        logger.info(
            "Loaded sheet '%s': %d rows × %d cols.", sheet_name, len(df), len(df.columns)
        )
        return df

    # ------------------------------------------------------------------
    # Fixed-name loader (backward-compatible)
    # ------------------------------------------------------------------

    def load_all(self) -> dict[str, pd.DataFrame]:
        """Load the three standard worksheets by their well-known names.

        Returns
        -------
        dict[str, pd.DataFrame]
            Keys are :data:`DEFAULT_SHEETS`; values are raw DataFrames.
        """
        logger.info("Loading %d standard sheets from %s …", len(DEFAULT_SHEETS), self.path)
        frames = {sheet: self.load_sheet(sheet) for sheet in DEFAULT_SHEETS}
        logger.info("All standard sheets loaded.")
        return frames

    # ------------------------------------------------------------------
    # Auto-discovery loader (NEW)
    # ------------------------------------------------------------------

    def discover_sheet_names(self) -> list[str]:
        """Return every worksheet name present in the workbook.

        Parameters
        ----------
        (none)

        Returns
        -------
        list[str]
            Sheet names in workbook order.
        """
        with pd.ExcelFile(self.path, engine="openpyxl") as xl:
            names = xl.sheet_names
        logger.info("Discovered %d sheet(s) in %s: %s", len(names), self.path, names)
        return names

    def load_all_sheets(self) -> dict[str, pd.DataFrame]:
        """Auto-discover and load **every** worksheet in the workbook.

        Unlike :meth:`load_all`, this method does not assume a fixed list of
        sheet names — it reads whatever is present in the file.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of ``sheet_name → raw DataFrame``.

        Example
        -------
        >>> loader = ExcelDatasetLoader(Path("data/workbook.xlsx"))
        >>> frames = loader.load_all_sheets()
        >>> list(frames.keys())
        ['Transaction_History', 'Customer_Loyalty_Profile', 'AI_Recommendations']
        """
        sheet_names = self.discover_sheet_names()
        frames: dict[str, pd.DataFrame] = {}
        for name in sheet_names:
            frames[name] = self.load_sheet(name)
        logger.info(
            "Loaded %d sheet(s) via auto-discovery: %s",
            len(frames),
            list(frames.keys()),
        )
        return frames
