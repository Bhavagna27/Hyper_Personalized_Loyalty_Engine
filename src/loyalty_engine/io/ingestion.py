"""IngestionPipeline — single entry-point for the full ingest workflow.

The pipeline executes four stages in order:

1. **Load** — auto-discover and read every worksheet via
   :class:`~loyalty_engine.io.ExcelDatasetLoader`.
2. **Validate** — run structural + data-quality checks via
   :func:`~loyalty_engine.validation.validate_workbook`.
3. **Clean** — apply safe transformations (trim, dedup, type-coercions) via the
   ``preprocessing`` module.
4. **Save** — write cleaned DataFrames to ``data/processed/`` as UTF-8 CSV
   files and persist the validation report as JSON to ``reports/``.

Usage
-----
Programmatic::

    from pathlib import Path
    from loyalty_engine.io import IngestionPipeline

    result = IngestionPipeline(Path("data/workbook.xlsx")).run()
    print(result.report.summary())

CLI::

    python -m loyalty_engine.cli ingest --input data/workbook.xlsx

Returned :class:`IngestionResult` carries:

* ``frames``        — raw DataFrames (dict)
* ``cleaned_frames``— cleaned DataFrames (dict)
* ``report``        — :class:`~loyalty_engine.validation.ValidationReport`
* ``saved_paths``   — list of :class:`pathlib.Path` objects written to disk
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from loyalty_engine.config import PATHS
from loyalty_engine.io.excel_loader import ExcelDatasetLoader
from loyalty_engine.io.persistence import ensure_dir, write_csv
from loyalty_engine.preprocessing import (
    clean_customer_profile,
    clean_recommendation_bank,
    clean_transaction_history,
)
from loyalty_engine.validation import ValidationReport, validate_workbook

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging setup helper
# ---------------------------------------------------------------------------


def configure_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> None:
    """Configure the root logger with a consistent format.

    Call this once at application start-up (the CLI does this automatically).

    Parameters
    ----------
    level:
        Logging level (e.g. ``logging.DEBUG``).
    log_file:
        Optional path for a file handler.  Parent directories are created.
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)
    logger.debug("Logging configured at level %s.", logging.getLevelName(level))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class IngestionResult(NamedTuple):
    """Immutable result returned by :meth:`IngestionPipeline.run`.

    Attributes
    ----------
    frames:
        Raw DataFrames keyed by sheet name.
    cleaned_frames:
        Cleaned DataFrames keyed by sheet name.
    report:
        Aggregated :class:`~loyalty_engine.validation.ValidationReport`.
    saved_paths:
        List of file paths written during :meth:`IngestionPipeline.save_processed`.
    """

    frames: dict[str, pd.DataFrame]
    cleaned_frames: dict[str, pd.DataFrame]
    report: ValidationReport
    saved_paths: list[Path]


# ---------------------------------------------------------------------------
# Cleaner registry
# ---------------------------------------------------------------------------

#: Maps each sheet name to its dedicated cleaning function.
_CLEANERS: dict[str, object] = {
    "Transaction_History": clean_transaction_history,
    "Customer_Loyalty_Profile": clean_customer_profile,
    "AI_Recommendations": clean_recommendation_bank,
}


def _clean_frames(
    frames: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Apply the appropriate cleaner to every sheet.

    Sheets that don't have a dedicated cleaner are passed through unchanged
    (copy), so the pipeline remains resilient to extra sheets from
    auto-discovery.

    Parameters
    ----------
    frames:
        Raw DataFrames keyed by sheet name.

    Returns
    -------
    dict[str, pd.DataFrame]
        Cleaned DataFrames with identical keys.
    """
    cleaned: dict[str, pd.DataFrame] = {}
    for name, df in frames.items():
        cleaner = _CLEANERS.get(name)
        if cleaner is not None:
            logger.info("Cleaning sheet '%s' …", name)
            cleaned[name] = cleaner(df)  # type: ignore[operator]
        else:
            logger.debug(
                "No dedicated cleaner for sheet '%s' — passing through unchanged.", name
            )
            cleaned[name] = df.copy()
    return cleaned


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class IngestionPipeline:
    """Orchestrates the full ingest → validate → clean → save workflow.

    Parameters
    ----------
    workbook_path:
        Path to the source ``.xlsx`` workbook.
    processed_dir:
        Directory where cleaned CSVs will be written.
        Defaults to ``data/processed/`` (from :data:`~loyalty_engine.config.PATHS`).
    report_dir:
        Directory where the JSON validation report will be written.
        Defaults to ``reports/`` (from :data:`~loyalty_engine.config.PATHS`).
    auto_discover:
        If ``True`` (default), use :meth:`~ExcelDatasetLoader.load_all_sheets`
        to load every sheet.  If ``False``, only the three standard sheets are
        loaded via :meth:`~ExcelDatasetLoader.load_all`.

    Example
    -------
    >>> from pathlib import Path
    >>> from loyalty_engine.io import IngestionPipeline
    >>> result = IngestionPipeline(Path("data/workbook.xlsx")).run()
    >>> print(result.report.summary())
    """

    workbook_path: Path
    processed_dir: Path = field(default_factory=lambda: PATHS.processed_dir)
    report_dir: Path = field(default_factory=lambda: PATHS.reports_dir)
    auto_discover: bool = True

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, pd.DataFrame]:
        """Stage 1 — load worksheets from the workbook."""
        loader = ExcelDatasetLoader(self.workbook_path)
        if self.auto_discover:
            return loader.load_all_sheets()
        return loader.load_all()

    def _validate(
        self, frames: dict[str, pd.DataFrame]
    ) -> ValidationReport:
        """Stage 2 — run structural + data-quality validation."""
        return validate_workbook(frames)

    def _clean(
        self, frames: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Stage 3 — apply safe cleaning transformations."""
        return _clean_frames(frames)

    # ------------------------------------------------------------------
    # Save helper
    # ------------------------------------------------------------------

    def save_processed(
        self,
        cleaned_frames: dict[str, pd.DataFrame],
        *,
        processed_dir: Path | None = None,
    ) -> list[Path]:
        """Write every cleaned DataFrame to a UTF-8 CSV file.

        Parameters
        ----------
        cleaned_frames:
            Dict of sheet name → cleaned DataFrame.
        processed_dir:
            Override the instance-level ``processed_dir`` if supplied.

        Returns
        -------
        list[Path]
            Paths of the written CSV files.
        """
        out_dir = ensure_dir(processed_dir or self.processed_dir)
        saved: list[Path] = []
        for name, df in cleaned_frames.items():
            dest = out_dir / f"{name}.csv"
            write_csv(df, dest, encoding="utf-8")
            logger.info("Saved cleaned '%s' → %s  (%d rows)", name, dest, len(df))
            saved.append(dest)
        return saved

    # ------------------------------------------------------------------
    # Report helper
    # ------------------------------------------------------------------

    def save_report(
        self,
        report: ValidationReport,
        *,
        report_dir: Path | None = None,
        filename: str = "validation_report.json",
    ) -> Path:
        """Persist the validation report as JSON.

        Parameters
        ----------
        report:
            The :class:`~loyalty_engine.validation.ValidationReport` to save.
        report_dir:
            Override the instance-level ``report_dir`` if supplied.
        filename:
            JSON filename inside *report_dir*.

        Returns
        -------
        Path
            Path of the written JSON file.
        """
        out_dir = report_dir or self.report_dir
        dest = out_dir / filename
        report.save_json(dest)
        return dest

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        save: bool = True,
        log_file: Path | None = None,
    ) -> IngestionResult:
        """Execute all four pipeline stages and return an :class:`IngestionResult`.

        Parameters
        ----------
        save:
            If ``True`` (default), write cleaned CSVs and validation JSON to
            disk.
        log_file:
            If provided, append pipeline logs to this file in addition to
            stderr.

        Returns
        -------
        IngestionResult
            Immutable named-tuple with ``frames``, ``cleaned_frames``,
            ``report``, and ``saved_paths``.
        """
        if log_file:
            configure_logging(log_file=log_file)

        logger.info("=" * 60)
        logger.info("IngestionPipeline starting: %s", self.workbook_path)
        logger.info("=" * 60)

        # Stage 1 — Load
        frames = self._load()

        # Stage 2 — Validate
        report = self._validate(frames)

        # Stage 3 — Clean
        cleaned_frames = self._clean(frames)

        # Stage 4 — Save
        saved_paths: list[Path] = []
        if save:
            saved_paths.extend(self.save_processed(cleaned_frames))
            report_path = self.save_report(report)
            saved_paths.append(report_path)
            logger.info("Pipeline complete. %d file(s) written.", len(saved_paths))
        else:
            logger.info("Pipeline complete (save=False — no files written).")

        return IngestionResult(
            frames=frames,
            cleaned_frames=cleaned_frames,
            report=report,
            saved_paths=saved_paths,
        )
