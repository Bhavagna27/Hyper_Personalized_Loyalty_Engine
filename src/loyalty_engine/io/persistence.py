"""Shared filesystem persistence helpers.

Centralizes the small, repeated file-I/O boilerplate used across the code
base: creating parent directories before writing, reading JSON/CSV artifacts
that may not exist, and serializing objects with ``joblib``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

__all__ = [
    "ensure_dir",
    "ensure_parent",
    "read_json",
    "write_json",
    "read_csv",
    "write_csv",
    "load_joblib",
    "dump_joblib",
]


def ensure_dir(path: str | Path) -> Path:
    """Create *path* (and parents) as a directory if it does not exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory of *path* if it does not exist.

    Returns the ``Path`` for *path* itself so callers can chain the result.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def write_json(data: Any, path: str | Path, *, indent: int = 2, default: Any = str) -> Path:
    """Serialize *data* to *path* as JSON, creating parent directories."""
    file_path = ensure_parent(path)
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, default=default)
    return file_path


def read_json(path: str | Path, *, default: Any = None) -> Any:
    """Load JSON from *path*, returning *default* if the file is absent."""
    file_path = Path(path)
    if not file_path.exists():
        return default
    with open(file_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(df: pd.DataFrame, path: str | Path, *, index: bool = False, **to_csv_kwargs: Any) -> Path:
    """Write *df* to *path* as CSV, creating parent directories."""
    file_path = ensure_parent(path)
    df.to_csv(file_path, index=index, **to_csv_kwargs)
    return file_path


def read_csv(path: str | Path, *, default: pd.DataFrame | None = None, **read_csv_kwargs: Any) -> pd.DataFrame | None:
    """Read a CSV from *path*, returning *default* if the file is absent."""
    file_path = Path(path)
    if not file_path.exists():
        return default
    return pd.read_csv(file_path, **read_csv_kwargs)


def dump_joblib(obj: Any, path: str | Path) -> Path:
    """Persist *obj* to *path* via ``joblib``, creating parent directories."""
    file_path = ensure_parent(path)
    joblib.dump(obj, file_path)
    return file_path


def load_joblib(path: str | Path, *, default: Any = None, must_exist: bool = False) -> Any:
    """Load a ``joblib`` artifact from *path*.

    If the file is absent, raise ``FileNotFoundError`` when *must_exist* is
    set, otherwise return *default*.
    """
    file_path = Path(path)
    if not file_path.exists():
        if must_exist:
            raise FileNotFoundError(f"Artifact not found at {file_path}")
        return default
    return joblib.load(file_path)
