"""Guardrails for loading serialized model artifacts.

``joblib``/pickle payloads execute arbitrary code while being deserialized, so
artifacts are only loaded from directories the project owns. Additional
directories can be allow-listed through the
``LOYALTY_ENGINE_TRUSTED_ARTIFACT_DIRS`` environment variable (``os.pathsep``
separated).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib

from loyalty_engine.config import PATHS

TRUSTED_DIRS_ENV = "LOYALTY_ENGINE_TRUSTED_ARTIFACT_DIRS"


def trusted_artifact_roots() -> tuple[Path, ...]:
    """Return the directories artifacts may be deserialized from."""
    roots = [PATHS.artifacts_dir, PATHS.root / "outputs"]
    extra = os.getenv(TRUSTED_DIRS_ENV, "")
    roots.extend(Path(entry).expanduser() for entry in extra.split(os.pathsep) if entry.strip())
    return tuple(root.resolve() for root in roots)


def is_trusted_artifact_path(path: str | Path) -> bool:
    resolved = Path(path).expanduser().resolve()
    return any(resolved == root or resolved.is_relative_to(root) for root in trusted_artifact_roots())


def load_artifact(path: str | Path) -> Any:
    """Deserialize a joblib artifact after verifying it lives in a trusted directory."""
    resolved = Path(path).expanduser().resolve()
    if not is_trusted_artifact_path(resolved):
        raise PermissionError(
            f"Refusing to deserialize artifact outside trusted directories: {resolved}. "
            f"Allow-list its directory via {TRUSTED_DIRS_ENV}."
        )
    return joblib.load(resolved)
