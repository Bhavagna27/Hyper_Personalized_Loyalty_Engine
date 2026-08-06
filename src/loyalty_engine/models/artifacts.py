from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loyalty_engine.io.persistence import dump_joblib, load_joblib


@dataclass
class ModelBundle:
    churn_model: Any
    health_model: Any
    feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


def save_bundle(bundle: ModelBundle, path: Path) -> None:
    dump_joblib(bundle, path)


def load_bundle(path: Path) -> ModelBundle:
    return load_joblib(path, must_exist=True)

