from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib


@dataclass
class ModelBundle:
    churn_model: Any
    health_model: Any
    feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


def save_bundle(bundle: ModelBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path) -> ModelBundle:
    return joblib.load(path)

