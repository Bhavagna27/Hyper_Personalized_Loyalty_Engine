from __future__ import annotations

import json
from pathlib import Path

from loyalty_engine.config import MODEL_CONFIG
from loyalty_engine.features import build_training_table
from loyalty_engine.io import ExcelDatasetLoader
from loyalty_engine.models import save_bundle, train_customer_models
from loyalty_engine.preprocessing import (
    clean_customer_profile,
    clean_recommendation_bank,
    clean_transaction_history,
)
from loyalty_engine.validation import validate_workbook


def run_training_pipeline(input_path: Path, artifact_path: Path) -> dict[str, str]:
    frames = ExcelDatasetLoader(input_path).load_all()
    validate_workbook(frames)

    transactions = clean_transaction_history(frames["Transaction_History"])
    profile = clean_customer_profile(frames["Customer_Loyalty_Profile"])
    recommendations = clean_recommendation_bank(frames["AI_Recommendations"])

    dataset = build_training_table(profile, transactions)

    bundle, reports, evaluation = train_customer_models(
        dataset=dataset,
        target_columns=MODEL_CONFIG.target_columns,
    )
    bundle.metadata["recommendation_bank_rows"] = len(recommendations)
    save_bundle(bundle, artifact_path)

    metrics_path = Path("outputs") / "models" / "model_evaluation.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(evaluation, fh, indent=2)

    return reports

