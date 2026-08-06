from __future__ import annotations

from pathlib import Path

from loyalty_engine.config import MODEL_CONFIG
from loyalty_engine.features import build_training_table
from loyalty_engine.io import ExcelDatasetLoader, write_json
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
    write_json(evaluation, metrics_path)

    return reports

