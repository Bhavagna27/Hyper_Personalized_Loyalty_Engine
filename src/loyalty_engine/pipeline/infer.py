from __future__ import annotations

from pathlib import Path

import pandas as pd

from loyalty_engine.features import build_training_table
from loyalty_engine.io import ExcelDatasetLoader
from loyalty_engine.models import load_bundle
from loyalty_engine.preprocessing import (
    clean_customer_profile,
    clean_recommendation_bank,
    clean_transaction_history,
)
from loyalty_engine.recommendations import RecommendationEngine
from loyalty_engine.validation import validate_workbook


def score_workbook(input_path: Path, artifact_path: Path) -> pd.DataFrame:
    frames = ExcelDatasetLoader(input_path).load_all()
    validate_workbook(frames)

    transactions = clean_transaction_history(frames["Transaction_History"])
    profile = clean_customer_profile(frames["Customer_Loyalty_Profile"])
    recommendations = clean_recommendation_bank(frames["AI_Recommendations"])
    bundle = load_bundle(artifact_path)

    features = build_training_table(profile, transactions)
    engine = RecommendationEngine(bundle=bundle, recommendation_bank=recommendations)
    scored = engine.predict_customer_states(features)
    return engine.generate_recommendations(scored)
