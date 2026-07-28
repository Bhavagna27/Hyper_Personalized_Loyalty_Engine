import pandas as pd

from loyalty_engine.models.trainers import train_customer_models


def test_supervised_training_excludes_target_leakage_columns():
    dataset = pd.DataFrame(
        {
            "Customer_ID": [1, 2, 3, 4, 5, 6],
            "Customer_Health": ["Healthy", "Stable", "Needs Attention", "Critical", "Healthy", "Stable"],
            "Churn_Risk": ["Low", "Medium", "High", "High", "Low", "Medium"],
            "customer_health_score": [100.0, 70.0, 40.0, 10.0, 100.0, 70.0],
            "churn_risk_score": [10.0, 50.0, 90.0, 90.0, 10.0, 50.0],
            "loyalty_score": [90.0, 70.0, 35.0, 20.0, 95.0, 75.0],
            "age": [25, 35, 45, 55, 30, 40],
            "income": [50_000, 60_000, 70_000, 80_000, 55_000, 65_000],
            "segment": ["A", "B", "A", "C", "B", "C"],
            "Churn": [0, 0, 1, 1, 0, 1],
            "Health": [0, 1, 0, 1, 0, 1],
        }
    )

    bundle, _, _ = train_customer_models(
        dataset=dataset,
        target_columns=("Churn", "Health"),
    )

    leaked_columns = {
        "Customer_Health",
        "Churn_Risk",
        "customer_health_score",
        "churn_risk_score",
        "loyalty_score",
    }
    assert leaked_columns.isdisjoint(bundle.feature_columns)
