import pandas as pd

from loyalty_engine.recommendations.engine import RecommendationEngine


def test_message_from_bank_handles_missing_business_issue_column():
    engine = RecommendationEngine.__new__(RecommendationEngine)
    engine.recommendation_bank = pd.DataFrame(
        [{"Recommended_Action": "retention offer", "Customer_Message": "Custom retention follow-up"}]
    )

    assert engine._message_from_bank("retention_offer") == "Custom retention follow-up"
