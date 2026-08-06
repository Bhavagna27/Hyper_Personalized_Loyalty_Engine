import json
import logging

import pandas as pd
import pytest

from loyalty_engine import cli
from loyalty_engine.llm.personalization import LLMRecommendationPersonalizer


def test_generate_message_returns_fallback_when_no_attempt_is_made(monkeypatch, caplog):
    personalizer = LLMRecommendationPersonalizer(api_key="test-key", max_retries=0)
    monkeypatch.setattr(personalizer, "_read_cache", lambda cache_key: None)
    monkeypatch.setattr(personalizer, "_write_cache", lambda cache_key, result: None)

    with caplog.at_level(logging.ERROR):
        result = personalizer.generate_message({"Customer_ID": "C1", "Persona": "Loyalist"})

    assert result == personalizer._fallback_message(
        personalizer._normalize_payload({"Customer_ID": "C1", "Persona": "Loyalist"})
    )
    assert "LLM generation failed" in caplog.text


def test_parse_response_logs_and_falls_back_on_invalid_json(caplog):
    personalizer = LLMRecommendationPersonalizer(api_key=None)
    payload = {"Customer_ID": "C1"}

    with caplog.at_level(logging.WARNING):
        result = personalizer._parse_response("not json", payload)

    assert result == personalizer._fallback_message(payload)
    assert "not valid JSON" in caplog.text


def test_unreadable_cache_is_reported_and_ignored(monkeypatch, tmp_path, caplog):
    cache_path = tmp_path / "cache.json"
    cache_path.write_text("{ broken", encoding="utf-8")
    personalizer = LLMRecommendationPersonalizer(api_key=None)
    monkeypatch.setattr(
        "loyalty_engine.llm.personalization.Path.resolve",
        lambda self: tmp_path / "personalization.py",
    )

    with caplog.at_level(logging.WARNING):
        assert personalizer._read_cache("missing-key") is None

    assert "unreadable LLM cache" in caplog.text

    personalizer._write_cache("key", {"customer_message": "hi"})
    assert json.loads(cache_path.read_text(encoding="utf-8")) == {"key": {"customer_message": "hi"}}


def test_cli_reports_missing_workbook_instead_of_traceback(monkeypatch, tmp_path, caplog):
    missing = tmp_path / "missing.xlsx"
    monkeypatch.setattr(
        "sys.argv", ["loyalty-engine", "ingest", "--input", str(missing)]
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as excinfo:
            cli.main()

    assert excinfo.value.code == 1
    assert "file not found" in caplog.text


def test_recommendation_engine_warns_when_model_bundle_is_missing(caplog):
    from loyalty_engine.recommendations.engine import RecommendationEngine

    engine = RecommendationEngine.__new__(RecommendationEngine)
    engine.bundle = None

    with caplog.at_level(logging.WARNING):
        states = engine.predict_customer_states(pd.DataFrame({"Customer_ID": ["C1"]}))

    assert states["predicted_churn_risk"].tolist() == ["Low"]
    assert "No trained model bundle available" in caplog.text
