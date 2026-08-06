"""Unit tests for :mod:`loyalty_engine.llm.personalization`.

The tests exercise the deterministic (no-API-key) code paths only. Caching is
disabled per-test via monkeypatch so the on-disk ``cache.json`` is never
touched.
"""
import pandas as pd
import pytest

from loyalty_engine.llm.personalization import LLMRecommendationPersonalizer


@pytest.fixture
def personalizer(monkeypatch):
    """A personalizer with no API key and caching stubbed out."""
    p = LLMRecommendationPersonalizer(api_key=None)
    monkeypatch.setattr(p, "_read_cache", lambda key: None)
    monkeypatch.setattr(p, "_write_cache", lambda key, result: None)
    return p


# --------------------------------------------------------------------------
# _extract_rewards
# --------------------------------------------------------------------------


def test_extract_rewards_from_list_of_strings(personalizer):
    payload = {"Top 3 Rewards": ["Cashback", "Miles"]}
    assert personalizer._extract_rewards(payload) == ["Cashback", "Miles"]


def test_extract_rewards_from_list_of_dicts(personalizer):
    payload = {
        "Top 3 Rewards": [
            {"reward_name": "Cashback"},
            {"name": "Miles"},
            {"reward": "Points"},
        ]
    }
    assert personalizer._extract_rewards(payload) == ["Cashback", "Miles", "Points"]


def test_extract_rewards_from_string(personalizer):
    assert personalizer._extract_rewards({"Top 3 Rewards": "Cashback"}) == ["Cashback"]


def test_extract_rewards_from_dict(personalizer):
    payload = {"recommended_rewards": {"reward_name": "Cashback"}}
    assert personalizer._extract_rewards(payload) == ["Cashback"]


def test_extract_rewards_empty_returns_empty_list(personalizer):
    assert personalizer._extract_rewards({}) == []


# --------------------------------------------------------------------------
# _format_scores / _safe_text
# --------------------------------------------------------------------------


def test_format_scores_joins_list(personalizer):
    assert personalizer._format_scores({"Reward Scores": [0.9, 0.5]}) == "0.9, 0.5"


def test_format_scores_stringifies_non_list(personalizer):
    assert personalizer._format_scores({"Scores": 42}) == "42"


def test_safe_text_uses_fallback_for_empty_and_whitespace(personalizer):
    assert personalizer._safe_text(None, "fb") == "fb"
    assert personalizer._safe_text("   ", "fb") == "fb"
    assert personalizer._safe_text("real", "fb") == "real"


def test_safe_text_stringifies_non_string(personalizer):
    assert personalizer._safe_text(123, "fb") == "123"


# --------------------------------------------------------------------------
# _normalize_payload
# --------------------------------------------------------------------------


def test_normalize_payload_maps_snake_case_aliases(personalizer):
    payload = personalizer._normalize_payload(
        {
            "customer_id": "C1",
            "persona": "Premium Traveler",
            "predicted_cluster": 2,
            "recommended_rewards": ["Miles"],
            "business_insight": "insight",
            "Membership_Tier": "Gold",
        }
    )
    assert payload["Customer_ID"] == "C1"
    assert payload["Persona"] == "Premium Traveler"
    assert payload["Cluster"] == 2
    assert payload["Top 3 Rewards"] == ["Miles"]
    assert payload["Business Insight"] == "insight"
    assert payload["Membership Tier"] == "Gold"


def test_normalize_payload_does_not_mutate_original(personalizer):
    original = {"customer_id": "C1"}
    personalizer._normalize_payload(original)
    assert "Customer_ID" not in original


# --------------------------------------------------------------------------
# fallback messages / prompt / parse
# --------------------------------------------------------------------------


def test_fallback_message_has_all_keys_and_grounded_content(personalizer):
    payload = {"Persona": "Value Shopper", "Cluster": 1, "Top 3 Rewards": ["Cashback"]}
    result = personalizer._fallback_message(payload)

    assert set(result) == {
        "customer_insight",
        "business_justification",
        "customer_message",
        "retention_strategy",
        "upsell_opportunity",
    }
    assert "Value Shopper" in result["customer_insight"]
    assert "Cashback" in result["customer_message"]


def test_generate_message_without_api_key_returns_fallback(personalizer):
    result = personalizer.generate_message(
        {"persona": "Digital Explorer", "predicted_cluster": 3}
    )
    assert "Digital Explorer" in result["customer_insight"]
    assert "cluster 3" in result["customer_insight"]


def test_build_prompt_includes_key_fields(personalizer):
    prompt = personalizer._build_prompt(
        {"Customer_ID": "C1", "Persona": "Luxury Lifestyle", "Top 3 Rewards": ["Miles"]}
    )
    assert "C1" in prompt
    assert "Luxury Lifestyle" in prompt
    assert "Miles" in prompt


def test_parse_response_falls_back_on_invalid_json(personalizer):
    payload = {"Persona": "Value Shopper", "Cluster": 1}
    result = personalizer._parse_response("not json", payload)
    # Falls back to deterministic content when JSON is unparseable.
    assert "Value Shopper" in result["customer_insight"]


def test_parse_response_uses_provided_fields(personalizer):
    content = '{"customer_insight": "Custom insight", "customer_message": "Hi there"}'
    result = personalizer._parse_response(content, {"Persona": "X", "Cluster": 0})
    assert result["customer_insight"] == "Custom insight"
    assert result["customer_message"] == "Hi there"
    # Unspecified keys still fall back.
    assert result["retention_strategy"]


# --------------------------------------------------------------------------
# generate_batch
# --------------------------------------------------------------------------


def test_generate_batch_returns_row_per_customer(personalizer):
    df = pd.DataFrame(
        {
            "Customer_ID": ["C1", "C2"],
            "persona": ["Value Shopper", "Digital Explorer"],
            "predicted_cluster": [0, 1],
        }
    )
    out = personalizer.generate_batch(df)

    assert len(out) == 2
    assert list(out["customer_id"]) == ["C1", "C2"]
    assert "customer_insight" in out.columns


def test_generate_batch_writes_csv(personalizer, tmp_path):
    df = pd.DataFrame({"Customer_ID": ["C1"], "persona": ["Value Shopper"]})
    out_path = tmp_path / "sub" / "out.csv"

    personalizer.generate_batch(df, output_path=out_path)

    assert out_path.exists()
    reloaded = pd.read_csv(out_path)
    assert reloaded.loc[0, "customer_id"] == "C1"
