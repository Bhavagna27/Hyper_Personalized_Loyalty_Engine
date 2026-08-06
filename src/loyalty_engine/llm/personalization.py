from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from loyalty_engine.io.persistence import read_json, write_csv, write_json

logger = logging.getLogger(__name__)


class LLMRecommendationPersonalizer:
    """Generate grounded customer-facing explanations from structured recommendation payloads.

    This layer never decides rewards. It only transforms the existing hybrid recommendation
    output into natural-language insights and messages. It uses the OpenAI API when available,
    otherwise falls back to a deterministic template.
    """

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini", max_retries: int = 3) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.max_retries = max_retries

    def generate_message(self, customer_payload: dict[str, Any]) -> dict[str, Any]:
        """Generate grounded copy for a single customer payload."""
        payload = self._normalize_payload(customer_payload)
        cache_key = self._cache_key(payload)
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        if not self.api_key:
            result = self._fallback_message(payload)
            self._write_cache(cache_key, result)
            return result

        try:
            import openai

            client = openai.OpenAI(api_key=self.api_key)
            last_error: Exception | None = None
            for attempt in range(self.max_retries):
                try:
                    response = client.responses.create(
                        model=self.model,
                        input=self._build_prompt(payload),
                        temperature=0.2,
                    )
                    content = getattr(response, "output_text", "") or ""
                    result = self._parse_response(content, payload)
                    self._write_cache(cache_key, result)
                    return result
                except Exception as exc:  # pragma: no cover - runtime fallback
                    last_error = exc
                    logger.warning("LLM attempt %s failed: %s", attempt + 1, exc)
            if last_error is not None:
                raise last_error
        except Exception as exc:  # pragma: no cover - runtime fallback
            logger.warning("LLM generation failed, using fallback: %s", exc)
            result = self._fallback_message(payload)
            self._write_cache(cache_key, result)
            return result

    def generate_batch(self, df: pd.DataFrame, output_path: str | Path | None = None) -> pd.DataFrame:
        """Generate grounded copy for a batch of recommendation payloads."""
        rows: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            payload = self.generate_message(row.to_dict())
            rows.append({
                "customer_id": row.get("Customer_ID") if "Customer_ID" in row else row.get("customer_id"),
                **payload,
            })

        output = pd.DataFrame(rows)
        if output_path is not None:
            write_csv(output, output_path)
        return output

    def _normalize_payload(self, customer_payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(customer_payload)
        if "Customer_ID" not in payload and "customer_id" in payload:
            payload["Customer_ID"] = payload["customer_id"]
        if "Persona" not in payload and "persona" in payload:
            payload["Persona"] = payload["persona"]
        if "Cluster" not in payload and "predicted_cluster" in payload:
            payload["Cluster"] = payload["predicted_cluster"]
        if "Top 3 Rewards" not in payload and "recommended_rewards" in payload:
            payload["Top 3 Rewards"] = payload["recommended_rewards"]
        if "Reward Scores" not in payload and "Scores" in payload:
            payload["Reward Scores"] = payload["Scores"]
        if "Business Insight" not in payload and "business_insight" in payload:
            payload["Business Insight"] = payload["business_insight"]
        if "Expected Impact" not in payload and "expected_impact" in payload:
            payload["Expected Impact"] = payload["expected_impact"]
        if "Favorite Category" not in payload and "favorite_category" in payload:
            payload["Favorite Category"] = payload["favorite_category"]
        if "Membership Tier" not in payload and "Membership_Tier" in payload:
            payload["Membership Tier"] = payload["Membership_Tier"]
        return payload

    def _build_prompt(self, payload: dict[str, Any]) -> str:
        approved_rewards = self._extract_rewards(payload)
        approved_reward_text = ", ".join(approved_rewards) if approved_rewards else "unavailable"
        return (
            "You are a marketing copy assistant for a loyalty program. "
            "Use ONLY the structured recommendation data provided. "
            "Do not invent rewards, percentages, customer behavior, loyalty tiers, or business insights. "
            "If a field is missing, say 'unavailable'. "
            f"Customer ID: {payload.get('Customer_ID', 'unavailable')}\n"
            f"Persona: {payload.get('Persona', 'unavailable')}\n"
            f"Cluster: {payload.get('Cluster', 'unavailable')}\n"
            f"Top 3 Rewards: {approved_reward_text}\n"
            f"Reward Scores: {self._format_scores(payload)}\n"
            f"Business Insight: {payload.get('Business Insight', 'unavailable')}\n"
            f"Expected Impact: {payload.get('Expected Impact', 'unavailable')}\n"
            f"Favorite Category: {payload.get('Favorite Category', 'unavailable')}\n"
            f"Membership Tier: {payload.get('Membership Tier', 'unavailable')}\n"
            "Return JSON with keys: customer_insight, business_justification, customer_message, retention_strategy, upsell_opportunity."
        )

    def _parse_response(self, content: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {}

        return {
            "customer_insight": self._safe_text(parsed.get("customer_insight"), self._fallback_customer_insight(payload)),
            "business_justification": self._safe_text(parsed.get("business_justification"), self._fallback_business_justification(payload)),
            "customer_message": self._safe_text(parsed.get("customer_message"), self._fallback_customer_message(payload)),
            "retention_strategy": self._safe_text(parsed.get("retention_strategy"), self._fallback_retention(payload)),
            "upsell_opportunity": self._safe_text(parsed.get("upsell_opportunity"), self._fallback_upsell(payload)),
        }

    def _fallback_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "customer_insight": self._fallback_customer_insight(payload),
            "business_justification": self._fallback_business_justification(payload),
            "customer_message": self._fallback_customer_message(payload),
            "retention_strategy": self._fallback_retention(payload),
            "upsell_opportunity": self._fallback_upsell(payload),
        }

    def _fallback_customer_insight(self, payload: dict[str, Any]) -> str:
        persona = payload.get("Persona") or payload.get("persona") or "unavailable"
        cluster = payload.get("Cluster") or payload.get("cluster") or "unavailable"
        return f"Customer profile indicates {persona} behavior in cluster {cluster}."

    def _fallback_business_justification(self, payload: dict[str, Any]) -> str:
        rewards = self._extract_rewards(payload)
        reward_text = ", ".join(rewards) if rewards else "unavailable"
        return f"These recommendations were selected because the structured profile supports the following rewards: {reward_text}."

    def _fallback_customer_message(self, payload: dict[str, Any]) -> str:
        rewards = self._extract_rewards(payload)
        reward_text = ", ".join(rewards) if rewards else "unavailable"
        return f"We have selected rewards tailored to your profile, including {reward_text}."

    def _fallback_retention(self, payload: dict[str, Any]) -> str:
        return "Maintain engagement with timely, relevant offers aligned to the customer profile."

    def _fallback_upsell(self, payload: dict[str, Any]) -> str:
        return "Use the recommended rewards to deepen engagement and encourage continued spending."

    def _extract_rewards(self, payload: dict[str, Any]) -> list[str]:
        raw_rewards = payload.get("Top 3 Rewards") or payload.get("recommended_rewards") or []
        if isinstance(raw_rewards, str):
            return [raw_rewards]
        if isinstance(raw_rewards, dict):
            return [str(raw_rewards.get("reward_name") or raw_rewards.get("name") or raw_rewards)]
        if isinstance(raw_rewards, list):
            rewards: list[str] = []
            for item in raw_rewards:
                if isinstance(item, dict):
                    reward_name = item.get("reward_name") or item.get("name") or item.get("reward")
                    if reward_name:
                        rewards.append(str(reward_name))
                else:
                    rewards.append(str(item))
            return rewards
        return []

    def _format_scores(self, payload: dict[str, Any]) -> str:
        scores = payload.get("Reward Scores") or payload.get("Scores") or []
        if isinstance(scores, list):
            return ", ".join(str(score) for score in scores)
        return str(scores)

    def _safe_text(self, value: Any, fallback: str) -> str:
        if not value:
            return fallback
        if isinstance(value, str):
            return value.strip() or fallback
        return str(value)

    def _cache_key(self, payload: dict[str, Any]) -> str:
        return json.dumps(payload, sort_keys=True, default=str)

    def _read_cache(self, cache_key: str) -> dict[str, Any] | None:
        cache_path = Path(__file__).resolve().parent / "cache.json"
        try:
            data = read_json(cache_path)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        return data.get(cache_key)

    def _write_cache(self, cache_key: str, result: dict[str, Any]) -> None:
        cache_path = Path(__file__).resolve().parent / "cache.json"
        try:
            data = read_json(cache_path, default={})
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        data[cache_key] = result
        write_json(data, cache_path)


def generate_message(customer_payload: dict[str, Any]) -> dict[str, Any]:
    return LLMRecommendationPersonalizer().generate_message(customer_payload)


def generate_batch(df: pd.DataFrame, output_path: str | Path | None = None) -> pd.DataFrame:
    return LLMRecommendationPersonalizer().generate_batch(df, output_path=output_path)
