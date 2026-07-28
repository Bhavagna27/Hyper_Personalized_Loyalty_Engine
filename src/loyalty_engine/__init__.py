"""Hyper-personalized loyalty recommendation engine."""

from loyalty_engine.llm import LLMRecommendationPersonalizer, generate_batch, generate_message

__all__ = ["__version__", "LLMRecommendationPersonalizer", "generate_message", "generate_batch"]
__version__ = "0.1.0"

