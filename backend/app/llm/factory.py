"""LLM provider factory."""

from app.config import settings
from app.llm.base import LLMProvider
from app.llm.venus_provider import VenusProvider
from app.llm.openai_provider import OpenAIProvider

# Singleton cache
_provider_instance: LLMProvider | None = None


def get_llm() -> LLMProvider:
    """Return a cached LLM provider instance based on config."""
    global _provider_instance
    if _provider_instance is not None:
        return _provider_instance

    provider_name = settings.llm_provider.lower()
    if provider_name == "venus":
        _provider_instance = VenusProvider()
    elif provider_name == "openai":
        _provider_instance = OpenAIProvider()
    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}")
    return _provider_instance
