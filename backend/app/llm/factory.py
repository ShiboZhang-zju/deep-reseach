"""LLM provider factory."""

from app.config import settings
from app.llm.base import LLMProvider
from app.llm.venus_provider import VenusProvider
from app.llm.openai_provider import OpenAIProvider
from app.llm.fallback_provider import FallbackLLMProvider

# Singleton cache
_provider_instance: LLMProvider | None = None


def get_llm() -> LLMProvider:
    """Return a cached LLM provider instance based on config."""
    global _provider_instance
    if _provider_instance is not None:
        return _provider_instance

    provider_name = settings.llm_provider.lower()
    if provider_name == "venus":
        primary = VenusProvider()
        if settings.fallback_venus_llm_model:
            fallback = VenusProvider(
                base_url=settings.fallback_venus_llm_proxy_url,
                model=settings.fallback_venus_llm_model,
            )
            _provider_instance = FallbackLLMProvider([primary, fallback])
        else:
            _provider_instance = primary
    elif provider_name == "openai":
        _provider_instance = OpenAIProvider()
    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}")
    return _provider_instance
