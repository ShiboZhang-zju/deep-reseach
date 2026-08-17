"""A provider that fails over to alternates when the primary backend is down."""

import logging
from typing import Type, TypeVar

from pydantic import BaseModel

from app.llm.base import LLMBudgetExceeded, LLMContextOverflow, LLMProvider

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class FallbackLLMProvider(LLMProvider):
    """Wrap a list of providers and fail over on transport/service errors.

    The primary backend's inference worker can go down (observed: the gateway
    returned a `connection refused` on its upstream :8021 port). A call then
    raises a plain ``RuntimeError("LLM call failed ...")``; this wrapper catches
    that and retries on the next provider, so a transient outage no longer aborts
    a whole research task.

    Deterministic errors are NOT retried: ``LLMContextOverflow`` (prompt too
    large) and ``LLMBudgetExceeded`` (per-task limit) would fail identically on
    every backend, so they are re-raised immediately.
    """

    def __init__(self, providers: list[LLMProvider]):
        super().__init__()
        if not providers:
            raise ValueError("FallbackLLMProvider needs at least one provider")
        self.providers = providers
        self._active_index = 0

    @property
    def active(self) -> LLMProvider:
        """The provider that last succeeded (or the primary, before any call)."""
        return self.providers[self._active_index]

    def _sync_state(self, provider: LLMProvider) -> None:
        """Mirror the successful provider's usage counters onto this wrapper."""
        self.last_usage = provider.last_usage
        self.total_tokens_used = provider.total_tokens_used
        self.call_count = provider.call_count

    async def _run(self, method_name: str, *args, **kwargs):
        last_error: Exception | None = None
        for i, provider in enumerate(self.providers):
            try:
                result = await getattr(provider, method_name)(*args, **kwargs)
                self._active_index = i
                self._sync_state(provider)
                return result
            except (LLMBudgetExceeded, LLMContextOverflow):
                raise
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "LLM provider %d/%d (%s) failed (%s); trying next",
                    i + 1, len(self.providers), type(provider).__name__, exc,
                )
        assert last_error is not None
        raise last_error

    async def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        return await self._run("chat", messages, temperature)

    async def chat_json(
        self, messages: list[dict], schema: Type[T], temperature: float = 0.3
    ) -> T:
        return await self._run("chat_json", messages, schema, temperature)
