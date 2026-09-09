"""The venus provider's hard outer timeout must fire even when httpx's own
timeout machinery silently never does (proactor IO completion loss,
2026-09-09: a request hung for 2.5+ hours inside client.post)."""

import asyncio

import pytest

from app.llm.venus_provider import VenusProvider


class _HangingClient:
    def __init__(self):
        self.calls = 0

    async def post(self, *args, **kwargs):
        self.calls += 1
        # Simulate the lost-completion hang: a future that never resolves and
        # is never cancelled from the inside.
        await asyncio.Future()


def _make_provider(monkeypatch, payload_ok=True):
    from app.llm.venus_provider import VenusProvider

    provider = VenusProvider()
    hanging = _HangingClient()

    async def _get_client():
        return hanging

    monkeypatch.setattr(provider, "_get_client", _get_client)
    return provider, hanging


@pytest.mark.asyncio
async def test_hard_timeout_fires_when_hung(monkeypatch):
    from app.llm import venus_provider as module

    monkeypatch.setattr(module, "_HARD_TIMEOUT_S", 0.05)
    provider, hanging = _make_provider(monkeypatch)

    with pytest.raises(RuntimeError, match="venus hard timeout"):
        await provider._post({"model": "m", "messages": []})
    assert hanging.calls == 1


@pytest.mark.asyncio
async def test_hard_timeout_does_not_break_normal_calls(monkeypatch):
    from app.llm import venus_provider as module

    monkeypatch.setattr(module, "_HARD_TIMEOUT_S", 5)
    provider = VenusProvider()

    class _FastClient:
        async def post(self, *args, **kwargs):
            import types

            resp = types.SimpleNamespace(
                status_code=200,
                text="",
                json=lambda: {"choices": [{"message": {"content": "ok"}}],
                              "usage": {"prompt_tokens": 5,
                                        "completion_tokens": 2,
                                        "total_tokens": 7}},
            )
            return resp

    async def _get_client():
        return _FastClient()

    monkeypatch.setattr(provider, "_get_client", _get_client)
    data = await provider._post({"model": "m", "messages": []})
    assert data["choices"][0]["message"]["content"] == "ok"
