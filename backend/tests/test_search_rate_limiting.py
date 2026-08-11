"""Tests for how paper search reacts to rate limiting (429)."""

import asyncio
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.paper_sources.base import parse_retry_after, retry_with_backoff


def _response(status, headers=None):
    return httpx.Response(status, headers=headers or {},
                          request=httpx.Request("GET", "https://example.org"))


def _raise(status, headers=None):
    response = _response(status, headers)
    raise httpx.HTTPStatusError("boom", request=response.request, response=response)


@pytest.mark.asyncio
async def test_429_is_not_retried_with_blind_backoff():
    """A rate limit means "unavailable now", so retrying in-round is waste.

    Measured against OpenAlex: every query spent its full retry budget only to
    fail again, while the source was rate limited at the IP level.
    """
    calls = []

    async def always_rate_limited():
        calls.append(1)
        _raise(429)

    with pytest.raises(httpx.HTTPStatusError):
        await retry_with_backoff(always_rate_limited, max_retries=3, base_delay=0.01)

    assert len(calls) == 1, "429 must fail fast and let the caller cool the source down"


@pytest.mark.asyncio
async def test_503_is_still_retried():
    calls = []

    async def flaky():
        calls.append(1)
        if len(calls) < 3:
            _raise(503)
        return {"ok": True}

    assert await retry_with_backoff(flaky, max_retries=3, base_delay=0.01) == {"ok": True}
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_short_retry_after_is_honoured_even_for_429():
    """An explicit Retry-After is evidence about when the source is usable."""
    calls = []

    async def rate_limited_once():
        calls.append(1)
        if len(calls) == 1:
            _raise(429, {"Retry-After": "0.01"})
        return {"ok": True}

    assert await retry_with_backoff(rate_limited_once, max_retries=3,
                                    base_delay=10.0) == {"ok": True}
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_long_retry_after_is_not_waited_out():
    calls = []

    async def rate_limited_long():
        calls.append(1)
        _raise(429, {"Retry-After": "600"})

    with pytest.raises(httpx.HTTPStatusError):
        await retry_with_backoff(rate_limited_long, max_retries=3, base_delay=0.01,
                                 max_retry_after=5.0)
    assert len(calls) == 1


def test_parse_retry_after_ignores_http_date_and_garbage():
    assert parse_retry_after(_response(429, {"Retry-After": "12"})) == 12.0
    assert parse_retry_after(_response(429, {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})) is None
    assert parse_retry_after(_response(429)) is None


class _CountingSource:
    """Fails with 429 on the first call, counts every call it receives."""

    name = "flaky-source"

    def __init__(self):
        self.calls = 0

    def is_available(self):
        return True

    async def search(self, query, limit=15):
        self.calls += 1
        await asyncio.sleep(0.02)
        _raise(429)


@pytest.mark.asyncio
async def test_queries_queued_behind_the_rate_limiter_skip_a_cooled_down_source(monkeypatch):
    """Queries that waited for a token must re-check the cooldown before calling.

    Scope note: this only helps when the token bucket actually makes callers
    queue (bucket drained). A full bucket releases a whole batch at once with no
    time gap, so nothing can be known yet and every query still issues one
    request — that waste is bounded by failing fast on 429 instead, which is
    covered by test_429_is_not_retried_with_blind_backoff.
    """
    from app.services import search_service as module

    service = module.SearchService.__new__(module.SearchService)
    service._cooldowns = {}
    service._cooldown_s = 60
    source = _CountingSource()
    service.sources = [source]

    gate = asyncio.Lock()

    async def _serialised_acquire(name):
        # A drained bucket releases callers one at a time; that gap is the window
        # in which an earlier query marks the source as rate limited.
        async with gate:
            await asyncio.sleep(0.05)

    monkeypatch.setattr(module.rate_limiter, "acquire", _serialised_acquire)

    results = await asyncio.gather(*[
        service._search_with_cache(source, f"query {index}", 5) for index in range(5)
    ])

    assert all(result == [] for result in results)
    assert source.calls == 1, "only the query that discovered the limit may call the source"
    assert service._is_cooled_down(source.name)


def test_cooldown_respects_a_longer_server_retry_after():
    from app.services import search_service as module

    service = module.SearchService.__new__(module.SearchService)
    service._cooldowns = {}
    service._cooldown_s = 60

    service._mark_cooldown("s2", retry_after=300.0)
    remaining = service._cooldowns["s2"] - __import__("time").time()
    assert 290 < remaining <= 300

    service._mark_cooldown("openalex", retry_after=1.0)
    remaining = service._cooldowns["openalex"] - __import__("time").time()
    assert 50 < remaining <= 60, "a shorter Retry-After must not weaken the default cooldown"
