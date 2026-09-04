"""Tests for the token-bucket wakeup race (P2: no negative-token burst)."""
import asyncio
import time

import pytest

from app.services.rate_limiter import TokenBucket


@pytest.mark.asyncio
async def test_wakeup_race_does_not_overshoot_quota(monkeypatch):
    """N waiters waking on the same refilled token: exactly one may take it.
    The old implementation let every loser subtract anyway (negative tokens),
    instantly firing N-1 requests over the quota."""
    # 1 token, refills 30/min = 0.5/s: after the first taker, the next token
    # needs ~2s. Fake sleep so the test stays fast: each sleep advances the
    # fake clock by 2s (enough for exactly one refill while capacity=1).
    clock = {"now": 0.0}
    bucket = TokenBucket(rate_per_min=30)
    bucket.tokens = 1.0
    bucket.last_refill = clock["now"]

    real_sleep = asyncio.sleep

    async def fake_sleep(seconds: float) -> None:
        clock["now"] += max(seconds, 2.0)
        await real_sleep(0)

    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def one() -> bool:
        await bucket.acquire()
        return True

    results = await asyncio.gather(*(one() for _ in range(3)))

    assert all(results)
    # Three sequential acquisitions, each gated by a real refill tick: tokens
    # must never dip below zero.
    assert bucket.tokens >= 0.0
