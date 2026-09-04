"""Global rate limiter for paper source API calls (P1-9).

Problem: Each source does its own retry_with_backoff, but there's no global
budget. Idea validation phase can trigger 5 ideas × 3 baselines × 6 sources = 90
requests on top of normal search, easily hitting S2/OpenAlex 429.

Solution: Per-source token bucket rate limiter. Each search call acquires a
token before hitting the API. Tokens refill at a configurable rate per minute.
"""

import asyncio
import logging
import time
from collections import defaultdict

from app.config import settings

logger = logging.getLogger(__name__)


class TokenBucket:
    """Async token bucket rate limiter.

    Tokens refill at `rate_per_min / 60` per second, up to `capacity` (burst).
    Acquire blocks until a token is available.
    """

    def __init__(self, rate_per_min: int, burst_multiplier: float = 1.5):
        self.rate_per_sec = rate_per_min / 60.0
        self.capacity = int(rate_per_min * burst_multiplier)
        self.tokens = float(self.capacity)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Acquire one token, blocking if necessary."""
        wait_time = 0.0
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self.last_refill
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_sec)
                self.last_refill = now

                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return

                # Need to wait for a token
                wait_time = (1.0 - self.tokens) / self.rate_per_sec
                logger.debug("Rate limit: waiting %.2fs for token (tokens=%.1f)",
                            wait_time, self.tokens)

            # Release lock while waiting, then RE-CHECK under the lock: when N
            # waiters wake simultaneously only one may take the refilled token.
            # The old code let every loser go negative — instant N-1 extra
            # requests over the quota, right at the 429 boundary.
            await asyncio.sleep(wait_time)


class RateLimiterRegistry:
    """Registry of per-source token buckets."""

    def __init__(self):
        self._buckets: dict[str, TokenBucket] = {}
        self._default_rate = settings.rate_limit_default_per_min

    def get_bucket(self, source_name: str) -> TokenBucket:
        """Get or create a token bucket for a source."""
        if source_name not in self._buckets:
            # Determine rate based on source
            rate = self._default_rate
            name_lower = source_name.lower()
            if "semantic" in name_lower or "s2" in name_lower:
                rate = settings.effective_s2_rate_per_min
            elif "openalex" in name_lower:
                rate = settings.effective_openalex_rate_per_min
            # arXiv, Crossref, IEEE, CORE use default
            self._buckets[source_name] = TokenBucket(rate_per_min=rate)
            logger.info("Rate limiter for '%s': %d req/min", source_name, rate)
        return self._buckets[source_name]

    async def acquire(self, source_name: str):
        """Acquire a token for the given source."""
        bucket = self.get_bucket(source_name)
        await bucket.acquire()


# Singleton
rate_limiter = RateLimiterRegistry()
