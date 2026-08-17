"""Paper source abstract base class."""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RawPaper:
    """Raw paper data from a single source before normalization."""
    title: str = ""
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    semantic_scholar_id: str = ""
    openalex_id: str = ""
    url: str = ""
    pdf_url: str = ""
    # Open-access flag: None = unknown, True = open access, False = paywalled.
    # A paywalled paper without an arXiv preprint or an OA pdf_url cannot be
    # downloaded, so it should be deprioritised at scoring time rather than
    # admitted into the evidence pool only to fail PDF fetch later.
    is_oa: bool | None = None
    citation_count: int = 0
    source: str = ""
    raw_data: dict = field(default_factory=dict)


def parse_retry_after(response) -> float | None:
    """Seconds to wait per the server's Retry-After header, if it gave one.

    Only the delta-seconds form is honoured; an HTTP-date is ignored because a
    date far in the future is useless to a single search round.
    """
    raw = (response.headers.get("Retry-After") or "").strip()
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


async def retry_with_backoff(
    func,
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    retry_status_codes: set[int] | None = None,
    max_retry_after: float = 5.0,
    **kwargs,
):
    """Retry an async function with exponential backoff.

    Args:
        func: Async function to retry.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds (doubles each retry).
        retry_status_codes: HTTP status codes that trigger blind backoff
            (default: 503 only — see below).
        max_retry_after: Longest server-requested Retry-After that is still
            worth waiting for inside a single search round.

    429 is deliberately NOT retried with blind backoff. A rate limit means the
    source is unavailable for now, not that it will recover in a few seconds:
    retrying within the same round cannot restore quota and only adds load. The
    caller (search_service) reacts by putting the whole source into cooldown, so
    the remaining queries skip it instead of each burning the full retry budget.
    Measured against OpenAlex, three retries per query were spent to always fail.
    The one exception is an explicit, short Retry-After: the server then states
    when it is usable again, which is evidence rather than guesswork.
    """
    if retry_status_codes is None:
        retry_status_codes = {503}

    import httpx

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            retry_after = parse_retry_after(e.response)
            honour_retry_after = (
                retry_after is not None
                and retry_after <= max_retry_after
                and attempt < max_retries
            )
            if honour_retry_after:
                logger.warning("Retry %d/%d after %.1fs (status %d, server Retry-After)",
                               attempt + 1, max_retries, retry_after, status)
                await asyncio.sleep(retry_after)
                continue
            if status in retry_status_codes and attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Retry %d/%d after %ds (status %d)",
                    attempt + 1, max_retries, delay, status,
                )
                await asyncio.sleep(delay)
                continue
            raise
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning("Retry %d/%d after %ds (%s)", attempt + 1, max_retries, delay, e)
                await asyncio.sleep(delay)
                continue
            raise
    # Should not reach here, but satisfy type checker
    return await func(*args, **kwargs)


class PaperSource(ABC):
    """Abstract base for paper data sources."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    async def search(self, query: str, limit: int = 15) -> list[RawPaper]:
        """Search papers by query string."""
        ...

    def is_available(self) -> bool:
        """Whether this source can actually return results in the current config.

        Sources that require an API key should override this to return False
        when the key is missing, so the search service can skip loading them
        (avoiding wasted requests) and report an honest active-source count.
        """
        return True

