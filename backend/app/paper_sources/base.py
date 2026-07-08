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
    citation_count: int = 0
    source: str = ""
    raw_data: dict = field(default_factory=dict)


async def retry_with_backoff(
    func,
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    retry_status_codes: set[int] | None = None,
    **kwargs,
):
    """Retry an async function with exponential backoff.
    
    Args:
        func: Async function to retry.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds (doubles each retry).
        retry_status_codes: HTTP status codes that trigger retry (default: 429).
    """
    if retry_status_codes is None:
        retry_status_codes = {429, 503}
    
    import httpx
    
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except httpx.HTTPStatusError as e:
            if e.response.status_code in retry_status_codes and attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Retry %d/%d after %ds (status %d)",
                    attempt + 1, max_retries, delay, e.response.status_code,
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

