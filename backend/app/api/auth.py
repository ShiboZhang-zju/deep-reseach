"""API Key authentication middleware.

P0-4: Minimal authentication to protect sensitive endpoints.
Configured via API_KEY environment variable. If not set, auth is disabled
(for local development only — a warning is logged at startup).
"""

import os
import logging

from fastapi import Request, HTTPException
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Header name for API key
API_KEY_HEADER = "X-API-Key"

# Paths that require authentication (mutating / resource-consuming endpoints)
_PROTECTED_PREFIXES = (
    "/api/tasks",  # Create, start, stop tasks (consumes LLM quota)
)

# Paths that are always public
_PUBLIC_PATHS = (
    "/", "/health", "/docs", "/openapi.json", "/redoc",
)

_api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def get_api_key() -> str | None:
    """Get the configured API key from environment. None means auth disabled."""
    return os.environ.get("API_KEY", "") or None


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Middleware that validates API key for protected endpoints.

    If API_KEY env var is not set, auth is disabled (with a warning).
    GET requests to non-task endpoints (papers, reports, ideas, etc.) are
    allowed without auth for read-only access.
    """

    def __init__(self, app, api_key: str | None = None):
        super().__init__(app)
        self.api_key = api_key or get_api_key()

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Always allow public paths
        if path in _PUBLIC_PATHS or path.startswith("/paper_assets"):
            return await call_next(request)

        # If no API key configured, auth is disabled
        if not self.api_key:
            return await call_next(request)

        # Protect task-related endpoints (create/start/stop consumes resources)
        is_protected = any(path.startswith(prefix) for prefix in _PROTECTED_PREFIXES)

        if not is_protected:
            return await call_next(request)

        # Only protect mutating methods on task endpoints
        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            provided_key = request.headers.get(API_KEY_HEADER)
            if provided_key != self.api_key:
                raise HTTPException(
                    status_code=401,
                    detail=f"Invalid or missing API key. Provide '{API_KEY_HEADER}' header.",
                )

        return await call_next(request)
