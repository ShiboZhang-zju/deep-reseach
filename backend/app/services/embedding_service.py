"""O5a: Pluggable embedding backend.

Provides a single `embed_texts()` entry point used by the RAG service. Two
backends:

  - "api": call an OpenAI-compatible /embeddings endpoint (Venus proxy or
              OpenAI). No PyTorch, so it is stable on Windows and lets us
              re-enable full-text RAG on every platform.
  - "local" : fall back to sentence-transformers (BAAI/bge-base). Kept for
              environments without API access.

All functions are synchronous and safe to call from asyncio.to_thread().
"""

from __future__ import annotations

import logging
import math

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


def _embedding_endpoint() -> str:
    if settings.embedding_api_url:
        return settings.embedding_api_url.rstrip("/")
    # Derive from the Venus LLM proxy base (…/llmproxy -> …/llmproxy/embeddings)
    base = settings.venus_llm_proxy_url.rstrip("/")
    return f"{base}/embeddings"


def _auth_header() -> dict:
    if settings.openai_api_key and "openai.com" in _embedding_endpoint():
        return {"Authorization": f"Bearer {settings.openai_api_key}"}
    token = settings.env_venus_openapi_secret_id
    if token:
        return {"Authorization": f"Bearer {token}@4083"}
    return {}


def _embed_texts_api(texts: list[str]) -> list[list[float]]:
    """Call the OpenAI-compatible embeddings API in batches."""
    endpoint = _embedding_endpoint()
    headers = {"Content-Type": "application/json", **_auth_header()}
    out: list[list[float]] = []
    batch_size = max(1, settings.embedding_batch_size)
    with httpx.Client(timeout=60) as client:
        for start in range(0, len(texts), batch_size):
            batch = [t[:8000] if t else " " for t in texts[start:start + batch_size]]
            payload = {"model": settings.embedding_model, "input": batch}
            resp = client.post(endpoint, headers=headers, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Embedding API failed: {resp.status_code} - {resp.text[:200]}"
                )
            data = resp.json()
            # OpenAI returns items in input order but include index to be safe.
            items = sorted(data["data"], key=lambda d: d.get("index", 0))
            batch_vecs = [item["embedding"] for item in items]
            # Guard: the returned dimension MUST match the configured dim, else
            # ChromaDB will later raise an opaque mismatch. Fail loud & early.
            for vec in batch_vecs:
                if len(vec) != settings.embedding_dim:
                    raise RuntimeError(
                        f"Embedding dim mismatch: API returned {len(vec)} but "
                        f"settings.embedding_dim={settings.embedding_dim}. "
                        f"Fix EMBEDDING_DIM to match model '{settings.embedding_model}'."
                    )
            out.extend(batch_vecs)
    return out


_local_model = None
_local_lock = None


def _embed_texts_local(texts: list[str]) -> list[list[float]]:
    global _local_model, _local_lock
    import threading
    if _local_lock is None:
        _local_lock = threading.Lock()
    if _local_model is None:
        with _local_lock:
            if _local_model is None:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading local embedding model BAAI/bge-base-en-v1.5 ...")
                _local_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    return _local_model.encode(texts, show_progress_bar=False, batch_size=32).tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, returning one vector per input.

    Respects settings.embedding_backend. On API failure, does NOT silently
    fall back to local (which would reintroduce the Windows segfault); instead
    the exception propagates so callers can degrade gracefully (e.g. RAG
    returns []).
    """
    if not texts:
        return []
    if settings.embedding_backend == "local":
        return _embed_texts_local(texts)
    return _embed_texts_api(texts)


def embed_text(text: str) -> list[float]:
    """Embed a single text."""
    result = embed_texts([text])
    return result[0] if result else []


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors;0.0 if either is empty/degenerate."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def embedding_enabled() -> bool:
    """Whether embedding/RAG should be considered available.

    API backend is always considered available (endpoint derived from config);
    local backend depends on sentence-transformers being importable at runtime.
    """
    if settings.embedding_backend == "api":
        return True
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False
