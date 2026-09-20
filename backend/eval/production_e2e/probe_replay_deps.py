"""Probe all replay/production external dependencies in one shot.

Answers "is it my code or is it the network" in ten seconds: the production
LLM provider path (get_llm — NOT raw HTTP; the current model variant routes
output through a reasoning field that raw calls misread as empty), arXiv,
OpenAlex, and the DashScope embedding endpoint.

Usage:
    cd backend
    python -m eval.production_e2e.probe_replay_deps
"""
from __future__ import annotations

import asyncio
import io
import sys

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import httpx


async def probe_llm() -> None:
    try:
        from app.llm.factory import get_llm
        from app.schemas.schemas import GapCandidateList

        llm = get_llm()
        result = await llm.chat_json([
            {"role": "system", "content": "You reply with strict JSON only."},
            {"role": "user", "content": 'Return {"gaps": []} exactly.'},
        ], GapCandidateList)
        print(f"LLM ({type(llm).__name__}): OK ->", result.model_dump_json()[:80])
    except Exception as exc:
        print(f"LLM: FAILED {type(exc).__name__}: {exc}")


def probe_http(name: str, url: str, timeout: float = 20.0) -> None:
    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True,
                         headers={"User-Agent": "Mozilla/5.0 (probe)"})
        print(f"{name}: HTTP {resp.status_code} ({len(resp.content)} bytes)")
    except Exception as exc:
        print(f"{name}: FAILED {type(exc).__name__}: {exc}")


def probe_embedding() -> None:
    try:
        from app.services.embedding_service import embed_texts

        vectors = embed_texts(["probe embedding"])
        print(f"DashScope embedding: OK dim={len(vectors[0]) if vectors else 0}")
    except Exception as exc:
        print(f"DashScope embedding: FAILED {type(exc).__name__}: {exc}")


def main() -> None:
    asyncio.run(probe_llm())
    probe_http("arXiv", "https://arxiv.org/abs/1706.03762")
    # 429 here is the unauthenticated public pool; the production client
    # sends the configured API key. Connected-but-throttled still counts as
    # "network up" for this probe.
    probe_http("OpenAlex", "https://api.openalex.org/works?per-page=1")
    probe_embedding()


if __name__ == "__main__":
    main()
