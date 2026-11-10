"""Pre-flight smoke test: LLM / embedding / paper-source connectivity.

Prints OK/FAIL per dependency without leaking secret values. Exit code 1 when
any hard dependency fails (a full run on a broken endpoint just burns an hour).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check(name, fn):
    try:
        result = fn()
        print(f"[OK]   {name}: {result}")
        return True
    except Exception as exc:
        print(f"[FAIL] {name}: {str(exc)[:200]}")
        return False


def llm_check():
    import asyncio
    from app.llm.factory import get_llm

    async def _call():
        llm = get_llm()
        out = await llm.chat([{"role": "user", "content": "Reply with the single word: ok"}])
        return (out or "")[:40].replace("\n", " ")

    return asyncio.run(_call())


def embedding_check():
    from app.services.embedding_service import embed_texts

    vecs = embed_texts(["connectivity probe"])
    if not vecs or not vecs[0]:
        raise RuntimeError("no vector returned")
    return f"dim={len(vecs[0])}"


def source_check():
    import asyncio
    from app.paper_sources.arxiv import ArxivSource

    async def _search():
        src = ArxivSource()
        return await src.search("test-time compute", limit=3)

    papers = asyncio.run(_search())
    return f"{len(papers)} papers from arXiv"


def settings_check():
    from app.config import settings

    lines = [
        f"llm_context={settings.llm_context_tokens}",
        f"max_rounds={settings.max_rounds}",
        f"precedent_check={settings.gap_mining_precedent_check_enabled}",
        f"evidence_concurrency={settings.evidence_extract_concurrency}",
        f"prewarm={settings.enable_pdf_prewarm}",
        f"openalex_key={'set' if settings.openalex_api_key else 'none'}",
        f"openalex_email={'set' if settings.openalex_email else 'none'}",
        f"s2_key={'set' if settings.semantic_scholar_api_key else 'none'}",
    ]
    return " ".join(lines)


if __name__ == "__main__":
    print(settings_check())
    ok = True
    ok &= check("LLM chat", llm_check)
    ok &= check("Embedding", embedding_check)
    ok &= check("arXiv search", source_check)
    sys.exit(0 if ok else 1)
