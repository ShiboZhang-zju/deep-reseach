"""Simulate real agent search: multiple queries, multiple rounds.

Tests the full search_service pipeline as used by the agent:
  search_service.search_multiple_queries(queries, limit)

Reports per-source yield, dedup rate, metadata completeness, and top papers.
"""
import asyncio
import sys
import os
import time
from collections import Counter, defaultdict

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.services.search_service import search_service
from app.services.scoring_service import normalize_paper, deduplicate_papers
from app.paper_sources.base import RawPaper


def assess_batch(papers: list[RawPaper], label: str) -> dict:
    n = len(papers)
    if n == 0:
        return {"label": label, "count": 0}
    return {
        "label": label,
        "count": n,
        "title%": round(sum(1 for p in papers if p.title) / n * 100),
        "abstract%": round(sum(1 for p in papers if p.abstract and len(p.abstract) > 50) / n * 100),
        "year%": round(sum(1 for p in papers if p.year) / n * 100),
        "venue%": round(sum(1 for p in papers if p.venue) / n * 100),
        "doi%": round(sum(1 for p in papers if p.doi) / n * 100),
        "pdf%": round(sum(1 for p in papers if p.pdf_url) / n * 100),
        "citation%": round(sum(1 for p in papers if p.citation_count > 0) / n * 100),
        "authors%": round(sum(1 for p in papers if p.authors) / n * 100),
    }


async def main():
    # Simulate agent queries for a topic like "graph neural network test oracle"
    # These mimic what generate_queries step would produce
    queries = [
        "graph neural network test oracle",
        "metamorphic testing neural network",
        "test oracle problem deep learning",
        "automated test generation graph neural network",
        "software testing oracle machine learning",
    ]

    print("=" * 80)
    print("REAL AGENT SEARCH SIMULATION")
    print(f"Queries: {len(queries)}")
    for i, q in enumerate(queries, 1):
        print(f"  Q{i}: {q}")
    print("=" * 80)

    all_raw = []
    per_source = defaultdict(list)
    t0 = time.time()

    # search_multiple_queries runs all queries × all sources concurrently
    # This is exactly what the agent does
    raw_papers = await search_service.search_multiple_queries(
        queries, limit=15
    )
    elapsed = time.time() - t0

    print(f"\nTotal raw papers: {len(raw_papers)} in {elapsed:.1f}s")

    # Group by source
    for p in raw_papers:
        per_source[p.source].append(p)

    print(f"\n--- Per-source yield ---")
    for src in sorted(per_source.keys()):
        papers = per_source[src]
        q = assess_batch(papers, src)
        print(f"\n[{src}] {q['count']} papers")
        if q["count"] > 0:
            print(f"  title={q['title%']}% abstract={q['abstract%']}% year={q['year%']}% "
                  f"venue={q['venue%']}% doi={q['doi%']}% pdf={q['pdf%']}% "
                  f"citation={q['citation%']}% authors={q['authors%']}%")

    # Deduplicate (same as agent does)
    deduped = deduplicate_papers(raw_papers)
    print(f"\n--- Dedup ---")
    print(f"Raw: {len(raw_papers)} -> Deduped: {len(deduped)} (removed {len(raw_papers) - len(deduped)})")

    # Cross-source overlap: papers found by multiple sources
    title_to_sources = defaultdict(set)
    for p in raw_papers:
        key = p.title.lower().strip()[:100]
        if key:
            title_to_sources[key].add(p.source)
    multi_source = {t: s for t, s in title_to_sources.items() if len(s) > 1}
    print(f"  Papers found by multiple sources: {len(multi_source)}")

    # Overall metadata completeness after dedup
    overall = assess_batch(deduped, "overall")
    print(f"\n--- Overall metadata (after dedup) ---")
    print(f"  title={overall['title%']}% abstract={overall['abstract%']}% year={overall['year%']}% "
          f"venue={overall['venue%']}% doi={overall['doi%']}% pdf={overall['pdf%']}% "
          f"citation={overall['citation%']}% authors={overall['authors%']}%")

    # Quality check: papers with citation_count > 0 (authority signal)
    cited = [p for p in deduped if p.citation_count > 0]
    cited.sort(key=lambda p: p.citation_count, reverse=True)
    print(f"\n--- Top 10 most-cited papers ---")
    for p in cited[:10]:
        yr = f"[{p.year}]" if p.year else ""
        src = p.source
        print(f"  {p.citation_count:>5} cites {yr} ({src}) {p.title[:75]}")

    # Papers with abstracts (usable for scoring/RAG)
    with_abstract = [p for p in deduped if p.abstract and len(p.abstract) > 50]
    print(f"\n--- Usability ---")
    print(f"  Papers with abstract (>50 chars): {len(with_abstract)}/{len(deduped)}")
    print(f"  Papers with DOI: {sum(1 for p in deduped if p.doi)}/{len(deduped)}")
    print(f"  Papers with PDF URL: {sum(1 for p in deduped if p.pdf_url)}/{len(deduped)}")
    print(f"  Papers with citation_count: {len(cited)}/{len(deduped)}")

    # Relevance spot-check: how many titles contain key terms?
    key_terms = ["test oracle", "metamorphic", "neural network", "graph neural", "testing", "oracle"]
    relevant = []
    for p in deduped:
        title_lower = p.title.lower()
        if any(term in title_lower for term in key_terms):
            relevant.append(p)
    print(f"\n--- Relevance spot-check ---")
    print(f"  Titles containing key terms: {len(relevant)}/{len(deduped)}")
    for p in relevant[:10]:
        print(f"    - {p.title[:80]}")

    # Verdict
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")
    issues = []
    if len(deduped) < 30:
        issues.append(f"LOW YIELD: only {len(deduped)} papers (expected 50+ for 5 queries)")
    if overall["abstract%"] < 60:
        issues.append(f"LOW ABSTRACT: only {overall['abstract%']}% have abstracts")
    if overall["citation%"] < 30:
        issues.append(f"LOW CITATION DATA: only {overall['citation%']}% have citation counts")
    if overall["doi%"] < 50:
        issues.append(f"LOW DOI: only {overall['doi%']}% have DOIs")
    if len(relevant) < len(deduped) * 0.3:
        issues.append(f"LOW RELEVANCE: only {len(relevant)}/{len(deduped)} titles match key terms")

    if issues:
        print("ISSUES FOUND:")
        for iss in issues:
            print(f"  [!] {iss}")
    else:
        print("[OK] All quality checks passed")

    # Source availability
    failed_sources = []
    for src_name in ["semantic_scholar", "openalex", "arxiv", "crossref", "ieee", "core"]:
        if src_name not in per_source or len(per_source[src_name]) == 0:
            failed_sources.append(src_name)
    if failed_sources:
        print(f"\nFailed/empty sources: {failed_sources}")


if __name__ == "__main__":
    asyncio.run(main())
