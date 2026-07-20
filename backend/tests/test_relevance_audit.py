"""Audit retrieved papers for topical relevance.

Writes full audit report to audit_report.txt for review.
"""
import asyncio
import os
import sys

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.services.search_service import search_service
from app.services.scoring_service import deduplicate_papers

DIRECTION = "graph neural network test oracle"


def classify(title: str, abstract: str) -> str:
    t = (title + " " + abstract).lower()
    has_gnn = any(k in t for k in ["graph neural", "gnn", "graph convolution", "message passing neural"])
    has_oracle = any(k in t for k in ["oracle", "metamorphic", "test oracle", "oracle problem"])
    has_testing = any(k in t for k in ["test", "testing", "verification", "fuzz", "mutation"])
    has_nn = any(k in t for k in ["neural network", "deep learning", "machine learning", "transformer"])

    if has_gnn and (has_oracle or has_testing):
        return "A"
    if has_oracle:
        return "B"
    if has_gnn and has_testing:
        return "B"
    if has_nn and has_testing:
        return "C"
    if has_gnn:
        return "C"
    if has_testing:
        return "D"
    if has_nn:
        return "D"
    return "E"


async def main():
    queries = [
        "graph neural network test oracle",
        "metamorphic testing neural network",
        "test oracle problem deep learning",
        "automated test generation graph neural network",
        "software testing oracle machine learning",
    ]

    lines = []
    def w(s=""):
        lines.append(s)

    w(f"Research direction: '{DIRECTION}'")
    w(f"Queries: {len(queries)}")
    w("=" * 90)

    raw = await search_service.search_multiple_queries(queries, 15)
    deduped = deduplicate_papers(raw)
    w(f"Total: {len(raw)} raw -> {len(deduped)} deduped")

    classified = {"A": [], "B": [], "C": [], "D": [], "E": []}
    for p in deduped:
        grade = classify(p.title, p.abstract or "")
        classified[grade].append(p)

    w("")
    w("--- Relevance Distribution ---")
    labels = {
        "A": "A: GNN + testing/oracle (core)",
        "B": "B: Oracle problem or GNN+testing (directly relevant)",
        "C": "C: NN testing or GNN (adjacent)",
        "D": "D: ML/graph (tangential)",
        "E": "E: Off-topic",
    }
    for g in "ABCDE":
        w(f"  {labels[g]}: {len(classified[g])}")

    # Full detail for A and B
    for g in ["A", "B"]:
        papers = classified[g]
        if not papers:
            continue
        w("")
        w("=" * 90)
        w(f"Grade {g} papers ({len(papers)}):")
        w("=" * 90)
        for i, p in enumerate(papers, 1):
            cite = f" [cited {p.citation_count}]" if p.citation_count else ""
            yr = f" [{p.year}]" if p.year else ""
            w(f"\n  {i}. {p.title}{yr}{cite} ({p.source})")
            if p.abstract:
                w(f"     Abstract: {p.abstract[:300]}")
            else:
                w(f"     Abstract: [MISSING]")
            if p.doi:
                w(f"     DOI: {p.doi}")
            if p.pdf_url:
                w(f"     PDF: {p.pdf_url[:80]}")

    # C titles
    if classified["C"]:
        w("")
        w("=" * 90)
        w(f"Grade C papers ({len(classified['C'])}) - titles only:")
        w("=" * 90)
        for p in classified["C"]:
            yr = f" [{p.year}]" if p.year else ""
            w(f"  - {p.title[:85]}{yr} ({p.source})")

    # D and E samples
    for g in ["D", "E"]:
        papers = classified[g]
        if papers:
            w(f"\nGrade {g} ({len(papers)}) samples:")
            for p in papers[:5]:
                w(f"  - {p.title[:85]} ({p.source})")
            if len(papers) > 5:
                w(f"  ... and {len(papers) - 5} more")

    # Verdict
    w("")
    w("=" * 90)
    w("VERDICT")
    w("=" * 90)
    a_b = len(classified["A"]) + len(classified["B"])
    total = len(deduped)
    w(f"  Core (A+B): {a_b}/{total} = {round(a_b/total*100)}%")
    w(f"  Usable (A+B+C): {a_b + len(classified['C'])}/{total} = {round((a_b + len(classified['C']))/total*100)}%")
    w(f"  Noise (D+E): {len(classified['D']) + len(classified['E'])}/{total} = {round((len(classified['D']) + len(classified['E']))/total*100)}%")

    if a_b < 10:
        w(f"  [!] LOW CORE COVERAGE: only {a_b} papers directly on-topic")
    if a_b >= 10 and (a_b + len(classified["C"])) >= total * 0.5:
        w(f"  [OK] Sufficient coverage for the research direction")
    if len(classified["E"]) > total * 0.3:
        w(f"  [!] HIGH NOISE: {round(len(classified['E'])/total*100)}% off-topic")

    report = "\n".join(lines)
    report_path = os.path.join(os.path.dirname(__file__), "audit_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report written to {report_path}")
    print(f"\n{report[:3000]}")  # Print first 3000 chars


if __name__ == "__main__":
    asyncio.run(main())
