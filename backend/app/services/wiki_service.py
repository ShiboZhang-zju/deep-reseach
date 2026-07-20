"""LLM Wiki service: Incremental knowledge compilation from papers.

Replaces GraphRAG with a "compile once, query many" approach.
Inspired by Karpathy's LLM Wiki pattern:

- Raw layer: Paper abstracts + RAG chunks (immutable, from doc 10)
- Wiki layer: Structured markdown pages (LLM-maintained, incrementally updated)
- Schema layer: Rules for wiki maintenance (in prompts.py)

Core operations:
- Ingest: New papers → LLM generates create/update actions → execute → update index
- Query: Read wiki pages for report/idea generation context
- Lint: Health check for contradictions, orphan pages, stale info

Key advantage over GraphRAG:
- Cross-references are pre-compiled (not re-derived per query)
- Contradictions are flagged during ingest
- Wiki pages are human-readable artifacts
- Concept pages naturally replace LLM-based clustering
"""

from __future__ import annotations

import json
import logging
import re

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

WIKI_BATCH_SIZE = 5  # papers per ingest batch


# === Main entry point: Ingest papers into wiki ===

async def ingest_papers_to_wiki(db: Session, papers: list, llm, task_id: str) -> dict:
    """Ingest papers into the wiki knowledge base.

    For each batch of papers:
    1. List existing wiki pages
    2. LLM generates create/update actions
    3. Execute actions (merge into existing pages)
    4. After all batches, regenerate index
    """
    from app.db.models import WikiPage, TaskPaper
    from app.services.event_service import emit_event
    from app.schemas.schemas import WikiActionList
    from app.agent.prompts import WIKI_INGEST_SYSTEM, WIKI_INGEST_USER

    if not papers:
        return {"total": 0, "pages_created": 0, "pages_updated": 0, "total_pages": 0}

    # Sort by priority (high first), then by citation count
    paper_priority = {}
    for p in papers:
        tp = db.query(TaskPaper).filter(
            TaskPaper.task_id == task_id,
            TaskPaper.paper_id == p.id,
        ).first()
        paper_priority[p.id] = tp.priority if tp else "medium"

    sorted_papers = sorted(papers, key=lambda p: (
        0 if paper_priority.get(p.id) == "high" else 1,
        -(p.citation_count or 0),
    ))

    total = len(sorted_papers)
    pages_created = 0
    pages_updated = 0

    emit_event(task_id, "status", {"status": "building_wiki", "total_papers": total})
    logger.info("Task %s: LLM Wiki ingest starting for %d papers", task_id[:8], total)

    # Process in batches
    num_batches = (total + WIKI_BATCH_SIZE - 1) // WIKI_BATCH_SIZE
    for batch_idx in range(num_batches):
        batch_start = batch_idx * WIKI_BATCH_SIZE
        batch = sorted_papers[batch_start:batch_start + WIKI_BATCH_SIZE]

        # Build paper context for this batch
        paper_context = _build_paper_context(batch, task_id, db)

        # Get RAG passages for batch papers
        rag_passages = await _get_batch_rag(batch)

        # List existing wiki pages (titles + types + brief summary)
        existing_pages = _list_wiki_pages(db, task_id)

        # LLM generates wiki actions
        try:
            messages = [
                {"role": "system", "content": WIKI_INGEST_SYSTEM},
                {"role": "user", "content": WIKI_INGEST_USER.format(
                    paper_context=paper_context,
                    rag_passages=rag_passages or "(no full-text available, use abstracts only)",
                    existing_pages=existing_pages or "(wiki is empty, this is the first batch)",
                    batch_info=(
                        f"Batch {batch_idx + 1}/{num_batches}, "
                        f"papers {batch_start + 1}-{min(batch_start + WIKI_BATCH_SIZE, total)} of {total}"
                    ),
                )},
            ]
            action_list = await llm.chat_json(messages, WikiActionList)
        except Exception as e:
            logger.error("Task %s: wiki ingest LLM call failed for batch %d: %s",
                        task_id[:8], batch_idx + 1, e)
            continue

        # Execute actions
        for action in action_list.actions:
            result = _execute_wiki_action(db, task_id, action)
            if result == "created":
                pages_created += 1
            elif result == "updated":
                pages_updated += 1

        db.commit()
        logger.info("Task %s: wiki batch %d/%d done (%d actions)",
                    task_id[:8], batch_idx + 1, num_batches, len(action_list.actions))

    # Regenerate index
    _regenerate_index(db, task_id)
    db.commit()

    total_pages = db.query(WikiPage).filter(WikiPage.task_id == task_id).count()
    summary = {
        "total": total,
        "pages_created": pages_created,
        "pages_updated": pages_updated,
        "total_pages": total_pages,
    }
    logger.info("Task %s: LLM Wiki ingest complete: %s", task_id[:8], summary)
    emit_event(task_id, "status", {"status": "wiki_built", **summary})

    return summary


# === Helpers: Build context for LLM ===

def _build_paper_context(papers: list, task_id: str, db: Session) -> str:
    """Build paper context text for wiki ingest prompt."""
    from app.db.models import TaskPaper

    lines = []
    for i, p in enumerate(papers):
        tp = db.query(TaskPaper).filter(
            TaskPaper.task_id == task_id,
            TaskPaper.paper_id == p.id,
        ).first()

        method_extract = ""
        if tp and tp.summary:
            parts = tp.summary.split("方法:")
            if len(parts) > 1:
                method_extract = parts[1].strip()

        lines.append(
            f"[P{i + 1}] ID:{p.id[:8]} | {p.title} ({p.year}) [{p.venue or 'N/A'}]\n"
            f"  摘要: {(p.abstract or 'N/A')[:500]}\n"
            f"  方法提取: {method_extract or 'N/A'}\n"
            f"  引用数: {p.citation_count or 0}"
        )

    return "\n\n".join(lines)


async def _get_batch_rag(papers: list) -> str:
    """Get RAG passages for batch papers.

    P1-1: Use abstract-based semantic query instead of just title concatenation.
    """
    from app.services.rag_service import rag_retrieve

    paper_ids = [p.id for p in papers]
    try:
        # Build a better query: use abstracts (more semantic than titles)
        query_parts = []
        for p in papers[:3]:
            if p.abstract:
                query_parts.append(p.abstract[:200])
            else:
                query_parts.append(p.title)
        query = " ".join(query_parts)

        results = await rag_retrieve(
            query=query,
            top_k=20,
            paper_ids=paper_ids,
            section_filter=["method", "experiment"],
        )
        if not results:
            return ""

        _fig_pattern = re.compile(r'\[FIGURE:.*?\]|\[TABLE\]|\[/TABLE\]')
        lines = []
        for r in results[:20]:
            clean = _fig_pattern.sub('', r["text"])[:500].strip()
            lines.append(f"[{r['paper_id'][:8]}] ({r['section']}) {clean}")
        return "\n".join(lines)
    except Exception as e:
        logger.warning("RAG retrieval for wiki ingest failed: %s", e)
        return ""


def _list_wiki_pages(db: Session, task_id: str) -> str:
    """List existing wiki pages as context for the LLM.

    P1-2: Provide enough context (300 chars + paper count) for the LLM to
    determine if a new paper belongs to an existing concept page.
    """
    from app.db.models import WikiPage

    pages = db.query(WikiPage).filter(
        WikiPage.task_id == task_id,
        WikiPage.page_type != "index",
    ).all()
    if not pages:
        return ""

    lines = []
    for p in pages:
        # Give more context: 300 chars + paper count for concept matching
        brief = (p.content_markdown or "")[:300].replace("\n", " ").strip()
        paper_count = len(json.loads(p.paper_ids_json or "[]"))
        lines.append(f"- [{p.page_type}] {p.title} ({paper_count} papers): {brief}...")
    return "\n".join(lines)


# === Execute wiki actions ===

def _find_similar_page(db, task_id: str, title: str, page_type: str):
    """Find an existing wiki page with a similar title (fuzzy match).

    Uses character-level Jaccard similarity on normalized titles.
    Threshold: 0.5 for concept pages (same theme, different wording).
    """
    from app.db.models import WikiPage

    pages = db.query(WikiPage).filter(
        WikiPage.task_id == task_id,
        WikiPage.page_type == page_type,
    ).all()

    if not pages:
        return None

    # Normalize: lowercase, strip punctuation, split into char set
    def _char_set(s: str) -> set:
        s = re.sub(r'[^\w\u4e00-\u9fff]', '', s.lower())
        return set(s)

    title_chars = _char_set(title)
    if not title_chars:
        return None

    best_match = None
    best_score = 0.0
    threshold = 0.5  # 50% character overlap → likely same concept

    for p in pages:
        p_chars = _char_set(p.title)
        if not p_chars:
            continue
        # Jaccard similarity
        intersection = len(title_chars & p_chars)
        union = len(title_chars | p_chars)
        score = intersection / union if union > 0 else 0.0

        if score > best_score:
            best_score = score
            best_match = p

    if best_score >= threshold:
        logger.info("Fuzzy matched '%s' -> '%s' (score: %.2f)", title, best_match.title, best_score)
        return best_match

    return None


def _execute_wiki_action(db: Session, task_id: str, action) -> str:
    """Execute a wiki action (create or update). Returns 'created', 'updated', or ''."""
    from app.db.models import WikiPage

    title = action.title.strip()

    # Check if page exists — exact match first
    existing = db.query(WikiPage).filter(
        WikiPage.task_id == task_id,
        WikiPage.title == title,
    ).first()

    # P0-2: Fuzzy title match for concept pages to prevent fragmentation
    # e.g., "记忆增强的大语言模型" vs "大语言模型记忆机制" should merge
    if not existing and action.page_type == "concept":
        existing = _find_similar_page(db, task_id, title, action.page_type)

    if action.op == "create":
        if existing:
            # Page already exists, treat as update (merge)
            return _merge_page(existing, action)

        page = WikiPage(
            task_id=task_id,
            page_type=action.page_type,
            title=title,
            content_markdown=action.content,
            paper_ids_json=json.dumps(action.paper_ids or [], ensure_ascii=False),
            links_json=json.dumps(action.links or [], ensure_ascii=False),
            contradictions_json=json.dumps(action.contradictions or [], ensure_ascii=False),
        )
        db.add(page)
        db.flush()
        return "created"

    elif action.op == "update":
        if existing:
            return _merge_page(existing, action)
        else:
            # Page doesn't exist, create it
            page = WikiPage(
                task_id=task_id,
                page_type=action.page_type,
                title=title,
                content_markdown=action.content,
                paper_ids_json=json.dumps(action.paper_ids or [], ensure_ascii=False),
                links_json=json.dumps(action.links or [], ensure_ascii=False),
                contradictions_json=json.dumps(action.contradictions or [], ensure_ascii=False),
            )
            db.add(page)
            db.flush()
            return "created"

    return ""


def _merge_page(existing, action) -> str:
    """Merge new content into existing wiki page using section-level merge.

    Instead of blindly appending, we parse both old and new content by
    markdown headings and merge section by section — new sections are added,
    existing sections are appended with de-dup.
    """
    # Merge paper_ids
    existing_paper_ids = set(json.loads(existing.paper_ids_json or "[]"))
    new_paper_ids = set(action.paper_ids or [])
    merged_paper_ids = list(existing_paper_ids | new_paper_ids)

    # Merge links
    existing_links = set(json.loads(existing.links_json or "[]"))
    new_links = set(action.links or [])
    merged_links = list(existing_links | new_links)

    # Merge contradictions
    existing_contras = set(json.loads(existing.contradictions_json or "[]"))
    new_contras = set(action.contradictions or [])
    merged_contras = list(existing_contras | new_contras)

    # === Section-level content merge ===
    old_content = existing.content_markdown or ""
    new_content = (action.content or "").strip()

    if new_content:
        old_sections = _parse_sections(old_content)
        new_sections = _parse_sections(new_content)

        for heading, body in new_sections.items():
            if heading in old_sections:
                # Section exists — merge with line-level deduplication
                old_body = old_sections[heading]
                old_lines = set(l.strip() for l in old_body.split("\n") if l.strip())
                new_lines = []
                for line in body.split("\n"):
                    line_stripped = line.strip()
                    # Only add lines that don't already exist
                    if line_stripped and line_stripped not in old_lines:
                        new_lines.append(line)
                        old_lines.add(line_stripped)
                    elif not line_stripped:
                        new_lines.append(line)  # Keep blank lines

                if new_lines:
                    old_sections[heading] = old_body.rstrip() + "\n\n" + "\n".join(new_lines).strip()
            else:
                # New section — add it
                old_sections[heading] = body

        # Reconstruct content
        merged_content = _reconstruct_sections(old_sections)
        existing.content_markdown = merged_content

    existing.paper_ids_json = json.dumps(merged_paper_ids, ensure_ascii=False)
    existing.links_json = json.dumps(merged_links, ensure_ascii=False)
    existing.contradictions_json = json.dumps(merged_contras, ensure_ascii=False)
    return "updated"


def _parse_sections(markdown: str) -> dict[str, str]:
    """Parse markdown into {heading: body} dict.

    Headings are the full heading line (e.g., "## Summary").
    Body is the content until the next heading.
    Content before the first heading is stored under key "__intro__".
    """
    if not markdown:
        return {}

    sections = {}
    current_heading = "__intro__"
    current_body = []

    for line in markdown.split("\n"):
        if re.match(r'^#+\s+', line):
            # Save previous section
            if current_body:
                sections[current_heading] = "\n".join(current_body).strip()
            current_heading = line.strip()
            current_body = []
        else:
            current_body.append(line)

    if current_body:
        sections[current_heading] = "\n".join(current_body).strip()

    return sections


def _reconstruct_sections(sections: dict[str, str]) -> str:
    """Reconstruct markdown from {heading: body} dict."""
    parts = []

    # Intro first (if exists)
    if "__intro__" in sections and sections["__intro__"]:
        parts.append(sections["__intro__"])

    # Then all other sections in order
    for heading, body in sections.items():
        if heading == "__intro__":
            continue
        parts.append(f"{heading}\n{body}" if body else heading)

    return "\n\n".join(parts)


# === Index management ===

def _regenerate_index(db: Session, task_id: str):
    """Regenerate the wiki index page."""
    from app.db.models import WikiPage

    pages = db.query(WikiPage).filter(
        WikiPage.task_id == task_id,
        WikiPage.page_type != "index",
    ).all()
    if not pages:
        return

    # Group by type
    by_type = {}
    for p in pages:
        by_type.setdefault(p.page_type, []).append(p)

    # Build index content
    lines = ["# Wiki Index\n"]
    type_labels = {
        "concept": "研究概念 (Concepts)",
        "method": "方法/技术 (Methods)",
        "dataset": "数据集 (Datasets)",
        "model": "模型 (Models)",
        "synthesis": "综合分析 (Synthesis)",
    }

    for ptype in ["concept", "method", "dataset", "model", "synthesis"]:
        if ptype not in by_type:
            continue
        label = type_labels.get(ptype, ptype)
        lines.append(f"\n## {label}\n")
        for p in sorted(by_type[ptype], key=lambda x: x.title):
            paper_count = len(json.loads(p.paper_ids_json or "[]"))
            lines.append(f"- **{p.title}** ({paper_count} papers)")

    index_content = "\n".join(lines)

    # Upsert index page
    existing_index = db.query(WikiPage).filter(
        WikiPage.task_id == task_id,
        WikiPage.page_type == "index",
    ).first()

    if existing_index:
        existing_index.content_markdown = index_content
    else:
        index_page = WikiPage(
            task_id=task_id,
            page_type="index",
            title="Wiki Index",
            content_markdown=index_content,
        )
        db.add(index_page)
    db.flush()


# === Query: Get wiki context for report/idea generation ===

def get_wiki_context(db: Session, task_id: str, page_types: list[str] | None = None,
                     max_chars: int = 15000) -> str:
    """Get wiki pages as context text for report/idea generation.

    Returns formatted markdown with all wiki pages (or filtered by type).
    Truncated to max_chars to prevent token overflow.
    """
    from app.db.models import WikiPage

    query = db.query(WikiPage).filter(
        WikiPage.task_id == task_id,
        WikiPage.page_type != "index",
    )
    if page_types:
        query = query.filter(WikiPage.page_type.in_(page_types))

    pages = query.all()
    if not pages:
        return ""

    # Sort: concept first, then synthesis, then method/dataset/model
    type_order = {"concept": 0, "synthesis": 1, "method": 2, "dataset": 3, "model": 4}
    pages.sort(key=lambda p: (type_order.get(p.page_type, 9), p.title))

    sections = ["# 知识库 Wiki（预编译的论文知识合成）\n"]
    total_len = len(sections[0])
    for p in pages:
        page_text = f"\n---\n\n## [{p.page_type}] {p.title}\n\n{p.content_markdown}"
        if total_len + len(page_text) > max_chars:
            # Truncate this page's content to fit
            remaining = max_chars - total_len - 100  # Leave room for truncation notice
            if remaining > 200:
                page_text = page_text[:remaining] + "\n...(截断)"
                sections.append(page_text)
                total_len += len(page_text)
            break
        sections.append(page_text)
        total_len += len(page_text)

    return "\n".join(sections)


# === Wiki → ClusterList conversion (backward compatibility) ===

def get_wiki_clusters(db: Session, task_id: str):
    """Convert wiki concept pages to ClusterList format.

    This replaces the LLM-based _build_paper_clusters with pre-compiled wiki pages.
    Returns ClusterList or None if no wiki pages exist.
    """
    from app.db.models import WikiPage, Paper
    from app.schemas.schemas import ClusterList, PaperCluster

    # Get concept pages (these are the "clusters")
    concept_pages = db.query(WikiPage).filter(
        WikiPage.task_id == task_id,
        WikiPage.page_type == "concept",
    ).all()

    if not concept_pages:
        return None

    clusters = []
    cross_gaps = []

    for page in concept_pages:
        paper_ids = json.loads(page.paper_ids_json or "[]")

        # Get paper titles from DB — LLM stores 8-char ID prefixes, need prefix match
        paper_titles = []
        seen_titles = set()
        for pid in paper_ids:
            # pid could be 8-char prefix or full UUID — handle both
            if len(pid) >= 32:
                # Full UUID
                p = db.query(Paper).filter(Paper.id == pid).first()
                if p and p.title not in seen_titles:
                    paper_titles.append(p.title)
                    seen_titles.add(p.title)
            else:
                # 8-char prefix — use LIKE match
                matches = db.query(Paper).filter(Paper.id.like(f"{pid}%")).all()
                for m in matches:
                    if m.title not in seen_titles:
                        paper_titles.append(m.title)
                        seen_titles.add(m.title)

        # Also try to match paper ID prefixes mentioned in content markdown
        if len(paper_titles) < len(paper_ids):
            id_pattern = re.findall(r'\b([0-9a-f]{8})\b', page.content_markdown or "")
            for prefix in id_pattern:
                matches = db.query(Paper).filter(Paper.id.like(f"{prefix}%")).all()
                for m in matches:
                    if m.title not in seen_titles:
                        paper_titles.append(m.title)
                        seen_titles.add(m.title)

        cluster = PaperCluster(
            cluster_name=page.title,
            core_method=(
                _extract_section(page.content_markdown, "Technical Details")
                or _extract_section(page.content_markdown, "Summary")
                or _extract_section(page.content_markdown, "总结")
                or "(see wiki page)"
            ),
            technique_details=(
                _extract_section(page.content_markdown, "Technical Details")
                or _extract_section(page.content_markdown, "技术细节")
                or (page.content_markdown or "")[:500]
            ),
            problem_addressed=(
                _extract_section(page.content_markdown, "Problem")
                or _extract_section(page.content_markdown, "问题")
                or "(see wiki page)"
            ),
            key_findings=(
                _extract_section(page.content_markdown, "Key Findings")
                or _extract_section(page.content_markdown, "关键发现")
                or _extract_section(page.content_markdown, "Findings")
                or "(see wiki page)"
            ),
            limitations=(
                _extract_section(page.content_markdown, "Limitations")
                or _extract_section(page.content_markdown, "局限")
                or "(see wiki page)"
            ),
            representative_papers=paper_titles[:5],
        )
        clusters.append(cluster)

    # Get synthesis pages for cross-cluster gaps
    synthesis_pages = db.query(WikiPage).filter(
        WikiPage.task_id == task_id,
        WikiPage.page_type == "synthesis",
    ).all()

    for sp in synthesis_pages:
        cross_gaps.append(f"[{sp.title}] {(sp.content_markdown or '')[:200]}")

    # Also check for contradictions across all pages
    all_pages = db.query(WikiPage).filter(
        WikiPage.task_id == task_id,
        WikiPage.page_type != "index",
    ).all()

    for p in all_pages:
        contradictions = json.loads(p.contradictions_json or "[]")
        for c in contradictions:
            cross_gaps.append(f"⚠️ 矛盾 [{p.title}]: {c}")

    return ClusterList(clusters=clusters, cross_cluster_gaps=cross_gaps)


def _extract_section(markdown: str, heading: str) -> str:
    """Extract content under a markdown heading."""
    if not markdown:
        return ""
    pattern = rf'(?:^|\n)#+\s*{re.escape(heading)}.*?\n(.*?)(?=\n#|\Z)'
    match = re.search(pattern, markdown, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()[:500]
    return ""


# === Stats and Lint ===

def get_wiki_stats(db: Session, task_id: str) -> dict:
    """Get wiki statistics for a task."""
    from app.db.models import WikiPage

    pages = db.query(WikiPage).filter(
        WikiPage.task_id == task_id,
        WikiPage.page_type != "index",
    ).all()

    by_type = {}
    for p in pages:
        by_type[p.page_type] = by_type.get(p.page_type, 0) + 1

    total_contradictions = sum(
        len(json.loads(p.contradictions_json or "[]")) for p in pages
    )

    return {
        "total_pages": len(pages),
        "by_type": by_type,
        "contradictions": total_contradictions,
    }


async def lint_wiki(db: Session, task_id: str, llm) -> dict:
    """Health check: find contradictions, orphan pages, stale info.

    Returns a report of issues found.
    """
    from app.db.models import WikiPage
    from app.agent.prompts import WIKI_LINT_SYSTEM, WIKI_LINT_USER
    from app.schemas.schemas import WikiLintResult

    pages = db.query(WikiPage).filter(
        WikiPage.task_id == task_id,
        WikiPage.page_type != "index",
    ).all()

    if not pages or len(pages) < 3:
        return {"issues": [], "total": 0}

    wiki_text = "\n\n".join(
        f"## [{p.page_type}] {p.title}\n{p.content_markdown}" for p in pages
    )

    try:
        messages = [
            {"role": "system", "content": WIKI_LINT_SYSTEM},
            {"role": "user", "content": WIKI_LINT_USER.format(wiki_content=wiki_text)},
        ]
        result = await llm.chat_json(messages, WikiLintResult)

        # Apply fixes: update contradictions field on relevant pages
        for issue in result.issues:
            if issue.issue_type == "contradiction" and issue.page_title:
                page = db.query(WikiPage).filter(
                    WikiPage.task_id == task_id,
                    WikiPage.title == issue.page_title,
                ).first()
                if page:
                    contras = json.loads(page.contradictions_json or "[]")
                    if issue.description not in contras:
                        contras.append(issue.description)
                        page.contradictions_json = json.dumps(contras, ensure_ascii=False)

        db.commit()
        return {
            "issues": [i.model_dump() for i in result.issues],
            "total": len(result.issues),
        }
    except Exception as e:
        logger.warning("Wiki lint failed: %s", e)
        return {"issues": [], "total": 0, "error": str(e)}
