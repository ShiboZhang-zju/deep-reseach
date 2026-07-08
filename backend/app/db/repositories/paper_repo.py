"""Database repository for papers."""

import json

from sqlalchemy.orm import Session

from app.db.models import Paper, TaskPaper, ResearchRound, Report, ResearchIdea, ExperimentPlan, AgentTrace, UserFeedback, PaperChunk
from app.services.scoring_service import normalize_title, title_hash


def find_paper_by_ids(
    db: Session,
    doi: str | None = None,
    arxiv_id: str | None = None,
    s2_id: str | None = None,
    openalex_id: str | None = None,
    title_hash_val: str | None = None,
) -> Paper | None:
    """Find existing paper by any unique identifier."""
    if doi:
        p = db.query(Paper).filter(Paper.doi == doi).first()
        if p:
            return p
    if arxiv_id:
        p = db.query(Paper).filter(Paper.arxiv_id == arxiv_id).first()
        if p:
            return p
    if s2_id:
        p = db.query(Paper).filter(Paper.semantic_scholar_id == s2_id).first()
        if p:
            return p
    if openalex_id:
        p = db.query(Paper).filter(Paper.openalex_id == openalex_id).first()
        if p:
            return p
    if title_hash_val:
        p = db.query(Paper).filter(Paper.title_hash == title_hash_val).first()
        if p:
            return p
    return None


def upsert_paper(db: Session, raw_data: dict) -> tuple[Paper, bool]:
    """Insert or update a paper. Returns (paper, is_new)."""
    existing = find_paper_by_ids(
        db,
        doi=raw_data.get("doi"),
        arxiv_id=raw_data.get("arxiv_id"),
        s2_id=raw_data.get("semantic_scholar_id"),
        openalex_id=raw_data.get("openalex_id"),
        title_hash_val=raw_data.get("title_hash"),
    )

    if existing:
        # Merge sources and fill missing fields
        existing_sources = json.loads(existing.sources_json) if existing.sources_json else []
        new_sources = json.loads(raw_data["sources_json"]) if raw_data.get("sources_json") else []
        all_sources = list(set(existing_sources + new_sources))
        existing.sources_json = json.dumps(all_sources, ensure_ascii=False)

        for attr in ["doi", "arxiv_id", "semantic_scholar_id", "openalex_id", "url", "pdf_url", "abstract"]:
            current_val = getattr(existing, attr, None)
            new_val = raw_data.get(attr)
            if not current_val and new_val:
                setattr(existing, attr, new_val)

        if raw_data.get("citation_count", 0) > (existing.citation_count or 0):
            existing.citation_count = raw_data["citation_count"]

        db.flush()
        return existing, False

    paper = Paper(**{k: v for k, v in raw_data.items() if hasattr(Paper, k)})
    db.add(paper)
    db.flush()
    return paper, True


def create_task_paper(
    db: Session,
    task_id: str,
    paper_id: str,
    discovered_round: int,
) -> TaskPaper:
    """Create a task-paper link (if not exists)."""
    existing = db.query(TaskPaper).filter(
        TaskPaper.task_id == task_id,
        TaskPaper.paper_id == paper_id,
    ).first()
    if existing:
        return existing

    tp = TaskPaper(task_id=task_id, paper_id=paper_id, discovered_round=discovered_round)
    db.add(tp)
    db.flush()
    return tp


def update_task_paper_scores(
    db: Session,
    task_paper_id: str,
    scores: dict,
    final_score: float,
    priority: str,
    reason: str,
    summary: str,
):
    """Update scoring for a task-paper link."""
    tp = db.get(TaskPaper, task_paper_id)
    if not tp:
        return
    tp.relevance_score = scores.get("relevance")
    tp.authority_score = scores.get("authority")
    tp.recency_score = scores.get("recency")
    tp.novelty_score = scores.get("novelty")
    tp.idea_potential_score = scores.get("idea_potential")
    tp.final_score = final_score
    tp.priority = priority
    tp.reason = reason
    tp.summary = summary
    db.flush()


def save_round(db: Session, task_id: str, round_number: int, queries: list[str],
               papers_found: int, new_papers: int, duplicate_rate: float,
               summary: str, gaps: list[str]) -> ResearchRound:
    """Save a research round record."""
    round_record = ResearchRound(
        task_id=task_id,
        round_number=round_number,
        queries_json=json.dumps(queries, ensure_ascii=False),
        papers_found=papers_found,
        new_papers=new_papers,
        duplicate_rate=duplicate_rate,
        summary=summary,
        knowledge_gaps_json=json.dumps(gaps, ensure_ascii=False),
    )
    db.add(round_record)
    db.flush()
    return round_record


def save_report(db: Session, task_id: str, content_markdown: str, content_json: str = None) -> Report:
    """Save a research report."""
    report = Report(task_id=task_id, content_markdown=content_markdown, content_json=content_json)
    db.add(report)
    db.flush()
    return report


def save_idea(db: Session, task_id: str, idea_data: dict) -> ResearchIdea:
    """Save a research idea."""
    idea = ResearchIdea(task_id=task_id, **{k: v for k, v in idea_data.items() if hasattr(ResearchIdea, k)})
    db.add(idea)
    db.flush()
    return idea


def update_idea_scores(db: Session, idea_id: str, scores: dict, final_score: float, decision: str):
    """Update idea evaluation scores."""
    idea = db.get(ResearchIdea, idea_id)
    if not idea:
        return
    idea.novelty = scores.get("novelty")
    idea.feasibility = scores.get("feasibility")
    idea.significance = scores.get("significance")
    idea.evidence_support = scores.get("evidence_support")
    idea.differentiation = scores.get("differentiation")
    idea.experimentability = scores.get("experimentability")
    idea.potential_impact = scores.get("potential_impact")
    idea.risk = scores.get("risk")
    idea.final_score = final_score
    idea.decision = decision
    db.flush()


def save_experiment(db: Session, task_id: str, idea_id: str, plan_data: dict) -> ExperimentPlan:
    """Save an experiment plan."""
    exp = ExperimentPlan(task_id=task_id, idea_id=idea_id,
                          **{k: v for k, v in plan_data.items() if hasattr(ExperimentPlan, k)})
    db.add(exp)
    db.flush()
    return exp


def save_trace(db: Session, task_id: str, step_name: str, step_type: str,
               round_number: int = None, input_data: dict = None, output_data: dict = None,
               tokens: int = None, duration_ms: int = None):
    """Save an agent trace record."""
    trace = AgentTrace(
        task_id=task_id,
        step_name=step_name,
        step_type=step_type,
        round_number=round_number,
        input_json=json.dumps(input_data, ensure_ascii=False, default=str) if input_data else None,
        output_json=json.dumps(output_data, ensure_ascii=False, default=str) if output_data else None,
        llm_tokens_used=tokens,
        duration_ms=duration_ms,
    )
    db.add(trace)
    db.flush()


def save_feedback(db: Session, task_id: str, feedback_type: str, content: str,
                  selected_idea_ids: list[str] = None, need_more_research: bool = False) -> UserFeedback:
    """Save user feedback."""
    fb = UserFeedback(
        task_id=task_id,
        feedback_type=feedback_type,
        content=content,
        selected_idea_ids_json=json.dumps(selected_idea_ids or [], ensure_ascii=False),
        need_more_research=need_more_research,
    )
    db.add(fb)
    db.flush()
    return fb


# === Paper chunk operations (RAG) ===

def save_chunks(db: Session, paper_id: str, chunks_data: list[dict]) -> list[PaperChunk]:
    """Save paper chunks to SQLite."""
    # Delete existing chunks for this paper
    db.query(PaperChunk).filter(PaperChunk.paper_id == paper_id).delete()
    saved = []
    for chunk_data in chunks_data:
        chunk = PaperChunk(
            paper_id=paper_id,
            chunk_index=chunk_data["chunk_index"],
            section=chunk_data.get("section", "unknown"),
            chunk_type=chunk_data.get("chunk_type", "text"),
            text=chunk_data["text"],
            image_paths_json=json.dumps(chunk_data.get("image_paths", []), ensure_ascii=False),
            page_number=chunk_data.get("page_number", 0),
            word_count=chunk_data.get("word_count", 0),
            has_pdf=chunk_data.get("has_pdf", False),
            extraction_method=chunk_data.get("extraction_method", "pymupdf_inline"),
        )
        db.add(chunk)
        saved.append(chunk)
    db.flush()
    return saved


def get_chunks_by_paper(db: Session, paper_id: str) -> list[PaperChunk]:
    """Get all chunks for a paper."""
    return db.query(PaperChunk).filter(PaperChunk.paper_id == paper_id).order_by(PaperChunk.chunk_index).all()


def has_chunks(db: Session, paper_id: str) -> bool:
    """Check if a paper already has chunks."""
    return db.query(PaperChunk).filter(PaperChunk.paper_id == paper_id).count() > 0


def get_paper_ids_with_chunks(db: Session, paper_ids: list[str]) -> set[str]:
    """Get set of paper IDs that already have chunks."""
    if not paper_ids:
        return set()
    results = db.query(PaperChunk.paper_id).filter(PaperChunk.paper_id.in_(paper_ids)).distinct().all()
    return {r[0] for r in results}
