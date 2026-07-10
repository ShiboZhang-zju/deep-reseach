"""SQLAlchemy ORM models."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Integer, Float, Text, Boolean, DateTime, ForeignKey, Index, text
from sqlalchemy.orm import relationship

from app.db.session import Base


def _utcnow() -> datetime:
    """Return current time in UTC. SQLite strips tzinfo on storage,
    so we return naive UTC to keep storage consistent.
    API layer re-appends +00:00 via isoformat_utc().
    """
    return datetime.utcnow()


def isoformat_utc(dt: datetime | None) -> str:
    """Format datetime as ISO 8601 with explicit UTC timezone marker.
    
    SQLite stores naive datetimes (strips tzinfo), so we re-append +00:00
    to ensure the frontend can correctly convert to local time.
    """
    if not dt:
        return ""
    # If already has tzinfo, use it as-is
    if dt.tzinfo is not None:
        return dt.isoformat()
    # Naive datetime — assume UTC and append +00:00
    return dt.isoformat() + "+00:00"


def _uuid() -> str:
    return str(uuid.uuid4())


class ResearchTask(Base):
    __tablename__ = "research_tasks"

    id = Column(String, primary_key=True, default=_uuid)
    user_input = Column(Text, nullable=False)
    normalized_topic = Column(Text)
    status = Column(String, nullable=False, default="pending")
    current_round = Column(Integer, default=0)
    max_rounds = Column(Integer, default=5)
    stop_reason = Column(Text)
    state_json = Column(Text)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    rounds = relationship("ResearchRound", back_populates="task", cascade="all, delete-orphan")
    task_papers = relationship("TaskPaper", back_populates="task", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="task", cascade="all, delete-orphan")
    ideas = relationship("ResearchIdea", back_populates="task", cascade="all, delete-orphan")
    traces = relationship("AgentTrace", back_populates="task", cascade="all, delete-orphan")
    feedbacks = relationship("UserFeedback", back_populates="task", cascade="all, delete-orphan")
    experiments = relationship("ExperimentPlan", back_populates="task", cascade="all, delete-orphan")


class Paper(Base):
    __tablename__ = "papers"

    id = Column(String, primary_key=True, default=_uuid)
    title = Column(Text, nullable=False)
    abstract = Column(Text)
    authors_json = Column(Text)
    year = Column(Integer)
    venue = Column(Text)
    doi = Column(Text)
    arxiv_id = Column(Text)
    semantic_scholar_id = Column(Text)
    openalex_id = Column(Text)
    url = Column(Text)
    pdf_url = Column(Text)
    citation_count = Column(Integer, default=0)
    sources_json = Column(Text)
    raw_json = Column(Text)
    normalized_title = Column(Text)
    title_hash = Column(String, index=True)
    created_at = Column(DateTime, default=_utcnow)

    task_papers = relationship("TaskPaper", back_populates="paper")
    chunks = relationship("PaperChunk", back_populates="paper", cascade="all, delete-orphan")
    citation_sources = relationship("PaperCitation", foreign_keys="PaperCitation.source_paper_id", cascade="all, delete-orphan")
    citation_targets = relationship("PaperCitation", foreign_keys="PaperCitation.target_paper_id", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_papers_doi", "doi", unique=True, sqlite_where=text("doi IS NOT NULL")),
        Index("idx_papers_arxiv", "arxiv_id", unique=True, sqlite_where=text("arxiv_id IS NOT NULL")),
        Index("idx_papers_s2", "semantic_scholar_id", unique=True, sqlite_where=text("semantic_scholar_id IS NOT NULL")),
        Index("idx_papers_openalex", "openalex_id", unique=True, sqlite_where=text("openalex_id IS NOT NULL")),
        Index("idx_papers_year", "year"),
        Index("idx_papers_citation", "citation_count"),
    )


class TaskPaper(Base):
    __tablename__ = "task_papers"

    id = Column(String, primary_key=True, default=_uuid)
    task_id = Column(String, ForeignKey("research_tasks.id"), nullable=False)
    paper_id = Column(String, ForeignKey("papers.id"), nullable=False)
    discovered_round = Column(Integer, nullable=False)
    relevance_score = Column(Float)
    authority_score = Column(Float)
    recency_score = Column(Float)
    novelty_score = Column(Float)
    idea_potential_score = Column(Float)
    final_score = Column(Float)
    priority = Column(String)
    reason = Column(Text)
    summary = Column(Text)
    created_at = Column(DateTime, default=_utcnow)

    task = relationship("ResearchTask", back_populates="task_papers")
    paper = relationship("Paper", back_populates="task_papers")

    __table_args__ = (
        Index("idx_task_papers_unique", "task_id", "paper_id", unique=True),
    )


class ResearchRound(Base):
    __tablename__ = "research_rounds"

    id = Column(String, primary_key=True, default=_uuid)
    task_id = Column(String, ForeignKey("research_tasks.id"), nullable=False)
    round_number = Column(Integer, nullable=False)
    queries_json = Column(Text)
    papers_found = Column(Integer, default=0)
    new_papers = Column(Integer, default=0)
    duplicate_rate = Column(Float)
    summary = Column(Text)
    knowledge_gaps_json = Column(Text)
    created_at = Column(DateTime, default=_utcnow)

    task = relationship("ResearchTask", back_populates="rounds")

    __table_args__ = (
        Index("idx_rounds_unique", "task_id", "round_number", unique=True),
    )


class Report(Base):
    __tablename__ = "reports"

    id = Column(String, primary_key=True, default=_uuid)
    task_id = Column(String, ForeignKey("research_tasks.id"), nullable=False)
    content_markdown = Column(Text)
    content_json = Column(Text)
    created_at = Column(DateTime, default=_utcnow)

    task = relationship("ResearchTask", back_populates="reports")


class ResearchIdea(Base):
    __tablename__ = "research_ideas"

    id = Column(String, primary_key=True, default=_uuid)
    task_id = Column(String, ForeignKey("research_tasks.id"), nullable=False)
    title = Column(Text)
    description = Column(Text)
    motivation = Column(Text)
    method_sketch = Column(Text)
    expected_contribution = Column(Text)
    novelty = Column(Float)
    feasibility = Column(Float)
    significance = Column(Float)
    evidence_support = Column(Float)
    differentiation = Column(Float)
    experimentability = Column(Float)
    potential_impact = Column(Float)
    risk = Column(Float)
    final_score = Column(Float)
    decision = Column(String)
    related_paper_ids_json = Column(Text)
    user_selected = Column(Boolean, default=False)
    created_at = Column(DateTime, default=_utcnow)

    task = relationship("ResearchTask", back_populates="ideas")


class ExperimentPlan(Base):
    __tablename__ = "experiment_plans"

    id = Column(String, primary_key=True, default=_uuid)
    task_id = Column(String, ForeignKey("research_tasks.id"), nullable=False)
    idea_id = Column(String, ForeignKey("research_ideas.id"), nullable=False)
    hypothesis = Column(Text)
    dataset = Column(Text)
    baselines = Column(Text)
    metrics = Column(Text)
    steps_markdown = Column(Text)
    steps_json = Column(Text)
    risks = Column(Text)
    created_at = Column(DateTime, default=_utcnow)

    task = relationship("ResearchTask", back_populates="experiments")


class AgentTrace(Base):
    __tablename__ = "agent_traces"

    id = Column(String, primary_key=True, default=_uuid)
    task_id = Column(String, ForeignKey("research_tasks.id"), nullable=False)
    step_name = Column(Text, nullable=False)
    step_type = Column(String, nullable=False)
    round_number = Column(Integer)
    input_json = Column(Text)
    output_json = Column(Text)
    llm_tokens_used = Column(Integer)
    duration_ms = Column(Integer)
    created_at = Column(DateTime, default=_utcnow)

    task = relationship("ResearchTask", back_populates="traces")


class PaperChunk(Base):
    __tablename__ = "paper_chunks"

    id = Column(String, primary_key=True, default=_uuid)
    paper_id = Column(String, ForeignKey("papers.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    section = Column(Text, default="unknown")
    chunk_type = Column(Text, default="text")  # text / figure / table / formula
    text = Column(Text, nullable=False)
    image_paths_json = Column(Text, default="[]")
    page_number = Column(Integer, default=0)
    word_count = Column(Integer, default=0)
    has_pdf = Column(Boolean, default=False)
    extraction_method = Column(Text, default="pymupdf_inline")
    created_at = Column(DateTime, default=_utcnow)

    paper = relationship("Paper", back_populates="chunks")

    __table_args__ = (
        Index("idx_paper_chunks_paper", "paper_id"),
        Index("idx_paper_chunks_section", "section"),
        Index("idx_paper_chunks_type", "chunk_type"),
    )


class PaperCitation(Base):
    """Paper-to-paper citation relationships.
    
    relation_type:
    - 'cites': source directly cites target (from S2/OpenAlex API)
    - 'co_cited': source and target are co-cited by a third paper
    - 'biblio_coupled': source and target share common references
    - 'semantic_similar': source and target have high embedding similarity
    """
    __tablename__ = "paper_citations"

    id = Column(String, primary_key=True, default=_uuid)
    source_paper_id = Column(String, ForeignKey("papers.id"), nullable=False)
    target_paper_id = Column(String, ForeignKey("papers.id"), nullable=False)
    relation_type = Column(String, nullable=False)
    weight = Column(Float, default=1.0)  # similarity score for co_cited/biblio/semantic
    source_task_id = Column(String, ForeignKey("research_tasks.id"))  # which task discovered this
    created_at = Column(DateTime, default=_utcnow)

    source_paper = relationship("Paper", foreign_keys=[source_paper_id])
    target_paper = relationship("Paper", foreign_keys=[target_paper_id])

    __table_args__ = (
        Index("idx_citations_source", "source_paper_id"),
        Index("idx_citations_target", "target_paper_id"),
        Index("idx_citations_type", "relation_type"),
        Index("idx_citations_unique", "source_paper_id", "target_paper_id", "relation_type", unique=True),
    )


class UserFeedback(Base):
    __tablename__ = "user_feedbacks"

    id = Column(String, primary_key=True, default=_uuid)
    task_id = Column(String, ForeignKey("research_tasks.id"), nullable=False)
    feedback_type = Column(String, nullable=False)
    content = Column(Text)
    selected_idea_ids_json = Column(Text)
    need_more_research = Column(Boolean, default=False)
    created_at = Column(DateTime, default=_utcnow)

    task = relationship("ResearchTask", back_populates="feedbacks")


class WikiPage(Base):
    """LLM Wiki pages — incrementally compiled knowledge from papers.

    Replaces GraphRAG's entity/relation/community approach with
    pre-compiled, human-readable markdown pages.

    page_type:
    - concept: Research theme/direction (replaces clustering)
    - method: Specific technique/algorithm
    - dataset: Specific dataset
    - model: Specific model
    - synthesis: Cross-cutting analysis
    - index: Auto-maintained wiki index
    """
    __tablename__ = "wiki_pages"

    id = Column(String, primary_key=True, default=_uuid)
    task_id = Column(String, ForeignKey("research_tasks.id"), nullable=False)
    page_type = Column(String, nullable=False)
    title = Column(Text, nullable=False)
    content_markdown = Column(Text, default="")
    paper_ids_json = Column(Text, default="[]")
    links_json = Column(Text, default="[]")
    contradictions_json = Column(Text, default="[]")
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        Index("idx_wiki_task_type", "task_id", "page_type"),
        Index("idx_wiki_task_title", "task_id", "title"),
    )
