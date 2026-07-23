"""SQLAlchemy ORM models."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Integer, Float, Text, Boolean, DateTime, ForeignKey, Index, text
from sqlalchemy.orm import relationship

from app.db.session import Base


def _utcnow() -> datetime:
    """Return current time as naive UTC (SQLite strips tzinfo on storage).

    Uses datetime.now(timezone.utc).replace(tzinfo=None) instead of the
    deprecated datetime.utcnow() (removed in Python 3.12+).
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


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
    # P1-5: soft-delete support for idea retry — 'active' (default) or 'superseded'
    idea_status = Column(String, default="active")
    created_at = Column(DateTime, default=_utcnow)

    task = relationship("ResearchTask", back_populates="ideas")

    __table_args__ = (
        Index("idx_ideas_task_status", "task_id", "idea_status"),
    )


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


class PaperAnalysis(Base):
    """Deep structured analysis of a paper, generated by LLM from full text or abstract.

    Replaces the shallow "500-char abstract + one-line summary" approach.
    Used as grounding knowledge for report generation, idea generation, and wiki ingest.

    Key design:
    - High-priority papers: analyzed from downloaded PDF full text
    - Medium/low papers: analyzed from full abstract (no truncation)
    - Structured output: problem, method_detail, experiment_setup, key_results, limitations, extendable_components
    - Source traceability: source_sections records where each piece of info came from
    """
    __tablename__ = "paper_analyses"

    id = Column(String, primary_key=True, default=_uuid)
    task_id = Column(String, ForeignKey("research_tasks.id"), nullable=False)
    paper_id = Column(String, ForeignKey("papers.id"), nullable=False)

    # Structured analysis fields (all in Chinese)
    problem = Column(Text, default="")                    # 解决什么具体问题
    method_detail = Column(Text, default="")              # 方法细节（架构、算法、创新点）
    experiment_setup = Column(Text, default="")           # 实验设置（数据集、基线、指标）
    key_results = Column(Text, default="")                # 关键结果（具体数值）
    limitations = Column(Text, default="")                # 局限性
    extendable_components = Column(Text, default="")      # 可复用/改进的组件

    # Traceability
    source_sections = Column(Text, default="{}")          # JSON: {"method": "Section 3", "results": "Table 2"}
    has_full_text = Column(Boolean, default=False)        # 是否用了PDF全文
    analysis_tokens = Column(Integer, default=0)          # 分析消耗的token数

    created_at = Column(DateTime, default=_utcnow)

    paper = relationship("Paper")

    __table_args__ = (
        Index("idx_paper_analyses_task", "task_id"),
        Index("idx_paper_analyses_paper", "paper_id"),
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


class PhaseRun(Base):
    """Phase execution record — tracks each phase of the refactored pipeline.

    Phase 0: Added for stage-level orchestration, retry, and resume support.
    Each phase (clarify, build_contract, decompose, search, evidence, coverage,
    gap_mining, gap_audit, feasibility_gate, idea_synthesis, idea_judgment)
    has its own PhaseRun record with status, timing, and error info.
    """
    __tablename__ = "phase_runs"

    id = Column(String, primary_key=True, default=_uuid)
    task_id = Column(String, ForeignKey("research_tasks.id"), nullable=False)
    phase_name = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending")
    # pending / running / completed / failed / skipped

    attempt_count = Column(Integer, default=0)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    # Versioning for resume: if input_version hasn't changed, can skip
    input_version = Column(Text)
    output_version = Column(Text)

    error_message = Column(Text)
    round_number = Column(Integer)  # for search rounds
    output_summary = Column(Text)  # brief JSON summary of outputs

    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        Index("idx_phase_runs_task", "task_id"),
        Index("idx_phase_runs_task_phase", "task_id", "phase_name"),
        Index("idx_phase_runs_status", "status"),
    )


# === Phase 1: Research Contract + Questions ===

class ResearchContract(Base):
    """Structured research direction contract.

    Replaces the old `normalized_topic` + `keywords` approach with a rich
    structured contract that captures user intent, constraints, and preferences.

    Phase 1: Added as part of the refactoring to evidence-grounded system.
    """
    __tablename__ = "research_contracts"

    id = Column(String, primary_key=True, default=_uuid)
    task_id = Column(String, ForeignKey("research_tasks.id"), nullable=False)

    # Core research direction
    topic = Column(Text, nullable=False)
    target_problem = Column(Text)
    target_setting = Column(Text)

    # Desired output type
    desired_output = Column(Text)  # method / system / benchmark / empirical_analysis

    # Novelty bar
    novelty_bar = Column(Text, default="conference")  # course_project / master_thesis / conference

    # Direction constraints
    preferred_directions_json = Column(Text, default="[]")
    excluded_directions_json = Column(Text, default="[]")

    # Resource constraints
    gpu_available = Column(Boolean)
    max_gpu_hours = Column(Float)
    max_api_budget = Column(Float)
    max_runtime_minutes = Column(Integer)
    allow_large_benchmark = Column(Boolean, default=True)
    allow_model_training = Column(Boolean, default=True)

    # Experiment preferences
    experiment_preferences_json = Column(Text, default="{}")

    # Search scope
    key_terms_json = Column(Text, default="[]")
    time_scope_start = Column(Integer)  # year
    time_scope_end = Column(Integer)    # year

    # Status
    status = Column(Text, default="active")
    confidence = Column(Float, default=0.5)

    # Phase 1.5: Versioning
    version = Column(Integer, nullable=False, default=1)
    input_hash = Column(Text, nullable=False, default="")
    superseded_at = Column(DateTime)
    source_feedback_ids_json = Column(Text, default="[]")

    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        Index("idx_contracts_task", "task_id"),
    )


class ResearchQuestion(Base):
    """A specific, searchable, answerable research question.

    Generated by decompose_research_space from a ResearchContract.
    Each question tracks its coverage status across search rounds.
    """
    __tablename__ = "research_questions"

    id = Column(String, primary_key=True, default=_uuid)
    task_id = Column(String, ForeignKey("research_tasks.id"), nullable=False)
    contract_id = Column(String, ForeignKey("research_contracts.id"))

    question = Column(Text, nullable=False)
    question_type = Column(Text, nullable=False)
    # problem / method / evaluation / dataset / resource / failure / application

    importance = Column(Float, default=0.5)
    searchability = Column(Float, default=0.5)
    status = Column(Text, default="open")
    # open / partially_covered / covered / unavailable / superseded

    axis_name = Column(Text)  # which research axis this question belongs to

    # Phase 1.5: Versioning support
    version = Column(Integer, nullable=False, default=1)
    superseded_at = Column(DateTime)

    created_at = Column(DateTime, default=_utcnow)

    __table_args__ = (
        Index("idx_rq_task", "task_id"),
        Index("idx_rq_status", "status"),
        Index("idx_rq_task_type", "task_id", "question_type"),
        Index("idx_rq_contract", "contract_id"),
    )


# === Phase 2: Evidence + Coverage ===

class EvidenceUnit(Base):
    """A single piece of evidence extracted from a paper.

    Phase 2.1: Added span_start, span_end, source_chunk_hash for provenance
    validation. Page numbers come from actual PDF pages, not chunk indices.
    """
    __tablename__ = "evidence_units"

    id = Column(String, primary_key=True, default=_uuid)
    task_id = Column(String, ForeignKey("research_tasks.id"), nullable=False)
    paper_id = Column(String, ForeignKey("papers.id"), nullable=False)

    evidence_type = Column(Text, nullable=False)

    normalized_claim = Column(Text, nullable=False)
    original_span = Column(Text)
    section = Column(Text)
    page_number = Column(Integer)      # Real PDF page number
    page_start = Column(Integer)       # Phase 2.1: for multi-page spans
    page_end = Column(Integer)

    # Phase 2.1: Byte-level provenance within source chunk
    span_start = Column(Integer)       # Character offset in source chunk
    span_end = Column(Integer)         # Character offset end in source chunk
    source_chunk_hash = Column(Text)   # SHA-256 of the source chunk text

    conditions_json = Column(Text, default="{}")
    dataset_name = Column(Text)
    metric_name = Column(Text)
    result_value = Column(Text)

    extraction_method = Column(Text, default="llm")
    extraction_confidence = Column(Float, default=0.5)

    verification_status = Column(Text, default="unverified")
    # unverified / verified / conflicted / rejected / abstract_only / upgraded

    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    paper = relationship("Paper")

    __table_args__ = (
        Index("idx_eu_task", "task_id"),
        Index("idx_eu_paper", "paper_id"),
        Index("idx_eu_type", "evidence_type"),
        Index("idx_eu_verification", "verification_status"),
        Index("idx_eu_chunk_hash", "source_chunk_hash"),
        Index("idx_eu_task_paper_hash", "task_id", "paper_id", "source_chunk_hash"),
    )


class QuestionEvidenceLink(Base):
    """Links an EvidenceUnit to a ResearchQuestion with a relationship type.

    Supports coverage tracking: each question accumulates supporting/contradicting evidence.
    """
    __tablename__ = "question_evidence_links"

    id = Column(String, primary_key=True, default=_uuid)
    question_id = Column(String, ForeignKey("research_questions.id"), nullable=False)
    evidence_id = Column(String, ForeignKey("evidence_units.id"), nullable=False)
    relation_type = Column(Text, default="supports")
    # supports / contradicts / partially_answers / background
    relevance_score = Column(Float, default=0.5)

    created_at = Column(DateTime, default=_utcnow)

    __table_args__ = (
        Index("idx_qel_question", "question_id"),
        Index("idx_qel_evidence", "evidence_id"),
    )


class PaperRole(Base):
    """Classification of a paper's role in the research context.

    A paper can have multiple roles (e.g., both 'survey' and 'benchmark').
    Replaces the single 'priority' field with multi-dimensional classification.
    """
    __tablename__ = "paper_roles"

    id = Column(String, primary_key=True, default=_uuid)
    task_id = Column(String, ForeignKey("research_tasks.id"), nullable=False)
    paper_id = Column(String, ForeignKey("papers.id"), nullable=False)
    role = Column(Text, nullable=False)
    # survey / seminal / direct_neighbor / benchmark / method /
    # negative_result / limitation_evidence / application
    confidence = Column(Float, default=0.5)
    reason = Column(Text)

    created_at = Column(DateTime, default=_utcnow)

    __table_args__ = (
        Index("idx_pr_task", "task_id"),
        Index("idx_pr_paper", "paper_id"),
        Index("idx_pr_role", "role"),
    )


class CoverageRecord(Base):
    """Coverage of a ResearchQuestion based on accumulated EvidenceUnits.

    Updated after each search round. Drives the next round's question selection.
    Phase 2.1: Added round_number for per-round snapshots.
    """
    __tablename__ = "coverage_records"

    id = Column(String, primary_key=True, default=_uuid)
    task_id = Column(String, ForeignKey("research_tasks.id"), nullable=False)
    question_id = Column(String, ForeignKey("research_questions.id"), nullable=False)

    coverage_score = Column(Float, default=0.0)
    confidence = Column(Float, default=0.0)
    supporting_evidence_count = Column(Integer, default=0)
    contradicting_evidence_count = Column(Integer, default=0)
    direct_neighbor_count = Column(Integer, default=0)
    unresolved_aspects_json = Column(Text, default="[]")
    unavailable_reason = Column(Text)

    # Phase 2.1: Round tracking for snapshots
    round_number = Column(Integer, nullable=False, default=0)

    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
    created_at = Column(DateTime, default=_utcnow)

    __table_args__ = (
        Index("idx_cr_task", "task_id"),
        Index("idx_cr_question", "question_id"),
        Index("idx_cr_task_question_round", "task_id", "question_id", "round_number"),
    )


class SearchQueryRecord(Base):
    """Structured search query with target question binding.

    Phase 2.2A: Added normalized_query_text, completed_at, and unique constraint.
    """
    __tablename__ = "search_query_records"

    id = Column(String, primary_key=True, default=_uuid)
    task_id = Column(String, ForeignKey("research_tasks.id"), nullable=False)
    query_text = Column(Text, nullable=False)
    normalized_query_text = Column(Text, default="")  # Phase 2.2A

    intent = Column(Text, nullable=False)
    target_question_id = Column(String, ForeignKey("research_questions.id"))
    expected_evidence_type = Column(Text)

    round_number = Column(Integer, nullable=False)
    status = Column(Text, default="pending")
    # pending / completed / failed

    result_count = Column(Integer, default=0)
    new_paper_count = Column(Integer, default=0)
    evidence_unit_count = Column(Integer, default=0)
    execution_error = Column(Text)

    created_at = Column(DateTime, default=_utcnow)
    completed_at = Column(DateTime)  # Phase 2.2A

    __table_args__ = (
        Index("idx_sqr_task", "task_id"),
        Index("idx_sqr_intent", "intent"),
        Index("idx_sqr_round", "round_number"),
        Index("idx_sqr_question", "target_question_id"),
        # Phase 2.2A: Unique constraint
        Index("idx_sqr_unique", "task_id", "round_number",
              "normalized_query_text", "target_question_id", unique=True),
    )
