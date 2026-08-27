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
    # Open-access flag: None = unknown, True = OA, False = paywalled. Used by
    # scoring to deprioritise paywalled papers that cannot be fetched.
    is_oa = Column(Boolean, nullable=True)
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
    contract_id = Column(String, ForeignKey("research_contracts.id"))
    gap_id = Column(String, ForeignKey("gap_candidates.id"))
    intervention_id = Column(String, ForeignKey("intervention_candidates.id"))
    pipeline_version = Column(Integer)
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
    score_status = Column(String, default="unscored")
    score_error = Column(Text)
    related_paper_ids_json = Column(Text)
    # O1: graded output inherited from the backing intervention —
    # A / B / C confidence tier so users see a ranked list of directions
    # instead of a binary go/reject.
    confidence_tier = Column(Text, default="C")
    user_selected = Column(Boolean, default=False)
    # P1-5: soft-delete support for idea retry — 'active' (default) or 'superseded'
    idea_status = Column(String, default="active")
    # P0/P1: deterministic quality classification and machine-readable blockers.
    quality_reason_codes_json = Column(Text, default="[]")
    created_at = Column(DateTime, default=_utcnow)

    task = relationship("ResearchTask", back_populates="ideas")

    __table_args__ = (
        Index("idx_ideas_task_status", "task_id", "idea_status"),
        Index("idx_ideas_contract", "contract_id"),
        Index("idx_ideas_intervention", "intervention_id"),
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
    # P1: structured experiment consistency contract.
    model_spec = Column(Text)
    dataset_provenance = Column(Text)
    oracle = Column(Text)
    statistical_analysis = Column(Text)
    resource_budget = Column(Text)
    scenario_atoms_json = Column(Text, default="[]")
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
    output_summary = Column(Text)  # brief JSON summary of outputs (truncated)
    output_json = Column(Text)  # Phase 2.2A: complete output payload (untruncated)

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

    # P1-1: round in which this evidence unit was extracted (per-round
    # marginal-evidence counting in Search Saturation). NULL for pre-P1-1 rows.
    discovered_round = Column(Integer, nullable=True)

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
    # Coverage breadth — how many *distinct papers* speak to the question. This
    # is the driver of coverage_score (not the evidence-unit counts), so it must
    # be queryable instead of living only inside a trace JSON blob.
    distinct_supporting_papers = Column(Integer, default=0)
    distinct_contradicting_papers = Column(Integer, default=0)
    direct_neighbor_count = Column(Integer, default=0)
    unresolved_aspects_json = Column(Text, default="[]")
    unavailable_reason = Column(Text)

    # P1-1: Evidence quality + search saturation (mechanical, per-RQ, per-round).
    evidence_quality = Column(Float, nullable=True)     # 0..1 aggregated quality
    fulltext_ratio = Column(Float, nullable=True)       # fraction of fulltext-locatable evidence
    directness = Column(Float, nullable=True)           # fraction of direct "supports" links
    search_saturation = Column(Text, nullable=True)     # INSUFFICIENT_OBSERVATION / STILL_GAINING / SATURATED
    last_round_marginal_papers = Column(Integer, nullable=True)
    last_round_marginal_evidence = Column(Integer, nullable=True)

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

    # Phase 3A: Gap-driven query binding
    target_gap_id = Column(String, ForeignKey("gap_candidates.id"))
    query_family = Column(Text, default="")
    search_policy_version = Column(Text, default="")

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
        Index("idx_sqr_gap", "target_gap_id"),
        # Phase 3A Closure: Two partial unique indexes replace the old
        # single unique index to support gap-driven queries. The search policy
        # is part of the identity so a policy revision can create a fresh
        # audit query without colliding with an older policy's record.
        # Discovery queries (target_gap_id IS NULL):
        Index("idx_sqr_unique_discovery", "task_id", "round_number",
              "normalized_query_text", "target_question_id",
              "search_policy_version", unique=True,
              sqlite_where=text("target_gap_id IS NULL")),
        # Gap queries (target_gap_id IS NOT NULL):
        Index("idx_sqr_unique_gap", "task_id", "round_number",
              "normalized_query_text", "target_gap_id",
              "search_policy_version", unique=True,
              sqlite_where=text("target_gap_id IS NOT NULL")),
    )


class SearchQueryPaper(Base):
    """Maps search queries to papers found — for yield tracking and provenance.

    Phase 2.2A Closure (#3): Query→Paper mapping for traceability.
    """
    __tablename__ = "search_query_papers"

    id = Column(String, primary_key=True, default=_uuid)
    query_id = Column(String, ForeignKey("search_query_records.id"), nullable=False)
    paper_id = Column(String, ForeignKey("papers.id"), nullable=False)
    rank = Column(Integer, default=0)
    source = Column(Text, default="unknown")
    is_new_for_task = Column(Boolean, default=True)

    created_at = Column(DateTime, default=_utcnow)

    __table_args__ = (
        Index("idx_sqp_query", "query_id"),
        Index("idx_sqp_paper", "paper_id"),
        Index("idx_sqp_unique", "query_id", "paper_id", "source", unique=True),
    )


class SearchRawResult(Base):
    """Minimal audit record of a paper as returned by a source, BEFORE any
    canonicalization or pre-filter.

    Persisting only the raw identity (external ids + title + year + raw rank)
    lets the overlap diagnostic reconstruct the recall waterfall for a query
    family's variants: raw -> canonical -> post-filter -> final Top-K. We
    deliberately do NOT store the full API payload here.
    """
    __tablename__ = "search_raw_results"

    id = Column(String, primary_key=True, default=_uuid)
    query_id = Column(String, ForeignKey("search_query_records.id"), nullable=False)
    source = Column(Text, default="unknown")
    # Position within this source's raw result list (pre-filter, per query).
    raw_rank = Column(Integer, default=0)
    # Best available external identifier for this paper (S2/OpenAlex/arXiv/DOI).
    external_paper_id = Column(Text)
    doi = Column(Text)
    arxiv_id = Column(Text)
    title = Column(Text)
    year = Column(Integer)
    # Backfilled after normalize/upsert maps the raw paper to a canonical row.
    canonical_paper_id = Column(String, ForeignKey("papers.id"), nullable=True)
    retrieved_at = Column(DateTime, default=_utcnow)

    __table_args__ = (
        Index("idx_srr_query", "query_id"),
        Index("idx_srr_query_src_rank", "query_id", "source", "raw_rank", unique=True),
        Index("idx_srr_canonical", "canonical_paper_id"),
    )


# === Phase 3A: Gap Control Plane ===

class GapCandidate(Base):
    """A candidate research gap identified from the Coverage Matrix.

    Phase 3A Closure: Structured falsifiable contract fields added.

    gap_type:
    - coverage_gap: A research question with low coverage score
    - contradiction: Evidence units conflict with each other
    - missing_method: No method evidence found for a question
    - missing_dataset: No dataset/benchmark evidence found
    - missing_evaluation: No evaluation/metric evidence found
    - boundary_gap: Edge of current knowledge, unexplored combination

    status (frozen enum):
    - candidate: Newly identified, not yet audited
    - auditing: Adversarial audit in progress
    - audited: Has been through adversarial audit (Phase 3C)
    - surviving: Passed audit, eligible for feasibility gate (Phase 4)
    - rejected: Failed audit or feasibility gate
    - superseded: Replaced by a newer version (contract change)
    - inconclusive: Search budget exhausted before novelty could be decided
      (distinct from `rejected`, which asserts the gap is disproven)

    provenance_status:
    - complete: All structured fields populated with evidence backing
    - partial: Some fields populated, some evidence missing
    - invalid: Structured fields inconsistent with evidence

    NOTE: supporting_evidence_ids_json and contradicting_evidence_ids_json
    are DEPRECATED snapshots. The authoritative source is gap_evidence_links.
    """
    __tablename__ = "gap_candidates"

    id = Column(String, primary_key=True, default=_uuid)
    task_id = Column(String, ForeignKey("research_tasks.id"), nullable=False)
    contract_id = Column(String, ForeignKey("research_contracts.id"))

    # Gap identification
    gap_type = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    # description is a human-readable summary; structured fields below are the
    # authoritative contract for Phase 3B/3C business logic.

    # Phase 3A Closure: Structured falsifiable contract
    target_setting = Column(Text)
    # What setting/scenario does this gap apply to?
    observed_problem = Column(Text)
    # What specific problem was observed in the evidence?
    existing_coverage = Column(Text)
    # What do existing papers already cover?
    missing_capability = Column(Text)
    # What specific capability/method/dataset is missing?
    claimed_delta = Column(Text)
    # What is the claimed improvement/difference from existing work?
    testable_hypothesis = Column(Text)
    # A falsifiable hypothesis that, if true, confirms the gap is real
    falsification_condition = Column(Text)
    # What evidence would falsify this gap (prove it's not a real gap)?
    provenance_status = Column(Text, default="partial")
    # complete / partial / invalid

    # Linkage to research questions (DEPRECATED — use gap_evidence_links for evidence)
    question_ids_json = Column(Text, default="[]")

    # DEPRECATED snapshots — use gap_evidence_links as authoritative source
    supporting_evidence_ids_json = Column(Text, default="[]")
    contradicting_evidence_ids_json = Column(Text, default="[]")

    # Mining context
    mining_round = Column(Integer, nullable=False, default=0)
    mining_policy_version = Column(Text, default="")
    # Evidence fingerprint the gap was mined from (evidence-sensitive
    # idempotency): a remediation round adds evidence, changes the fingerprint,
    # and the existing-gap short-circuit no longer matches, so mining re-runs
    # and the new evidence can surface new gaps. NULL for legacy rows (never
    # matches a non-empty fingerprint -> resume re-mines once).
    mining_input_version = Column(Text, default="")

    # Assessment (populated in Phase 3C audit)
    novelty_score = Column(Float)
    feasibility_score = Column(Float)
    significance_score = Column(Float)
    risk_score = Column(Float)

    # Status (frozen enum)
    status = Column(Text, nullable=False, default="candidate")
    # candidate / auditing / audited / surviving / rejected / superseded

    # Versioning
    version = Column(Integer, nullable=False, default=1)
    superseded_at = Column(DateTime)

    # Lineage (Phase 3D): narrowing a gap creates a new version of the SAME
    # canonical gap, not a sibling row. canonical_gap_id points at the lineage
    # root (NULL on the root itself — resolve with `canonical_gap_id or id`);
    # parent_gap_id points at the immediately superseded version. This is what
    # lets the report show one row per canonical gap (its latest version) and
    # keeps a rejected gap from being reworded into a fresh "novel" gap.
    canonical_gap_id = Column(String, nullable=True)
    parent_gap_id = Column(String, nullable=True)

    # Nearest-prior-art provenance (Phase 3E): a surviving gap's novelty is
    # reported as "closest prior work + residual gap + search confidence"
    # instead of a bare novelty_confidence, so the claim is traceable to a
    # specific paper and the audit's retrieval quality is explicit.
    nearest_prior_art_paper_id = Column(String, nullable=True)
    nearest_prior_art_title = Column(String, nullable=True)
    residual_gap = Column(Text, nullable=True)
    search_confidence = Column(String, nullable=True)  # INSUFFICIENT_OBSERVATION / high / medium / low

    # P1-1: Novelty search coverage + nearest-prior-art stability (mechanical).
    npa_stability = Column(Float, nullable=True)          # cross-round Top-K stability 0..1
    family_coverage = Column(Float, nullable=True)        # query-family coverage 0..1
    residual_claim_ids_json = Column(Text, default="[]")  # atomic claim ids left uncovered

    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        Index("idx_gc_task", "task_id"),
        Index("idx_gc_contract", "contract_id"),
        Index("idx_gc_status", "status"),
        Index("idx_gc_task_status", "task_id", "status"),
        Index("idx_gc_canonical", "canonical_gap_id"),
    )


class GapPhenomenonPlan(Base):
    """The phenomenon-validation plan for a surviving gap (Phase 3F).

    Before any intervention or method is designed, the gap's underlying
    empirical claim is pinned down: what phenomenon it rests on, the cheapest
    experiment that could falsify it, and the kill criterion below which the
    phenomenon is too small to be worth a method paper. This gates method
    generation on "the problem is real and measurable", not "the story is
    plausible".
    """
    __tablename__ = "gap_phenomenon_plans"

    id = Column(String, primary_key=True, default=_uuid)
    task_id = Column(String, ForeignKey("research_tasks.id"), nullable=False)
    contract_id = Column(String, ForeignKey("research_contracts.id"))
    gap_id = Column(String, ForeignKey("gap_candidates.id"), nullable=False)

    phenomenon = Column(Text, nullable=False)
    mechanism_under_test = Column(Text)
    supports_gap_claim = Column(Text)
    critical_unknown = Column(Text)
    expected_observation = Column(Text)
    alternative_explanation = Column(Text)
    comparator = Column(Text)
    oracle_experiment = Column(Text)
    kill_criterion = Column(Text)
    kill_criterion_basis = Column(Text)
    measurement = Column(Text)
    pipeline_version = Column(Integer)
    created_at = Column(DateTime, default=_utcnow)

    __table_args__ = (
        Index("idx_gpp_task", "task_id"),
        Index("idx_gpp_gap", "gap_id"),
    )


class InterventionCandidate(Base):
    """A proposed mechanism that addresses one surviving gap."""
    __tablename__ = "intervention_candidates"

    id = Column(String, primary_key=True, default=_uuid)
    task_id = Column(String, ForeignKey("research_tasks.id"), nullable=False)
    gap_id = Column(String, ForeignKey("gap_candidates.id"), nullable=False)
    contract_id = Column(String, ForeignKey("research_contracts.id"))
    intervention_type = Column(Text, nullable=False)
    failure_mechanism = Column(Text, nullable=False)
    proposed_intervention = Column(Text, nullable=False)
    intermediate_effect = Column(Text, nullable=False)
    measurable_outcome = Column(Text, nullable=False)
    required_components_json = Column(Text, default="[]")
    dependency_paper_ids_json = Column(Text, default="[]")
    implementation_cost = Column(Text)
    mechanism_confidence = Column(Float)
    evidence_gate = Column(Text, default="UNKNOWN")
    novelty_gate = Column(Text, default="UNKNOWN")
    feasibility_gate = Column(Text, default="UNKNOWN")
    gate_rationale_json = Column(Text, default="{}")
    # O1: graded output — A (all gates PASS + full-text gap), B (some gate
    # UNKNOWN or abstract-only gap; needs human/experiment confirmation),
    # C (a gate FAILed but the intervention is kept as a speculative direction).
    confidence_tier = Column(Text, default="C")
    status = Column(Text, default="candidate")
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        Index("idx_ic_task", "task_id"),
        Index("idx_ic_gap", "gap_id"),
        Index("idx_ic_contract", "contract_id"),
        Index("idx_ic_status", "status"),
    )


class GapEvidenceLink(Base):
    """Links a GapCandidate to EvidenceUnit with a relationship type.

    Supports provenance: each gap must be traceable to specific evidence.
    """
    __tablename__ = "gap_evidence_links"

    id = Column(String, primary_key=True, default=_uuid)
    gap_id = Column(String, ForeignKey("gap_candidates.id"), nullable=False)
    evidence_id = Column(String, ForeignKey("evidence_units.id"), nullable=False)
    relation_type = Column(Text, default="suggests")
    # suggests / contradicts / limits / motivates / background

    relevance_score = Column(Float, default=0.5)

    created_at = Column(DateTime, default=_utcnow)

    __table_args__ = (
        Index("idx_gel_gap", "gap_id"),
        Index("idx_gel_evidence", "evidence_id"),
        Index("idx_gel_unique", "gap_id", "evidence_id", unique=True),
    )


class GapAudit(Base):
    """Adversarial audit result for a GapCandidate.

    Phase 3A Closure: Added decision fields and recommended_action enum.

    audit_result:
    - pending: Audit not yet started
    - confirmed: Gap is genuine — no existing work fully addresses it
    - partially_closed: Some existing work partially addresses it
    - closed: Existing work already addresses it — reject
    - uncertain: Insufficient evidence to determine

    recommended_action (frozen enum):
    - continue: Gap confirmed, proceed to feasibility gate → surviving
    - narrow: Gap partially closed, needs narrowing → audited
    - more_search: Insufficient evidence, need more adversarial search → auditing
    - reject: Gap is closed or invalid → rejected

    State mapping (Phase 3C business logic, NOT implemented here):
    - continue → surviving
    - narrow → audited
    - more_search → auditing
    - reject → rejected
    """
    __tablename__ = "gap_audits"

    id = Column(String, primary_key=True, default=_uuid)
    gap_id = Column(String, ForeignKey("gap_candidates.id"), nullable=False)
    task_id = Column(String, ForeignKey("research_tasks.id"), nullable=False)

    # Audit queries
    adversarial_queries_json = Column(Text, default="[]")
    search_policy_version = Column(Text, default="")
    search_admission_status = Column(Text, default="UNKNOWN")
    search_admission_reasons_json = Column(Text, default="[]")
    search_query_ids_json = Column(Text, default="[]")

    # Audit result
    audit_result = Column(Text, nullable=False, default="pending")

    nearest_neighbor_summary = Column(Text)
    differentiation_summary = Column(Text)
    neighbor_paper_ids_json = Column(Text, default="[]")

    # Phase 3A Closure: Decision fields
    evidence_for_gap_json = Column(Text, default="[]")
    # Evidence IDs supporting the gap's existence
    evidence_against_gap_json = Column(Text, default="[]")
    # Evidence IDs suggesting the gap is already addressed
    remaining_delta = Column(Text)
    # After audit, what delta remains between the gap and existing work?
    novelty_confidence = Column(Float)
    # 0-1, confidence that the gap is truly novel
    audit_confidence = Column(Float)
    # 0-1, confidence in the audit result itself
    recommended_action = Column(Text, default="continue")
    # continue / narrow / more_search / reject
    rejection_reason = Column(Text)
    # If recommended_action == reject, why?
    audited_claimed_delta = Column(Text)
    # The gap's claimed_delta at the time of this audit. An audit judges exactly
    # that claim, so a later run needs it to tell "nothing changed, the verdict
    # would be identical" apart from "the claim was narrowed, judge it again".

    # Audit diagnostics: machine-readable cause and marginal information. These
    # are separate from rejection_reason because an uncertain audit need not be
    # a rejection and may have several independent retrieval causes.
    failure_reason_codes_json = Column(Text, default="[]")
    evidence_delta_json = Column(Text, default="{}")

    # Audit metadata
    audit_round = Column(Integer, nullable=False, default=0)

    created_at = Column(DateTime, default=_utcnow)

    __table_args__ = (
        Index("idx_ga_gap", "gap_id"),
        Index("idx_ga_task", "task_id"),
        Index("idx_ga_result", "audit_result"),
    )


class NeighborComparison(Base):
    """Detailed comparison between a GapCandidate and a neighboring paper.

    Phase 3A Closure: Added structured comparison fields.
    shared_aspects_json and differentiating_aspects_json are kept for backward
    compatibility, but Phase 3C's formal judgment uses structured fields below.
    """
    __tablename__ = "neighbor_comparisons"

    id = Column(String, primary_key=True, default=_uuid)
    gap_id = Column(String, ForeignKey("gap_candidates.id"), nullable=False)
    paper_id = Column(String, ForeignKey("papers.id"), nullable=False)
    task_id = Column(String, ForeignKey("research_tasks.id"), nullable=False)

    # Comparison
    similarity_score = Column(Float, default=0.0)

    # DEPRECATED: use structured fields below
    shared_aspects_json = Column(Text, default="[]")
    differentiating_aspects_json = Column(Text, default="[]")

    overlap_risk = Column(Float, default=0.0)

    # Phase 3A Closure: Structured comparison fields
    shared_problem = Column(Text)
    # What problem does this paper share with the gap?
    shared_mechanism = Column(Text)
    # What mechanism/method is shared?
    shared_evaluation = Column(Text)
    # What evaluation approach is shared?
    covered_claims_json = Column(Text, default="[]")
    # JSON array of gap claims that this paper already covers
    uncovered_claims_json = Column(Text, default="[]")
    # JSON array of gap claims that this paper does NOT cover
    overlap_ratio = Column(Float, default=0.0)
    # 0-1, ratio of gap claims covered by this paper

    # P1-1: one-line statement of why this neighbor, despite its overlap, does
    # NOT close the gap (the residual difference, grounded in claim coverage).
    why_not_closed = Column(Text)

    created_at = Column(DateTime, default=_utcnow)

    __table_args__ = (
        Index("idx_nc_gap", "gap_id"),
        Index("idx_nc_paper", "paper_id"),
        Index("idx_nc_unique", "gap_id", "paper_id", unique=True),
    )


class GapPaperRelevance(Base):
    """Structured relevance of one paper to ONE specific gap (0027).

    Gap-specific prior-art screening (false-novelty audit, fd688ba6):
    TaskPaper.final_score measures task/RQ relevance, but the NPA audit needs
    "how close is this paper to THIS gap". Audit-recalled papers were stored
    with final_score=NULL and then out-ranked by broad surveys in neighbor
    selection, so direct prior art never reached the NPA pool. This is a
    many-to-many (gap, paper) table scored cheaply on title+abstract BEFORE
    the deep NPA audit; overlap fields are qualitative (yes/partial/no) and
    relevance_score is aggregated by code (never taken verbatim from the LLM).
    """

    __tablename__ = "gap_paper_relevance"

    id = Column(String, primary_key=True, default=_uuid)
    gap_id = Column(String, ForeignKey("gap_candidates.id"), nullable=False)
    paper_id = Column(String, ForeignKey("papers.id"), nullable=False)
    task_id = Column(String, ForeignKey("research_tasks.id"), nullable=False)

    relevance_score = Column(Float, nullable=False, default=0.0)
    problem_overlap = Column(String)          # yes / partial / no
    mechanism_overlap = Column(String)        # yes / partial / no
    evaluation_overlap = Column(String)       # yes / partial / no
    claim_overlap = Column(String)            # yes / partial / no
    addresses_claim_ids_json = Column(Text, default="[]")
    rationale = Column(Text)
    scoring_version = Column(String, default="gap-rel-v1")

    created_at = Column(DateTime, default=_utcnow)

    __table_args__ = (
        Index("idx_gpr_gap", "gap_id"),
        Index("idx_gpr_paper", "paper_id"),
        Index("idx_gpr_unique", "gap_id", "paper_id", unique=True),
    )


class GapAtomicClaim(Base):
    """A canonical atomic claim derived from a gap's claimed delta (Phase 3H).

    Each claim is a positive capability/delta statement (e.g. "dense hidden
    tests distinguish accidental from robust correctness"), NOT a literature-gap
    conclusion like "existing work lacks X". A prior-art paper is judged against
    these atomic claims, and the residual gap is the set of claims no neighbor
    fully covers — computed as a set subtraction, never a string intersection.
    """
    __tablename__ = "gap_atomic_claims"

    id = Column(String, primary_key=True, default=_uuid)
    task_id = Column(String, ForeignKey("research_tasks.id"), nullable=False)
    gap_id = Column(String, ForeignKey("gap_candidates.id"), nullable=False)

    claim_index = Column(Integer, nullable=False)
    claim_text = Column(Text, nullable=False)

    created_at = Column(DateTime, default=_utcnow)

    __table_args__ = (
        Index("idx_gac_gap", "gap_id"),
        Index("idx_gac_task_gap", "task_id", "gap_id"),
    )


class NeighborClaimCoverage(Base):
    """A neighbor paper's coverage judgment for one atomic claim (Phase 3H).

    coverage ∈ {FULL, PARTIAL, NONE, UNCERTAIN}. FULL removes the claim from the
    residual set; PARTIAL marks it partially addressed (narrowing); NONE means an
    effective NPA explicitly judged it uncovered; UNCERTAIN means the judgment
    is missing/insufficient (never aggregated into NONE).
    """
    __tablename__ = "neighbor_claim_coverage"

    id = Column(String, primary_key=True, default=_uuid)
    task_id = Column(String, ForeignKey("research_tasks.id"), nullable=False)
    gap_id = Column(String, ForeignKey("gap_candidates.id"), nullable=False)
    neighbor_paper_id = Column(String, ForeignKey("papers.id"), nullable=False)
    claim_id = Column(String, ForeignKey("gap_atomic_claims.id"), nullable=False)

    coverage = Column(Text, nullable=False)  # FULL / PARTIAL / NONE / UNCERTAIN
    rationale = Column(Text)

    created_at = Column(DateTime, default=_utcnow)

    __table_args__ = (
        Index("idx_ncc_gap", "gap_id"),
        Index("idx_ncc_claim", "claim_id"),
        Index("idx_ncc_unique", "gap_id", "neighbor_paper_id", "claim_id", unique=True),
    )
