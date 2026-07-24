"""Pydantic schemas for API request/response and LLM structured output."""

from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field, field_validator


# === Task schemas ===

class TaskCreate(BaseModel):
    user_input: str = Field(..., min_length=2, description="研究方向描述")


class TaskOut(BaseModel):
    id: str
    user_input: str
    normalized_topic: str | None = None
    status: str
    current_round: int
    max_rounds: int
    stop_reason: str | None = None
    state_json: str | None = None
    created_at: str
    updated_at: str


class ClarifyRequest(BaseModel):
    answers: list[str] = Field(..., description="用户对澄清问题的回答")


class FeedbackRequest(BaseModel):
    content: str = ""
    need_more_research: bool = False


class IdeaSelectRequest(BaseModel):
    idea_ids: list[str] = Field(..., description="用户选择的 Idea ID 列表")


# === Paper schemas ===

class PaperOut(BaseModel):
    id: str
    title: str
    abstract: str | None = None
    authors_json: str | None = None
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    citation_count: int = 0
    sources_json: str | None = None
    final_score: float | None = None
    priority: str | None = None
    reason: str | None = None
    summary: str | None = None


class RoundOut(BaseModel):
    id: str
    round_number: int
    queries_json: str | None = None
    papers_found: int
    new_papers: int
    duplicate_rate: float | None = None
    summary: str | None = None
    knowledge_gaps_json: str | None = None
    created_at: str


# === Report schemas ===

class ReportOut(BaseModel):
    id: str
    task_id: str
    content_markdown: str
    content_json: str | None = None
    created_at: str


# === Idea schemas ===

class IdeaOut(BaseModel):
    id: str
    task_id: str
    title: str | None = None
    description: str | None = None
    motivation: str | None = None
    method_sketch: str | None = None
    expected_contribution: str | None = None
    novelty: float | None = None
    feasibility: float | None = None
    significance: float | None = None
    evidence_support: float | None = None
    differentiation: float | None = None
    experimentability: float | None = None
    potential_impact: float | None = None
    risk: float | None = None
    final_score: float | None = None
    decision: str | None = None
    related_paper_ids_json: str | None = None
    user_selected: bool = False
    idea_status: str = "active"  # P1-5: active / superseded
    created_at: str


# === Experiment schemas ===

class ExperimentOut(BaseModel):
    id: str
    task_id: str
    idea_id: str
    hypothesis: str | None = None
    dataset: str | None = None
    baselines: str | None = None
    metrics: str | None = None
    steps_markdown: str | None = None
    steps_json: str | None = None
    risks: str | None = None
    created_at: str


# === Trace schemas ===

class TraceOut(BaseModel):
    id: str
    step_name: str
    step_type: str
    round_number: int | None = None
    input_json: str | None = None
    output_json: str | None = None
    llm_tokens_used: int | None = None
    duration_ms: int | None = None
    created_at: str


# === LLM structured output schemas ===

class ClarityResult(BaseModel):
    is_clear: bool = Field(..., description="研究方向是否足够明确")
    normalized_topic: str | None = Field(None, description="标准化研究方向")
    keywords: list[str] = Field(default_factory=list, description="关键词列表")
    questions: list[str] = Field(default_factory=list, description="澄清问题列表")


# Phase 2: Evidence types (defined here for forward references)
EvidenceType = Literal[
    "problem", "method", "result", "limitation", "dataset", "metric",
    "negative_result", "future_work", "comparison"
]

SearchIntent = Literal[
    "survey", "seminal", "recent_work", "benchmark",
    "direct_neighbor", "limitation", "negative_result",
    "question_answering", "gap_falsification",
]


class GeneratedQuery(BaseModel):
    """Structured search query — LLM must explicitly bind to a Question."""
    query_text: str = Field(..., description="检索 query 文本 (3-8 words)")
    intent: SearchIntent = Field(..., description="查询意图")
    target_question_id: str = Field(..., description="目标 ResearchQuestion ID")
    expected_evidence_type: EvidenceType | None = Field(None, description="期望证据类型")


class QueryList(BaseModel):
    queries: list[GeneratedQuery] = Field(..., description="结构化 query 列表")


class PaperScore(BaseModel):
    relevance: float = Field(..., ge=0, le=1)
    authority: float = Field(..., ge=0, le=1)
    recency: float = Field(..., ge=0, le=1)
    novelty: float = Field(..., ge=0, le=1)
    idea_potential: float = Field(..., ge=0, le=1)
    reason: str = ""
    summary: str = ""
    method_extract: str = Field("", description="论文使用的具体方法/模型/算法/数据集（中文）")


class RoundSummary(BaseModel):
    summary: str = Field(..., description="本轮研究摘要")
    knowledge_gaps: list[str] = Field(default_factory=list, description="知识缺口列表")


class ResearchReport(BaseModel):
    title: str
    sections: list[dict] = Field(..., description="报告各章节")


class ResearchIdeaItem(BaseModel):
    title: str
    description: str
    motivation: str
    method_sketch: str
    expected_contribution: str
    related_paper_ids: list[str] = Field(default_factory=list, description="Related paper IDs from the provided list")
    related_paper_titles: list[str] = Field(default_factory=list, description="Related paper titles for display")


class IdeaList(BaseModel):
    ideas: list[ResearchIdeaItem]


class IdeaScore(BaseModel):
    novelty: float = Field(..., ge=0, le=1)
    feasibility: float = Field(..., ge=0, le=1)
    significance: float = Field(..., ge=0, le=1)
    evidence_support: float = Field(..., ge=0, le=1)
    differentiation: float = Field(..., ge=0, le=1)
    experimentability: float = Field(..., ge=0, le=1)
    potential_impact: float = Field(..., ge=0, le=1)
    risk: float = Field(..., ge=0, le=1)
    reason: str = ""


class ExperimentPlanSchema(BaseModel):
    hypothesis: str
    dataset: str
    baselines: str
    metrics: str
    steps: list[str]
    risks: str


class IdeaMethodExtract(BaseModel):
    """P0-3: Structured extraction of method_sketch components via LLM.
    Replaces regex-based extraction with LLM-based parsing for accuracy."""
    baselines: list[str] = Field(default_factory=list, description="Baseline method names mentioned (real, verifiable methods only)")
    datasets: list[str] = Field(default_factory=list, description="Dataset names mentioned (real, well-known datasets only)")
    metrics: list[str] = Field(default_factory=list, description="Evaluation metrics mentioned")
    model_architecture: str = Field(default="", description="Model architecture description")
    algorithm: str = Field(default="", description="Algorithm description")
    has_fake_content: bool = Field(default=False, description="True if any baseline/dataset appears fabricated or non-existent")
    fake_items: list[str] = Field(default_factory=list, description="Names that appear fabricated or non-existent")


class PaperAnalysisSchema(BaseModel):
    """Structured deep analysis of a single paper."""
    problem: str = Field(..., description="论文解决的具体问题（中文）")
    method_detail: str = Field(..., description="方法细节：架构、算法、关键创新点（中文，具体到技术层级）")
    experiment_setup: str = Field(..., description="实验设置：数据集、基线、评估指标（中文）")
    key_results: str = Field(..., description="关键结果：具体数值（中文，如'Top-10: 45%'而非'效果很好'）")
    limitations: str = Field(..., description="局限性：作者承认的或显而易见的局限（中文）")
    extendable_components: str = Field(..., description="可复用/改进的组件：哪些模块可以被复用、改进或与其他方法组合（中文）")
    source_sections: dict = Field(default_factory=dict, description="信息来源章节，如 {'method': 'Section 3', 'results': 'Table 2'}")


# === Self-feedback schemas (P0) ===

class ReportFeedback(BaseModel):
    needs_improvement: bool = Field(..., description="报告是否需要改进")
    suggestions: list[str] = Field(default_factory=list, description="具体改进建议")
    missing_content: str = Field("", description="缺失的内容描述")


class ReportOutlineSection(BaseModel):
    """One section of the report outline."""
    title: str = Field(..., description="章节标题（中文）")
    description: str = Field("", description="章节内容描述")
    paper_indices: list[int] = Field(default_factory=list, description="相关论文编号 [P1]=1")


class ReportOutline(BaseModel):
    """Full report outline for two-step generation."""
    sections: list[ReportOutlineSection] = Field(default_factory=list)


class NoveltyCheck(BaseModel):
    is_novel: bool = Field(..., description="创意是否新颖")
    similar_papers: list[str] = Field(default_factory=list, description="已有类似工作的论文标题")
    novelty_reason: str = Field("", description="新颖性判断理由（中文）")


# === Paper clustering schemas ===

class PaperCluster(BaseModel):
    cluster_name: str = Field(..., description="聚类名称（中文）")
    core_method: str = Field(..., description="核心方法/技术（中文）")
    technique_details: str = Field(..., description="具体技术细节：用什么模型/算法/数据集/架构（中文）")
    problem_addressed: str = Field(..., description="解决的问题（中文）")
    key_findings: str = Field(..., description="关键发现/结论（中文）")
    limitations: str = Field(..., description="局限性/未解决的问题（中文）")
    representative_papers: list[str] = Field(default_factory=list, description="代表论文标题")


class ClusterList(BaseModel):
    clusters: list[PaperCluster] = Field(default_factory=list)
    cross_cluster_gaps: list[str] = Field(default_factory=list, description="跨聚类的知识空白（中文）")


# === RAG schemas ===

class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    section: str = "unknown"
    paper_id: str = ""
    score: float = 0.0
    image_paths: list[str] = Field(default_factory=list)


class MethodExtract(BaseModel):
    method_extract: str = Field(..., description="从论文段落中提取的具体方法/模型/算法/数据集（中文）")


# === Idea validation schemas ===

class IdeaValidation(BaseModel):
    """Validation result for a single idea."""
    is_duplicate: bool = Field(False, description="是否与其他idea重复")
    duplicate_of: str = Field("", description="重复的idea标题")
    baseline_issues: list[str] = Field(default_factory=list, description="基线问题列表（编造/不存在的基线名）")
    metric_issues: list[str] = Field(default_factory=list, description="指标问题列表（指标与假设不匹配）")
    has_issues: bool = Field(False, description="是否存在任何问题")
    severity: float = Field(0.0, ge=0, le=1, description="问题严重程度 0-1")


class IdeaValidationList(BaseModel):
    """Validation results for all ideas."""
    validations: list[IdeaValidation] = Field(default_factory=list)


# === LLM Wiki schemas ===

class WikiAction(BaseModel):
    """A single wiki action: create or update a page."""
    op: str = Field(..., description="create or update")
    page_type: str = Field(..., description="concept, method, dataset, model, synthesis")
    title: str = Field(..., description="Page title (concise)")
    content: str = Field(..., description="Full markdown content for this page")
    paper_ids: list[str] = Field(default_factory=list, description="Paper ID prefixes referenced")
    links: list[str] = Field(default_factory=list, description="Outbound wikilink targets")
    contradictions: list[str] = Field(default_factory=list, description="Contradictions found")


class WikiActionList(BaseModel):
    """List of wiki actions from LLM."""
    actions: list[WikiAction] = Field(default_factory=list)


class WikiLintIssue(BaseModel):
    """A single issue found during wiki lint."""
    issue_type: str = Field(..., description="contradiction, orphan, stale, missing_link")
    page_title: str = Field("", description="Affected page title")
    description: str = Field("", description="Issue description")


class WikiLintResult(BaseModel):
    """Result of wiki health check."""
    issues: list[WikiLintIssue] = Field(default_factory=list)


# === Phase 0: New task statuses and PhaseRun schemas ===

# New legitimate task statuses (in addition to existing ones)
NEW_TASK_STATUSES = [
    "insufficient_evidence",    # no credible ideas found after retries
    "more_research_required",   # coverage gaps remain, user should add more direction
    "auditing_gaps",             # adversarial gap audit in progress
    "checking_feasibility",      # feasibility gates in progress
    "synthesizing_ideas",        # idea synthesis from surviving gaps
    "judging_ideas",             # independent idea judgment in progress
]

# Legitimate idea decisions (in addition to go/revise/reject)
NEW_IDEA_DECISIONS = [
    "insufficient_evidence",     # cannot determine due to lack of evidence
]


class PhaseRunOut(BaseModel):
    """Phase execution record for API responses."""
    id: str
    task_id: str
    phase_name: str
    status: str
    attempt_count: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    input_version: str | None = None
    output_version: str | None = None
    error_message: str | None = None
    round_number: int | None = None
    output_summary: str | None = None
    created_at: str
    updated_at: str


# === Phase 1: Research Contract + Question Decomposition ===

class ResourceConstraintsSchema(BaseModel):
    """Resource constraints for the research."""
    gpu_available: bool | None = None
    max_gpu_hours: float | None = None
    max_api_budget: float | None = None
    max_runtime_minutes: int | None = None
    allow_large_benchmark: bool = True
    allow_model_training: bool = True


class ResearchContractSchema(BaseModel):
    """Structured research contract for LLM output."""
    topic: str = Field(..., description="研究主题（英文，用于检索）")
    target_problem: str = Field("", description="目标问题（中文）")
    target_setting: str = Field("", description="目标场景（中文）")
    desired_output: Literal["method", "system", "benchmark", "empirical_analysis"] = Field(
        "method", description="期望输出类型")
    novelty_bar: Literal["course_project", "master_thesis", "conference"] = Field(
        "conference", description="创新门槛")
    preferred_directions: list[str] = Field(default_factory=list, description="偏好方向")
    excluded_directions: list[str] = Field(default_factory=list, description="排除方向")
    gpu_available: bool | None = None
    max_gpu_hours: float | None = None
    max_api_budget: float | None = None
    max_runtime_minutes: int | None = None
    allow_large_benchmark: bool = True
    allow_model_training: bool = True
    experiment_preferences: dict = Field(default_factory=dict, description="实验偏好")
    key_terms: list[str] = Field(default_factory=list, description="关键术语（英文，用于检索）")
    time_scope_start: int | None = None
    time_scope_end: int | None = None
    confidence: float = Field(0.5, ge=0, le=1, description="置信度")


class ResearchAxisSchema(BaseModel):
    """A research axis for decomposition."""
    axis_name: str = Field(..., description="轴名称")
    values: list[str] = Field(default_factory=list, description="该轴上的可选值")


class ResearchQuestionSchema(BaseModel):
    """A single research question."""
    question: str = Field(..., min_length=5, description="研究问题（具体、可检索、可回答）")
    question_type: Literal["problem", "method", "evaluation", "dataset", "resource", "failure", "application"] = Field(
        ..., description="问题类型")
    importance: float = Field(0.5, ge=0, le=1, description="重要性")
    searchability: float = Field(0.5, ge=0, le=1, description="可检索性")
    axis_name: str = Field("", description="所属研究轴")


class ResearchDecompositionSchema(BaseModel):
    """Output of research space decomposition."""
    axes: list[ResearchAxisSchema] = Field(default_factory=list, description="研究轴")
    questions: list[ResearchQuestionSchema] = Field(..., description="研究问题列表（5-12个）")

    @field_validator("questions")
    @classmethod
    def validate_questions(cls, v):
        if len(v) < 5:
            raise ValueError(f"Must generate at least 5 questions, got {len(v)}")
        if len(v) > 12:
            raise ValueError(f"Must generate at most 12 questions, got {len(v)}")
        # Check at least 3 different axes
        axes = set(q.axis_name for q in v if q.axis_name)
        if len(axes) < 3:
            raise ValueError(f"Must cover at least 3 different axes, got {len(axes)}: {axes}")
        # Check no empty questions
        for q in v:
            if not q.question or not q.question.strip():
                raise ValueError("Question text must not be empty")
        # Check no exact duplicates
        texts = [q.question.strip().lower() for q in v]
        if len(texts) != len(set(texts)):
            raise ValueError("Questions must not be exact duplicates")
        return v


class ContractOut(BaseModel):
    """Research contract for API response — returns structured fields, not JSON strings."""
    id: str
    task_id: str
    topic: str
    target_problem: str | None = None
    target_setting: str | None = None
    desired_output: str | None = None
    novelty_bar: str | None = None
    preferred_directions: list[str] = Field(default_factory=list)
    excluded_directions: list[str] = Field(default_factory=list)
    gpu_available: bool | None = None
    max_gpu_hours: float | None = None
    max_api_budget: float | None = None
    max_runtime_minutes: int | None = None
    allow_large_benchmark: bool = True
    allow_model_training: bool = True
    key_terms: list[str] = Field(default_factory=list)
    experiment_preferences: dict = Field(default_factory=dict)
    time_scope_start: int | None = None
    time_scope_end: int | None = None
    status: str = "active"
    confidence: float = 0.5
    version: int = 1
    input_hash: str | None = None
    created_at: str
    updated_at: str


class ResearchQuestionOut(BaseModel):
    """Research question for API response."""
    id: str
    task_id: str
    contract_id: str | None = None
    question: str
    question_type: str
    importance: float = 0.5
    searchability: float = 0.5
    status: str = "open"
    axis_name: str | None = None
    version: int = 1
    created_at: str


# === Phase 2: Evidence + Coverage schemas ===
# EvidenceType is already defined above (before GeneratedQuery)

VerificationStatus = Literal[
    "unverified", "verified", "conflicted", "rejected", "abstract_only"
]

PaperRoleType = Literal[
    "survey", "seminal", "direct_neighbor", "benchmark", "method",
    "negative_result", "limitation_evidence", "application"
]

RelationType = Literal[
    "supports", "contradicts", "partially_answers", "background"
]


class EvidenceUnitOut(BaseModel):
    """Evidence unit for API response."""
    id: str
    task_id: str
    paper_id: str
    evidence_type: str
    normalized_claim: str
    original_span: str | None = None
    section: str | None = None
    page_number: int | None = None
    dataset_name: str | None = None
    metric_name: str | None = None
    result_value: str | None = None
    extraction_method: str = "llm"
    extraction_confidence: float = 0.5
    verification_status: str = "unverified"
    created_at: str


class CoverageRecordOut(BaseModel):
    """Coverage record for API response."""
    id: str
    task_id: str
    question_id: str
    coverage_score: float = 0.0
    confidence: float = 0.0
    supporting_evidence_count: int = 0
    contradicting_evidence_count: int = 0
    direct_neighbor_count: int = 0
    unresolved_aspects: list[str] = Field(default_factory=list)
    unavailable_reason: str | None = None
    updated_at: str
    created_at: str


class PaperRoleOut(BaseModel):
    """Paper role for API response."""
    id: str
    task_id: str
    paper_id: str
    role: str
    confidence: float = 0.5
    reason: str | None = None
    created_at: str


class EvidenceExtractionSchema(BaseModel):
    """LLM output schema for evidence extraction from a paper chunk."""
    evidence_type: EvidenceType
    normalized_claim: str = Field(..., min_length=5, description="归一化的claim（中文）")
    original_span: str = Field("", description="原文片段")
    dataset_name: str | None = None
    metric_name: str | None = None
    result_value: str | None = None
    conditions: dict = Field(default_factory=dict, description="条件信息")


class EvidenceExtractionList(BaseModel):
    """List of evidence units extracted from a paper chunk."""
    evidence_units: list[EvidenceExtractionSchema] = Field(default_factory=list)


class PaperRoleClassificationSchema(BaseModel):
    """LLM output for paper role classification."""
    roles: list[PaperRoleType] = Field(default_factory=list)
    confidence: float = Field(0.5, ge=0, le=1)
    reason: str = ""


# === Phase 3A: Gap Control Plane schemas ===

GapType = Literal[
    "coverage_gap", "contradiction", "missing_method",
    "missing_dataset", "missing_evaluation", "boundary_gap",
]

GapStatus = Literal[
    "candidate", "auditing", "audited", "surviving", "rejected", "superseded",
]

GapAuditResult = Literal[
    "pending", "confirmed", "partially_closed", "closed", "uncertain",
]

GapRecommendedAction = Literal[
    "continue", "narrow", "more_search", "reject",
]

GapProvenanceStatus = Literal[
    "complete", "partial", "invalid",
]


class GapCandidateOut(BaseModel):
    """Gap candidate for API response."""
    id: str
    task_id: str
    contract_id: str | None = None
    gap_type: str
    description: str
    # Phase 3A Closure: structured fields
    target_setting: str | None = None
    observed_problem: str | None = None
    existing_coverage: str | None = None
    missing_capability: str | None = None
    claimed_delta: str | None = None
    testable_hypothesis: str | None = None
    falsification_condition: str | None = None
    provenance_status: str = "partial"
    question_ids: list[str] = Field(default_factory=list)
    # DEPRECATED snapshots — use gap_evidence_links as authoritative source
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    mining_round: int = 0
    novelty_score: float | None = None
    feasibility_score: float | None = None
    significance_score: float | None = None
    risk_score: float | None = None
    status: str = "candidate"
    version: int = 1
    created_at: str
    updated_at: str


class GapCandidateSchema(BaseModel):
    """LLM output schema for gap mining (Phase 3B — defined now for forward compat).

    Phase 3A Closure: Structured falsifiable contract — each gap must have
    a testable hypothesis and falsification condition, not just a description.
    """
    gap_type: GapType = Field(..., description="Gap 类型")
    description: str = Field(..., min_length=10, description="Gap 描述（中文，人类可读摘要）")
    target_setting: str = Field(..., description="目标场景（中文）")
    observed_problem: str = Field(..., description="在证据中观察到的具体问题（中文）")
    existing_coverage: str = Field(..., description="现有论文已覆盖的内容（中文）")
    missing_capability: str = Field(..., description="缺失的具体能力/方法/数据集（中文）")
    claimed_delta: str = Field(..., description="与现有工作的声称差异（中文）")
    testable_hypothesis: str = Field(..., description="可检验假设（中文，可被实验验证或证伪）")
    falsification_condition: str = Field(..., description="证伪条件（中文，什么证据能证明此 Gap 不成立）")
    question_ids: list[str] = Field(..., min_length=1, description="关联的 ResearchQuestion ID 列表（至少 1 个）")
    supporting_evidence_ids: list[str] = Field(..., min_length=1, description="支撑此 Gap 的 Evidence ID 列表（至少 1 个）")
    contradicting_evidence_ids: list[str] = Field(default_factory=list, description="产生矛盾的 Evidence ID 列表")
    novelty_score: float = Field(0.5, ge=0, le=1, description="新颖性")
    feasibility_score: float = Field(0.5, ge=0, le=1, description="可行性")
    significance_score: float = Field(0.5, ge=0, le=1, description="重要性")


class GapCandidateList(BaseModel):
    """LLM output for gap mining — list of gap candidates."""
    gaps: list[GapCandidateSchema] = Field(default_factory=list)


class GapAuditOut(BaseModel):
    """Gap audit for API response."""
    id: str
    gap_id: str
    task_id: str
    audit_result: str = "pending"
    nearest_neighbor_summary: str | None = None
    differentiation_summary: str | None = None
    neighbor_paper_ids: list[str] = Field(default_factory=list)
    # Phase 3A Closure: Decision fields
    evidence_for_gap: list[str] = Field(default_factory=list)
    evidence_against_gap: list[str] = Field(default_factory=list)
    remaining_delta: str | None = None
    novelty_confidence: float | None = None
    audit_confidence: float | None = None
    recommended_action: str = "continue"
    rejection_reason: str | None = None
    audit_round: int = 0
    created_at: str


class NeighborComparisonOut(BaseModel):
    """Neighbor comparison for API response."""
    id: str
    gap_id: str
    paper_id: str
    task_id: str
    similarity_score: float = 0.0
    # DEPRECATED
    shared_aspects: list[str] = Field(default_factory=list)
    differentiating_aspects: list[str] = Field(default_factory=list)
    overlap_risk: float = 0.0
    # Phase 3A Closure: Structured fields
    shared_problem: str | None = None
    shared_mechanism: str | None = None
    shared_evaluation: str | None = None
    covered_claims: list[str] = Field(default_factory=list)
    uncovered_claims: list[str] = Field(default_factory=list)
    overlap_ratio: float = 0.0
    created_at: str


class GapEvidenceLinkOut(BaseModel):
    """Gap-evidence link for API response."""
    id: str
    gap_id: str
    evidence_id: str
    relation_type: str = "suggests"
    relevance_score: float = 0.5
    created_at: str
