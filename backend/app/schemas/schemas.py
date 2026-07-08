"""Pydantic schemas for API request/response and LLM structured output."""

from pydantic import BaseModel, Field


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


class GeneratedQuery(BaseModel):
    query: str
    source_hint: str | None = None


class QueryList(BaseModel):
    queries: list[str] = Field(..., description="本轮检索 query 列表")


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


# === Self-feedback schemas (P0) ===

class ReportFeedback(BaseModel):
    needs_improvement: bool = Field(..., description="报告是否需要改进")
    suggestions: list[str] = Field(default_factory=list, description="具体改进建议")
    missing_content: str = Field("", description="缺失的内容描述")


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
