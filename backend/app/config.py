"""Application configuration loaded from environment variables."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from pydantic_settings import BaseSettings, SettingsConfigDict


# Look for .env in current dir, then parent dirs
_env_file = Path(".env")
if not _env_file.exists():
    _env_file = Path("../.env")
    if not _env_file.exists():
        _env_file = Path("../../.env")


@dataclass
class ModelConfig:
    """A concrete LLM backend configuration (OpenAI-compatible endpoint).

    base_url must be the OpenAI-compatible root; the provider appends
    '/chat/completions'. extra_body carries vendor sampling/template params
    forwarded verbatim to the backend (e.g. local Qwen: top_k,
    repetition_penalty, chat_template_kwargs.enable_thinking).
    """

    name: str
    base_url: str
    model: str
    extra_body: dict[str, Any] = field(default_factory=dict)


# Project-level model registry. The first entry is the default backend used
# by the Settings defaults below; .env values still override at runtime.
MODELS: list[ModelConfig] = [
    ModelConfig(
        name="Qwen3.5-397B-A17B",
        base_url="http://28.251.176.200:8080/openapi",
        model="Qwen3.5-397B-A17B-W8A8-P800-Functional-Agent",
        extra_body={
            "top_k": 50,
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    ),
    # Fallback backend used when the primary's inference worker is down
    # (observed: connection refused on its upstream :8021 port). Same
    # Qwen3.6-35B-A3B family, deployed on a separate host.
    ModelConfig(
        name="Qwen3.6-35B-A3B-P800-test-image",
        base_url="http://28.252.230.68:8080/openapi",
        model="Qwen3.6-35B-A3B-P800-test-image",
        extra_body={
            "top_k": 50,
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    ),
]

_DEFAULT_MODEL = MODELS[0]
_FALLBACK_MODEL = MODELS[1]


class Settings(BaseSettings):
    # LLM Provider
    llm_provider: str = "venus"
    env_venus_openapi_secret_id: str = ""
    venus_llm_proxy_url: str = _DEFAULT_MODEL.base_url
    venus_llm_model: str = _DEFAULT_MODEL.model
    # Fallback backend used when the primary endpoint fails (transport/service
    # errors only, not context/budget errors). Empty model string disables the
    # fallback so the primary behaves exactly as before.
    fallback_venus_llm_proxy_url: str = _FALLBACK_MODEL.base_url
    fallback_venus_llm_model: str = _FALLBACK_MODEL.model

    # OpenAI fallback
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o"

    # Extra request-body params forwarded to the LLM backend (OpenAI-compatible
    # backends that accept vendor params, e.g. local Qwen: top_k,
    # repetition_penalty, chat_template_kwargs). JSON object; empty = none.
    llm_extra_body: dict[str, Any] = dict(_DEFAULT_MODEL.extra_body)

    # Context window of the configured model and the output budget reserved
    # inside it. The local Qwen deployment rejects any request whose input
    # exceeds llm_context_tokens with a bare HTTP 400, and it derives the
    # allowed output length from what is left of the window: a prompt that
    # nearly fills the context silently leaves no room for the answer, which
    # arrives truncated mid-string. Reserving the output budget up front keeps
    # both failure modes out of the pipeline.
    llm_context_tokens: int = 40960
    llm_max_output_tokens: int = 4096

    # Paper source API keys / polite-pool emails
    semantic_scholar_api_key: str = ""  # 可选，免费申请: https://www.semanticscholar.org/product/api#api-key-form
    openalex_email: str = ""            # 进 OpenAlex polite pool
    openalex_api_key: str = ""          # 可选，免费注册: https://openalex.org （$1/天预算 vs 无 key $0.10/天）
    crossref_email: str = ""            # 进 Crossref polite pool (50 req/s)
    ieee_api_key: str = ""              # 需 IEEE 订购，可留空跳过
    core_api_key: str = ""              # 可留空跳过

    # Agent params
    max_rounds: int = 5
    queries_per_round: int = 5
    papers_per_source_per_query: int = 15
    high_priority_target: int = 15

    # Concurrency limits (P0-1: prevent SQLite write contention)
    max_concurrent_agents: int = 2

    # Hard wall-clock budget for one agent run. Evidence extraction dominates
    # runtime (roughly 1 min per paper with a local model), so a run with a
    # wide evidence budget can legitimately exceed one hour; making this
    # configurable avoids losing a whole run to a hard-coded ceiling. Progress
    # is committed incrementally, so a timed-out task can be resumed.
    agent_timeout_seconds: int = 3600

    # LLM budget per task (0 = unlimited). When exceeded, the runner degrades
    # gracefully to existing output (emits a landscape brief) instead of
    # running unbounded cost or hard-failing.
    max_llm_calls_per_task: int = 800
    max_llm_tokens_per_task: int = 0# 0 = no token cap; set to enforce token budget

    # O2: Targeted remediation — when a pipeline gate fails (no gap / no
    # surviving gap / no intervention / readiness), run up to this many extra
    # directed-search rounds aimed at the specific failure reason before giving
    # up and emitting a landscape brief. Set to 0 to disable remediation.
    max_remediation_attempts: int = 2
    # Total directed-search rounds allowed across ALL gates for one task —
    # a global budget so remediation cannot balloon the runtime.
    max_remediation_rounds_total: int = 3

    # RAG / PDF download
    enable_scihub: bool = False  # P0-4: Sci-Hub disabled by default for legal compliance
    # RAG full-text indexing does synchronous PDF parsing (PyMuPDF/pdfplumber)
    # which can hard-crash (native segfault, not catchable by try/except) on some
    # malformed PDFs under Windows. Set to False to skip full-text RAG indexing and
    # fall back to abstract-only grounding (gap mining relies on extracted Evidence
    # Units, not on RAG chunks, so the pipeline can still produce gaps/ideas).
    enable_rag_indexing: bool = True

    # O5a: Embedding backend — pluggable so we can use an OpenAI-compatible
    # embedding API (Venus / OpenAI) instead of local sentence-transformers,
    # which segfaults under PyTorch on Windows. Values: "api" | "local".
    # When "api", RAG full-text retrieval is re-enabled on all platforms.
    embedding_backend: str = "api"
    embedding_api_url: str = ""          # OpenAI-compatible embeddings endpoint; empty -> derive from venus_llm_proxy_url
    embedding_key: str = ""              # dedicated embedding API key (e.g. Aliyun DashScope); falls back to openai/venus if empty
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536            # dimension of the chosen embedding model
    embedding_batch_size: int = 10       # Aliyun DashScope caps embeddings batch at 20; keep <=20
    # O7: drop retrieved papers whose title+abstract cosine similarity to the
    # research topic is below this threshold, before they enter the store.
    # Set to 0 to disable prefiltering. Raised from 0.35: a run with 401 papers
    # saw 77% land as low-priority noise (only top-N ever get LLM-scored and
    # top-30 ever get evidence-extracted), so the pool was far too loose.
    search_prefilter_min_similarity: float = 0.45
    # Papers without an abstract cannot contribute evidence unless their PDF is
    # fetched (which frequently fails), so demand a stronger title-only match
    # before admitting them. Falls back to the base threshold if lower.
    search_prefilter_no_abstract_min_similarity: float = 0.55

    # Evidence extraction (P0/P1/P2/P4)
    # Max papers to extract evidence from per round. Lower for quick validation
    # runs; the historical hard-coded value was 30.
    evidence_max_papers: int = 30
    # Papers are processed in batches so that finished papers are committed
    # (and thus survive a crash/OOM) before the next batch starts. This is the
    # per-batch size AND the concurrency within a batch.
    evidence_batch_size: int = 4
    # Per-paper wall-clock timeout (seconds) for evidence extraction. Prevents a
    # single slow paper from stalling the whole round. Must be generous enough
    # for a paper's chunks to finish; with chunk-level concurrency a paper
    # typically needs a few LLM round-trips, not one per chunk serially.
    evidence_paper_timeout_s: int = 300
    # Skip chunks shorter than this many characters (likely references/boilerplate).
    evidence_min_chunk_chars: int = 200
    # Cap the number of chunks per paper sent to the LLM (the most information-
    # dense sections come first). Bounds per-paper cost and latency so a paper
    # cannot blow the timeout by having dozens of chunks.
    evidence_max_chunks_per_paper: int = 6
    # How many chunks of a single paper to extract concurrently. Turns the old
    # serial per-chunk loop (N x 15s) into ceil(N/concurrency) x 15s.
    evidence_chunk_concurrency: int = 4
    # Scoring prefilter: only send the top-N papers (by a cheap heuristic:
    # citations, recency, source authority) to the LLM for full scoring; the
    # rest are marked low-priority without an LLM call. 0 disables the prefilter
    # (score everything). This bounds the very slow LLM scoring stage.
    score_llm_top_n: int = 40
    # Background metadata enrichment re-queries S2/OpenAlex for newly found
    # papers. Without an S2 key these share the same rate-limited quota as the
    # main search loop and slow it down for little gain. Disable to skip it.
    enable_metadata_enrichment: bool = True

    # Scoring weights (P1-7: authority bumped from 0.25 to 0.30)
    score_weight_relevance: float = 0.25
    score_weight_authority: float = 0.30
    score_weight_recency: float = 0.15
    score_weight_novelty: float = 0.15
    score_weight_idea_potential: float = 0.15
    # Cross-paper score calibration: when a round scores >= this many papers,
    # rescale final scores by batch z-score to widen the (historically ~0.05)
    # priority separation. Set to 0 to disable calibration.
    score_calibration_min_batch: int = 5
    score_calibration_strength: float = 0.15  # fraction of z-spread to add

    # Rate limiting (P1-9: per-source request budget)
    # S2 无 key 限速 100 req/5min ≈ 20/min；OpenAlex 无 key $0.10/天预算，请求不宜过密
    # 有 key 时可在 .env 覆盖为更高值
    rate_limit_s2_per_min: int = 20       # 无 key 保守值；有 key 可设 5000
    rate_limit_openalex_per_min: int = 10 # 无 key 保守值；有 key 可设 100
    rate_limit_default_per_min: int = 60
    # 429 后冷却秒数：期间跳过该源不发请求
    rate_limit_cooldown_s: int = 60
    # 多 query 检索时并发数（避免瞬间打爆限速源）
    search_query_concurrency: int = 2

    # Database
    database_url: str = "sqlite:///./deep_research.db"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Authentication (P0-4): if set, POST/PUT/DELETE on /api/tasks require X-API-Key header
    api_key: str = ""

    # O4: When an API key/email is configured, the source allows much higher
    # throughput. These "effective" rates are auto-selected so users do not
    # have to hand-tune rate_limit_* values in .env after obtaining a key.
    # A user-supplied override (rate_limit_s2_per_min raised above the no-key
    # default) is always respected.
    s2_rate_with_key: int = 5000        # S2 with API key: 1 req/s sustained, bursts higher
    openalex_rate_with_key: int = 100   # OpenAlex with polite-pool email/key

    @property
    def effective_s2_rate_per_min(self) -> int:
        """S2 rate: high when a key is present, else the conservative no-key value.

        If the user explicitly raised rate_limit_s2_per_min in .env, honor it.
        """
        if self.rate_limit_s2_per_min > 20:
            return self.rate_limit_s2_per_min  # explicit user override
        if self.semantic_scholar_api_key.strip():
            return self.s2_rate_with_key
        return self.rate_limit_s2_per_min

    @property
    def constrained_retrieval_mode(self) -> bool:
        """True when high-quality source supply is likely constrained (no S2 key
        and no polite-pool emails), so gap-audit admission thresholds should be
        relaxed to avoid systematically stalling at'uncertain'."""
        if self.semantic_scholar_api_key.strip():
            return False
        # An OpenAlex/Crossref polite-pool email materially improves neighbor
        # supply; if the user provided one, don't treat retrieval as constrained.
        if self.openalex_email.strip() or self.crossref_email.strip() or self.openalex_api_key.strip():
            return False
        return True

    # Gap-audit search admission thresholds (relaxed automatically under
    # constrained_retrieval_mode to keep the no-key path usable).
    gap_admission_min_completed_queries: int = 2
    gap_admission_min_query_families: int = 2
    gap_admission_min_gap_papers: int = 3
    gap_admission_min_completed_queries_constrained: int = 1
    gap_admission_min_query_families_constrained: int = 1
    gap_admission_min_gap_papers_constrained: int = 2

    # P1-1: Search Saturation (per-RQ, three-state). Configurable heuristics —
    # defaults are STARTING POINTS, not scientific ground truth; calibrate on
    # historical end-to-end tasks. States:
    #   INSUFFICIENT_OBSERVATION / STILL_GAINING / SATURATED
    saturation_min_marginal_papers: int = 2      # high-value papers added this round
    saturation_min_marginal_evidence: int = 3    # distinct evidence-bearing papers added this round
    saturation_consecutive_rounds: int = 2       # rounds of observation before SATURATED is allowed
    saturation_gain_rate_threshold: float = 1.0  # cumulative-relative gain < this -> decaying
    saturation_recall_stability: float = 0.6     # RQ top-K recall stability for the SATURATED arm

    # P1-1: Nearest-prior-art stability + search confidence (four-state).
    npa_stability_high: float = 0.6
    npa_stability_medium: float = 0.4
    family_coverage_high: float = 0.8
    family_coverage_medium: float = 0.6
    family_stability_floor: float = 0.3         # per-family floor -> flag that family, not the whole gap

    @property
    def effective_openalex_rate_per_min(self) -> int:
        """OpenAlex rate: high when an email/key is present (polite pool)."""
        if self.rate_limit_openalex_per_min > 10:
            return self.rate_limit_openalex_per_min  # explicit user override
        if self.openalex_email.strip() or self.openalex_api_key.strip():
            return self.openalex_rate_with_key
        return self.rate_limit_openalex_per_min

    model_config = SettingsConfigDict(
        env_file=str(_env_file),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
