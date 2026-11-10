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
    # DeepSeek-V4.1-Flash on the infer-proxy host. Verified 2026-10-28:
    # /chat/completions answers without an Authorization header, accepts
    # response_format=json_object, tools and the vendor params below, and
    # reports max_model_len=557056. enable_thinking=False is honored (the
    # `reasoning` field comes back empty and completion_tokens drops from 20
    # to 2 on a trivial prompt), which keeps the reasoning chain from eating
    # the output budget the pipeline reserved for JSON.
    ModelConfig(
        name="DeepSeek-V4.1-Flash-1017-W8A8-P800-1028test",
        base_url="http://28.251.176.246:8080/openapi",
        model="DeepSeek-V4.1-Flash-1017-W8A8-P800-1028test",
        extra_body={
            "top_k": 50,
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    ),
    # Fallback backend used when the primary's inference worker is down.
    # NOTE (2026-10-28): the previous Qwen hosts are all unreachable right now
    # (28.252.232.9 -> 502, 28.252.232.11 -> 404, 28.251.176.200 -> 502), so
    # .env disables failover by leaving FALLBACK_VENUS_LLM_MODEL empty. This
    # entry is kept as the default only for environments that still serve it.
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
    # Per-provider circuit breaker for transient transport/service failures.
    llm_circuit_failure_threshold: int = 2
    llm_circuit_cooldown_seconds: float = 30.0
    llm_circuit_max_cooldown_seconds: float = 300.0

    # OpenAI fallback
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o"

    # Extra request-body params forwarded to the LLM backend (OpenAI-compatible
    # backends that accept vendor params, e.g. local Qwen: top_k,
    # repetition_penalty, chat_template_kwargs). JSON object; empty = none.
    llm_extra_body: dict[str, Any] = dict(_DEFAULT_MODEL.extra_body)

    # Context window of the configured model and the output budget reserved
    # inside it. The backend rejects any request whose input exceeds the window
    # with a bare HTTP 400, and it derives the allowed output length from what
    # is left of the window: a prompt that nearly fills the context silently
    # leaves no room for the answer, which arrives truncated mid-string.
    # Reserving the output budget up front keeps both failure modes out of the
    # pipeline.
    #
    # The DeepSeek-V4.1-Flash deployment reports max_model_len=557056, but this
    # is deliberately declared far below that: under-declaring only makes the
    # admission guard more conservative, while over-declaring lets doomed
    # requests through. It also bounds `_output_allowance`, which asks for
    # "context - prompt upper bound" output tokens — declaring the full window
    # would request half a million output tokens for a short prompt.
    llm_context_tokens: int = 131072
    llm_max_output_tokens: int = 8192

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
    # Per-canonical-gap-family remediation budget (evidence-sensitive
    # idempotency, 0027). A remediation round answers "how much more retrieval
    # does THIS gap's novelty need?", so the budget is scoped to the canonical
    # gap family that consumed it — not the whole task. A new gap family that
    # appears after a mining revision must not inherit the spent budget of an
    # old family (which previously let an old gap eat a new gap's only chance,
    # see task 08005641). Narrowed versions (v1 -> v2) share the family budget.
    max_remediation_attempts_per_gap: int = 2
    # Total deterministic gap-narrowing passes in the opportunity pipeline.
    # Narrowing is cheaper than remediation but still re-runs adversarial search;
    # keep a task-level cap so multiple gap families cannot create an unbounded
    # sequence of re-audits.
    max_narrowing_passes_total: int = 3

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

    # PDF prewarm: fetch the round's PDFs *before* extraction starts, rather than
    # discovering availability paper-by-paper inside the extraction loop.
    #
    # Rationale (measured 2026-09-18): retrieval failures are transient
    # (arXiv rate-limit windows) rather than properties of the papers — of arXiv
    # papers that had never been cached, a plain retry recovered 40/40 and then
    # 30/30. Downloading inside the extraction loop therefore couples a
    # recoverable network condition to evidence loss, and makes the round's cost
    # unpredictable. Prewarming separates "can we get the corpus" from "what does
    # the corpus say", so a rate-limit window costs a bounded delay instead of a
    # silently truncated corpus.
    enable_pdf_prewarm: bool = True
    # Concurrency for prewarm downloads. Deliberately below the extraction-time
    # concurrency (3): prewarm is the only thing running at that moment, and
    # arXiv throttles aggressively above ~3 req/s. Raising this recovers nothing
    # and triggers the very 429s the retry logic then has to absorb.
    pdf_prewarm_concurrency: int = 3
    # Padding added on top of the retry budget when sizing the prewarm timeout,
    # so retries and slow large PDFs are not cut off by the batch timeout.
    pdf_prewarm_timeout_padding_s: int = 60

    # Extraction concurrency: how many papers are in flight simultaneously.
    #
    # This is the knob that makes full-corpus coverage affordable. Previously
    # concurrency was pinned to `evidence_batch_size` *and* applied per batch with
    # a barrier between batches, so a round of N papers took ceil(N/batch) rounds
    # of the slowest paper. A single shared pool removes the barrier, which is why
    # the per-round cap on how many papers get read can be dropped without the
    # round time exploding.
    #
    # Each in-flight paper = 1 PDF in memory + `evidence_chunk_concurrency`
    # concurrent LLM calls. Raising this multiplies both.
    evidence_extract_concurrency: int = 4
    # Scoring prefilter: only send the top-N papers (by a cheap heuristic:
    # citations, recency, source authority) to the LLM for full scoring; the
    # rest are marked low-priority without an LLM call. 0 disables the prefilter
    # (score everything). This bounds the very slow LLM scoring stage.
    score_llm_top_n: int = 40
    # Metadata enrichment re-queries S2/OpenAlex for newly found papers. It is
    # awaited within the search phase and uses the phase session, so it does not
    # create a concurrent SQLite writer. Disable to skip its network latency.
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
    family_instability_more_search_budget: int = 2  # max family-instability-triggered more_search rounds
    variant_invariance_embedding_threshold: float = 0.7  # variant vs canonical intent cosine floor (cheap guardrail)

    # 0027: Gap-specific prior-art screening (false-novelty audit, fd688ba6).
    # TaskPaper.final_score measures task/RQ relevance; the NPA audit needs
    # "how close is this paper to THIS gap". Audit-recalled papers were stored
    # with final_score=NULL and then out-ranked by broad surveys in neighbor
    # selection, so direct prior art never reached the NPA pool. These weights
    # aggregate the qualitative overlap judgments into gap_relevance — kept as
    # configurable heuristics, not scientific truth, pending calibration on
    # historical tasks.
    gap_relevance_claim_weight: float = 0.4
    gap_relevance_problem_weight: float = 0.35
    gap_relevance_evaluation_weight: float = 0.25
    # Max audit-recalled papers to score per gap in the cheap screen; the deep
    # NPA audit then runs on the Top-M only (two-stage, keeps 200+ pools cheap).
    gap_relevance_screen_top_m: int = 20
    gap_relevance_scoring_version: str = "gap-rel-v1"
    # P2 (2026-10-29): a neighbour below this gap-relevance floor cannot CERTIFY
    # novelty. Its FULL/PARTIAL/NONE judgment is treated as UNCERTAIN
    # (insufficient evidence) instead of a decided verdict.
    #
    # Measured on the v7 replay pilot (2026-10-28): all 33 neighbours behind the
    # 11 mechanical-novelty overrides screened a median gap_relevance of 0.17,
    # 17/33 were below 0.3, and the three neighbours behind gap 4e409776 were
    # off-domain (gravitational waves, general relativity; relevance 0.00). Each
    # was judged NONE — correctly, they do cover nothing about LoRA interference
    # — so decided_ratio reached 1.0 and mechanical novelty 1.0: "three
    # irrelevant papers were silent" was recorded as the strongest possible
    # novelty evidence. The screen score already existed; only the floor was
    # missing.
    #
    # Calibrated on that pilot before switching on (all 11 overrides re-scored
    # by the production function): the neighbour relevance pool splits into
    # "off-domain" (<= 0.17) and "relevant" (>= 0.38) with nothing in between,
    # so any floor in 0.2..0.3 gives the identical outcome — 8/11 overrides
    # survive, and the 3 that drop out are the ones whose entire pool scored
    # <= 0.17. A 0.4 floor would start discarding relevant neighbours (0.38)
    # and leave only 4/11, so 0.3 is the widest setting that keeps real evidence.
    #
    # Set to 0.0 to restore legacy behaviour; the per-audit trace always records
    # neighbour relevance diagnostics either way.
    gap_relevance_min_neighbor: float = 0.3
    # How many relevance-qualified neighbours the mechanical novelty override
    # needs before it may certify novelty at all (2026-10-29).
    #
    # With the 0.3 floor applied to the v7 replay pilot, 3 of the 8 surviving
    # override gaps rested on exactly ONE qualified neighbour — and two of
    # those scored mechanical_novelty 1.00. One neighbour judged NONE is one
    # paper that did not happen to mention the claim; it is not evidence that
    # nobody has. An independent literature review of those 8 gaps (single
    # reviewer, one search round, abstract-level) rated exactly those
    # one-neighbour gaps as its least confident partial-coverage findings.
    #
    # Two independent routes — the neighbour-count structure and the external
    # literature check — fail the same three gaps, so the floor alone is not
    # sufficient and this count is the second half of the fix.
    #
    # COST, stated plainly: this suppresses discovery in fields where two
    # relevant neighbours are genuinely hard to retrieve. Set to 1 to keep the
    # floor without the count requirement; the final calibration belongs on a
    # human-labelled set, not on this 6-topic pilot.
    mech_novelty_min_qualified_neighbors: int = 2
    variant_regenerate_budget: int = 2          # per-variant regenerate attempts before dropping

    # Per-Gap audit guardrails. These are operational heuristics: they cap
    # external-search cost without changing the scientific verdict thresholds.
    gap_audit_max_queries: int = 12
    gap_audit_max_candidate_papers: int = 20
    gap_audit_timeout_seconds: int = 600
    # v15 渐进式对抗审计（2026-09-02，用户裁决「渐进式对抗审计 + 增量重审」）。
    # 默认 off 保持 v14 语义——production_e2e 24 题正式跑须同版本完成。
    # 开启后：P0-1 重复轮/零新邻的停止判断提前到近邻全文抽取之前；
    # P0-2 近邻全文按 wave_size 一波波抽取、凑足 audit_ceiling_min_fulltext_neighbors
    # 篇 verified 即停（ceiling 本来只要求 1 篇，v14 却最多一次跑 5 篇）。
    # 激活时 GAP_SEARCH_POLICY_VERSION 自动升 v15；须跑 test_research_validity_regression。
    gap_audit_progressive: bool = False
    audit_neighbor_evidence_wave_size: int = 2
    # v16 Budgeted Falsification（2026-09-03 用户裁决）：放弃「穷尽式查新」作为必经
    # 门禁，改为固定预算内尽力找 killer prior art；预算内找不到 → survived_budgeted_
    # audit（措辞：在当前检索预算与审计范围内未发现直接覆盖该 claimed delta 的
    # prior art），绝不声称 confirmed novel。预算硬上限：
    #   1 次主审计/gap + 1 次 killer search，无 re-audit/remediation；
    #   4 查询（砍 benchmark 族）× 3 源（S2/OpenAlex/arXiv）→ Top-10 摘要筛选 →
    #   Top-3 邻 claim 判断 → 0-2 篇全文（疑似 killer 优先，按 relevance_score）→
    #   killer search ≤3 查询 → STOP；Top-2 gap 竞争（max_gaps_to_deep_audit）；
    #   超时（GAP_AUDIT_TIMEOUT_SECONDS=600）→ insufficient_evidence，不补救。
    gap_audit_budgeted: bool = False
    budgeted_audit_max_queries: int = 4
    budgeted_audit_max_candidate_papers: int = 10
    budgeted_audit_neighbors: int = 3
    budgeted_audit_fulltext_max_papers: int = 2
    budgeted_killer_search_max_queries: int = 3
    budgeted_audit_sources: str = "semantic_scholar,openalex,arxiv"

    # --- Neighbour-budget experiment (2026-11-03) ---------------------------
    # EXPERIMENT ARM SWITCH — leave None outside the experiment. 0 disables.
    #
    # Why: the audit judges novelty against `budgeted_audit_neighbors` papers
    # (production: 3). Measured on the fixed-budget pilot's 37 reviewed gaps,
    # the corpus-recall ordering puts the reviewer's own killer paper at
    # (n=71 in-corpus killers):
    #
    #     k=1  14.1%   k=2  28.2%   k=3  36.6%   k=5  53.5%
    #     k=6  56.3%   k=8  60.6%   k=10 63.4%   k=15 69.0%   k=20 76.1%
    #
    # So at the production budget the audit cannot see roughly two thirds of
    # the papers that would falsify it. The biggest single step is k=3 -> k=5
    # (+12 killers, +16.9pp); k=8 captures 60.6% and is where the curve bends
    # hard (k=8 -> k=10 buys only +2.8pp).
    #
    # Cost is NOT linear in this knob and that matters for the comparison:
    # the audit decision is ONE llm.chat_json call with every neighbour in the
    # prompt (`_AUDIT_USER.neighbors=_format_neighbors(db, neighbors)`), so a
    # larger budget lengthens one prompt instead of adding calls or timeout
    # exposure. Full-text extraction is separately capped by
    # `budgeted_audit_fulltext_max_papers` and does not scale with neighbours.
    #
    # When set, overrides `budgeted_audit_neighbors` for this process and stamps
    # audit traces with the value so the two arms stay separable.
    audit_neighbors_override: int = 0

    # Noise guard for the larger budget: NOT a new mechanism.
    #
    # Extending the neighbour list reaches further down the relevance ordering,
    # which raises an obvious objection — "more neighbours just add low-quality
    # papers, and a PARTIAL judgment from a junk paper is enough to make a real
    # gap look covered". The existing `gap_relevance_min_neighbor=0.3` already
    # answers this: in `_mechanical_novelty_score`, any judgment from a
    # neighbour below the floor is demoted to UNCERTAIN (never coverage, never
    # novelty). So the guard is ALREADY in force and must stay identical in
    # both arms, otherwise the comparison would move two variables at once
    # (count AND relevance policy).
    #
    # What the larger budget does need is VISIBILITY: at k=3 the demotion count
    # is almost always 0, at k=8 it can be non-zero and could silently cap the
    # arm's novelty. `audit_neighbors_override` traces therefore always carry
    # the effective neighbour count plus the below-floor count, so "the arm did
    # worse" can be separated from "the arm's extra neighbours were demoted".
    audit_neighbor_noise_guard_documented: bool = True

    # --- Multi-service experiment isolation (2026-11-03) --------------------
    # `recover_interrupted_tasks()` runs on EVERY service startup and resets
    # every task in an interrupted status (`searching`, `extracting_evidence`,
    # ...) back to `pending` — it has no notion of WHICH service owns which
    # task. In a two-service experiment (arms on 8001/8002 sharing one DB) that
    # means restarting arm C marks arm D's in-flight task as interrupted and
    # flips its running phase rows to failed. The damage is silent: the task
    # simply stops advancing, or resumes later under whatever arm happens to
    # pick it up — i.e. an arm's result can be produced by the other arm's
    # configuration.
    #
    # Set to false for experiment services. Recovery is a convenience for
    # interactive use; during a controlled comparison the protocol's
    # interruption rule (pause, fix, re-run the pair) must take precedence over
    # automatic resume. Leave true for normal single-service operation.
    recover_tasks_on_startup: bool = True
    # 2 (2026-10-30, probe df744606): Top-1 clipped the 7-lens fan-out to a
    # single audited gap per cycle — 21 accepted candidates saw 3 audits and
    # ALL ideas hung off one surviving gap (100% single-gap concentration).
    # A full audit costs ~250s against a 119-minute run, so the clip's savings
    # were immaterial next to the breadth it destroyed. Top-2 keeps both
    # audits complete before the winner is chosen (no "first through the
    # gate wins"), same as the non-budgeted competition default.
    max_gaps_to_deep_audit: int = 2

    # Starvation control (probe 0ae1a146, 2026-10-30): 16 of 20 gaps stayed at
    # `candidate` because each competition re-ranked the same stable pool and
    # the same losers lost again. Two mechanisms:
    #   1. a small deferred-rounds bonus in the cheap rank (capped +0.09) so a
    #      near-tied candidate eventually gets its turn;
    #   2. retirement after this many competitions without selection, so
    #      "never audited" becomes an explicit terminal state instead of an
    #      ever-growing backlog of auditable-looking candidates. 0 disables
    #      retirement (bonus stays active).
    gap_starvation_deferral_limit: int = 3

    # Incremental coverage matching (2026-10-30, probe df744606): coverage
    # re-matched the whole evidence pool against every question each round
    # (measured 246 -> 398 -> 509 -> 667s, monotonic with pool size), although
    # (question, evidence) verdicts are stable — evidence rows are immutable
    # and question text never changes after decompose. When enabled, only
    # evidence created after the previous coverage snapshot goes to the LLM;
    # already-judged pairs are replayed from their links. Turn off to restore
    # full re-matching (e.g. after changing the matching prompt semantics).
    coverage_incremental_matching: bool = True
    # 低证据补搜（2026-09-04，用户选项 A：目的是产出可用 Idea，证据不足先补证据）。
    # v16 budgeted 快进快出：低证据主题（冷门/检索面窄）带着很薄的证据池进入
    # mining，只产出 1 个薄 gap，被审计拒绝后直接 abstain（task 0eca528a：
    # 30 证据 / 1 gap / 15min 弃权）。开启后：readiness 通过、进入 mining 之前，
    # 若证据池低于 min_evidence_units_for_idea，先做一次有界定向补搜
    # （low_evidence_topup playbook，limitation 意图）再继续。
    # 默认 off 保持 v16 语义；每任务至多一次，受 remediation 全局预算约束。
    low_evidence_remediation_enabled: bool = False
    min_evidence_units_for_idea: int = 100
    # 审计判决补救（2026-09-08，用户批准）：budgeted 审计 verdict=more_search 时，
    # gap 直接落入 inconclusive 弃权——审计的 killer 预算只服务 confirmed 判决，
    # uncertain gap 一次定向补搜的机会都没有（task 407b0359 watermarking 实测：
    # 差的就是一次 bigram 边界验证）。开启后：budgeted 无存活时，对 auditing
    # 状态的 gap 先做一次 claim 锚定的定向补搜（复用 O2 机制），再做一轮
    # 重审（remediation_round 变化使 audit input_version 更新，PhaseRun 真正
    # 重判而非重放），仍无存活才弃权。每任务至多一次，受全局补救预算约束。
    audit_verdict_remediation_enabled: bool = False
    # 低新颖性暂缓弃权（2026-09-08，A+B 方案）：budgeted 审计 confirmed 但
    # novelty_confidence <= 0.4 时，不再降级 uncertain（budgeted 下无补搜入口，
    # 降级=直接弃权=假阴性工厂），而是让 confirmed 原样 survive——下游
    # intervention 的 novelty gate（<0.3 FAIL / 0.3..0.5 WARN tier-B）自动把
    # 这类 gap 的 Idea 限制在 conditional_review，四级体系接管分层。同时记录
    # 机械主张覆盖度（claimed_delta 内容词 vs 近邻 title+abstract 的覆盖率，
    # 仅分析用不参与门禁——替代 LLM 自报分数作为判据的实验数据）。
    audit_low_novelty_provisional_survive: bool = False
    audit_low_novelty_coverage_floor: float = 0.25
    # 机械新颖性门禁（2026-09-08，选项 B）：自报 novelty_confidence 波动大
    # （同主题同证据量跨轮 0.3..0.85），≤0.4 一刀切把"审计搜索弱"当成"不新颖"。
    # 开启后：低自报分分支先聚合持久化的 claim 覆盖矩阵（LLM 结构化事实，
    # 每邻居×每原子主张 FULL/PARTIAL/NONE/UNCERTAIN），gated_novelty =
    # 未覆盖主张比例 × 已决定主张比例——搜索不弱且明确未覆盖（decided 高、
    # uncovered 高）才背书 survive 并单调上修自报分；全 UNCERTAIN（搜索弱）
    # 的 gated=0，不会假阳性。矩阵缺失时 fail-open 回退旧行为。默认关。
    audit_mechanical_novelty_gate_enabled: bool = False
    # Idea 多样性门禁（2026-09-08）：同一 surviving gap 的多个 intervention idea
    # 常为同一机制的词面变体（watermarking 实测：5 idea 中 4 个 entropy 变体，
    # 高度重叠不构成独立贡献）。embedding 相似度 >= idea_diversity_sim_threshold
    # 的近重复 idea 按 final_score 保留代表，其余降 conditional_review 并标
    # DUPLICATE_VARIANT_OF（不删除，人工可复核）。默认开（确定性降档而非硬拒）。
    idea_diversity_gate_enabled: bool = True
    idea_diversity_sim_threshold: float = 0.80
    # Evidence-funnel repair (E2E 2026-08-26: audit-recalled papers stayed
    # priority=NULL and never entered evidence extraction, so 94/94 round-3
    # papers were invisible downstream and NO_FULLTEXT_EVIDENCE was structural).
    # Score audit-recalled papers like main-round papers, and extract full-text
    # evidence for the selected NPA neighbors before the evidence delta is
    # computed, bounded to keep the per-gap audit timeout meaningful.
    audit_score_retrieved_papers: bool = True
    audit_neighbor_evidence_extraction: bool = True
    audit_neighbor_evidence_max_papers: int = 5

    # P0-B (run8 review, task 03e0f59a): audit verdict ceiling — retrieval
    # diagnostics recorded since P1.2 must CONSTRAIN the verdict, not merely
    # document it. Run8 stamped confirmed/0.9 while its own diagnostics said
    # search_confidence=INSUFFICIENT_OBSERVATION, zero neighbors carried
    # verified full-text evidence, and the killer search recalled only 2
    # off-topic papers (medical classification / video OCR) using the gap's
    # self-invented terminology. A confirmed verdict requires an audit basis
    # that could plausibly have found prior work.
    # >= this many neighbors must carry verified full-text evidence.
    audit_ceiling_min_fulltext_neighbors: int = 1
    # Median family stability@20 below this + no cross-round validation
    # (first audit) = unvalidated unstable retrieval.
    audit_ceiling_stability_floor: float = 0.15
    # A killer search that recalled fewer fresh papers than this tested
    # nothing (its terms were too narrow to find ANY paper).
    audit_ceiling_killer_min_recall: int = 3
    # After the more_search budget is exhausted the gap may survive, but its
    # novelty_confidence is capped here (weak survive, explicitly recorded).
    audit_ceiling_novelty_cap: float = 0.6
    # Max queries executed for the (terminology-expanded) killer search.
    killer_search_max_queries: int = 6

    # P0 (run10 review): construct-declared control implementation check.
    # A construct_identification that declares confounders AND names a
    # control/oracle must actually implement it in the experiment design;
    # otherwise the schema-complete string is compliance theater (run10
    # review: "control for model capability" with no matched arm anywhere).
    # The check is token-overlap first, embedding-similarity second; both
    # are engineering heuristics for "the design references the control",
    # NOT scientific thresholds — calibrate on historical tasks.
    # >= this fraction of the control declaration's content tokens must
    # appear in steps/controls/contrast (substring match, so "clean" hits
    # "clean-control").
    construct_control_token_match_min: float = 0.5
    # Or: the control declaration must embed at least this similar to SOME
    # step/control segment.
    construct_control_embedding_min: float = 0.55

    # P2-C idea-level novelty quick-check (generate_minimal_experiments):
    # retrieval against each hypothesis cluster's hypothesis + primary
    # mechanism. already_implemented verdicts demote the idea to
    # conditional_review (METHOD_ALREADY_PUBLISHED); retrieval-infrastructure
    # failures degrade to a trace, never a demotion.
    idea_novelty_check_enabled: bool = True
    idea_novelty_check_top_k: int = 5

    # v7.2 (pilot adjudication 2026-10-29): scaling_regime candidates undergo a
    # claim-level precedent retrieval at mining admission — 4/7 pilot
    # scaling_regime opportunities were false-open (the claimed "untested
    # range" was already measured outside the evidence pool). A "covered"
    # verdict rejects the candidate (SCALING_RANGE_ALREADY_MEASURED);
    # retrieval-infrastructure failures degrade to a trace, never a rejection.
    gap_mining_precedent_check_enabled: bool = True
    gap_mining_precedent_check_top_k: int = 5

    # Semantic recall (2026-11-02, fixed-budget pilot adjudication failure).
    # Keyword search cannot express a conjunction such as "router intervention
    # x hydra-effect compensation": the pilot's clearest case (fbp-089, MoE
    # routing) retrieved 12 results and ZERO were about its topic, while the
    # queries themselves were reasonable. So: over-retrieve by keyword, then
    # filter/rerank by embedding similarity to the claim.
    #
    # The 0.41 threshold was measured, not chosen (scripts/_calibrate_sim.py,
    # 37 real gaps): claim-vs-own-audit-neighbour similarity has median 0.507
    # (p10 0.376), while claim-vs-known-noise has max 0.408. At 0.41 the filter
    # keeps 82% of real neighbours and admits 0% of the known off-topic pool.
    # Because 18% of true neighbours fall below it, this is a SOFT filter whose
    # kept/filtered counts must be disclosed by every consumer — lowering it
    # trades noise for recall, and 0.0 disables filtering while keeping rerank.
    semantic_recall_enabled: bool = True
    semantic_recall_threshold: float = 0.41
    semantic_recall_overfetch: int = 3

    # --- Audit comparison-pool supply (2026-11-02, killer-funnel diagnosis) ---
    # Measured on the fixed-budget pilot's 37 reviewed gaps: of the 82 killer
    # papers the independent reviewer named that were ALREADY in the task corpus
    # (title match), only 4 (4.9%) were ever shown to the audit as neighbours.
    # The rest split into: screened but not selected (6), in the audit pool but
    # cut by the 10-paper cap BEFORE the gap-relevance screen (17), and — the
    # dominant loss — in the corpus but NEVER admitted to any pool (55, 67.1%).
    # Root cause: the candidate pool is built from the gap's adversarial search
    # plus `collect_same_question_neighbors`, which only walks the evidence links
    # of the gap's own questions. A killer that the task corpus contains but that
    # never got linked to those questions was therefore invisible, and the audit
    # reported "confirmed" on a 3-paper pool that omitted it. Hence the pilot's
    # 33 confirmed -> 0 NONE collapse.
    #
    # Fix: top up the pool with a corpus-wide semantic recall over the task's own
    # papers (title+abstract vs the gap's claim). Cheap (one embedding pass) and
    # it directly attacks the 67.1% bucket. Threshold is the soft, disclosed kind
    # — recall diagnostics go into the trace, never silently dropped.
    audit_corpus_recall_enabled: bool = True
    audit_corpus_recall_threshold: float = 0.45
    # Hard ceiling on papers embedded per gap; the task corpus can hold 150-420
    # papers, all of which is still affordable in one batch.
    audit_corpus_recall_max_papers: int = 450
    # Number of corpus-recall hits unioned into the audit candidate pool.
    audit_corpus_recall_top_k: int = 15
    # Screen audit candidates with the gap-relevance model BEFORE the candidate
    # cap, and cap by gap relevance instead of TaskPaper.final_score. The old
    # order capped first (by a TASK-level relevance score) and screened the
    # survivors, so a direct killer ranked 11th by task relevance was discarded
    # without ever being judged against THIS gap (17 of 82 in the pilot).
    audit_screen_before_cap: bool = True

    # --- V2-no-audit ablation (capability attribution, 2026-11-02) ---
    # EXPERIMENT ARM SWITCH — leave False outside the ablation.
    #
    # production_e2e_v1 only ever measured the V2 *bundle* against simple
    # baselines, so "how much does the adversarial audit itself contribute" has
    # never been identified. The pilot makes this urgent: the audit's own
    # `confirmed` verdict predicted publishable novelty at essentially zero rate
    # (33 confirmed -> 0 NONE), and its `novelty_confidence` does not separate
    # covered from uncovered gaps (AUC 0.61 over 37 gaps).
    #
    # When True, `audit_gap_candidates` performs NO adversarial retrieval and NO
    # falsification judgement: every gap selected by the competition is marked
    # surviving and stamped with its own search-policy version, and the
    # intervention novelty gate is made explicitly PASS-by-ablation rather than
    # silently UNKNOWN. That last part is what makes the arm interpretable: an
    # un-audited gap must not land in tier B merely because a field is missing,
    # otherwise the arm is guaranteed to lose on executable count for a purely
    # mechanical reason and the comparison measures bookkeeping, not the gate.
    gap_audit_disabled_ablation: bool = False

    # Legacy v6 mining mode (fixed-budget experiment, 2026-10-31; see
    # eval_results/fixed_budget_pilot/EXPERIMENT_DESIGN.md). EXPERIMENT ARM
    # SWITCH — leave False outside the experiment.
    #
    # When True, gap mining reproduces the v6 two-type policy on the current
    # (debugged) stack: only boundary_gap / missing_evaluation lenses with
    # v6-faithful prompts (no v7.2 outcome-target wording), question-level
    # blockers restored (NO_LIMITATION_SIGNAL / UNRESOLVED_VERIFIED_
    # CONTRADICTION), the global limitation-or-negative_result citation rule,
    # and the v7.2 admission gates disabled. Everything downstream (audit,
    # budget, competition, interventions, ideas) is shared with arm A so the
    # mining policy is the only difference between arms. Candidates are
    # stamped `evidence-admission-v6-exp` so experiment rows stay separable
    # from the 281 historical `evidence-admission-v6` rows.
    #
    # Structural differences from historical v6 (disclosed, deliberate):
    #   - two per-type lens calls instead of one two-type menu call (the
    #     shared validation/dedup/trace machinery is per-lens);
    #   - runs on the current stack, i.e. WITH all post-v6 bug fixes
    #     (novelty-gate short-circuit, winner selection, Top-2 competition) —
    #     that is the point: the comparison isolates the mining policy, not
    #     the era's bugs.
    gap_mining_legacy_v6_mode: bool = False

    # v7.3 CANDIDATE GATE — OFF by default, deliberately.
    #
    # Rejects a claimed_delta that is an unexecuted measurement template rather
    # than a mechanism gap: measurement-verb lead + multi-deliverable conjunction
    # + no mechanism vocabulary. Motivation (fixed-budget pilot, 37 reviewed
    # gaps): 26 of 37 deltas had exactly that shape, and independent review gave
    # them 3 FULL / 23 PARTIAL / 0 NONE — the reviewer's residual notes repeated
    # "the specific operationalisation is not directly reported, but the
    # substantive delta is executed by [n]".
    #
    # It ships disabled because the 60% false-open precision it showed was
    # measured on the SAME 37 units the rule was derived from. That is post-hoc
    # fitting, not validation. Turning it on requires an out-of-sample check on
    # the 24-topic production_e2e pool; without that, enabling it would repeat
    # the exact failure this project's pre-registration discipline exists to
    # prevent (shipping a mechanism because it looks good on the sample that
    # suggested it).
    gap_mining_operationalization_gate: bool = False

    # Conjunction-only delta gate (2026-11-06, neighbour-budget experiment).
    #
    # OFF by default, deliberately. Independent blind review of 28 audited gaps
    # returned NONE=0 / PARTIAL=26 / FULL=2, so the primary metric
    # T=NONE/(...) was 0.0% in BOTH arms and the 3-vs-8 contrast could not be
    # measured at all.
    #
    # CORRECTION (2026-11-09): an earlier version of this comment said the
    # ingredients-covered-not-the-combination argument was the top PARTIAL
    # reason at 12/26 = 46%. That was inflated by a loose regex (counting any
    # "both"/"together"/"combines") and by summing non-exclusive categories.
    # Strict matching: 2/26 = 8%. The actual top reasons are "partially covers
    # the claim" (50%) and "adjacent work exists, but in a different setting"
    # (50%) — the latter being the rubric's own catch-all clause.
    #
    # A calibration probe settled the question: an absurd claim with zero
    # adjacent work (LLM inference under lunar gravity) was returned as NONE,
    # while a realistic claim came back PARTIAL. And an out-of-sample run with
    # the new mining prompt produced 24 deltas with no numbers and no
    # configuration phrasing (was 74% / 59%), yet blind review still returned
    # 6/6 PARTIAL, NONE=0. SHAPE CHANGED, THE VERDICT DID NOT.
    #
    # So this gate is NOT the fix for NONE=0, and no gate at the mining layer
    # can be: the unreachability lives in the adjudication rubric. A pure
    # conjunction remains structurally unanswerable as NONE, which is why the
    # rule is kept, but it is a secondary effect. See
    # eval_results/neighbor_budget/GAP_SUPPLY_VALIDATION.md.
    #
    # The detector fires on 5.2% of the 116 gaps (6/6 of them genuinely
    # conjunction-shaped), so it is precise but LOW-RECALL. NOT enabled: the
    # rule was derived from the same 28 units it would be scored on, so
    # enabling it without an out-of-sample check would repeat the v7.3 mistake
    # documented above the operationalization gate.
    gap_mining_conjunction_gate: bool = False

    # Idea score go-line (2026-10-31, distribution analysis over 595 scored
    # ideas / 94 tasks — see scripts/idea_score_distribution.py).
    #
    # This was a hardcoded 0.70 in three places. The analysis shows it is
    # calibrated to the v3–v7 population and NOT to v7.2:
    #   historical (v3-v7, n=579): median 0.711, 61.3% clear 0.70
    #   v7.2 probes  (n=16)     : median 0.654, 12.5% clear 0.70
    #     - and 0 of the 4 v7.2 executable_candidates clear it
    # So in v7.2 the line fires almost never — it reads as "the rubric says
    # everything is mediocre" when it actually says "the line moved".
    #
    # It is now one setting instead of three magic numbers. Left at 0.70 on
    # purpose: lowering it to the v7.2 median (~0.65) is a POLICY decision
    # (absolute-quality bar vs relative ranking within a run) that needs its
    # own justification, not a side effect of a refactor. The point of this
    # change is that the number is now visible, shared, and adjustable.
    idea_score_go_line: float = 0.70
    idea_score_revise_line: float = 0.50

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
