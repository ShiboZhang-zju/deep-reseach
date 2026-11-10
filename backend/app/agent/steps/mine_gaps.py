"""Step: Opportunity discovery — mine evidence-backed research opportunities.

v7 (2026-09-20): the gap ontology widened from two limitation-anchored types
(boundary_gap / missing_evaluation) to a seven-type Research Opportunity
ontology, each type with its OWN evidence admission contract, discovered via
a lens fan-out (one LLM call per opportunity type — the fan-out pattern the
intervention stage has used since intervention-lens-fanout-v2, moved forward
to discovery time so creativity is not filtered by a single narrow admission
narrative before it ever runs). Verified contradictions no longer block
mining: they are the raw material of the contradiction opportunity type.
"""

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from app.agent.evidence_relations import (
    CONTRADICTING_RELATIONS,
    MIN_RELATION_RELEVANCE,
    SUPPORTING_RELATIONS,
)
from app.agent.state import ResearchState
from app.db.models import EvidenceUnit, QuestionEvidenceLink, ResearchContract, ResearchQuestion
from app.db.repositories import gap_repo, paper_repo
from app.schemas.schemas import GapCandidateList
from app.services.embedding_service import cosine_similarity, embed_texts
from app.services.timing import StageTimings

logger = logging.getLogger(__name__)

# The Research Opportunity ontology (v7). Legacy types (coverage_gap,
# missing_method, missing_dataset) remain valid DB values for historical rows
# but can no longer be produced by mining.
_SUPPORTED_GAP_TYPES = {
    "boundary_gap",
    "missing_evaluation",
    "contradiction",
    "untested_assumption",
    "mechanism_gap",
    "scaling_regime",
    "cross_domain_transfer",
}
_ADMISSIBLE_STATUSES = {"verified", "upgraded", "abstract_only"}
_LIMITATION_SIGNAL_TYPES = {"limitation", "negative_result"}
# The admitted evidence pool grows with every search round, but the prompt has
# to stay inside the model's context window. Observed on a real run: 9 admitted
# units in round 1, 74 in round 3, and a round-4 prompt the backend rejected at
# 40961 tokens against a 40960-token window — which failed a two-hour task
# outright. Selection is therefore bounded per question, per paper and overall,
# and rotates across questions so one heavily covered question cannot crowd the
# others out of the prompt.
_MAX_EVIDENCE_PER_QUESTION = 8
_MAX_EVIDENCE_PER_PAPER_PER_QUESTION = 2
_MAX_PROMPT_EVIDENCE = 40
# Contradicting (verified conflict) units are rare and high-signal: they are
# the raw material of contradiction opportunities, so each question's shortlist
# reserves slots for them ahead of the type-interleaved bulk.
_MAX_CONTRADICTING_PER_QUESTION = 2
# Bumped to v3 when "covered" questions became eligible for mining: the policy
# version is what invalidates gaps already stamped for a round, so a change to
# the admission rules must bump it or the new rules never take effect on a
# resumed task.
# v5 (2026-08-27): already-proposed gaps are injected into the mining prompt
# (complementary claims over reworded duplicates) and the semantic-dedup
# threshold tightened 0.85 -> 0.78. Task 9e56a131: 9 candidates collapsed into
# 3-4 semantic clusters; remediation rounds re-mined reworded duplicates while
# the undetermined originals were never re-audited.
# v6 (2026-08-28, task d6f64087): placeholder-hypothesis guard — a
# testable_hypothesis or claimed_delta containing literal placeholder operands
# (">X%", "<Y parameters") is rejected at mining time.
# v7 (2026-09-20, Research Opportunity ontology): two-type menu -> seven-type
# opportunity ontology with per-type evidence admission contracts; lens
# fan-out (one LLM call per type, bounded candidates per lens); verified
# contradictions admitted as opportunity signal instead of blocking the
# question (UNRESOLVED_VERIFIED_CONTRADICTION / NO_LIMITATION_SIGNAL removed
# from question-level blocking, UNEXPECTED_CONTRADICTING_EVIDENCE removed from
# candidate rejection — the limitation requirement is now per-type: boundary /
# missing_evaluation still demand it, assumption / contradiction / transfer
# never did need it).
# v7.1 (2026-09-20, replay-readiness tightening): the >=2-paper floor is no
# longer uniform. untested_assumption requires assumption PREVALENCE — the
# same implicit assumption shared by >=3 independent papers (two methods
# sharing a habit is coincidence, three is a pattern worth testing);
# cross_domain_transfer requires BOTH sides of the transfer to be cited
# explicitly (>=1 source-domain mechanism unit + >=1 target-domain problem
# unit via the source_domain_evidence_ids / target_evidence_ids schema
# fields), not merely two arbitrary papers.
# v7.2 (2026-10-30, pilot adjudication 2026-10-29): two admission fixes driven
# by the blind-review false-open breakdown —
#   1. boundary_gap rejects PERFORMANCE_TARGET_DELTA: a claimed_delta led by
#      an outcome verb ("achieves >10% accuracy", "improves X by >25%") states
#      a performance target, not a boundary condition. Both human-true
#      survivors of the pilot were exactly this shape, and the reviewer
#      flagged them target_only ("NONE does not mean a credible gap").
#   2. scaling_regime admission gains a claim-level precedent retrieval
#      (SCALING_RANGE_ALREADY_MEASURED): 4 of 7 v7-only scaling_regime
#      opportunities were false-open because the claimed "untested range" had
#      already been measured by work outside the evidence pool. Adversarial
#      queries -> retrieval -> conservative verdict; infra failures degrade
#      to a trace and never reject (same contract as the idea novelty check).
GAP_MINING_POLICY_VERSION = "opportunity-ontology-v7.2"

# Legacy v6 mining mode (fixed-budget experiment arm B, 2026-10-31). Distinct
# from the historical "evidence-admission-v6" stamp so experiment rows are
# separable from the 281 historical ones.
V6_EXPERIMENTAL_POLICY_VERSION = "evidence-admission-v6-exp"


def current_mining_policy_version() -> str:
    """The policy stamp this process stamps and filters candidates by.

    Import-time constants cannot express a mode switch, and every consumer
    (mining stamps, audit candidate filters, runner gap selection) must agree
    on ONE value per process — the experiment's arm services resolve this
    differently (arm A: v7.2, arm B: v6-exp) but never mix within a process.
    """
    from app.config import settings
    if settings.gap_mining_legacy_v6_mode:
        return V6_EXPERIMENTAL_POLICY_VERSION
    return GAP_MINING_POLICY_VERSION


def _settings_v6_mode() -> bool:
    """Read the v6 control-arm switch (imported lazily for test monkeypatching)."""
    from app.config import settings
    return bool(settings.gap_mining_legacy_v6_mode)


def _settings_operationalization_gate() -> bool:
    """Read the v7.3 candidate gate switch (OFF by default, see the rule)."""
    from app.config import settings
    return bool(settings.gap_mining_operationalization_gate)


def _settings_conjunction_gate() -> bool:
    """Read the conjunction-only gate switch (OFF by default, see the rule)."""
    from app.config import settings
    return bool(settings.gap_mining_conjunction_gate)

# Max candidates accepted from a single opportunity lens — the fan-out's total
# breadth stays bounded (7 lenses x 2 = 14 per round before gates), and the
# runner's top-K gap competition decides which actually get audited.
_MAX_CANDIDATES_PER_LENS = 2

# Placeholder operands: comparisons or quantifiers whose operand is a bare
# letter (X/Y/N) instead of a number. Real hypotheses quantify ("noise above
# 10%"); a letter operand means the mining LLM deferred a decision it was
# asked to make, and the placeholder propagates downstream verbatim.
_PLACEHOLDER_OPERAND_RE = re.compile(
    r"[><=≤≥]\s*[XYN]\b"
    r"|\b[XYN]\s*%"
    r"|\b[XYN]\s+(?:samples?|papers?|parameters?|examples?|labels?|units?)\b",
)

# v7.2 (pilot adjudication 2026-10-29): a boundary_gap's claimed_delta must
# state WHERE/WHEN existing work fails or was never tested. A delta led by an
# outcome verb ("Achieves >10% higher accuracy ... while maintaining <5%
# latency overhead") is a performance target — the two human-true survivors of
# the pilot's false-open-heavy pool were both exactly this shape, and the
# reviewer flagged them target_only ("NONE does not mean a credible gap": an
# unprecedent performance number is not a research boundary). The check is
# deliberately conservative: it only fires on an outcome-verb LEAD and exempts
# any delta that also carries boundary language ("fails", "not evaluated",
# "no prior work", ...), so legitimate boundary statements phrased around a
# measurement ("methods degrade beyond 10% noise") pass untouched.
_PERFORMANCE_TARGET_LEAD_RE = re.compile(
    r"^\W*(?:achiev\w*|attain\w*|reach(?:es)?|improv\w*|outperform\w*|boost\w*|"
    r"increas\w*|reduc\w*|lower\w*|deliver\w*|yield\w*|obtain\w*|enhanc\w*)\b",
    re.IGNORECASE,
)
_BOUNDARY_LANGUAGE_RE = re.compile(
    r"\b(?:fail\w*|degrad\w*|break\w*|boundar\w*|limitation\w*|untested|unexplored|"
    r"not\s+(?:been\s+)?(?:evaluat|test|stud|explor|measure|report)\w*|"
    r"no\s+(?:prior\s+|previous\s+)?(?:work|study|studies|evaluation|report)\w*|"
    r"remains?\s+(?:untested|unknown|unexplored)|unknown\s+(?:whether|if)|"
    r"never\s+(?:been\s+)?(?:evaluat|test|stud)\w*|stops?\s+working|"
    r"未(?:曾|经)?(?:评测|评估|测试|研究|探索)|无(?:先例|先前工作|相关工作)|"
    r"(?:失|退|崩|泛化)化|边界|局限)\b",
    re.IGNORECASE,
)


def _is_performance_target_delta(claimed_delta: str) -> bool:
    """True when a boundary_gap delta is an outcome target, not a boundary."""
    text = (claimed_delta or "").strip()
    if not text:
        return False
    if _BOUNDARY_LANGUAGE_RE.search(text):
        return False
    return bool(_PERFORMANCE_TARGET_LEAD_RE.search(text))


# --- v7.3 candidate: operationalization-only delta (OFF by default) ---------
# Diagnosis (2026-11-02, fixed-budget pilot's 37 reviewed gaps): 26 of 37 deltas
# lead with a measurement verb ("Provide the first quantitative estimate...",
# "Quantify ... for at least 3 methods") and carry no mechanism vocabulary.
# Independent review returned 3 FULL / 23 PARTIAL / 0 NONE on that group: the
# reviewer kept writing "the specific operationalisation is not directly
# reported, but the substantive delta is executed by [n]". That is not a gap —
# it is an unexploited template over material that already exists.
#
# The rule below fires on measurement-verb lead + at least one conjunction of
# deliverables + no mechanism vocabulary. On the same 37 units it fires on 5
# (3 FULL, 2 PARTIAL) with 60.0% false-open vs a 10.8% base rate, and loses no
# NONE. That is a SMALL, POST-HOC-FIT sample: the rule was derived from the same
# units it is scored on, so it is not evidence that it generalises.
#
# Therefore it ships DISABLED. Enabling it requires an out-of-sample check
# (the 24-topic production_e2e pool), and it must never be enabled together
# with a claim that it improves credibility — that is exactly the unvalidated
# mechanism the project's own pre-registration discipline forbids.
_OPERATIONAL_LEAD_RE = re.compile(
    r"^\W*(?:provide|quantify|measure|characteri[sz]e|estimate|report|map|"
    r"extend|establish|offer|deliver|present|evaluate|benchmark)\b",
    re.IGNORECASE,
)
_MECHANISM_VOCAB_RE = re.compile(
    r"\b(?:mechanism|causal|because|why|attribut\w*|mediate\w*|drives?|"
    r"govern\w*|underlying|explain\w*|intervention|procedure|protocol)\b",
    re.IGNORECASE,
)
_MULTI_ELEMENT_RE = re.compile(
    r"\b(?:and\s+(?:test|show|compare|evaluate|quantify|report)|as well as|"
    r"for at least \w+|across (?:at least )?\w+ "
    r"(?:methods|domains|families|scales|disciplines)|combined|jointly)\b",
    re.IGNORECASE,
)


# --- conjunction-only delta (2026-11-06, neighbour-budget experiment) -------
# Diagnosis. In the 3-vs-8 run, independent blind review returned
#   NONE=0, PARTIAL=26, FULL=2 over 28 audited gaps (RESULTS.md §2),
# so the primary metric T=NONE/(...) was 0.0% in BOTH arms: no headroom.
#
# CORRECTION (2026-11-09). An earlier version of this comment claimed the
# ingredients-covered-but-not-the-combination argument was the dominant PARTIAL
# reason at "12/26 = 46%". That figure was inflated by a loose regex that
# counted any occurrence of "both" / "together" / "combines", and by summing
# non-exclusive categories. Strict matching gives 2/26 = 8%.
#
# What the PARTIAL rationales actually say (32 PARTIALs, incl. out-of-sample):
#   "partially covers the claim"                     50%
#   "adjacent work exists, but in a different setting" 50%   <-- the rubric's
#                                                              catch-all
#   ingredients-covered-not-combination (strict)      8%
#
# So the conjunction shape is NOT the main driver. The main driver is the
# rubric clause "covers it in a different setting", which any adjacent paper
# satisfies — and every real topic has adjacent papers. A calibration probe
# settled it: an absurd claim with ZERO adjacent work (LLM inference under
# lunar gravity) was returned as NONE; a realistic claim was returned PARTIAL.
# See eval_results/neighbor_budget/GAP_SUPPLY_VALIDATION.md §4.
#
# Out-of-sample check of the prompt fix: 24 newly mined deltas carried no
# numbers and no configuration phrasing (was 74%/59%), yet blind review returned
# 6/6 PARTIAL, NONE=0. Shape changed; the verdict did not.
#
# The rule below is therefore NOT the fix for NONE=0. It is kept only because a
# pure conjunction is still structurally unanswerable as NONE (every ingredient
# is published) — a narrow, real, but secondary effect.
#
# An earlier attempt in this session stripped numeric/configuration material
# and fired on 0/116 — it was measuring the wrong shape and is not shipped.
#
# SHIPS DISABLED, same discipline as the operationalization gate above: the
# rule was derived from the same 28 units it would be scored on, and enabling
# it without an out-of-sample check would repeat the v7.3 mistake.
_CONJ_ENUM_RE = re.compile(
    r"(?:\(\s*(?:i{1,3}|iv|v|[a-d]|[1-4])\s*\))"      # (i) (ii) / (1) (2)
    # 3+ conjoined clauses within one sentence. The gaps between the "and"s are
    # generous (160 chars) because real deltas pad each clause with its own
    # qualifier ("on 2 tasks and under 3 graded OOD-shift levels"). The
    # separator excludes sentence-final punctuation but NOT a period that is
    # part of a decimal ("1.5x"), otherwise ordinary numeric deltas never fire.
    r"|(?:\band\b(?:(?<!\d)\.|[\s,;:()\w%<>=~\-]){0,160}?\band\b"
    r"(?:(?<!\d)\.|[\s,;:()\w%<>=~\-]){0,160}?\band\b)",
    re.IGNORECASE,
)
_CONJ_WORD_RE = re.compile(
    r"\b(?:combin\w*|conjunct\w*|jointly|simultaneous\w*|at once|"
    r"all of (?:the )?(?:three|four)|"
    r"(?:two|three|four)\s+(?:components?|aspects?|dimensions?|signals?|"
    r"conditions?|channels?|stages?|axes))\b",
    re.IGNORECASE,
)


def _is_conjunction_only_delta(claimed_delta: str, missing_capability: str = "") -> bool:
    """True when a delta is an arrangement of already-published ingredients.

    Fires on an explicit enumeration/conjunction marker in `claimed_delta`.
    Deliberately does NOT fire on a delta that merely contains one "and": the
    three-clause pattern and the enumeration/combination vocabulary are what
    the reviewed failures actually had.
    """
    text = (claimed_delta or "").strip()
    if not text:
        return False
    if _CONJ_WORD_RE.search(text):
        return True
    return bool(_CONJ_ENUM_RE.search(text))


def _is_operationalization_only_delta(claimed_delta: str) -> bool:
    """True when a delta asks for a particular measurement, not a mechanism.

    Conservative on purpose: requires a measurement-verb lead AND a
    multi-deliverable conjunction AND no mechanism vocabulary. A delta that
    merely quantifies ONE thing ("measure ECE under drift") does not fire — that
    can be a real missing evaluation.
    """
    text = (claimed_delta or "").strip()
    if not text:
        return False
    if _MECHANISM_VOCAB_RE.search(text):
        return False
    if not _OPERATIONAL_LEAD_RE.search(text):
        return False
    return bool(_MULTI_ELEMENT_RE.search(text))


# --- v7.2: claim-level precedent retrieval for scaling_regime --------------
# Modelled on the idea novelty quick-check (generate_minimal_experiments):
# adversarial queries -> retrieval -> conservative verdict. ONLY a positive
# "covered" verdict rejects the candidate; every infrastructure failure
# degrades to a trace (an external-source outage is not evidence about the
# claim) — mirroring the idea novelty check's fail-open contract.

_SCALING_PRECEDENT_QUERIES_SYSTEM = """You generate adversarial search queries that would retrieve prior work already measuring inside a claimed untested scale or regime range.
Rules:
- Output English only.
- At most 2 queries, 3-8 words each.
- Anchor on the claimed range's dimensions (scale, sequence length, memory/budget, model size, frequency) together with the behaviour, e.g. "long documents beyond 32k tokens efficiency", "sub-1B parameter retrieval augmentation".
- The goal is FALSIFICATION: find work that already measured in the claimed range."""

_SCALING_PRECEDENT_VERDICT_SYSTEM = """You are a conservative judge deciding whether a retrieved paper already reports measurements inside a claimed untested scale/regime range.
Set range_already_measured=true ONLY if the paper clearly reports evaluating the same behaviour in the claimed range or an overlapping one (the abstract must show the measurement, not just propose or discuss it).
A paper studying the behaviour in a DIFFERENT range, or only proposing methods without measuring, is NOT coverage — answer false.
If true, set evidence_paper_id to that paper's id from the provided list."""


class ScalingPrecedentQueriesSchema(BaseModel):
    queries: list[str] = Field(default_factory=list)


class ScalingPrecedentVerdictSchema(BaseModel):
    range_already_measured: bool = False
    evidence_paper_id: str | None = None


async def _scaling_regime_precedent_check(db, state, llm, task_id, candidate):
    """Claim-level precedent retrieval for one scaling_regime candidate.

    Pilot adjudication (2026-10-29): 4 of 7 v7-only scaling_regime opportunities
    were blind-judged false-open — the claimed "untested range" had already
    been measured by prior work the evidence pool never saw. The per-type
    admission contract therefore gains this check.

    Returns (verdict, matched_paper_id) with verdict one of:
    - "disabled":           check turned off by configuration
    - "covered":            a retrieved paper already measures in the claimed
                            range -> reject SCALING_RANGE_ALREADY_MEASURED
    - "passed":             retrieval succeeded, no covering measurement found
    - "passed_no_results":  retrieval succeeded but returned nothing
    - "degraded":           infrastructure failure (LLM/search). NEVER rejects.
    """
    from app.config import settings

    if not settings.gap_mining_precedent_check_enabled:
        return "disabled", None
    from app.agent.steps.generate_queries import SearchQueryExecution
    from app.agent.steps.search_papers import search_and_save_papers
    from app.db.models import Paper, SearchQueryPaper
    from app.db.repositories.search_query_repo import save_search_query

    claim = (candidate.claimed_delta or "").strip()
    if not claim:
        return "passed", None

    queries: list[str] = []
    try:
        spec = await llm.chat_json([
            {"role": "system", "content": _SCALING_PRECEDENT_QUERIES_SYSTEM},
            {"role": "user", "content": json.dumps({
                "claimed_delta": claim,
                "target_setting": candidate.target_setting or "",
                "observed_problem": candidate.observed_problem or "",
            }, ensure_ascii=False)},
        ], ScalingPrecedentQueriesSchema)
        queries = [q.strip() for q in (spec.queries if spec else [])
                   if q and q.strip()][:2]
    except Exception as exc:
        logger.warning("Task %s: scaling precedent query generation degraded: %s",
                       task_id[:8], exc)
        paper_repo.save_trace(db, task_id, "scaling_precedent_check", "decision", output_data={
            "claimed_delta": claim[:200], "verdict": "degraded",
            "stage": "query_generation", "error": str(exc)[:200],
        })
        return "degraded", None
    if not queries:
        paper_repo.save_trace(db, task_id, "scaling_precedent_check", "decision", output_data={
            "claimed_delta": claim[:200], "verdict": "degraded",
            "stage": "query_generation", "error": "no queries returned",
        })
        return "degraded", None

    round_num = state.current_round
    # Bind each query to the candidate's first cited question (the execution
    # record requires a non-null target; the gap row does not exist yet, so
    # target_gap_id cannot be used here).
    target_question_id = candidate.question_ids[0] if candidate.question_ids else None
    executions = []
    for idx, text in enumerate(queries):
        record = save_search_query(
            db, task_id, text, f"scaling_precedent_{idx + 1}",
            target_question_id, "scaling_precedent", round_num,
        )
        executions.append(SearchQueryExecution(
            query_id=record.id, query_text=text,
            intent=f"scaling_precedent_{idx + 1}",
            target_question_id=target_question_id,
            expected_evidence_type=None,
        ))
    db.commit()
    try:
        await search_and_save_papers(db, state, executions, task_id, round_num)
    except Exception as exc:
        logger.warning("Task %s: scaling precedent retrieval degraded: %s",
                       task_id[:8], exc)
        paper_repo.save_trace(db, task_id, "scaling_precedent_check", "decision", output_data={
            "claimed_delta": claim[:200], "queries": queries,
            "verdict": "degraded", "stage": "retrieval", "error": str(exc)[:200],
        })
        return "degraded", None

    top_rows = (
        db.query(SearchQueryPaper, Paper)
        .join(Paper, SearchQueryPaper.paper_id == Paper.id)
        .filter(SearchQueryPaper.query_id.in_([e.query_id for e in executions]))
        .order_by(SearchQueryPaper.rank)
        .limit(settings.gap_mining_precedent_check_top_k * 3)
        .all()
    )
    seen_ids: set[str] = set()
    top_papers: list[Paper] = []
    for _sqp, paper in top_rows:
        if paper.id in seen_ids:
            continue
        seen_ids.add(paper.id)
        top_papers.append(paper)
        if len(top_papers) >= settings.gap_mining_precedent_check_top_k:
            break
    if not top_papers:
        paper_repo.save_trace(db, task_id, "scaling_precedent_check", "decision", output_data={
            "claimed_delta": claim[:200], "queries": queries,
            "verdict": "passed_no_results",
        })
        return "passed_no_results", None

    paper_list = [{
        "id": p.id, "title": (p.title or "")[:150],
        "abstract": (p.abstract or "")[:500],
    } for p in top_papers]
    try:
        verdict = await llm.chat_json([
            {"role": "system", "content": _SCALING_PRECEDENT_VERDICT_SYSTEM},
            {"role": "user", "content": json.dumps({
                "claimed_untested_range": claim,
                "papers": paper_list,
            }, ensure_ascii=False)},
        ], ScalingPrecedentVerdictSchema)
    except Exception as exc:
        logger.warning("Task %s: scaling precedent verdict degraded: %s",
                       task_id[:8], exc)
        paper_repo.save_trace(db, task_id, "scaling_precedent_check", "decision", output_data={
            "claimed_delta": claim[:200], "queries": queries,
            "verdict": "degraded", "stage": "verdict", "error": str(exc)[:200],
        })
        return "degraded", None

    matched = None
    if (verdict is not None and verdict.range_already_measured
            and verdict.evidence_paper_id and verdict.evidence_paper_id in seen_ids):
        matched = verdict.evidence_paper_id
    final_verdict = "covered" if matched else "passed"
    paper_repo.save_trace(db, task_id, "scaling_precedent_check", "decision", output_data={
        "claimed_delta": claim[:200], "queries": queries,
        "verdict": final_verdict, "covering_paper_id": matched,
        "examined_paper_ids": sorted(seen_ids),
    })
    return final_verdict, matched

# Semantic-dedup threshold: two gap candidates whose fingerprint (observed
# problem + missing capability + claimed delta) embeds to a cosine similarity at
# or above this bound are the same gap reworded. Without this, the same gap
# enters the audit twice under different wording and can be rejected once and
# "confirmed" once — the single most damaging false signal a research-idea
# system can emit.
# 0.78 (2026-08-27): at 0.85 the filter only caught verbatim-level rewordings —
# task 9e56a131 slipped 9 candidates into 3-4 semantic clusters (only 1 was
# caught as DUPLICATE_GAP). 0.78 keeps distinct-mechanism gaps apart while
# merging same-claim rewordings; misclassified candidates get another shot via
# the runner's re-audit channel.
_GAP_DEDUP_SIMILARITY = 0.78


@dataclass(frozen=True)
class OpportunityContract:
    """One opportunity type: its discovery directive and admission contract.

    Different opportunities must NOT share one admission rule. The old uniform
    rule (>=2 papers + limitation signal) structurally biased mining toward
    "limitations the authors themselves admitted" — untested assumptions,
    unexplained mechanisms and inter-paper contradictions can never appear
    there, yet they are among the most valuable research opportunities. The
    contract therefore varies per type:

    - required_evidence_types: at least one cited evidence unit must be of one
      of these types (None = no type requirement for this type);
    - signal_rejection_code: rejection reason when that requirement fails;
    - min_usable_units / min_units_rejection_code: floor on distinct usable
      cited evidence units (a contradiction needs the two conflicting claims);
    - min_independent_papers / paper_floor_rejection_code: per-type floor on
      the distinct papers behind the cited support. The universal default is
      2, but it is NOT a uniform absolute: untested_assumption demands
      prevalence (>=3 papers sharing the assumption), because "two methods
      happen to share a habit" is not a research opportunity on its own;
    - requires_both_domains: cross_domain_transfer must cite BOTH sides of
      the transfer explicitly (source-domain mechanism + target-domain
      problem), not merely two arbitrary papers that mention the field.
    """
    gap_type: str
    definition: str
    discovery_directive: str
    citation_rule: str
    required_evidence_types: frozenset[str] | None = None
    signal_rejection_code: str = ""
    min_usable_units: int = 1
    min_units_rejection_code: str = ""
    min_independent_papers: int = 2
    paper_floor_rejection_code: str = "INSUFFICIENT_CANDIDATE_PAPER_SUPPORT"
    requires_both_domains: bool = False


# Order matters: the two legacy types come first so historical behaviour (and
# the prompt-shape contract tests that inspect the first lens call) stay
# stable, then the five new opportunity types.
_OPPORTUNITY_CONTRACTS: tuple[OpportunityContract, ...] = (
    OpportunityContract(
        gap_type="boundary_gap",
        definition=(
            "an existing method or evaluation has a documented boundary, "
            "failure condition, or untested setting."
        ),
        discovery_directive=(
            "Look for limitation or negative-result evidence: a method that "
            "stops working under a specific condition, a scale where it "
            "degrades, or a setting its evaluation never covered. The "
            "claimed_delta must state the boundary condition — WHERE or WHEN "
            "existing work fails or was never tested. NEVER phrase it as an "
            "outcome target ('achieves >10% accuracy improvement', 'improves "
            "X by >25%'): a performance number nobody has hit yet is a goal, "
            "not a documented boundary, and will be rejected as "
            "PERFORMANCE_TARGET_DELTA."
        ),
        citation_rule=(
            "cite at least one evidence unit whose Type is 'limitation' or "
            "'negative_result' — this opportunity needs a documented "
            "shortcoming, not only comparisons"
        ),
        required_evidence_types=frozenset({"limitation", "negative_result"}),
        signal_rejection_code="CANDIDATE_LACKS_LIMITATION_SIGNAL",
    ),
    OpportunityContract(
        gap_type="missing_evaluation",
        definition=(
            "existing work does not evaluate a concrete, important "
            "capability, condition, cost, or risk."
        ),
        discovery_directive=(
            "Look for evidence that a capability is claimed or assumed but "
            "never measured: unevaluated risks, unmeasured costs, "
            "announced-but-missing evaluations."
        ),
        citation_rule=(
            "cite at least one evidence unit whose Type is 'limitation', "
            "'negative_result' or 'future_work' — the unevaluated capability "
            "must be documented somewhere, not merely assumed"
        ),
        required_evidence_types=frozenset(
            {"limitation", "negative_result", "future_work"}),
        signal_rejection_code="CANDIDATE_LACKS_LIMITATION_SIGNAL",
    ),
    OpportunityContract(
        gap_type="contradiction",
        definition=(
            "two or more verified results conflict under comparable settings, "
            "and the disagreement itself is unexplained."
        ),
        discovery_directive=(
            "Look for questions where supporting and conflicting evidence "
            "coexist: paper A reports a method works in setting X while paper "
            "B reports it does not. The opportunity is explaining or "
            "resolving the flip — which variable changes the conclusion. "
            "Evidence marked 'Verified conflict' is the strongest raw "
            "material for this type."
        ),
        citation_rule=(
            "cite BOTH sides of the conflict in supporting_evidence_ids (at "
            "least two evidence units from different papers) — they jointly "
            "establish that the contradiction exists; use "
            "contradicting_evidence_ids only for evidence showing the "
            "conflict is already resolved"
        ),
        min_usable_units=2,
        min_units_rejection_code="CONTRADICTION_NEEDS_CONFLICTING_CLAIM_PAIR",
    ),
    OpportunityContract(
        gap_type="untested_assumption",
        definition=(
            "multiple works share an implicit assumption that no supplied "
            "evidence tests."
        ),
        discovery_directive=(
            "Look for an assumption SEVERAL papers inherit without verifying "
            "(e.g. 'retrieval similarity ranks future usefulness'). "
            "Prevalence is the signal: an assumption shared by only two "
            "methods may be coincidence, while one inherited across many "
            "independent methods is a pattern worth testing. The opportunity "
            "is testing the inverted or untested assumption. A limitation "
            "admission is NOT required for this type."
        ),
        citation_rule=(
            "cite evidence units from at least THREE independent papers that "
            "share or rely on the same implicit assumption (assumption "
            "prevalence is the admission contract for this type); method and "
            "result units are admissible"
        ),
        min_independent_papers=3,
        paper_floor_rejection_code="ASSUMPTION_NEEDS_BROADER_PREVALENCE",
    ),
    OpportunityContract(
        gap_type="mechanism_gap",
        definition=(
            "a phenomenon is documented (an effect or a reproducible failure) "
            "but its cause is not established."
        ),
        discovery_directive=(
            "Look for results where something works or fails without an "
            "explanation: 'improves X' or 'breaks under Y' with no verified "
            "mechanism. The opportunity is isolating and testing the causal "
            "mechanism."
        ),
        citation_rule=(
            "cite at least one evidence unit whose Type is 'result', "
            "'negative_result', 'comparison' or 'limitation' — the phenomenon "
            "itself must be documented"
        ),
        required_evidence_types=frozenset(
            {"result", "negative_result", "comparison", "limitation"}),
        signal_rejection_code="CANDIDATE_LACKS_OBSERVED_EFFECT_EVIDENCE",
    ),
    OpportunityContract(
        gap_type="scaling_regime",
        definition=(
            "a behaviour is established in one scale or regime range while an "
            "important range is untested."
        ),
        discovery_directive=(
            "Look for documented tested ranges (scale, sequence length, "
            "memory size, budget, frequency) and name the adjacent regime "
            "that nobody reports. The opportunity is measuring the boundary "
            "of the established range."
        ),
        citation_rule=(
            "cite at least one evidence unit whose Type is 'result', "
            "'negative_result', 'comparison', 'limitation' or 'dataset' — the "
            "tested range must be documented, and the untested regime must be "
            "named explicitly"
        ),
        required_evidence_types=frozenset(
            {"result", "negative_result", "comparison", "limitation", "dataset"}),
        signal_rejection_code="CANDIDATE_LACKS_TESTED_RANGE_EVIDENCE",
    ),
    OpportunityContract(
        gap_type="cross_domain_transfer",
        definition=(
            "a mechanism proven in a neighbouring field solves an isomorphic "
            "problem that the target field has not imported."
        ),
        discovery_directive=(
            "Look for a mature mechanism in an adjacent area (databases, "
            "systems, signal processing, control theory, statistics, ...) "
            "that maps onto an unsolved problem in the target setting. The "
            "admission contract is TWO-SIDED: the source-domain mechanism "
            "must be documented, the target-domain problem must be "
            "documented, and the mapping between them must be explicit. A "
            "limitation admission is NOT required for this type."
        ),
        citation_rule=(
            "cite the evidence that documents the source mechanism AND the "
            "evidence that shows the target setting lacks it, then ALSO list "
            "the source evidence IDs in source_domain_evidence_ids and the "
            "target evidence IDs in target_evidence_ids, and make the "
            "source->target mapping explicit in the description"
        ),
        requires_both_domains=True,
    ),
)

_CONTRACTS_BY_TYPE = {spec.gap_type: spec for spec in _OPPORTUNITY_CONTRACTS}


# --- v6 experimental control arm (fixed-budget experiment, 2026-10-31) ------
# Reproduces the v6 two-type mining policy on the current stack. Contract text
# is v6-faithful (checked against the git snapshot at 686f6d1):
#   - definitions are IDENTICAL to v6 (v7 kept the wording verbatim);
#   - discovery directives are deliberately NEUTRAL — v6 had no per-type
#     directive, and v7.2's boundary directive carries the outcome-target
#     prohibition which is part of the v7.2 hypothesis under test and must not
#     leak into the control arm;
#   - the citation rule is v6's GLOBAL rule (limitation or negative_result for
#     BOTH types — v7.2's missing_evaluation contract widened it to include
#     future_work, which is also part of the hypothesis).
_V6_EXPERIMENT_CONTRACTS: tuple[OpportunityContract, ...] = (
    OpportunityContract(
        gap_type="boundary_gap",
        definition=(
            "an existing method or evaluation has a documented boundary, "
            "failure condition, or untested setting."
        ),
        discovery_directive=(
            "Identify candidate gaps of this fixed type that the supplied "
            "evidence supports."
        ),
        citation_rule=(
            "cite at least one evidence unit whose Type is 'limitation' or "
            "'negative_result': a gap needs a documented shortcoming, not "
            "only comparisons"
        ),
        required_evidence_types=frozenset({"limitation", "negative_result"}),
        signal_rejection_code="CANDIDATE_LACKS_LIMITATION_SIGNAL",
    ),
    OpportunityContract(
        gap_type="missing_evaluation",
        definition=(
            "existing work does not evaluate a concrete, important "
            "capability, condition, cost, or risk."
        ),
        discovery_directive=(
            "Identify candidate gaps of this fixed type that the supplied "
            "evidence supports."
        ),
        citation_rule=(
            "cite at least one evidence unit whose Type is 'limitation' or "
            "'negative_result': a gap needs a documented shortcoming, not "
            "only comparisons"
        ),
        required_evidence_types=frozenset({"limitation", "negative_result"}),
        signal_rejection_code="CANDIDATE_LACKS_LIMITATION_SIGNAL",
    ),
)


def _active_contracts() -> tuple[OpportunityContract, ...]:
    """The contract set for THIS process: v7.2's seven lenses, or the v6
    experimental control arm's two."""
    from app.config import settings
    if settings.gap_mining_legacy_v6_mode:
        return _V6_EXPERIMENT_CONTRACTS
    return _OPPORTUNITY_CONTRACTS


def compute_evidence_fingerprint(db, task_id: str) -> str:
    """Stable content fingerprint of the evidence pool + question links.

    Evidence-sensitive idempotency for gap mining: the runner hashes this
    fingerprint into the mining phase's input_version, so an O2 remediation
    round that adds evidence changes the fingerprint and therefore the input
    version, which makes PhaseRun idempotency re-run admission + mining on the
    richer pool instead of skipping them as "already done".

    The fingerprint is content-based (stable identity fields, sorted rows,
    floats rounded) — NOT a count — so the same pool always hashes identically
    while any added/replaced evidence or relinked question changes it.
    """
    evidence = db.query(EvidenceUnit).filter(EvidenceUnit.task_id == task_id).all()
    ev_rows = sorted(
        (e.id, e.paper_id, e.evidence_type, e.verification_status,
         round(e.extraction_confidence or 0.0, 3))
        for e in evidence
    )
    question_ids = {
        q.id for q in db.query(ResearchQuestion)
        .filter(ResearchQuestion.task_id == task_id).all()
    }
    links = db.query(QuestionEvidenceLink).all()
    link_rows = sorted(
        (l.question_id, l.evidence_id, l.relation_type,
         round(l.relevance_score or 0.0, 3))
        for l in links if l.question_id in question_ids
    )
    payload = json.dumps(
        {"evidence": ev_rows, "links": link_rows},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _gap_fingerprint(observed_problem, missing_capability, claimed_delta) -> str:
    """The stable semantic identity of a gap, used for dedup."""
    return " ".join(p for p in (observed_problem, missing_capability, claimed_delta)
                    if p).strip()


async def _embed_fingerprints(texts: list[str]) -> list[list[float]]:
    """Embed a batch of gap fingerprints; empty on failure (dedup is non-fatal)."""
    if not texts:
        return []
    try:
        return await asyncio.to_thread(embed_texts, texts)
    except Exception as exc:
        logger.warning("gap dedup embedding failed (%s); dedup disabled", exc)
        return []

@dataclass(frozen=True)
class QuestionEvidenceAdmission:
    question_id: str
    status: str
    reason_codes: list[str] = field(default_factory=list)
    admissible_evidence_ids: list[str] = field(default_factory=list)
    supporting_evidence_ids: list[str] = field(default_factory=list)
    limiting_evidence_ids: list[str] = field(default_factory=list)
    contradicting_evidence_ids: list[str] = field(default_factory=list)
    supporting_paper_ids: list[str] = field(default_factory=list)
    verified_span_count: int = 0
    # O5(b): evidence confidence tier — "A" when full-text-locatable evidence
    # backs the gap, "B" when only abstract-strength evidence is available.
    # B-tier admissions still PASS but produce provenance_status="partial" gaps.
    evidence_tier: str = "A"

# O5(b): minimum abstract-strength supporting papers required to admit a
# question at B-tier when no full-text-locatable evidence is present.
_MIN_ABSTRACT_TIER_PAPERS = 2

def _is_fulltext_locatable(item: EvidenceUnit) -> bool:
    return (
        item.verification_status in {"verified", "upgraded"}
        and bool(item.original_span)
        and bool(item.source_chunk_hash)
        and (item.page_number is not None or item.page_start is not None)
        and item.span_start is not None
        and item.span_end is not None
    )

def evaluate_gap_mining_admission(question, links, evidence_by_id) -> QuestionEvidenceAdmission:
    # "covered" is eligible: it means the question is well evidenced, which is a
    # prerequisite for claiming a gap, not a reason to exclude it. Only questions
    # retired from the active set (e.g. superseded) are ineligible here.
    if question.status not in {"open", "partially_covered", "covered"}:
        return QuestionEvidenceAdmission(question.id, "FAIL", ["QUESTION_NOT_ELIGIBLE"])
    if not links:
        return QuestionEvidenceAdmission(question.id, "UNKNOWN", ["NO_LINKED_EVIDENCE"])
    supporting, contradicting = [], []
    for link in links:
        item = evidence_by_id.get(link.evidence_id)
        if not item or (link.relevance_score or 0) < MIN_RELATION_RELEVANCE:
            continue
        if link.relation_type in SUPPORTING_RELATIONS and item.verification_status in _ADMISSIBLE_STATUSES:
            supporting.append(item)
        if link.relation_type in CONTRADICTING_RELATIONS and item.verification_status in {"verified", "upgraded"}:
            contradicting.append(item)
    # v7: the independent-papers floor spans BOTH sides of a conflict. A
    # question with one supporting paper and one contradicting paper is a
    # genuine inter-paper conflict — exactly the material a contradiction
    # opportunity needs — and used to be blocked here.
    papers = sorted(
        {item.paper_id for item in supporting}
        | {item.paper_id for item in contradicting})
    spans = sum(_is_fulltext_locatable(item) for item in supporting)
    limiting = [item for item in supporting if item.evidence_type in _LIMITATION_SIGNAL_TYPES]
    reasons = []
    if len(papers) < 2: reasons.append("INSUFFICIENT_INDEPENDENT_PAPERS")
    # v7: neither the limitation signal nor the verified contradiction blocks
    # admission any more. The limitation requirement is enforced per
    # opportunity TYPE at candidate validation (boundary_gap /
    # missing_evaluation demand it; assumption / contradiction / transfer
    # never needed it), and a verified contradiction is the raw material of
    # the contradiction opportunity, not a defect of the question. Both
    # signals remain recorded on the admission for the trace and the prompt
    # ("Verified conflict" annotations).
    #
    # v6 experimental control arm (2026-10-31): the question-level blockers
    # ARE restored — historical v6 refused to mine a question with no
    # limitation signal or with an unresolved verified contradiction, and the
    # control arm must reproduce that policy, not v7's per-type relaxation.
    from app.config import settings as _settings
    if _settings.gap_mining_legacy_v6_mode:
        if not limiting:
            reasons.append("NO_LIMITATION_SIGNAL")
        if contradicting:
            reasons.append("UNRESOLVED_VERIFIED_CONTRADICTION")

    # O5(b): full-text is no longer a hard requirement. Determine an evidence
    # tier instead:
    #   A-tier: at least one full-text-locatable span  -> high-confidence gap
    #   B-tier: no full-text, but >= _MIN_ABSTRACT_TIER_PAPERS independent
    #           papers with admissible evidence -> admissible but flagged
    #           partial (needs downstream audit / follow-up to confirm)
    # Only when neither condition holds do we withhold admission.
    if spans >= 1:
        evidence_tier = "A"
    elif len(papers) >= _MIN_ABSTRACT_TIER_PAPERS:
        evidence_tier = "B"
    else:
        evidence_tier = "B"
        reasons.append("NO_FULLTEXT_LOCATABLE_EVIDENCE")

    # Mid-priority #5: down-weight abstract-only evidence. Even if some
    # full-text span exists, if the *limitation signal itself* is only
    # abstract-strength (never full-text-locatable), the gap should not be
    # A-tier — the core "problem" claim is weakly grounded. Demote to B so
    # downstream (O1 tiering / audit) treats it as needing confirmation.
    # NOTE: this only affects the tier, NOT the PASS/UNKNOWN admission decision
    # (so it never blocks an otherwise-admissible gap).
    if evidence_tier == "A" and limiting and not any(_is_fulltext_locatable(item) for item in limiting):
        evidence_tier = "B"
        logger.info("Question %s: demoting gap evidence A->B (limitation signal is abstract-only)",
                    question.id[:8])

    status = "PASS" if not reasons else "UNKNOWN"
    # v7: the admissible pool includes the verified contradicting units — they
    # are offered to the model (annotated "Verified conflict") so the
    # contradiction lens can actually see the conflict it is asked to find.
    admissible = supporting + contradicting
    return QuestionEvidenceAdmission(
        question.id, status, reasons,
        [item.id for item in admissible], [item.id for item in supporting],
        [item.id for item in limiting], [item.id for item in contradicting],
        papers, spans, evidence_tier,
    )

_GAP_MINING_SYSTEM = """You identify candidate research opportunities of ONE fixed type from supplied paper evidence.
Opportunity type for this call: {gap_type} — {definition}
How to discover it: {discovery_directive}

Return only opportunities that are supported by the supplied evidence. An opportunity is not merely a topic with little evidence.
For every opportunity:
- cite only supplied question IDs and evidence IDs;
- every evidence ID you cite must list one of the opportunity's question IDs under
  "Linked questions", so the opportunity stays grounded in the questions it claims;
- cite evidence from at least two different papers (see "Paper ID");
- {citation_rule};
- state what existing work already covers and the smallest missing capability;
- write a falsifiable condition that would close the opportunity;
- quantifiers in testable_hypothesis and claimed_delta must be concrete numbers
  (">10% label noise", "<3B parameters"). NEVER emit placeholder operands like
  ">X%", "<Y parameters" or "N samples" — a hypothesis with an unquantified
  letter operand is rejected outright;
- CLAIMED DELTA MUST NAME A MISSING MECHANISM OR CAPABILITY, NOT AN
  ARRANGEMENT. Two shapes are FORBIDDEN because independent review can only
  ever answer "partially covered" to them, never "nobody has done this":
    (a) CONFIGURATION-ONLY: "at <=1B parameters", "on NarrativeQA",
        "at >30% vs <10% compression rate", "with 3 seeds". These are grid
        cells over a topic that is already fixed by existing work.
    (b) CONJUNCTION-ONLY: "combine A + B + C", "jointly measure X, Y and Z",
        "the first work to do (1) and (2) and (3)". Each ingredient is
        individually published; the reviewer will (correctly) reply that the
        ingredients are covered and only the combination is new. A combination
        is a project plan, not a gap.
  Measured on 116 mined gaps: 55% carry a numeric bound and 49% enumerate
  >=3 conjoined elements. Both shapes are structurally unanswerable as NONE
  (the topic exists; the cell/combination does not), so review can only reply
  "partially covered".
  If what you actually found is a configuration or a combination, write it in
  `missing_capability` and DROP the candidate — do not re-badge it as a
  capability;
- State the delta as ONE assertion about how something works:
  "X causes Y through Z" / "Z is not the mediator it is assumed to be" /
  "the effect reverses because W". Mechanism, causal attribution, or a
  refuted assumption — one of them, singly;
- SELF-CHECK before returning: (1) delete every number, dataset name and model
  size from `claimed_delta`; the remainder must still assert a mechanism.
  (2) delete any one of its clauses; the delta must be FALSE or uninteresting
  without it — if it survives intact as "A+B+C", it was a conjunction.
  Fail either check -> drop the candidate;
- do not propose a solution or invent paper findings;
- write every field in English (target setting, observed problem, existing
  coverage, missing capability, claimed delta, falsification condition). The
  downstream adversarial search and audit run against English academic
  databases, and Chinese opportunity text degrades their retrieval recall;
- gap_type MUST be exactly "{gap_type}" for every opportunity you return —
  candidates of any other type are discarded without audit.
Return an empty list when the evidence is insufficient for THIS type."""

_GAP_MINING_USER = """Research topic: {topic}
Target problem: {target_problem}
Target setting: {target_setting}

Opportunity type for this call: {gap_type} (produce ONLY this type).

Research questions:
{questions}

Evidence (only these IDs may be cited):
{evidence}

Gaps already proposed in earlier rounds (do NOT re-propose a gap whose claimed
delta substantially overlaps any of these — reworded duplicates are dropped by
a similarity filter; propose a complementary claim instead):
{existing_gaps}

Generate at most {max_gaps} evidence-backed candidate gaps of type {gap_type}.
If the supplied evidence does not support this type, return an empty list."""


def _format_evidence(evidence: list[EvidenceUnit],
                     question_links_by_evidence: dict[str, list[tuple[str, str]]] | None = None) -> str:
    """Render the evidence block.

    Every attribute that downstream validation tests must be visible here,
    otherwise the model is judged on constraints it was never shown and valid
    candidates are silently discarded: `Paper ID` backs the "at least two
    independent papers" gate, `Type` backs the per-type signal gates, and
    `question_links_by_evidence` backs the question-grounding gate.

    v7: units linked to a question through a verified `contradicts` relation
    carry a "Verified conflict" line — they are the raw material of
    contradiction opportunities and must be distinguishable from ordinary
    corroboration in the prompt.
    """
    lines = []
    for item in evidence:
        conditions = json.loads(item.conditions_json or "{}")
        condition_text = json.dumps(conditions, ensure_ascii=False) if conditions else "(none)"
        entry = (
            f"- Evidence ID: {item.id}\n"
            f"  Paper ID: {item.paper_id}\n"
            f"  Type: {item.evidence_type}\n"
            f"  Claim: {item.normalized_claim}\n"
            f"  Section: {item.section or '(unknown)'}\n"
            f"  Conditions: {condition_text}\n"
            f"  Verification: {item.verification_status or 'unverified'}"
        )
        linked = (question_links_by_evidence or {}).get(item.id)
        if linked:
            entry += "\n  Linked questions: " + ", ".join(q for q, _ in linked)
            conflicts = [q for q, relation in linked if relation == "contradicts"]
            if conflicts:
                entry += ("\n  Verified conflict: this evidence contradicts the "
                          "premise of " + ", ".join(conflicts))
        lines.append(entry)
    return "\n".join(lines)


def _evidence_priority(item: EvidenceUnit) -> tuple:
    """Rank evidence by how much it can contribute to a defensible gap.

    A gap needs a documented shortcoming and independent papers behind it, so
    limitation-type and full-text-locatable units come first; the ID breaks ties
    to keep selection deterministic across runs.
    """
    return (
        0 if item.evidence_type in _LIMITATION_SIGNAL_TYPES else 1,
        0 if _is_fulltext_locatable(item) else 1,
        0 if item.verification_status in {"verified", "upgraded"} else 1,
        -float(item.extraction_confidence or 0.0),
        item.id,
    )


def select_prompt_evidence(
    passed_admissions: dict[str, "QuestionEvidenceAdmission"],
    evidence_by_id: dict[str, EvidenceUnit],
    per_question: int = _MAX_EVIDENCE_PER_QUESTION,
    per_paper: int = _MAX_EVIDENCE_PER_PAPER_PER_QUESTION,
    total: int = _MAX_PROMPT_EVIDENCE,
) -> tuple[list[EvidenceUnit], dict[str, list[tuple[str, str]]]]:
    """Pick a bounded evidence subset to offer the model, and its question links.

    Every downstream gate still has to be satisfiable from what is offered, so
    each question contributes its highest-value units first while a per-paper
    cap keeps at least two independent papers in view. Questions are then
    interleaved round-robin, so the global cap truncates the long tail rather
    than whole questions.

    v7: within a question the shortlist is interleaved BY EVIDENCE TYPE
    (limitation-family first). The old pure priority ordering let one heavily
    populated type crowd every other type out of the per-question slots, which
    was harmless when all gaps needed a limitation but now would starve the
    assumption / mechanism / scaling lenses of the method- and result-type
    evidence they mine. Verified-conflict units are reserved slots first: they
    are rare, high-signal, and the contradiction lens cannot find a conflict
    it is never shown.

    Returns (selected units, question links) where each link is a
    (question_id, relation) pair and relation is "supports" or "contradicts".
    """
    shortlists: dict[str, list[EvidenceUnit]] = {}
    for question_id in sorted(passed_admissions):
        admission = passed_admissions[question_id]
        candidates = sorted(
            (evidence_by_id[evidence_id]
             for evidence_id in admission.admissible_evidence_ids
             if evidence_id in evidence_by_id),
            key=_evidence_priority,
        )
        contradicting_ids = set(admission.contradicting_evidence_ids)
        conflict_leads = [item for item in candidates
                          if item.id in contradicting_ids][:_MAX_CONTRADICTING_PER_QUESTION]
        lead_ids = {item.id for item in conflict_leads}
        remaining = [item for item in candidates if item.id not in lead_ids]

        by_type: dict[str, list[EvidenceUnit]] = {}
        for item in remaining:
            by_type.setdefault(item.evidence_type, []).append(item)
        ordered_types = sorted(
            by_type,
            key=lambda t: (0 if t in _LIMITATION_SIGNAL_TYPES else 1, t))
        interleaved: list[EvidenceUnit] = []
        max_type_depth = max((len(group) for group in by_type.values()), default=0)
        for depth in range(max_type_depth):
            for type_name in ordered_types:
                if depth < len(by_type[type_name]):
                    interleaved.append(by_type[type_name][depth])

        per_paper_seen: dict[str, int] = {}
        shortlist: list[EvidenceUnit] = []
        for item in conflict_leads + interleaved:
            if per_paper_seen.get(item.paper_id, 0) >= per_paper:
                continue
            per_paper_seen[item.paper_id] = per_paper_seen.get(item.paper_id, 0) + 1
            shortlist.append(item)
            if len(shortlist) >= per_question:
                break
        shortlists[question_id] = shortlist

    selected: dict[str, EvidenceUnit] = {}
    max_depth = max((len(items) for items in shortlists.values()), default=0)
    for depth in range(max_depth):
        if len(selected) >= total:
            break
        for shortlist in shortlists.values():
            if len(selected) >= total:
                break
            if depth < len(shortlist):
                item = shortlist[depth]
                selected.setdefault(item.id, item)

    question_links_by_evidence: dict[str, list[tuple[str, str]]] = {}
    for question_id in sorted(passed_admissions):
        admission = passed_admissions[question_id]
        supporting_ids = set(admission.supporting_evidence_ids)
        for evidence_id in admission.admissible_evidence_ids:
            if evidence_id in selected:
                relation = "supports" if evidence_id in supporting_ids else "contradicts"
                question_links_by_evidence.setdefault(evidence_id, []).append(
                    (question_id, relation))
    return list(selected.values()), question_links_by_evidence


async def mine_gap_candidates(
    db,
    state: ResearchState,
    llm,
    task_id: str,
    max_gaps: int = _MAX_CANDIDATES_PER_LENS,
    input_version: str = "",
) -> list:
    """Create evidence-backed research-opportunity candidates via lens fan-out.

    v7: one LLM call per opportunity type (seven calls), each restricted to
    that type and judged against that type's admission contract. A single
    call's candidates are correlated — they all think along the first
    mechanism that came to mind — so the fan-out decorrelates discovery the
    same way intervention-lens-fanout-v2 decorrelates intervention design.
    The grounding gates, semantic dedup and the runner's top-K gap
    competition remain the arbiter.

    max_gaps is the per-lens candidate cap (not a global one).

    input_version is the evidence-sensitive fingerprint (the same hash the
    runner embeds in the mining phase's input_version). The existing-gap
    short-circuit binds to it: when remediation adds evidence, the fingerprint
    changes, the short-circuit misses, and mining re-runs on the richer pool
    instead of reusing gaps mined from the old pool. Legacy calls without an
    input_version keep the round-based short-circuit for backward compatibility.
    """
    if not state.contract_id:
        logger.warning("Task %s: skip gap mining without an active contract", task_id[:8])
        return []

    contract = db.get(ResearchContract, state.contract_id)
    if not contract or contract.status != "active":
        logger.warning("Task %s: skip gap mining without an active contract record", task_id[:8])
        return []

    if input_version:
        existing = [gap for gap in gap_repo.list_gaps_for_contract(db, task_id, contract.id)
                    if gap.mining_policy_version == current_mining_policy_version()
                    and gap.mining_input_version == input_version
                    and gap.status not in {"rejected", "superseded"}]
    else:
        existing = [gap for gap in gap_repo.list_gaps_for_contract(db, task_id, contract.id)
                    if gap.mining_policy_version == current_mining_policy_version()
                    and gap.mining_round == state.current_round
                    and gap.status not in {"rejected", "superseded"}]
    if existing:
        state.active_gap_ids = [gap.id for gap in existing if gap.status != "rejected"]
        return existing

    # A gap can only be claimed where the existing work is understood, so the
    # questions with the *most* evidence are the ones worth mining. Restricting
    # this to open/partially_covered questions inverted that: `update_coverage`
    # marks a question "covered" once evidence accumulates, so the best-supported
    # questions were excluded and mining ran on the weakest ones. Observed on a
    # real run: 10 active questions, 8 high-importance ones covered, 191 evidence
    # units — yet only the single question at 0.15 coverage was mined, yielding
    # one fragile candidate that the audit could not confirm, which then burned a
    # remediation round. "covered" means well-evidenced, not gap-free.
    #
    # Quality is still enforced per question by evaluate_gap_mining_admission
    # (distinct papers, verified spans) and per candidate by the per-type
    # admission contracts, so widening the pool lets those gates do their job
    # instead of pre-filtering on a signal that means the opposite of what
    # mining needs.
    questions = db.query(ResearchQuestion).filter(
        ResearchQuestion.task_id == task_id,
        ResearchQuestion.contract_id == contract.id,
        ResearchQuestion.status.in_(["open", "partially_covered", "covered"]),
    ).order_by(ResearchQuestion.importance.desc()).all()
    if not questions:
        return []

    question_ids = [question.id for question in questions]
    links = db.query(QuestionEvidenceLink).filter(
        QuestionEvidenceLink.question_id.in_(question_ids),
    ).all()
    links_by_question = {question_id: [] for question_id in question_ids}
    for link in links:
        links_by_question.setdefault(link.question_id, []).append(link)
    linked_evidence_ids = {link.evidence_id for link in links}
    evidence_by_id = {
        item.id: item for item in db.query(EvidenceUnit).filter(
            EvidenceUnit.task_id == task_id,
            EvidenceUnit.id.in_(linked_evidence_ids),
        ).all()
    } if linked_evidence_ids else {}
    admissions = [
        evaluate_gap_mining_admission(question, links_by_question[question.id], evidence_by_id)
        for question in questions
    ]
    paper_repo.save_trace(db, task_id, "gap_mining_admission", "decision", output_data={
        "contract_id": contract.id,
        "passed_question_ids": [item.question_id for item in admissions if item.status == "PASS"],
        "question_results": [{
            "question_id": item.question_id, "status": item.status, "reason_codes": item.reason_codes,
            "supporting_paper_count": len(item.supporting_paper_ids),
            "verified_span_count": item.verified_span_count,
            "limitation_evidence_count": len(item.limiting_evidence_ids),
            "contradiction_count": len(item.contradicting_evidence_ids),
        } for item in admissions],
    })
    passed_admissions = {item.question_id: item for item in admissions if item.status == "PASS"}
    if not passed_admissions:
        db.commit()
        return []

    admissible_evidence_ids = {
        evidence_id for admission in passed_admissions.values()
        for evidence_id in admission.admissible_evidence_ids
    }
    evidence, question_links_by_evidence = select_prompt_evidence(
        passed_admissions, evidence_by_id)
    if not evidence:
        db.commit()
        return []
    # Only what the model was actually shown may be cited. Validating against
    # the whole admitted pool would accept IDs the prompt never offered, which
    # is indistinguishable from a fabricated citation.
    allowed_evidence_ids = {item.id for item in evidence}
    question_text = "\n".join(
        f"- Question ID: {question.id}\n  Type: {question.question_type}\n  Question: {question.question}"
        for question in questions if question.id in passed_admissions
    )
    # P0-2: surface already-proposed gaps to the generator. Task 9e56a131
    # produced 9 candidates forming only 3-4 semantic clusters — remediation
    # rounds re-mined reworded duplicates while the undetermined originals
    # were never re-audited. Showing the model what is already on the table
    # pushes new candidates toward complementary claims; the embedding dedup
    # below remains the hard backstop.
    existing_gaps = [g for g in gap_repo.list_gaps_for_contract(
        db, task_id, contract.id, include_superseded=True)
        if g.status != "rejected"]
    existing_claims = [f"- {(g.claimed_delta or '').strip()[:220]}"
                       for g in existing_gaps[:10] if (g.claimed_delta or "").strip()]
    existing_gaps_block = "\n".join(existing_claims) if existing_claims else "(none yet)"

    # Semantic dedup prep: embed every already-accepted gap's fingerprint ONCE
    # (this batch excluded). Per lens, only that lens's type-matching
    # candidates are embedded. Dedup is non-fatal — if embedding is
    # unavailable we skip it rather than blocking mining — so a duplicate can
    # only slip through when embeddings fail.
    # `existing_gaps` is the same pool loaded above for the prompt — every
    # non-rejected gap in the contract, INCLUDING superseded versions: a gap
    # that was narrowed (v1 superseded -> v2) must still block a later
    # candidate that rewords v1's original broad claim, otherwise "reword the
    # rejected broad claim" defeats the lineage. Only genuinely rejected gaps
    # (audit said closed) leave the comparison pool.
    existing_fps = [_gap_fingerprint(g.observed_problem, g.missing_capability,
                                     g.claimed_delta) for g in existing_gaps]
    existing_vecs_raw = await _embed_fingerprints(existing_fps)
    existing_dedup_ok = bool(existing_vecs_raw) and len(existing_vecs_raw) == len(existing_fps)
    accepted_vecs = list(existing_vecs_raw) if existing_dedup_ok else []

    created = []
    rejected_candidates = []
    accepted_candidates = []
    lens_fanout = []
    # Sub-stage attribution (probe df744606: `mine_gaps` cost 221-409s with no
    # breakdown). Buckets: per-lens LLM call, per-lens embedding dedup, and the
    # whole admission loop.
    timings = StageTimings()

    # --- Opportunity lens fan-out (v7) ---------------------------------------
    # One call per opportunity type. A candidate returned by the wrong lens is
    # rejected as TYPE_LENS_MISMATCH without consuming an audit slot.
    # v6 experimental control arm: `_active_contracts()` returns the two-type
    # v6 set, so the fan-out (and everything downstream in this loop) only
    # ever sees boundary_gap / missing_evaluation.
    for contract_spec in _active_contracts():
        with timings.stage(f"lens_llm:{contract_spec.gap_type}"):
            result = await llm.chat_json([{
                "role": "system", "content": _GAP_MINING_SYSTEM.format(
                    gap_type=contract_spec.gap_type,
                    definition=contract_spec.definition,
                    discovery_directive=contract_spec.discovery_directive,
                    citation_rule=contract_spec.citation_rule,
                ),
            }, {
                "role": "user", "content": _GAP_MINING_USER.format(
                    topic=contract.topic, target_problem=contract.target_problem or "(not specified)",
                    target_setting=contract.target_setting or "(not specified)",
                    gap_type=contract_spec.gap_type,
                    questions=question_text,
                    evidence=_format_evidence(evidence, question_links_by_evidence),
                    existing_gaps=existing_gaps_block, max_gaps=max_gaps,
                ),
            }], GapCandidateList)

        candidate_objs = result.gaps[:max_gaps]
        typed_candidates = []
        for candidate in candidate_objs:
            if candidate.gap_type != contract_spec.gap_type:
                rejected_candidates.append({
                    "reason": "TYPE_LENS_MISMATCH",
                    "lens": contract_spec.gap_type,
                    "gap_type": candidate.gap_type,
                    "question_ids": candidate.question_ids,
                    "cited_evidence_ids": candidate.supporting_evidence_ids,
                })
            else:
                typed_candidates.append(candidate)

        candidate_fps = [_gap_fingerprint(c.observed_problem, c.missing_capability,
                                          c.claimed_delta) for c in typed_candidates]
        with timings.stage("dedup_embedding"):
            lens_vecs_raw = await _embed_fingerprints(candidate_fps)
        lens_dedup_enabled = bool(lens_vecs_raw) and len(lens_vecs_raw) == len(candidate_fps)
        lens_vecs = lens_vecs_raw if lens_dedup_enabled else []

        lens_accepted = 0
        for idx, candidate in enumerate(typed_candidates):
            candidate_question_ids = set(candidate.question_ids)
            allowed_for_candidate = {
                evidence_id for question_id in candidate_question_ids
                for evidence_id in passed_admissions.get(question_id, QuestionEvidenceAdmission("", "FAIL")).admissible_evidence_ids
                if evidence_id in allowed_evidence_ids
            }
            cited_evidence_ids = list(dict.fromkeys(candidate.supporting_evidence_ids))
            # Drop citations that were never offered instead of discarding the whole
            # candidate: a single stale or fabricated ID used to invalidate every
            # other correctly cited, admitted evidence unit. Whatever remains must
            # still satisfy every substantive gate below on its own, and the dropped
            # IDs are recorded, so nothing is laundered.
            unoffered_evidence_ids = [item for item in cited_evidence_ids
                                      if item not in allowed_evidence_ids]
            usable_evidence_ids = [item for item in cited_evidence_ids
                                   if item in allowed_evidence_ids]
            support = [evidence_by_id[evidence_id] for evidence_id in usable_evidence_ids]
            reason = None
            if not candidate_question_ids or not candidate_question_ids.issubset(passed_admissions): reason = "UNKNOWN_QUESTION_ID"
            elif not usable_evidence_ids:
                # Nothing the candidate cites was ever offered (fabricated, or from a
                # question that did not pass admission).
                reason = "UNKNOWN_EVIDENCE_ID"
            elif not (set(usable_evidence_ids) & allowed_for_candidate):
                # Every usable citation is admissible, but none is linked to a
                # question the candidate claims — the gap is not grounded in its own
                # questions. Corroborating evidence from sibling questions is kept,
                # since the prompt deliberately offers the whole admitted pool.
                reason = "EVIDENCE_NOT_LINKED_TO_CITED_QUESTION"
            elif len(usable_evidence_ids) < contract_spec.min_usable_units:
                # v7: a contradiction is established by the conflicting claims
                # themselves — at least two cited units are required. Checked
                # before the paper floor so the rejection names the actual
                # defect (only one side of the conflict cited); for every other
                # type min_usable_units is 1 and this never fires.
                reason = contract_spec.min_units_rejection_code
            elif len({item.paper_id for item in support}) < contract_spec.min_independent_papers:
                # v7.1: the paper floor is per-type. The universal default
                # stays 2, but untested_assumption demands PREVALENCE (>=3
                # papers sharing the assumption) — two methods sharing a
                # habit is coincidence, not a research opportunity.
                reason = contract_spec.paper_floor_rejection_code
            elif (contract_spec.required_evidence_types is not None
                  and not any(item.evidence_type in contract_spec.required_evidence_types
                              for item in support)):
                # v7 per-type admission contract: the signal this opportunity
                # type is built on must actually be cited. boundary /
                # missing_evaluation demand a documented shortcoming; mechanism
                # / scaling demand a documented phenomenon or tested range.
                reason = contract_spec.signal_rejection_code
            elif (contract_spec.requires_both_domains
                  and (not [item for item in candidate.source_domain_evidence_ids
                            if item in allowed_evidence_ids]
                       or not [item for item in candidate.target_evidence_ids
                               if item in allowed_evidence_ids]
                       or not (set(candidate.source_domain_evidence_ids)
                               | set(candidate.target_evidence_ids))
                       <= set(usable_evidence_ids))):
                # v7.1: cross_domain_transfer is two-sided by contract — a
                # documented source-domain mechanism AND a documented
                # target-domain problem, both cited among the supporting
                # evidence. Two arbitrary papers that mention the field do
                # not establish a transfer opportunity.
                reason = "TRANSFER_NEEDS_BOTH_DOMAIN_EVIDENCE"
            elif (_PLACEHOLDER_OPERAND_RE.search(candidate.testable_hypothesis or "")
                    or _PLACEHOLDER_OPERAND_RE.search(candidate.claimed_delta or "")):
                # P1-1 (task d6f64087): ">X% label noise or <Y parameters" is a
                # hypothesis the mining LLM never finished quantifying. Accepting it
                # lets the placeholder flow through narrowing into intervention and
                # experiment prompts (observed verbatim on the surviving gap).
                reason = "PLACEHOLDER_OPERAND_IN_HYPOTHESIS"
            elif (contract_spec.gap_type == "boundary_gap"
                    and not _settings_v6_mode()
                    and _is_performance_target_delta(candidate.claimed_delta or "")):
                # v7.2 (pilot adjudication 2026-10-29): "Achieves >10% higher
                # accuracy ..." is an outcome target, not a boundary condition.
                # Both human-true pilot survivors had exactly this shape and were
                # reviewer-flagged target_only — an unprecedent performance
                # number is a goal, not a documented boundary.
                # Disabled in the v6 control arm: this gate is part of the
                # v7.2 hypothesis under test.
                reason = "PERFORMANCE_TARGET_DELTA"
            elif (_settings_operationalization_gate()
                    and not _settings_v6_mode()
                    and _is_operationalization_only_delta(candidate.claimed_delta or "")):
                # v7.3 candidate, DISABLED by default (see the rule's comment:
                # the calibration sample is the same sample the rule was
                # derived from). Fires only when explicitly enabled.
                reason = "OPERATIONALIZATION_ONLY_DELTA"
            elif (_settings_conjunction_gate()
                    and not _settings_v6_mode()
                    and _is_conjunction_only_delta(candidate.claimed_delta or "",
                                                   candidate.missing_capability or "")):
                # Conjunction-only delta, DISABLED by default for the same
                # reason as the gate above: the 46%-of-PARTIAL figure and the
                # detector were derived from the same 28 reviewed units.
                reason = "CONJUNCTION_ONLY_DELTA"
            # v7: candidate.contradicting_evidence_ids no longer rejects the
            # candidate. Evidence that undermines the gap is recorded as a
            # "contradicts" gap-evidence link below and consumed by the
            # adversarial audit — surface the tension instead of silently
            # discarding the opportunity.
            if reason:
                rejected_candidates.append({
                    "reason": reason,
                    "lens": contract_spec.gap_type,
                    "gap_type": candidate.gap_type,
                    "question_ids": candidate.question_ids,
                    "cited_evidence_ids": candidate.supporting_evidence_ids,
                    "unoffered_evidence_ids": unoffered_evidence_ids,
                })
                continue

            # Semantic dedup: a candidate that is the same gap as one already
            # accepted (this batch or a previous round) under different wording is
            # dropped, so it cannot enter the audit twice and earn a contradictory
            # verdict. The fingerprint is observed problem + missing capability +
            # claimed delta.
            dup = False
            if lens_dedup_enabled and lens_vecs[idx]:
                for vec in accepted_vecs:
                    if vec and cosine_similarity(lens_vecs[idx], vec) >= _GAP_DEDUP_SIMILARITY:
                        dup = True
                        break
            if dup:
                rejected_candidates.append({
                    "reason": "DUPLICATE_GAP",
                    "lens": contract_spec.gap_type,
                    "gap_type": candidate.gap_type,
                    "question_ids": candidate.question_ids,
                    "cited_evidence_ids": candidate.supporting_evidence_ids,
                })
                continue

            # v7.2: scaling_regime candidates undergo a claim-level precedent
            # retrieval before consuming an audit slot — 4/7 pilot
            # scaling_regime opportunities were false-open because prior work
            # outside the evidence pool had already measured the claimed
            # range. Runs AFTER dedup so a reworded duplicate never spends
            # retrieval budget. Only a positive "covered" verdict rejects;
            # infrastructure failures fail open (degraded -> accept).
            if contract_spec.gap_type == "scaling_regime":
                with timings.stage("scaling_precedent_check"):
                    precedent_verdict, covering_paper_id = await _scaling_regime_precedent_check(
                        db, state, llm, task_id, candidate)
                if precedent_verdict == "covered":
                    rejected_candidates.append({
                        "reason": "SCALING_RANGE_ALREADY_MEASURED",
                        "lens": contract_spec.gap_type,
                        "gap_type": candidate.gap_type,
                        "question_ids": candidate.question_ids,
                        "cited_evidence_ids": candidate.supporting_evidence_ids,
                        "covering_paper_id": covering_paper_id,
                    })
                    continue

            # O5(b): full-text presence sets provenance tier rather than rejecting.
            # A-tier (complete) has a full-text-locatable span; B-tier (partial)
            # is abstract-strength only and must be confirmed by the downstream
            # adversarial audit before it can survive.
            has_fulltext = any(_is_fulltext_locatable(item) for item in support)
            provenance_status = "complete" if has_fulltext else "partial"
            gap = gap_repo.create_gap_candidate(
                db, task_id=task_id, contract_id=contract.id, gap_type=candidate.gap_type,
                description=candidate.description, target_setting=candidate.target_setting,
                observed_problem=candidate.observed_problem, existing_coverage=candidate.existing_coverage,
                missing_capability=candidate.missing_capability, claimed_delta=candidate.claimed_delta,
                testable_hypothesis=candidate.testable_hypothesis,
                falsification_condition=candidate.falsification_condition, provenance_status=provenance_status,
                question_ids=candidate.question_ids, mining_round=state.current_round,
                novelty_score=candidate.novelty_score, feasibility_score=candidate.feasibility_score,
                significance_score=candidate.significance_score,
                mining_policy_version=current_mining_policy_version(),
                mining_input_version=input_version,
            )
            # Only admitted evidence gets linked; an unoffered ID would otherwise
            # create a gap-evidence link pointing at nothing.
            for evidence_id in usable_evidence_ids:
                gap_repo.create_gap_evidence_link(db, gap.id, evidence_id, "suggests", 0.8)
            for evidence_id in candidate.contradicting_evidence_ids:
                gap_repo.create_gap_evidence_link(db, gap.id, evidence_id, "contradicts", 0.8)
            created.append(gap)
            lens_accepted += 1
            if lens_dedup_enabled and lens_vecs[idx]:
                accepted_vecs.append(lens_vecs[idx])
            accepted_candidates.append({
                "gap_id": gap.id, "gap_type": candidate.gap_type,
                "lens": contract_spec.gap_type,
                "linked_evidence_ids": usable_evidence_ids,
                "dropped_evidence_ids": unoffered_evidence_ids,
            })
        lens_fanout.append({
            "lens": contract_spec.gap_type,
            "llm_candidate_count": len(candidate_objs),
            "type_matched_count": len(typed_candidates),
            "accepted_count": lens_accepted,
        })

    paper_repo.save_trace(db, task_id, "gap_mining_candidates", "decision", output_data={
        "llm_candidate_count": sum(item["llm_candidate_count"] for item in lens_fanout),
        "accepted_candidate_count": len(created),
        "lens_fanout": lens_fanout,
        "accepted_candidates": accepted_candidates,
        "rejected_candidates": rejected_candidates,
        "stage_timings": timings.as_dict(),
    })
    state.active_gap_ids = [gap.id for gap in created]
    paper_repo.save_trace(db, task_id, "mine_gap_candidates", "action", output_data={
        "contract_id": contract.id,
        "candidate_count": len(created),
        "evidence_count": len(evidence),
        "admissible_evidence_count": len(admissible_evidence_ids),
        "opportunity_lens_count": len(_OPPORTUNITY_CONTRACTS),
        "supported_gap_types": sorted(_SUPPORTED_GAP_TYPES),
    })
    db.commit()
    return created
