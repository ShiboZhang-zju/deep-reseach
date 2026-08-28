"""Step: Run a lightweight adversarial audit for evidence-backed gap candidates."""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, replace
from collections import defaultdict

from pydantic import BaseModel, Field

from app.config import settings
from app.agent.state import ResearchState
from app.agent.steps.generate_queries import SearchQueryExecution
from app.agent.steps.search_papers import search_and_save_papers
from app.agent.steps.mine_gaps import GAP_MINING_POLICY_VERSION
from app.agent.steps.gap_relevance import score_all_gap_candidates
from app.db.models import (EvidenceUnit, GapAudit, GapCandidate, Paper, QuestionEvidenceLink,
                          SearchQueryPaper, SearchQueryRecord, SearchRawResult, TaskPaper)
from app.db.repositories import gap_repo, paper_repo, task_repo
from app.db.repositories.search_query_repo import save_search_query

logger = logging.getLogger(__name__)

_MAX_NEIGHBORS = 5
# Neighbors whose highest semantic similarity falls below this bound did not
# meaningfully cover the gap's domain, so a "confirmed" verdict resting on such
# neighbors is a retrieval miss rather than a novelty signal.
_MIN_NEIGHBOR_SIMILARITY = 0.3
# P1.2 killer search: embedding cosine between the audit's killer description
# and a freshly recalled paper's title+abstract above which the paper counts
# as a hit. Workflow heuristic, not a calibrated truth (calibration rule
# applies).
_KILLER_HIT_SIMILARITY = 0.82
# A high novelty_confidence combined with a low-similarity neighbor set is the
# contradictory signal that flags retrieval failure (observed: 5 application-
# domain neighbors at max similarity 0.15 yet novelty_confidence 0.95).
_HIGH_NOVELTY = 0.8
# P0-1b (2026-08-28, task d6f64087): symmetric guard at the LOW end. The v4
# guard above distrusts high-confidence confirms with distant neighbors; the
# mirror image — a confirmed verdict whose own novelty_confidence is very low —
# is the auditor admitting its search was too weak to rule prior work out
# ("no neighbor covers the claim" by absence, not by evidence). Observed:
# surviving gap 475c8acd confirmed with novelty_confidence=0.3, which then
# produced three tier-A interventions. Downgrade to uncertain/more_search so a
# later round retrries with better retrieval instead of stamping a hollow
# "novel".
_LOW_NOVELTY_CONFIRMED = 0.4
# E2E 2026-08-26: the surviving gap's audit output had structured fields
# confirmed/continue while its own remaining_delta concluded verbatim
# "Therefore, the decision is uncertain." Match only unambiguous
# conclusion-of-uncertainty phrasings; anything weaker is left to the verdict
# fields so this guard cannot fire on hedged-but-committed text.
_AUDIT_TEXT_CONFLICT_RE = re.compile(
    r"\bdecision is (?:uncertain|inconclusive)\b"
    r"|\b(?:cannot|unable to) (?:confirm|conclude)\b"
    r"|\binsufficient evidence to confirm\b"
    r"|\bnovelty (?:cannot|can not) be (?:confirmed|established)\b",
    re.IGNORECASE,
)
# v2 made corpus papers from the same research questions admissible comparison
# material; v3 closes a gap as undecidable when a repeated adversarial round
# brings no new material for the same claim; v4 generates English adversarial
# queries (anchored on mechanism/capability rather than surface keywords) and
# downgrades confirmed verdicts whose neighbors are semantically too distant to
# have ruled prior work out. Like the mining policy version, this is what
# invalidates audits already stamped for a round, so an admission-or-verdict
# rule change must bump it or resumed tasks keep their old verdicts.
# v8 repairs the evidence funnel: audit-recalled papers are scored (priority
# was NULL, so they never entered evidence extraction) and NPA neighbors get
# full-text evidence extracted + injected into the audit prompt.
# v9 (2026-08-27): variant-validation failure no longer drops a query family —
# raw LLM variants are accepted flagged LOW_CONFIDENCE_VARIANTS. v8 runs with
# 4/5 families dropped (task 9e56a131) structurally failed admission on
# INSUFFICIENT_QUERY_FAMILIES before any search even ran.
# v10 (2026-08-27): third tier — a family whose variants list is EMPTY but
# whose structured intent is complete gets variants synthesized from its
# structured fields (task 23ec8f20 re-audit: default_factory=list bypasses
# min_length, 4/5 families empty, admission failed twice on family count).
# v11 (2026-08-28, task d6f64087): low-novelty confirmed guard — a confirmed
# verdict whose own novelty_confidence <= 0.4 is downgraded to uncertain/
# more_search (search-absence novelty, observed: surviving gap at 0.3 feeding
# three tier-A interventions). Verdict rules changed, so previously stamped
# audits must be invalidated on resume.
GAP_SEARCH_POLICY_VERSION = "gap-search-admission-v12"


# --- Canonical atomic claims (Phase 3H) ---
# A gap's claimed delta is decomposed into atomic positive capability/delta
# statements BEFORE the NPA audit, so each neighbor is judged per-claim and the
# residual gap is a set subtraction (claims - union(FULL-covered claims)), never
# a natural-language string intersection. Claim verdict handles UNCERTAIN first.

class AtomicClaimSchema(BaseModel):
    claim_text: str = Field(min_length=5)


class AtomicClaimList(BaseModel):
    claims: list[AtomicClaimSchema] = Field(default_factory=list, min_length=1)


class ClaimCoverageSchema(BaseModel):
    claim_index: int = Field(ge=0)
    coverage: str = Field(description="One of: FULL, PARTIAL, NONE, UNCERTAIN")
    rationale: str = ""


_ATOMIC_CLAIM_SYSTEM = """You decompose a research gap's claimed delta into canonical atomic claims.

Each atomic claim must be a POSITIVE capability/delta statement — something a prior-art paper
could independently be judged to cover or not cover. It must NOT be a literature-gap conclusion
like "existing work lacks X" or "no work does Y".

Bad: "existing work lacks dense hidden tests"
Good: "dense hidden tests distinguish accidental from robust correctness"

Produce 3-5 atomic claims that together capture the gap's claimed delta. Each claim is atomic
(one testable capability), self-contained, and in English."""

_ATOMIC_CLAIM_USER = """Gap claimed delta: {claimed_delta}
Missing capability: {missing_capability}

Produce the atomic claims."""


async def _ensure_atomic_claims(db, llm, gap: GapCandidate, task_id: str) -> list:
    """Idempotently decompose a gap's claimed delta into atomic claims.

    Called when the gap enters the NPA audit (not after surviving). Returns the
    existing claims on a re-audit; a decomposition failure is non-fatal and
    leaves the old LLM-decision verdict path intact.
    """
    claims = gap_repo.list_atomic_claims(db, gap.id)
    if claims:
        return claims
    if not (gap.claimed_delta or "").strip():
        return []
    try:
        result = await llm.chat_json([
            {"role": "system", "content": _ATOMIC_CLAIM_SYSTEM},
            {"role": "user", "content": _ATOMIC_CLAIM_USER.format(
                claimed_delta=gap.claimed_delta or "",
                missing_capability=gap.missing_capability or "",
            )},
        ], AtomicClaimList)
        result_claims = list(result.claims)
    except Exception as exc:
        logger.warning("Gap %s: atomic claim decomposition failed (%s); non-fatal",
                       gap.id[:8], exc)
        return []
    for idx, claim in enumerate(result_claims):
        gap_repo.create_atomic_claim(db, task_id, gap.id, idx, claim.claim_text)
    db.flush()
    return gap_repo.list_atomic_claims(db, gap.id)


def _format_atomic_claims(claims) -> str:
    if not claims:
        return "(no atomic claims; judge against the claimed delta directly)"
    return "\n".join(f"- claim[{c.claim_index}]: {c.claim_text}" for c in claims)


def _derive_verdict_from_claims(db, gap: GapCandidate) -> tuple[str, str, list[str]] | None:
    """Derive (audit_result, recommended_action, residual_claim_ids) from claim
    coverage. Returns None when atomic claims are absent (fall back to the LLM's
    decision).

    UNCERTAIN is handled first: a claim with no effective NPA judgment, or one
    judged UNCERTAIN, forces more_search rather than a false confirmed. NONE is
    only "every effective NPA explicitly judged it uncovered". residual only
    subtracts FULL; PARTIAL stays partially-addressed (drives narrowing).
    """
    claims = gap_repo.list_atomic_claims(db, gap.id)
    if not claims:
        return None
    coverages = gap_repo.list_neighbor_claim_coverage(db, gap.id)
    by_claim = defaultdict(list)
    for cov in coverages:
        by_claim[cov.claim_id].append(cov.coverage)

    full: list[str] = []
    partial: list[str] = []
    none: list[str] = []
    uncertain: list[str] = []
    for claim in claims:
        covs = by_claim.get(claim.id, [])
        if not covs:
            uncertain.append(claim.id)          # no judgment -> insufficient evidence
        elif any(c == "FULL" for c in covs):
            full.append(claim.id)
        elif any(c == "PARTIAL" for c in covs):
            partial.append(claim.id)
        elif any(c == "UNCERTAIN" for c in covs):
            uncertain.append(claim.id)
        else:
            none.append(claim.id)               # every NPA explicitly judged NONE

    residual_ids = none + partial + uncertain   # claims not FULL-covered
    if uncertain:
        return ("uncertain", "more_search", residual_ids)
    if not none and not partial:
        return ("closed", "reject", residual_ids)
    if none and not partial:
        return ("confirmed", "continue", residual_ids)
    return ("partially_closed", "narrow", residual_ids)


class AuditDecisionInvalid(ValueError):
    """The model returned an audit decision that violates the audit contract.

    Kept distinct from other errors so that one malformed decision degrades a
    single gap's audit instead of aborting the whole research run.
    """


_AUDIT_SYSTEM = """You are conducting an adversarial research-gap audit.
Your job is to find whether the supplied neighboring papers already close the candidate gap.
Do not infer novelty from absence alone. Return uncertain when the evidence is insufficient.

A gap is:
- confirmed when neighbors do not cover its core claimed delta;
- partially_closed when they cover part of it and a concrete remaining delta exists;
- closed when a neighbor covers the core problem, capability, and evaluation claim;
- uncertain when the supplied evidence cannot decide.

Important: absence of coverage is only evidence of novelty when the neighbors are
actually relevant prior work on the gap's own research problem. If every neighbor
belongs to a different research domain than the gap (for example, the gap is about
systems-level training efficiency but all neighbors are application-domain papers
such as medical segmentation or deepfake detection), then the comparison material
is insufficient — the audit has not actually ruled out prior work. In that case you
MUST return uncertain / more_search and assign a low novelty_confidence, rather than
confirmed. Also set a low similarity_score for such off-domain neighbors.

recommended_action must be exactly one of these four values, and nothing else:
- "continue" for a confirmed gap;
- "narrow" for a partially_closed gap;
- "reject" for a closed gap;
- "more_search" when the decision is uncertain.

For each neighbor, compare only the supplied gap claims and paper content. Do not invent papers or evidence IDs.
evidence_for_gap_ids and evidence_against_gap_ids may only contain IDs copied verbatim
from the gap's supporting/contradicting evidence lists; neighbor paper IDs are not
evidence IDs and must never appear there. Copy every ID character by character.

For each neighbor, also output claim_coverage: for EACH atomic claim (identified by its
claim_index), judge whether that neighbor covers it — FULL (covers the capability
completely), PARTIAL (covers part of it), NONE (clearly does not cover it), or UNCERTAIN
(cannot tell from the supplied content). Give a short rationale. Do not skip any claim.
A claim that no effective neighbor judges is treated as insufficient evidence, NOT as
uncovered — return UNCERTAIN for it rather than NONE when you genuinely cannot tell.

EPISTEMIC CONTRACT (what your numbers mean): novelty_confidence is a WORKFLOW RANKING
HEURISTIC — how confident you are that the CURRENT SEARCH COVERAGE has not missed a
direct hit — NOT a probability that the gap is novel. audit_result "confirmed" means
"survived the current audit", never "novelty proven". To make that basis explicit:
- closest_killer_work: describe the study that, if it existed, would DIRECTLY close
  this gap's claimed delta (concrete method + setting + comparison). This is the
  audit's own falsifier — name it honestly, even when you believe it does not exist.
- killer_query_terms: 2-4 search terms that would find that killer work.
- killer_found: true only if one of the supplied neighbors IS that killer.
- residual_uncertainty: what this audit could not rule out given the retrieval
  coverage you actually saw (e.g. "one abstract-only near neighbor left unread")."""

_AUDIT_USER = """Candidate gap:
- ID: {gap_id}
- Type: {gap_type}
- Setting: {target_setting}
- Observed problem: {observed_problem}
- Existing coverage: {existing_coverage}
- Missing capability: {missing_capability}
- Claimed delta: {claimed_delta}
- Falsification condition: {falsification_condition}

Supporting evidence IDs: {supporting_evidence_ids}
Contradicting evidence IDs: {contradicting_evidence_ids}

Atomic claims (judge each neighbor's claim_coverage against these indices):
{atomic_claims}

Neighbor papers:
{neighbors}

Return the audit decision and comparisons."""


class NeighborAuditSchema(BaseModel):
    paper_id: str
    similarity_score: float = Field(ge=0, le=1)
    shared_problem: str = ""
    shared_mechanism: str = ""
    shared_evaluation: str = ""
    covered_claims: list[str] = Field(default_factory=list)
    uncovered_claims: list[str] = Field(default_factory=list)
    overlap_ratio: float = Field(ge=0, le=1)
    overlap_risk: float = Field(ge=0, le=1)
    why_not_closed: str = ""
    claim_coverage: list[ClaimCoverageSchema] = Field(default_factory=list)


class GapAuditDecisionSchema(BaseModel):
    # Kept as plain strings on purpose: the provider injects this schema into the
    # prompt, so the allowed values are stated here, but an out-of-contract value
    # must reach _validate_audit_decision (which degrades a single gap) instead of
    # raising a ValidationError that would abort the whole run.
    audit_result: str = Field(
        description="One of: confirmed, partially_closed, closed, uncertain")
    recommended_action: str = Field(
        description="One of: continue, narrow, more_search, reject")
    remaining_delta: str = ""
    nearest_neighbor_summary: str = ""
    differentiation_summary: str = ""
    evidence_for_gap_ids: list[str] = Field(default_factory=list)
    evidence_against_gap_ids: list[str] = Field(default_factory=list)
    novelty_confidence: float = Field(ge=0, le=1)
    audit_confidence: float = Field(ge=0, le=1)
    rejection_reason: str = ""
    comparisons: list[NeighborAuditSchema] = Field(default_factory=list)
    # P1.2 (run7 review): the killer question — "what paper, if found, would
    # directly close this gap?" A good novelty audit names its own falsifier;
    # the pipeline then runs one final adversarial search for exactly that.
    closest_killer_work: str = Field(
        default="",
        description="The study that, if found, would directly kill this gap's "
                    "claimed delta (describe it concretely; empty if nothing "
                    "short of a full implementation would).")
    killer_query_terms: list[str] = Field(default_factory=list)
    killer_found: bool = False
    residual_uncertainty: str = Field(
        default="",
        description="What the audit could NOT rule out with the current "
                    "retrieval coverage.")


@dataclass
class GapAuditResult:
    gap_id: str
    audit_result: str
    recommended_action: str
    remaining_delta: str

    def to_phase_payload(self) -> dict:
        return {
            "gap_id": self.gap_id,
            "audit_result": self.audit_result,
            "recommended_action": self.recommended_action,
            "remaining_delta": self.remaining_delta,
        }


@dataclass(frozen=True)
class AdversarialQuerySpec:
    family: str
    query_text: str
    variant_index: int = 0

@dataclass(frozen=True)
class GapSearchAdmission:
    gap_id: str
    status: str
    reason_codes: list[str]
    query_ids: list[str]
    completed_query_ids: list[str]
    failed_query_ids: list[str]
    completed_families: list[str]
    source_count: int
    candidate_paper_ids: list[str]
    external_neighbor_ids: list[str]

VALID_QUERY_FAMILIES = ("exact_gap", "synonym", "mechanism", "benchmark", "method_neighbor")


def build_adversarial_queries(gap: GapCandidate) -> list[AdversarialQuerySpec]:
    """Fallback: raw concatenation across the 5 families (no synonym expansion)."""
    base = " ".join(filter(None, [gap.target_setting, gap.missing_capability, gap.claimed_delta]))
    specs = [
        AdversarialQuerySpec("exact_gap", base),
        AdversarialQuerySpec("synonym", " ".join(filter(None, [gap.missing_capability, "standard terminology prior work"]))),
        AdversarialQuerySpec("mechanism", " ".join(filter(None, [gap.observed_problem, "failure mechanism"]))),
        AdversarialQuerySpec("benchmark", " ".join(filter(None, [gap.target_setting, "benchmark evaluation"]))),
        AdversarialQuerySpec("method_neighbor", " ".join(filter(None, [gap.missing_capability, "method"]))),
    ]
    seen = set()
    return [spec for spec in specs if spec.query_text and not (spec.query_text.lower() in seen or seen.add(spec.query_text.lower()))]


def _cap_query_specs_family_balanced(query_specs: list, cap: int) -> list:
    """Family-balanced query cap (E2E 2026-08-26 fix).

    Sequential truncation let a small cap collapse all executed queries into
    the first one or two families (observed: cap=4 -> 3 exact_gap + 1 synonym,
    or 0 exact_gap + 4 synonym after exact_gap variants were dropped by
    dedup/validation), which made INSUFFICIENT_QUERY_FAMILIES structural and
    left family-internal stability uncomputable. Round-robin across families
    instead, preserving in-family order, so any cap >= 2 retains as many
    distinct families as the generated spec set actually contains.
    """
    if cap >= len(query_specs):
        return list(query_specs)
    by_family: dict[str, list] = {}
    family_order: list[str] = []
    for spec in query_specs:
        fam = spec.family
        if fam not in by_family:
            by_family[fam] = []
            family_order.append(fam)
        by_family[fam].append(spec)
    selected: list = []
    depth = 0
    while len(selected) < cap:
        progressed = False
        for fam in family_order:
            if len(selected) >= cap:
                break
            family_specs = by_family.get(fam, [])
            if depth < len(family_specs):
                selected.append(family_specs[depth])
                progressed = True
        if not progressed:
            break
        depth += 1
    return selected


class IntentWithVariantsSchema(BaseModel):
    family: str = Field(description="One of: exact_gap, synonym, mechanism, benchmark, method_neighbor")
    problem: str = Field(min_length=3)
    mechanism: str = Field(min_length=3)
    intervention: str = Field(min_length=3)
    evaluation_setting: str = Field(min_length=3)
    task_scope: str = Field(min_length=3)
    standard_terms: list[str] = Field(
        default_factory=list, max_length=5,
        description="Synonym family only: the DISTINCT standard terms of art used in the "
                    "literature for this gap's core mechanism/capability, one per variant.")
    variants: list[str] = Field(default_factory=list, min_length=2, max_length=4)


class GapQueryGenList(BaseModel):
    intents: list[IntentWithVariantsSchema] = Field(default_factory=list)


_QUERY_GEN_SYSTEM = """You generate English academic search queries for an adversarial gap audit in TWO
conceptual steps. The goal is to retrieve PRIOR WORK that could already close this gap, but the
queries must be SEMANTICALLY STABLE so that a family-internal stability check is meaningful.

STEP 1 — fix a canonical search INTENT per family. For each family, first pin down these five
structured fields (they are the invariant that every variant must preserve):
- problem: the concrete problem being solved.
- mechanism: the exact failure/working mechanism under investigation.
- intervention: the exact method/technique being evaluated (or the target of evaluation).
- evaluation_setting: the benchmark / task / evaluation protocol.
- task_scope: the application scope (e.g. code generation, memory, QA).

Produce EXACTLY these 5 families:
- exact_gap: the gap's own claimed delta / missing capability.
- synonym: the SAME intent rewritten with STANDARD TERMS OF ART from the literature
  (defends against lexical mismatch — prior work using different terms). Communities
  rename the same concept: a system refusing to answer may be called abstention,
  unanswerability, selective prediction, or calibrated "I don't know"; insufficient
  evidence may be called evidence sufficiency or context sufficiency. FIRST enumerate
  the distinct standard names actually used for THIS gap's core mechanism and
  capability into `standard_terms` (as many as apply, up to 5), THEN produce exactly
  ONE variant per enumerated term, anchoring that variant on its term — never two
  variants on the same term.
- mechanism: anchored on the SAME failure mechanism.
- benchmark: anchored on the SAME evaluation setting / benchmark / task family.
- method_neighbor: anchored on the SAME intervention / method family.

STEP 2 — generate 2-4 query variants PER family. Each variant must be a PARAPHRASE of that
family's canonical intent: ONLY terminology and wording may change. The problem, mechanism,
intervention, evaluation setting, and task scope MUST NOT drift — a "benchmark" family variant
must not become an "efficiency analysis", and a "method_neighbor" family variant must not swap
the intervention for a different method. Different variants differ only in phrasing/terminology.
Use the full budget (4 variants) for the synonym family: more distinct terms recalled is the
whole point of that family.

Rules:
- Output English only.
- Translate concepts into standard English technical vocabulary; do not invent new terms.
- Terminology diversity is the synonym family's purpose: its variants must each anchor
  on a different standard term of art for the same concept — never repeat a term across
  the family's variants, and cover as many distinct standard names as the budget allows."""

_QUERY_GEN_USER = """Candidate gap:
- Setting: {target_setting}
- Observed problem: {observed_problem}
- Missing capability: {missing_capability}
- Claimed delta: {claimed_delta}
- Falsification condition: {falsification_condition}

First fix each family's canonical intent (problem/mechanism/intervention/evaluation_setting/
task_scope), then produce 2-3 paraphrase variants per family that preserve it exactly."""


_VARIANT_REGENERATE_SYSTEM = """You are rewriting ONE academic search query so it is a faithful paraphrase of a fixed
canonical search intent. A previous variant drifted away from the intent (it changed the problem,
mechanism, intervention, evaluation setting, or task scope). Produce a corrected query that ONLY
changes terminology and wording — never the mechanism, method, benchmark, or objective."""

_VARIANT_REGENERATE_USER = """Canonical intent:
- Problem: {problem}
- Mechanism: {mechanism}
- Intervention: {intervention}
- Evaluation setting: {evaluation_setting}
- Task scope: {task_scope}

Drifted variant to replace: {bad_variant}

Return a corrected query that is a faithful paraphrase of the intent above."""


class RegenerateVariantSchema(BaseModel):
    variant: str = Field(min_length=5)


def _intent_canonical_text(intent) -> str:
    return " ".join(filter(None, [intent.problem, intent.mechanism, intent.intervention,
                                  intent.evaluation_setting, intent.task_scope]))


async def _variant_is_invariant(canonical_text: str, variant: str, threshold: float) -> bool:
    """Cheap embedding guardrail: is `variant` still a paraphrase of the intent?

    This is NOT the sole judge — a borderline/drifted variant is fed to the LLM
    for a corrected regeneration. If embedding is unavailable we do not block.
    """
    if not canonical_text or not variant:
        return False
    try:
        from app.services.embedding_service import cosine_similarity, embed_texts
        vecs = await asyncio.to_thread(embed_texts, [canonical_text, variant])
        if len(vecs) == 2 and vecs[0] and vecs[1]:
            return cosine_similarity(vecs[0], vecs[1]) >= threshold
    except Exception as exc:
        logger.warning("variant invariance embedding failed (%s); not blocking", exc)
    return True


async def _regenerate_variant(llm, intent, bad_variant: str) -> str:
    """LLM-directed regeneration of a drifted variant (the non-embedding judge).

    A failed regeneration (e.g. the LLM returns malformed JSON or a schema blob
    instead of a `variant` string) must NOT abort the whole task. It degrades to
    an empty string, which the caller treats as "keep this variant dropped",
    ultimately marking QUERY_GENERATION_INVALID if <2 valid variants remain.
    """
    try:
        result = await llm.chat_json([
            {"role": "system", "content": _VARIANT_REGENERATE_SYSTEM},
            {"role": "user", "content": _VARIANT_REGENERATE_USER.format(
                problem=intent.problem, mechanism=intent.mechanism,
                intervention=intent.intervention, evaluation_setting=intent.evaluation_setting,
                task_scope=intent.task_scope, bad_variant=bad_variant)},
        ], RegenerateVariantSchema)
        return (result.variant or "").strip()
    except Exception as exc:
        logger.warning("variant regeneration failed (non-fatal, dropping variant): %s", exc)
        return ""


async def _validate_and_regenerate_variants(llm, intent, budget: int) -> list[str]:
    """Validate semantic invariance per variant; regenerate drifted ones up to a
    budget, dropping still-drifted variants. Returns the valid variants (>=2
    required, else the caller marks QUERY_GENERATION_INVALID)."""
    from app.config import settings
    canonical_text = _intent_canonical_text(intent)
    threshold = settings.variant_invariance_embedding_threshold
    valid: list[str] = []
    for variant in intent.variants:
        text = (variant or "").strip()
        if not text:
            continue
        accepted = False
        for _ in range(budget + 1):
            if await _variant_is_invariant(canonical_text, text, threshold):
                valid.append(text)
                accepted = True
                break
            regenerated = await _regenerate_variant(llm, intent, text)
            if not regenerated or regenerated == text:
                break
            text = regenerated
        if not accepted:
            logger.info("dropped drifted variant for family '%s' after %d regen attempts",
                        intent.family, budget)
    # De-duplicate within the family (keep first).
    seen = set()
    deduped = []
    for v in valid:
        key = v.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(v)
    return deduped


def _synthesize_family_variants(intent) -> list[str]:
    """Build family queries from the structured intent when the LLM returned
    no usable variants.

    Task 23ec8f20 re-audit (2026-08-27): the two-step generator returned a
    complete canonical intent (problem/mechanism/intervention/... all present)
    but an EMPTY variants list for 4 of 5 families — the schema's
    default_factory=list bypasses min_length validation, so the structured
    fallback never triggers. The structured fields fully determine the
    family's queries (same construction philosophy as
    build_adversarial_queries), so synthesize two field-combination variants
    instead of dropping the family; the verdict-side NPA convergence gate
    still guards retrieval stability.
    """
    combinations = {
        "exact_gap": (
            " ".join(filter(None, [intent.problem, intent.intervention])),
            " ".join(filter(None, [intent.task_scope, intent.problem])),
        ),
        "synonym": (
            " ".join(filter(None, [intent.mechanism, "standard terminology prior work"])),
            " ".join(filter(None, [intent.intervention, "related work terminology"])),
        ),
        "mechanism": (
            " ".join(filter(None, [intent.mechanism, "failure mechanism"])),
            " ".join(filter(None, [intent.problem, "mechanism analysis"])),
        ),
        "benchmark": (
            " ".join(filter(None, [intent.evaluation_setting, "benchmark evaluation"])),
            " ".join(filter(None, [intent.task_scope, "benchmark dataset"])),
        ),
        "method_neighbor": (
            " ".join(filter(None, [intent.intervention, "method"])),
            " ".join(filter(None, [intent.mechanism, "alternative approach"])),
        ),
    }
    seen: set[str] = set()
    out: list[str] = []
    for text in combinations.get(intent.family, ()):
        text = " ".join(text.split())
        if len(text) >= 20 and text.lower() not in seen:
            seen.add(text.lower())
            out.append(text)
    return out


def _cover_standard_terms(intent, variants: list[str]) -> list[str]:
    """Deterministic term-coverage enforcement for the synonym family.

    The LLM enumerates the standard terms of art (``standard_terms``) but does
    not reliably anchor one variant per term — observed on fd688ba6 replays:
    4 terms enumerated (including "unanswerability"), yet 2 of 4 variants
    drifted back to the gap's own wording. An enumerated-but-unused term is a
    guaranteed retrieval blind spot: UAEval4RAG ("Unanswerability Evaluation
    for RAG") was missed exactly because no query ever contained it.

    For every enumerated term no variant covers, append a code-constructed
    query anchoring that term. Sources match on keywords, so coverage beats
    phrasing elegance; construction from the intent's own fields keeps the
    query a faithful paraphrase by definition (it IS the intent plus the term).
    """
    if not intent.standard_terms:
        return variants
    lowered = [v.lower() for v in variants]
    missing = []
    for term in intent.standard_terms:
        t = (term or "").strip()
        if not t:
            continue
        if not any(t.lower() in v for v in lowered):
            missing.append(t)
    if not missing:
        return variants
    scope = (intent.task_scope or "").strip()
    out = list(variants)
    for t in missing:
        out.append(f"{t} {scope}".strip())
    return out


async def generate_english_adversarial_queries(db, llm, gap: GapCandidate) -> list[AdversarialQuerySpec]:
    """Two-step generation: fix each family's canonical intent first (structured
    problem/mechanism/intervention/evaluation_setting/task_scope), then produce
    paraphrase variants that must preserve it. Variants are validated for
    semantic invariance (embedding guardrail + LLM regeneration) so a family
    never silently drifts into a different mechanism/method/benchmark — which is
    what made family-internal stability meaningless (scope drift, not lexical
    mismatch). Falls back to raw concatenation on total failure.
    """
    from app.config import settings
    try:
        gen = await llm.chat_json([
            {"role": "system", "content": _QUERY_GEN_SYSTEM},
            {"role": "user", "content": _QUERY_GEN_USER.format(
                target_setting=gap.target_setting or "",
                observed_problem=gap.observed_problem or "",
                missing_capability=gap.missing_capability or "",
                claimed_delta=gap.claimed_delta or "",
                falsification_condition=gap.falsification_condition or "",
            )},
        ], GapQueryGenList)
        intents = list(gen.intents)
    except Exception as exc:
        logger.warning("Gap %s: English query generation failed (%s), falling back to raw concatenation", gap.id[:8], exc)
        return build_adversarial_queries(gap)

    specs: list[AdversarialQuerySpec] = []
    intents_summary = []
    budget = settings.variant_regenerate_budget
    for intent in intents:
        if intent.family not in VALID_QUERY_FAMILIES:
            continue
        try:
            valid_variants = await _validate_and_regenerate_variants(llm, intent, budget)
        except Exception as exc:
            logger.warning("family '%s': variant validation/regeneration failed "
                           "(non-fatal, marking QUERY_GENERATION_INVALID): %s",
                           intent.family, exc)
            valid_variants = []
        if len(valid_variants) < 2:
            # P1-2 fallback: dropping the whole family is worse than running it
            # unvalidated. Observed on task 9e56a131: 4 of 5 families hit
            # QUERY_GENERATION_INVALID, the gap was left with a single synonym
            # family and failed search admission on INSUFFICIENT_QUERY_FAMILIES
            # before any search even ran. The raw LLM variants still encode the
            # canonical intent (the generator received the structured intent),
            # so run them flagged LOW_CONFIDENCE_VARIANTS — the verdict-side NPA
            # convergence gate still guards against unstable retrieval.
            raw_variants = [v.strip() for v in (intent.variants or []) if v and v.strip()]
            if len(raw_variants) >= 2:
                valid_variants = raw_variants[:3]
                intents_summary.append({
                    "family": intent.family, "status": "LOW_CONFIDENCE_VARIANTS",
                    "valid_variant_count": len(valid_variants),
                    "raw_variant_count": len(raw_variants),
                    "standard_terms": intent.standard_terms,
                    "problem": intent.problem, "mechanism": intent.mechanism,
                    "intervention": intent.intervention,
                    "evaluation_setting": intent.evaluation_setting,
                    "task_scope": intent.task_scope,
                })
            else:
                # Structured-intent fallback (task 23ec8f20 re-audit): the
                # generator returned a complete canonical intent but NO usable
                # variants for 4/5 families, and the gap failed admission on
                # INSUFFICIENT_QUERY_FAMILIES before any search ran — twice.
                synthesized = _synthesize_family_variants(intent)
                if synthesized:
                    valid_variants = synthesized
                    intents_summary.append({
                        "family": intent.family, "status": "SYNTHESIZED_VARIANTS",
                        "valid_variant_count": len(valid_variants),
                        "standard_terms": intent.standard_terms,
                        "problem": intent.problem, "mechanism": intent.mechanism,
                        "intervention": intent.intervention,
                        "evaluation_setting": intent.evaluation_setting,
                        "task_scope": intent.task_scope,
                    })
                else:
                    # QUERY_GENERATION_INVALID: cannot compute a meaningful family-internal
                    # stability, and must NOT be read as SEARCH_UNSTABLE.
                    intents_summary.append({
                        "family": intent.family, "status": "QUERY_GENERATION_INVALID",
                        "valid_variant_count": len(valid_variants),
                        "standard_terms": intent.standard_terms,
                        "problem": intent.problem, "mechanism": intent.mechanism,
                        "intervention": intent.intervention,
                        "evaluation_setting": intent.evaluation_setting,
                        "task_scope": intent.task_scope,
                    })
                    continue
        for vi, variant in enumerate(valid_variants):
            specs.append(AdversarialQuerySpec(intent.family, variant, variant_index=vi))
        constructed = (valid_variants if intent.family != "synonym"
                       else _cover_standard_terms(intent, valid_variants))
        for vi, variant in enumerate(constructed[len(valid_variants):], start=len(valid_variants)):
            specs.append(AdversarialQuerySpec(intent.family, variant, variant_index=vi))
        intents_summary.append({
            "family": intent.family, "status": "VALID",
            "variant_count": len(valid_variants),
            "standard_terms": intent.standard_terms,
            "constructed_term_queries": constructed[len(valid_variants):],
            "problem": intent.problem, "mechanism": intent.mechanism,
            "intervention": intent.intervention,
            "evaluation_setting": intent.evaluation_setting,
            "task_scope": intent.task_scope,
        })

    # Structured persistence of the canonical intents (for diagnosis/calibration).
    paper_repo.save_trace(db, gap.task_id, "gap_query_intents", "generation",
                          output_data={"gap_id": gap.id, "intents": intents_summary})
    # QUERY_GENERATION_INVALID (or no valid family): return empty, do NOT fall back
    # to raw concatenation — raw concatenation is exactly the scope-drift source
    # that made family-internal stability meaningless. The caller treats an empty
    # query set as "cannot search" (more_search), never as SEARCH_UNSTABLE.
    return specs


def collect_same_question_neighbors(db, gap) -> list[str]:
    """Papers already in the corpus that speak to the same research questions.

    The adversarial search decides whether due diligence was done; it should not
    also be the only supply of comparison material. Rounds of searching already
    put the most relevant papers into this task's corpus, and the papers behind
    the evidence for a gap's own questions are, by construction, existing work on
    that exact question — precisely what "has this been done already?" needs.

    Observed on a real run: three of four gaps were stuck at
    INSUFFICIENT_GAP_SPECIFIC_PAPERS with source_count=1 and only two candidate
    papers, because the academic sources were rate limited at the IP level, while
    112 papers and 375 evidence units sat unused in the corpus.

    A gap's own supporting papers are excluded: a claim cannot be checked against
    the very evidence it was derived from.
    """
    question_ids = json.loads(gap.question_ids_json or "[]")
    if not question_ids:
        return []
    support_papers = {
        db.get(EvidenceUnit, link.evidence_id).paper_id
        for link in gap_repo.list_gap_evidence(db, gap.id)
        if db.get(EvidenceUnit, link.evidence_id)
    }
    links = db.query(QuestionEvidenceLink).filter(
        QuestionEvidenceLink.question_id.in_(question_ids)).all()
    if not links:
        return []
    units = db.query(EvidenceUnit).filter(
        EvidenceUnit.id.in_({link.evidence_id for link in links})).all()
    return sorted({
        unit.paper_id for unit in units
        if unit.paper_id and unit.paper_id not in support_papers
    })


def evaluate_gap_search_admission(db, gap, query_ids):
    from app.config import settings
    constrained = settings.constrained_retrieval_mode
    min_completed = (settings.gap_admission_min_completed_queries_constrained
                     if constrained else settings.gap_admission_min_completed_queries)
    min_families = (settings.gap_admission_min_query_families_constrained
                    if constrained else settings.gap_admission_min_query_families)
    min_papers = (settings.gap_admission_min_gap_papers_constrained
                  if constrained else settings.gap_admission_min_gap_papers)

    queries = db.query(SearchQueryRecord).filter(
        SearchQueryRecord.id.in_(query_ids),
        SearchQueryRecord.target_gap_id == gap.id,
        SearchQueryRecord.search_policy_version == GAP_SEARCH_POLICY_VERSION,
    ).all()
    reasons = []
    if not queries:
        return GapSearchAdmission(gap.id, "UNKNOWN", ["NO_GAP_QUERIES"], [], [], [], [], 0, [], [])
    completed = [item for item in queries if item.status == "completed"]
    failed = [item for item in queries if item.status == "failed"]
    families = sorted({item.query_family for item in completed if item.query_family})
    # Search-quality gates stay strict: they attest that due diligence happened,
    # and no local corpus can substitute for that.
    if len(completed) < min_completed: reasons.append("INSUFFICIENT_COMPLETED_QUERIES")
    if len(families) < min_families: reasons.append("INSUFFICIENT_QUERY_FAMILIES")
    if len(completed) / max(len(queries), 1) < 0.5: reasons.append("SEARCH_SUCCESS_RATE_TOO_LOW")
    mappings = db.query(SearchQueryPaper).filter(SearchQueryPaper.query_id.in_([item.id for item in completed])).all()
    retrieved_paper_ids = {item.paper_id for item in mappings}
    sources = {item.source for item in mappings if item.source and item.source != "unknown"}
    if not sources: reasons.append("NO_SUCCESSFUL_SOURCE")
    # Only the *amount* of comparison material may be topped up from the corpus.
    corpus_paper_ids = collect_same_question_neighbors(db, gap)
    paper_ids = sorted({paper_id for paper_id in retrieved_paper_ids | set(corpus_paper_ids)
                        if db.get(Paper, paper_id) is not None})
    if len(paper_ids) < min_papers: reasons.append("INSUFFICIENT_GAP_SPECIFIC_PAPERS")
    support_papers = {db.get(EvidenceUnit, link.evidence_id).paper_id for link in gap_repo.list_gap_evidence(db, gap.id) if link.relation_type == "suggests" and db.get(EvidenceUnit, link.evidence_id)}
    external = [item for item in paper_ids if item not in support_papers]
    if not external: reasons.append("NO_EXTERNAL_NEIGHBOR")
    if constrained and reasons:
        logger.info("Gap %s admission evaluated under constrained retrieval mode "
                    "(relaxed thresholds: completed>=%d, families>=%d, papers>=%d)",
                    gap.id[:8], min_completed, min_families, min_papers)
    return GapSearchAdmission(gap.id, "PASS" if not reasons else "UNKNOWN", reasons, [item.id for item in queries], [item.id for item in completed], [item.id for item in failed], families, len(sources), paper_ids, external)


def _cap_gap_candidate_papers(db, paper_ids: list[str], limit: int) -> list[str]:
    if len(paper_ids) <= limit:
        return list(paper_ids)
    rows = db.query(TaskPaper).filter(TaskPaper.paper_id.in_(paper_ids)).all()
    score_by_paper: dict[str, tuple[float, int]] = {}
    for row in rows:
        value = (
            float(row.final_score if row.final_score is not None else -1.0),
            int(row.discovered_round or 0),
        )
        if value > score_by_paper.get(row.paper_id, (-1.0, 0)):
            score_by_paper[row.paper_id] = value
    return sorted(
        paper_ids,
        key=lambda paper_id: score_by_paper.get(paper_id, (-1.0, 0)),
        reverse=True,
    )[:limit]


def select_gap_specific_neighbors(db, gap, query_ids, limit=_MAX_NEIGHBORS,
                                  candidate_paper_ids: list[str] | None = None):
    allowed_ids = set(candidate_paper_ids) if candidate_paper_ids is not None else None
    mappings = db.query(SearchQueryPaper).filter(SearchQueryPaper.query_id.in_(query_ids)).all()
    query_family = {item.id: item.query_family for item in db.query(SearchQueryRecord).filter(SearchQueryRecord.id.in_(query_ids)).all()}
    stats = defaultdict(lambda: {"hits": 0, "families": set(), "sources": set(), "best_rank": 10**6})
    for mapping in mappings:
        if allowed_ids is not None and mapping.paper_id not in allowed_ids:
            continue
        item = stats[mapping.paper_id]
        item["hits"] += 1
        item["families"].add(query_family.get(mapping.query_id, ""))
        item["sources"].add(mapping.source)
        item["best_rank"] = min(item["best_rank"], mapping.rank)
    ranked = []
    for paper_id, item in stats.items():
        paper = db.get(Paper, paper_id)
        if not paper: continue
        tp = db.query(TaskPaper).filter(TaskPaper.task_id == gap.task_id, TaskPaper.paper_id == paper_id).first()
        score = 0.4 * item["hits"] + 0.3 * len(item["families"]) + 0.2 / (item["best_rank"] + 1) + 0.1 * len(item["sources"]) + 0.1 * (tp.final_score or 0 if tp else 0)
        ranked.append((score, paper))
    # 0027: gap-specific relevance gate (two-stage). Cheap title/abstract gap
    # relevance is the PRIMARY screening signal: direct prior art must not be
    # out-ranked by a broad survey that merely gets hit by many queries. We keep
    # the legacy formula untouched but restrict it to the gap-relevance Top-M,
    # so a generic survey cannot crowd the NPA pool any more. Papers without a
    # score (never screened) keep their legacy rank rather than being dropped.
    if ranked:
        from app.config import settings
        rel_map = {row.paper_id: row.relevance_score
                   for row in gap_repo.list_gap_paper_relevance(db, gap.id)}
        # Gate only activates when at least one candidate was actually scored;
        # otherwise this degrades to the legacy formula untouched (e.g. when
        # the screening LLM call failed or the score is disabled).
        if rel_map and settings.gap_relevance_screen_top_m > 0:
            m = settings.gap_relevance_screen_top_m
            # PRIMARY: gap-specific relevance (desc) — a directly relevant paper
            # must NOT be crowded out of the NPA pool by a broad survey that
            # merely gets hit by many queries. SECONDARY: legacy score as a
            # tie-break only. Unscored papers sort below any scored one
            # (rel_map -1.0). Top-M bounds how many candidates the deep NPA
            # audit has to look at; the final Top-K keeps this relevance-first
            # order (relevant papers first, legacy rank as tie-break).
            ranked.sort(key=lambda item: (-rel_map.get(item[1].id, -1.0),
                                          -item[0], item[1].id))
            ranked = ranked[:max(limit, m)]
        else:
            ranked.sort(key=lambda item: item[0], reverse=True)
    neighbors = [paper for _, paper in ranked[:limit]]
    if len(neighbors) >= limit:
        return neighbors
    # Admission may pass on corpus papers, so the comparison step has to be able
    # to see them too — otherwise the audit is asked to judge novelty with fewer
    # neighbours than it was admitted on. These rank after retrieved papers
    # because they carry no adversarial-search signal, and are ordered by the
    # task's own relevance score.
    seen = {paper.id for paper in neighbors}
    fallback = []
    for paper_id in collect_same_question_neighbors(db, gap):
        if allowed_ids is not None and paper_id not in allowed_ids:
            continue
        if paper_id in seen or paper_id in stats:
            continue
        paper = db.get(Paper, paper_id)
        if not paper:
            continue
        tp = db.query(TaskPaper).filter(TaskPaper.task_id == gap.task_id,
                                       TaskPaper.paper_id == paper_id).first()
        fallback.append(((tp.final_score or 0.0) if tp else 0.0, paper))
    fallback.sort(key=lambda item: item[0], reverse=True)
    neighbors.extend(paper for _, paper in fallback[:limit - len(neighbors)])
    return neighbors

def _save_search_admission_trace(db, task_id, admission):
    paper_repo.save_trace(db, task_id, "gap_search_admission", "decision", output_data={
        "gap_id": admission.gap_id, "search_policy_version": GAP_SEARCH_POLICY_VERSION,
        "query_ids": admission.query_ids, "query_families": admission.completed_families,
        "completed_query_count": len(admission.completed_query_ids), "failed_query_count": len(admission.failed_query_ids),
        "success_rate": len(admission.completed_query_ids) / max(len(admission.query_ids), 1),
        "source_count": admission.source_count, "candidate_paper_count": len(admission.candidate_paper_ids),
        "external_neighbor_count": len(admission.external_neighbor_ids), "status": admission.status,
        "reason_codes": admission.reason_codes,
    })


def _audit_evidence_delta(db, gap, query_ids: list[str], admission, neighbors: list[Paper]) -> tuple[list[str], dict]:
    """Compute marginal papers and evidence for the current admitted audit."""
    current_query_ids = set(query_ids)
    current_paper_ids = set(admission.candidate_paper_ids)
    current_neighbor_ids = {paper.id for paper in neighbors}
    previous = (db.query(GapAudit)
                .filter(GapAudit.gap_id == gap.id,
                        GapAudit.search_admission_status == "PASS")
                .order_by(GapAudit.created_at.desc()).first())
    previous_query_ids = set(json.loads(previous.search_query_ids_json or "[]")) if previous else set()
    previous_paper_ids = set(json.loads(previous.neighbor_paper_ids_json or "[]")) if previous else set()
    if previous_query_ids:
        previous_paper_ids.update(row.paper_id for row in db.query(SearchQueryPaper).filter(
            SearchQueryPaper.query_id.in_(previous_query_ids)).all())
    new_paper_ids = sorted(current_paper_ids - previous_paper_ids)
    evidence = db.query(EvidenceUnit).filter(
        EvidenceUnit.task_id == gap.task_id,
        EvidenceUnit.paper_id.in_(current_paper_ids or {"__none__"}),
    ).all()
    new_evidence = [item for item in evidence if item.paper_id in set(new_paper_ids)]
    fulltext_evidence = [item for item in evidence if item.verification_status in {"verified", "upgraded"}]
    new_neighbor_ids = sorted(current_neighbor_ids - previous_paper_ids)
    delta = {
        "query_count": len(current_query_ids),
        "completed_query_count": len(admission.completed_query_ids),
        "candidate_paper_count": len(current_paper_ids),
        "neighbor_paper_count": len(current_neighbor_ids),
        "new_paper_count": len(new_paper_ids),
        "new_paper_ids": new_paper_ids,
        "new_neighbor_count": len(new_neighbor_ids),
        "new_neighbor_ids": new_neighbor_ids,
        "evidence_count": len(evidence),
        "new_evidence_count": len(new_evidence),
        "new_evidence_ids": sorted(item.id for item in new_evidence),
        "fulltext_evidence_count": len(fulltext_evidence),
        "previous_audit_id": previous.id if previous else None,
    }
    codes = list(admission.reason_codes or [])
    if current_paper_ids and not new_paper_ids and previous:
        codes.append("NO_NEW_PAPERS")
    if evidence and not new_evidence and previous:
        codes.append("NO_NEW_EVIDENCE")
    if current_paper_ids and not fulltext_evidence:
        codes.append("NO_FULLTEXT_EVIDENCE")
    if admission.status == "PASS" and not current_neighbor_ids:
        codes.append("NO_COMPARABLE_PRIOR_ART")
    return sorted(set(codes)), delta


def _record_audit_timeout(db, task_id: str, gap: GapCandidate, audit_round: int) -> GapAuditResult:
    gap.status = "auditing"
    gap_repo.create_gap_audit(
        db, gap_id=gap.id, task_id=task_id, adversarial_queries=[],
        audit_result="uncertain", neighbor_paper_ids=[], recommended_action="more_search",
        audit_round=audit_round, search_policy_version=GAP_SEARCH_POLICY_VERSION,
        search_admission_status="AUDIT_TIMEOUT", search_admission_reasons=["AUDIT_TIMEOUT"],
        search_query_ids=[], audited_claimed_delta=gap.claimed_delta or "",
        failure_reason_codes=["AUDIT_TIMEOUT"], evidence_delta={
            "query_count": 0, "candidate_paper_count": 0,
            "new_paper_count": 0, "new_evidence_count": 0,
        },
    )
    paper_repo.save_trace(db, task_id, "gap_audit_timeout", "decision", output_data={
        "gap_id": gap.id, "timeout_seconds": settings.gap_audit_timeout_seconds,
    })
    db.commit()
    return GapAuditResult(gap.id, "uncertain", "more_search", "")


async def audit_gap_candidates(
    db,
    state: ResearchState,
    llm,
    task_id: str,
    gap_ids: list[str] | None = None,
    perform_search: bool = True,
) -> list[GapAuditResult]:
    """Audit only current-contract, current-policy gap candidates."""
    query = db.query(GapCandidate).filter(
        GapCandidate.task_id == task_id,
        GapCandidate.contract_id == state.contract_id,
        GapCandidate.mining_policy_version == GAP_MINING_POLICY_VERSION,
        GapCandidate.status.in_(["candidate", "auditing", "audited"]),
    )
    if gap_ids is not None:
        query = query.filter(GapCandidate.id.in_(gap_ids))
    gaps = query.all()
    results = []
    from app.config import settings
    timeout_seconds = max(float(settings.gap_audit_timeout_seconds), 0.001)
    for gap in gaps:
        try:
            results.append(await asyncio.wait_for(
                audit_gap_candidate(db, state, llm, task_id, gap, perform_search),
                timeout=timeout_seconds,
            ))
        except asyncio.TimeoutError:
            db.rollback()
            refreshed_gap = gap_repo.get_gap(db, gap.id)
            results.append(_record_audit_timeout(db, task_id, refreshed_gap, state.current_round + 1))
            logger.warning("Gap %s audit exceeded %.1fs and was capped", gap.id[:8], timeout_seconds)
    # Read the surviving set back from the database rather than from this batch:
    # when only some gaps are re-audited (e.g. after narrowing), gaps confirmed
    # by an earlier batch are still surviving and must not be dropped from the
    # state.
    state.surviving_gap_ids = [gap.id for gap in db.query(GapCandidate).filter(
        GapCandidate.task_id == task_id,
        GapCandidate.contract_id == state.contract_id,
        GapCandidate.mining_policy_version == GAP_MINING_POLICY_VERSION,
        GapCandidate.status == "surviving",
    ).all()]
    task_repo.save_state(db, task_id, state)
    db.commit()
    return results

def _latest_decided_audit(db, gap_id: str) -> GapAudit | None:
    """The most recent audit that actually had comparison material to judge.

    Audits blocked at search admission are excluded: they describe a retrieval
    problem, not a verdict, and their neighbor list has a different meaning.
    """
    return (db.query(GapAudit)
            .filter(GapAudit.gap_id == gap_id,
                    GapAudit.search_admission_status == "PASS",
                    GapAudit.search_policy_version == GAP_SEARCH_POLICY_VERSION)
            .order_by(GapAudit.created_at.desc())
            .first())


def _audit_input_repeats(previous: GapAudit | None, gap: GapCandidate,
                         neighbors: list[Paper]) -> bool:
    """True when re-judging this gap cannot reach a different verdict.

    `more_search` asks for new comparison material. If a fresh adversarial round
    surfaced exactly the same neighbors for exactly the same claim, that request
    has been answered with "there is nothing new" — repeating it replays
    identical queries for an identical verdict.

    An audit recorded before the claim was tracked (`audited_claimed_delta` is
    NULL) is treated as different input on purpose: it cannot be shown that the
    claim is unchanged, and guessing here would close a gap that deserves a
    verdict.
    """
    if previous is None or previous.recommended_action != "more_search":
        return False
    if previous.audited_claimed_delta is None:
        return False
    if previous.audited_claimed_delta != (gap.claimed_delta or ""):
        return False
    return set(json.loads(previous.neighbor_paper_ids_json or "[]")) == {
        paper.id for paper in neighbors}


async def audit_gap_candidate(
    db,
    state: ResearchState,
    llm,
    task_id: str,
    gap: GapCandidate,
    perform_search: bool = True,
) -> GapAuditResult:
    """Run one bounded audit with configurable query, paper and time limits."""
    if gap.task_id != task_id:
        raise ValueError("Gap does not belong to task")

    gap.status = "auditing"
    atomic_claims = await _ensure_atomic_claims(db, llm, gap, task_id)
    audit_round = state.current_round + 1
    query_specs = await generate_english_adversarial_queries(db, llm, gap)
    if not query_specs:
        # QUERY_GENERATION_INVALID: every family's variants drifted and could not
        # be regenerated into a valid set. Cannot search — mark more_search, and
        # do NOT read this as SEARCH_UNSTABLE (the search never ran).
        paper_repo.save_trace(db, task_id, "gap_query_generation_invalid", "decision",
                              output_data={"gap_id": gap.id})
        gap.status = "auditing"
        gap_repo.create_gap_audit(
            db, gap_id=gap.id, task_id=task_id, adversarial_queries=[],
            audit_result="uncertain", neighbor_paper_ids=[],
            recommended_action="more_search", audit_round=audit_round,
            search_policy_version=GAP_SEARCH_POLICY_VERSION,
            search_admission_status="QUERY_GENERATION_INVALID",
            search_admission_reasons=["QUERY_GENERATION_INVALID"],
            search_query_ids=[], audited_claimed_delta=gap.claimed_delta or "",
            failure_reason_codes=["QUERY_GENERATION_INVALID"],
            evidence_delta={"query_count": 0, "candidate_paper_count": 0, "new_paper_count": 0, "new_evidence_count": 0},
        )
        db.commit()
        return GapAuditResult(gap.id, "uncertain", "more_search", "")
    # Query identity is task + round + normalized text + target gap + policy;
    # generated families are descriptive metadata. Deduplicate globally before
    # persistence so equivalent variants from different families cannot create
    # multiple executions for one database identity.
    unique_specs = []
    seen_normalized = set()
    for spec in query_specs:
        normalized = " ".join((spec.query_text or "").split()).casefold()
        if not normalized or normalized in seen_normalized:
            continue
        seen_normalized.add(normalized)
        unique_specs.append(spec)
    query_specs = unique_specs
    query_cap = max(settings.gap_audit_max_queries, 1)
    generated_query_count = len(query_specs)
    if generated_query_count > query_cap:
        query_specs = _cap_query_specs_family_balanced(query_specs, query_cap)
        paper_repo.save_trace(db, task_id, "gap_audit_query_cap", "decision", output_data={
            "gap_id": gap.id,
            "configured_limit": query_cap,
            "generated_count": generated_query_count,
            "executed_count": len(query_specs),
            "skipped_count": generated_query_count - len(query_specs),
            "executed_families": sorted({spec.family for spec in query_specs}),
            "reason_code": "AUDIT_QUERY_CAP_REACHED",
        })
    queries = [spec.query_text for spec in query_specs]
    executions = []
    for spec in query_specs:
        record = save_search_query(
            db, task_id, spec.query_text, f"gap_{spec.family}", None, None, audit_round,
            target_gap_id=gap.id, query_family=spec.family,
            search_policy_version=GAP_SEARCH_POLICY_VERSION,
        )
        executions.append(SearchQueryExecution(
            query_id=record.id, query_text=spec.query_text, intent=f"gap_{spec.family}",
            target_question_id=None, expected_evidence_type=None, target_gap_id=gap.id,
        ))
    db.commit()
    if perform_search and executions:
        try:
            await search_and_save_papers(db, state, executions, task_id, audit_round)
            # Evidence-funnel repair: audit-recalled papers used to stay
            # priority=NULL (main-round scoring never saw them), so they were
            # invisible to extract_evidence (priority in [high, medium]) and
            # downstream phases. Score them under the same policy so the cap
            # ranking and the evidence pipeline both see real priorities.
            if settings.audit_score_retrieved_papers:
                try:
                    from app.agent.steps.score_papers import score_papers
                    await score_papers(db, state, llm, task_id, audit_round)
                except Exception as exc:
                    logger.warning("Gap %s: audit-round paper scoring failed "
                                   "(non-fatal, papers stay unscored): %s",
                                   gap.id[:8], exc)
        except RuntimeError as exc:
            logger.info("Gap %s search admission deferred: %s", gap.id[:8], exc)
    # P0-1: persist search + scoring work as soon as it completes. The audit
    # runs under a hard per-gap timeout whose handler rolls the session back;
    # without this checkpoint a timeout discards the whole adversarial search
    # round (queries, recalled papers, scores) and a re-audit starts from zero.
    # (Task 9e56a131: 7/9 audits timed out with everything lost.)
    db.commit()
    admission = evaluate_gap_search_admission(db, gap, [item.query_id for item in executions])
    candidate_cap = max(settings.gap_audit_max_candidate_papers, 1)
    original_candidate_count = len(admission.candidate_paper_ids)
    if original_candidate_count > candidate_cap:
        capped_ids = _cap_gap_candidate_papers(db, admission.candidate_paper_ids, candidate_cap)
        capped_set = set(capped_ids)
        capped_external = [pid for pid in admission.external_neighbor_ids if pid in capped_set]
        capped_reasons = list(admission.reason_codes) + ["AUDIT_CANDIDATE_CAP_REACHED"]
        constrained = settings.constrained_retrieval_mode
        min_admitted_papers = (
            settings.gap_admission_min_gap_papers_constrained
            if constrained else settings.gap_admission_min_gap_papers
        )
        if len(capped_ids) < min_admitted_papers:
            capped_reasons.append("INSUFFICIENT_GAP_SPECIFIC_PAPERS")
        if not capped_external:
            capped_reasons.append("NO_EXTERNAL_NEIGHBOR")
        blocking_reasons = [code for code in capped_reasons
                            if code != "AUDIT_CANDIDATE_CAP_REACHED"]
        admission = replace(
            admission,
            status="PASS" if not blocking_reasons else "UNKNOWN",
            candidate_paper_ids=capped_ids,
            external_neighbor_ids=capped_external,
            reason_codes=sorted(set(capped_reasons)),
        )
        paper_repo.save_trace(db, task_id, "gap_audit_candidate_cap", "decision", output_data={
            "gap_id": gap.id, "configured_limit": candidate_cap,
            "original_count": original_candidate_count, "capped_count": len(capped_ids),
            "reason_code": "AUDIT_CANDIDATE_CAP_REACHED",
        })
    _save_search_admission_trace(db, task_id, admission)
    if admission.status != "PASS":
        gap.status = "auditing"
        failure_codes, evidence_delta = _audit_evidence_delta(
            db, gap, admission.query_ids, admission, [])
        gap_repo.create_gap_audit(
            db, gap_id=gap.id, task_id=task_id, adversarial_queries=queries,
            audit_result="uncertain", neighbor_paper_ids=[],
            recommended_action="more_search", audit_round=audit_round,
            search_policy_version=GAP_SEARCH_POLICY_VERSION, search_admission_status=admission.status,
            search_admission_reasons=admission.reason_codes, search_query_ids=admission.query_ids,
            audited_claimed_delta=gap.claimed_delta or "",
            failure_reason_codes=failure_codes,
            evidence_delta=evidence_delta,
        )
        db.commit()
        return GapAuditResult(gap.id, "uncertain", "more_search", "")
    # 0027: gap-specific prior-art screening. Score every audit-recalled paper
    # on title+abstract against THIS gap (cheap), persist gap_paper_relevance,
    # and expose the ranked list to the NPA selector. Direct prior art that a
    # broad survey out-ranks on raw query hits must not be invisible to the
    # audit because it never got a gap-specific score. Safe on replay: already
    # scored papers are reused, so this never re-hits the LLM for the same
    # (gap, paper) under the same scoring version.
    try:
        scored = await score_all_gap_candidates(
            db, llm, gap, admission.candidate_paper_ids, task_id)
        if scored:
            paper_repo.save_trace(db, task_id, "gap_paper_relevance_screen", "decision",
                                  output_data={
                                      "gap_id": gap.id,
                                      "scoring_version": settings.gap_relevance_scoring_version,
                                      "top": [
                                          {"paper_id": pid, "gap_relevance": score,
                                           "claim_overlap": getattr(s, "claim_overlap", None),
                                           "problem_overlap": getattr(s, "problem_overlap", None),
                                           "evaluation_overlap": getattr(s, "evaluation_overlap", None)}
                                          for pid, score, s in scored[:20]
                                      ],
                                  })
    except Exception as exc:
        logger.warning("Gap %s: gap-relevance screening failed (non-fatal): %s",
                       gap.id[:8], exc)
    # P0-1: same checkpoint rationale as the search phase — relevance scores are
    # (gap, paper)-cached and reused on re-audit, so they must survive a
    # downstream timeout rollback.
    db.commit()
    neighbors = select_gap_specific_neighbors(
        db, gap, admission.completed_query_ids,
        candidate_paper_ids=admission.candidate_paper_ids,
    )
    # Evidence-funnel repair: neighbors without verified full-text evidence
    # used to make NO_FULLTEXT_EVIDENCE structural (abstract-only comparison
    # material). Extract bounded full-text evidence for the selected neighbors
    # BEFORE the evidence delta is computed, so the audit's fulltext count and
    # the claim-level verdict see real extracted material.
    if settings.audit_neighbor_evidence_extraction and neighbors:
        await _ensure_neighbor_evidence(db, state, llm, task_id, gap, neighbors, audit_round)
    failure_codes, evidence_delta = _audit_evidence_delta(
        db, gap, admission.query_ids, admission, neighbors)
    if not neighbors:
        gap.status = "auditing"
        failure_codes = sorted(set(failure_codes + ["NO_COMPARABLE_PRIOR_ART"]))
        gap_repo.create_gap_audit(
            db, gap_id=gap.id, task_id=task_id, adversarial_queries=queries,
            audit_result="uncertain", neighbor_paper_ids=[],
            recommended_action="more_search", audit_round=audit_round,
            search_policy_version=GAP_SEARCH_POLICY_VERSION,
            search_admission_status=admission.status,
            search_admission_reasons=admission.reason_codes,
            search_query_ids=admission.query_ids,
            audited_claimed_delta=gap.claimed_delta or "",
            failure_reason_codes=failure_codes,
            evidence_delta=evidence_delta,
        )
        paper_repo.save_trace(db, task_id, "gap_audit_no_comparable_prior_art", "decision",
                              output_data={"gap_id": gap.id, "candidate_paper_count": len(admission.candidate_paper_ids),
                                           "neighbor_paper_count": 0})
        db.commit()
        return GapAuditResult(gap.id, "uncertain", "more_search", "")
    # Recall waterfall diagnostic: record per-family raw/canonical/post-filter/
    # final overlap on every admitted audit (not only confirmed gaps), so the
    # instability root cause can be located regardless of the verdict.
    try:
        _waterfall = _compute_overlap_waterfall(db, gap, [paper.id for paper in neighbors])
        paper_repo.save_trace(db, task_id, "gap_overlap_waterfall", "decision",
                              output_data={"gap_id": gap.id, "waterfall": _waterfall})
        # Cumulative NPA Top-K convergence is the stability signal that matters
        # for novelty audit (variants complement each other, so their pairwise
        # Jaccard is NOT the right gate). Also record the final_score
        # distribution so a relevance cutoff can be chosen from data later.
        _cumulative = _compute_cumulative_npa_stability(db, gap, admission.completed_query_ids)
        _score_dist = _compute_final_score_distribution(db, gap, admission.completed_query_ids)
        paper_repo.save_trace(db, task_id, "gap_cumulative_npa_stability", "decision",
                              output_data={"gap_id": gap.id,
                                           "cumulative": _cumulative,
                                           "final_score_distribution": _score_dist})
    except Exception as exc:
        logger.warning("Gap %s: NPA diagnostics failed (non-fatal): %s",
                       gap.id[:8], exc)
    if _audit_input_repeats(_latest_decided_audit(db, gap.id), gap, neighbors):
        # A previous audit asked for more search and got none: same claim, same
        # neighbors. Observed cost of not detecting this: one gap audited four
        # times with an identical query set for an identical "uncertain", about
        # nineteen minutes of a run spent re-deciding a settled question while
        # the external sources were rate-limited. Close it as undecidable so the
        # remaining budget goes to gaps that can still be judged; the gap is
        # reported as unproven rather than silently dropped.
        gap.status = "rejected"
        reason = ("novelty_undecidable: adversarial round "
                  f"{audit_round} surfaced no new comparison material for the same claim")
        gap_repo.create_gap_audit(
            db, gap_id=gap.id, task_id=task_id, adversarial_queries=queries,
            audit_result="uncertain", neighbor_paper_ids=[paper.id for paper in neighbors],
            recommended_action="reject", rejection_reason=reason, audit_round=audit_round,
            search_policy_version=GAP_SEARCH_POLICY_VERSION, search_admission_status=admission.status,
            search_admission_reasons=admission.reason_codes, search_query_ids=admission.query_ids,
            audited_claimed_delta=gap.claimed_delta or "",
            failure_reason_codes=sorted(set(failure_codes + ["NO_NEW_COMPARISON_MATERIAL"])),
            evidence_delta=evidence_delta,
        )
        paper_repo.save_trace(db, task_id, "gap_audit_undecidable", "decision", output_data={
            "gap_id": gap.id,
            "neighbor_count": len(neighbors),
            "reason": reason,
        })
        logger.info("Gap %s: closing as undecidable — no new comparison material "
                    "after a repeated adversarial round", gap.id[:8])
        db.commit()
        return GapAuditResult(gap.id, "uncertain", "reject", "")
    decision = await llm.chat_json([
        {"role": "system", "content": _AUDIT_SYSTEM},
        {"role": "user", "content": _AUDIT_USER.format(
            gap_id=gap.id, gap_type=gap.gap_type, target_setting=gap.target_setting or "(not specified)",
            observed_problem=gap.observed_problem or "(not specified)",
            existing_coverage=gap.existing_coverage or "(not specified)",
            missing_capability=gap.missing_capability or "(not specified)",
            claimed_delta=gap.claimed_delta or "(not specified)",
            falsification_condition=gap.falsification_condition or "(not specified)",
            supporting_evidence_ids=_gap_evidence_ids(db, gap.id, "suggests"),
            contradicting_evidence_ids=_gap_evidence_ids(db, gap.id, "contradicts"),
            atomic_claims=_format_atomic_claims(atomic_claims),
            neighbors=_format_neighbors(db, neighbors),
        )},
    ], GapAuditDecisionSchema)

    try:
        dropped_evidence_ids, dropped_comparisons = _validate_audit_decision(decision, gap, neighbors, db)
    except AuditDecisionInvalid as err:
        # A malformed decision is a failure of this one audit, not of the run.
        # Raising here used to abort the whole task and discard every gap,
        # every piece of evidence and the entire report. Degrade this gap to
        # uncertain (the same outcome as a blocked search admission) and record
        # what was wrong, so the remaining gaps still get audited and a later
        # round can retry.
        logger.warning("Gap %s: rejecting malformed audit decision (%s)", gap.id[:8], err)
        gap.status = "auditing"
        gap_repo.create_gap_audit(
            db, gap_id=gap.id, task_id=task_id, adversarial_queries=queries,
            audit_result="uncertain", neighbor_paper_ids=[paper.id for paper in neighbors],
            recommended_action="more_search", audit_round=audit_round,
            rejection_reason=f"invalid_audit_decision: {err}",
            search_policy_version=GAP_SEARCH_POLICY_VERSION,
            search_admission_status=admission.status,
            search_admission_reasons=admission.reason_codes, search_query_ids=admission.query_ids,
            audited_claimed_delta=gap.claimed_delta or "",
            failure_reason_codes=failure_codes + ["INVALID_AUDIT_DECISION"],
            evidence_delta=evidence_delta,
        )
        paper_repo.save_trace(db, task_id, "gap_audit_invalid_decision", "decision", output_data={
            "gap_id": gap.id, "error": str(err),
            "audit_result": decision.audit_result,
            "recommended_action": decision.recommended_action,
        })
        db.commit()
        return GapAuditResult(gap.id, "uncertain", "more_search", "")
    if dropped_comparisons:
        logger.warning("Gap %s: dropped %d comparison(s) citing unknown neighbor "
                       "paper(s): %s", gap.id[:8], len(dropped_comparisons),
                       dropped_comparisons)
        paper_repo.save_trace(db, task_id, "gap_audit_dropped_comparisons", "decision",
                              output_data={"gap_id": gap.id,
                                           "dropped_comparison_paper_ids": dropped_comparisons,
                                           "audit_result": decision.audit_result})
    if dropped_evidence_ids:
        logger.warning("Gap %s: dropped %d unusable evidence ID(s) from the audit "
                       "decision: %s", gap.id[:8], len(dropped_evidence_ids),
                       dropped_evidence_ids)
        paper_repo.save_trace(db, task_id, "gap_audit_dropped_evidence_ids", "decision",
                              output_data={"gap_id": gap.id,
                                           "dropped_evidence_ids": dropped_evidence_ids,
                                           "audit_result": decision.audit_result})
    # Relevance guard (v4): a confirmed verdict is only meaningful when the
    # neighbors are actual prior work on the gap's own problem. If every
    # neighbor is semantically distant (low similarity) yet the model still
    # claims high novelty, the audit almost certainly failed to retrieve the
    # real comparison material. Downgrade to uncertain so a later round retries
    # with better queries instead of stamping a false "novel".
    if (decision.audit_result == "confirmed"
            and decision.novelty_confidence > _HIGH_NOVELTY
            and decision.comparisons
            and max((c.similarity_score for c in decision.comparisons), default=1.0) < _MIN_NEIGHBOR_SIMILARITY):
        max_sim = max(c.similarity_score for c in decision.comparisons)
        original_novelty = decision.novelty_confidence
        decision.audit_result = "uncertain"
        decision.recommended_action = "more_search"
        decision.novelty_confidence = min(original_novelty, 0.5)
        decision.rejection_reason = (
            "neighbor semantic relevance too low to rule out prior work "
            f"(max similarity {max_sim:.2f}); retrieval may have missed mechanism-level work"
        )
        paper_repo.save_trace(db, task_id, "gap_audit_low_relevance_downgrade", "decision",
                              output_data={
                                  "gap_id": gap.id,
                                  "max_neighbor_similarity": max_sim,
                                  "original_novelty_confidence": original_novelty,
                                  "downgraded_to": "uncertain",
                              })
        logger.warning("Gap %s: downgrading confirmed->uncertain (max neighbor "
                       "similarity %.2f below %.2f)", gap.id[:8], max_sim,
                       _MIN_NEIGHBOR_SIMILARITY)
    # P1.2 killer search (run7 review): a confirmed verdict is a
    # SURVIVED_CURRENT_AUDIT statement — the audit names the paper that would
    # kill it, and the pipeline spends one final adversarial search looking
    # for exactly that. A strong match degrades the verdict instead of letting
    # it survive on a retrieval blind spot.
    killer_work = await _run_killer_search(
        db, state, task_id, gap, decision, neighbors, audit_round,
        perform_search=perform_search)
    if killer_work.get("killer_hits"):
        failure_codes.append("KILLER_WORK_FOUND")
        decision.audit_result = "uncertain"
        decision.recommended_action = "more_search"
        decision.novelty_confidence = min(decision.novelty_confidence, 0.5)
        decision.rejection_reason = (
            "final killer search recalled a strongly matching paper for the "
            "audit's own killer description: "
            + "; ".join(h["paper_id"] for h in killer_work["killer_hits"][:3])
        )
        paper_repo.save_trace(db, task_id, "gap_audit_killer_found", "decision",
                              output_data={
                                  "gap_id": gap.id,
                                  "hits": killer_work["killer_hits"],
                                  "original_verdict": "confirmed",
                                  "downgraded_to": "uncertain",
                              })
        logger.warning("Gap %s: killer search hit %d paper(s); downgrading "
                       "confirmed->uncertain", gap.id[:8],
                       len(killer_work["killer_hits"]))
    for comparison in decision.comparisons:
        gap_repo.create_neighbor_comparison(
            db,
            gap_id=gap.id,
            paper_id=comparison.paper_id,
            task_id=task_id,
            similarity_score=comparison.similarity_score,
            shared_problem=comparison.shared_problem,
            shared_mechanism=comparison.shared_mechanism,
            shared_evaluation=comparison.shared_evaluation,
            covered_claims=comparison.covered_claims,
            uncovered_claims=comparison.uncovered_claims,
            overlap_ratio=comparison.overlap_ratio,
            overlap_risk=comparison.overlap_risk,
            why_not_closed=comparison.why_not_closed or None,
        )
        # P1-1: record per-claim coverage (FULL/PARTIAL/NONE/UNCERTAIN).
        for cov in comparison.claim_coverage:
            if 0 <= cov.claim_index < len(atomic_claims):
                gap_repo.create_neighbor_claim_coverage(
                    db, task_id=task_id, gap_id=gap.id,
                    neighbor_paper_id=comparison.paper_id,
                    claim_id=atomic_claims[cov.claim_index].id,
                    coverage=cov.coverage.upper(), rationale=cov.rationale,
                )

    # P1-1: derive the verdict from claim-level coverage when atomic claims are
    # present — grounding confirmed/narrow/closed in per-claim judgments that
    # handle UNCERTAIN first, instead of the LLM's free-form audit_result.
    claim_verdict = _derive_verdict_from_claims(db, gap)
    if claim_verdict is not None:
        derived_result, derived_action, residual_ids = claim_verdict
        # A model can correctly identify a concrete partial remaining delta while
        # omitting one per-claim coverage row. Do not promote it to confirmed;
        # retain only the safe partial/narrow path so the explicit delta gets a
        # fresh audit. This prevents an output-format omission from turning a
        # usable narrowing signal into an expensive blind remediation search.
        if (derived_result == "uncertain"
                and decision.audit_result == "partially_closed"
                and decision.recommended_action == "narrow"
                and (decision.remaining_delta or "").strip()
                and residual_ids):
            derived_result = "partially_closed"
            derived_action = "narrow"
            failure_codes.append("CLAIM_COVERAGE_INCOMPLETE_RETAINED_FOR_NARROWING")
            paper_repo.save_trace(
                db, task_id, "gap_audit_partial_salvage", "decision",
                output_data={
                    "gap_id": gap.id,
                    "residual_claim_ids": residual_ids,
                    "remaining_delta": decision.remaining_delta,
                    "reason_code": "CLAIM_COVERAGE_INCOMPLETE_RETAINED_FOR_NARROWING",
                })
        decision.audit_result = derived_result
        decision.recommended_action = derived_action
        gap.residual_claim_ids_json = json.dumps(residual_ids)
        if residual_ids:
            id_set = set(residual_ids)
            gap.residual_gap = " ".join(
                c.claim_text for c in atomic_claims if c.id in id_set)

        # P1-1 blind-spot fix: a claim-level "confirmed" is not trustworthy when
        # the multi-query union has NOT converged — i.e. adding more variants
        # still changes the global NPA Top-K, so the audit may have missed the
        # real prior art. Downgrade to more_search, bounded by a budget so a
        # persistently non-converging union cannot loop forever. We deliberately
        # do NOT gate on family-internal Jaccard: variants complement each
        # other, so low pairwise overlap is expected and uninformative.
        # E2E 2026-08-26: convergence is None means fewer than two completed
        # queries, so union stability is UNMEASURED. Treating unmeasured as
        # pass-through let a single-query audit (constrained mode) promote a
        # confirmed verdict with no convergence evidence at all. Both
        # "measured below threshold" and "unmeasured" now downgrade, still
        # bounded by the more_search budget.
        if derived_result == "confirmed":
            diagnostics = _compute_npa_diagnostics(db, gap)
            convergence_unstable = (
                diagnostics.cumulative_convergence is None
                or diagnostics.cumulative_convergence < settings.npa_stability_medium
            )
            if convergence_unstable:
                more_search_count = sum(
                    1 for a in gap_repo.list_gap_audits(db, gap.id)
                    if a.recommended_action == "more_search")
                if more_search_count < settings.family_instability_more_search_budget:
                    decision.audit_result = "uncertain"
                    decision.recommended_action = "more_search"
                    failure_codes.append(
                        "NPA_UNMEASURED" if diagnostics.cumulative_convergence is None
                        else "NPA_UNCONVERGED")
                    paper_repo.save_trace(
                        db, task_id, "gap_audit_npa_unconverged", "decision",
                        output_data={
                            "gap_id": gap.id,
                            "cumulative_convergence": diagnostics.cumulative_convergence,
                            "instable_families": diagnostics.instable_families,
                            "median_family_stability": diagnostics.median_family_stability,
                        })

    action = decision.recommended_action
    if action == "continue" and decision.audit_result == "confirmed":
        # E2E 2026-08-26: text-verdict consistency guard. A confirmed/continue
        # decision whose own remaining_delta concludes "the decision is
        # uncertain" is the model contradicting its structured fields; promoting
        # it to surviving stamps a verdict the audit text itself does not
        # support (observed verbatim on the surviving gap of run 2026-08-26_b).
        # Runs AFTER claim-verdict derivation and the NPA gate so it guards the
        # final decision regardless of which path produced "confirmed".
        if _AUDIT_TEXT_CONFLICT_RE.search(decision.remaining_delta or ""):
            decision.audit_result = "uncertain"
            decision.recommended_action = "more_search"
            action = "more_search"
            gap.status = "auditing"
            decision.rejection_reason = (
                "audit text-verdict conflict: remaining_delta concludes the "
                "decision is uncertain while structured fields said confirmed/continue"
            )
            failure_codes.append("AUDIT_TEXT_VERDICT_CONFLICT")
            paper_repo.save_trace(db, task_id, "gap_audit_text_verdict_conflict",
                                  "decision", output_data={"gap_id": gap.id})
            logger.warning("Gap %s: downgrading confirmed->uncertain (remaining_delta "
                           "text contradicts the structured verdict)", gap.id[:8])
        elif (decision.novelty_confidence is not None
              and decision.novelty_confidence <= _LOW_NOVELTY_CONFIRMED):
            # P0-1b (task d6f64087): confirmed but the auditor itself reports
            # novelty_confidence <= 0.4 — "no neighbor covered the claim" by
            # search absence, not by evidence. Promoting this to surviving
            # lets a hollow-novelty gap reach tier-A interventions (observed:
            # surviving gap confirmed at 0.3 → 3 tier-A interventions →
            # experiment plans built on an unconfirmed premise). Send it back
            # for retrieval instead.
            original_novelty = decision.novelty_confidence
            decision.audit_result = "uncertain"
            decision.recommended_action = "more_search"
            action = "more_search"
            gap.status = "auditing"
            decision.rejection_reason = (
                f"confirmed verdict carries novelty_confidence={original_novelty:.2f} "
                f"(<= {_LOW_NOVELTY_CONFIRMED}): the audit's own search was too weak "
                f"to establish novelty; retry with better retrieval"
            )
            failure_codes.append("LOW_NOVELTY_CONFIDENCE_CONFIRMED")
            paper_repo.save_trace(db, task_id, "gap_audit_low_novelty_downgrade",
                                  "decision", output_data={
                                      "gap_id": gap.id,
                                      "novelty_confidence": original_novelty,
                                      "downgraded_to": "uncertain",
                                  })
            logger.warning("Gap %s: downgrading confirmed->uncertain "
                           "(novelty_confidence %.2f <= %.2f)",
                           gap.id[:8], original_novelty, _LOW_NOVELTY_CONFIRMED)
        else:
            gap.status = "surviving"
            _record_nearest_prior_art(db, gap, decision)
    elif action == "reject" or decision.audit_result == "closed":
        gap.status = "rejected"
        action = "reject"
    elif action == "narrow" or decision.audit_result == "partially_closed":
        gap.status = "audited"
        action = "narrow"
    else:
        gap.status = "auditing"
        action = "more_search"

    # P1.2 search coverage: the mechanical facts the verdict rests on. The
    # verdict is a SURVIVED_CURRENT_AUDIT statement — this snapshot makes its
    # retrieval basis explicit instead of letting a bare 0.85 read as
    # "probability of novelty".
    search_coverage = {
        "queries_executed": len(queries),
        "query_families": sorted({spec.family for spec in query_specs}),
        "closest_neighbors_reviewed": len(neighbors),
        "fulltext_neighbors_reviewed": (
            min(len(neighbors), settings.audit_neighbor_evidence_max_papers)
            if settings.audit_neighbor_evidence_extraction else 0),
        "candidate_pool_size": int(original_candidate_count or 0),
        "search_admission_status": admission.status,
    }
    gap_repo.create_gap_audit(
        db,
        gap_id=gap.id,
        task_id=task_id,
        adversarial_queries=queries,
        audit_result=decision.audit_result,
        nearest_neighbor_summary=decision.nearest_neighbor_summary,
        differentiation_summary=decision.differentiation_summary,
        neighbor_paper_ids=[paper.id for paper in neighbors],
        evidence_for_gap=decision.evidence_for_gap_ids,
        evidence_against_gap=decision.evidence_against_gap_ids,
        remaining_delta=decision.remaining_delta,
        novelty_confidence=decision.novelty_confidence,
        audit_confidence=decision.audit_confidence,
        recommended_action=action,
        rejection_reason=decision.rejection_reason or None,
        audit_round=audit_round,
        search_policy_version=GAP_SEARCH_POLICY_VERSION,
        search_admission_status=admission.status,
        search_admission_reasons=admission.reason_codes,
        search_query_ids=admission.query_ids,
        audited_claimed_delta=gap.claimed_delta or "",
        failure_reason_codes=failure_codes,
        evidence_delta=evidence_delta,
        killer_work=killer_work,
        search_coverage=search_coverage,
    )
    paper_repo.save_trace(db, task_id, "audit_gap_candidate", "decision", output_data={
        "gap_id": gap.id,
        "audit_result": decision.audit_result,
        "recommended_action": action,
        "neighbor_count": len(neighbors),
        "query_count": len(queries),
        "search_coverage": search_coverage,
        "killer_work": killer_work,
    })
    db.commit()
    return GapAuditResult(gap.id, decision.audit_result, action, decision.remaining_delta)


def _select_neighbors(db, task_id: str) -> list[Paper]:
    return db.query(Paper).join(TaskPaper).filter(
        TaskPaper.task_id == task_id,
    ).order_by(
        TaskPaper.final_score.desc().nullslast(),
        Paper.citation_count.desc().nullslast(),
    ).limit(_MAX_NEIGHBORS).all()


_EVIDENCE_TYPE_AUDIT_ORDER = {
    "limitation": 0, "future_work": 1, "negative_result": 2,
    "comparison": 3, "result": 4, "method": 5,
}


def _neighbor_verified_evidence(db, paper_id: str, limit: int = 2) -> list[EvidenceUnit]:
    """Bounded full-text evidence excerpts per neighbor for the audit prompt.

    Limitation-type signals come first: they are the material a novelty audit
    actually needs to decide claim coverage. The bound (2 per neighbor) keeps
    the prompt injection small regardless of how much evidence exists.
    """
    units = db.query(EvidenceUnit).filter(
        EvidenceUnit.paper_id == paper_id,
        EvidenceUnit.verification_status.in_(("verified", "upgraded")),
    ).limit(12).all()
    units.sort(key=lambda u: _EVIDENCE_TYPE_AUDIT_ORDER.get(u.evidence_type or "", 9))
    return units[:limit]


async def _run_killer_search(db, state, task_id, gap, decision, neighbors,
                             audit_round, perform_search: bool = True) -> dict:
    """One final adversarial search for the audit's own named killer (P1.2).

    The verdict names the paper that would kill the gap; searching for exactly
    that is the cheapest novelty guard there is. Freshly recalled papers that
    embed close to the killer description count as hits (the caller degrades
    the verdict). Degraded infrastructure returns the record without hits —
    an outage is not evidence that the killer exists.
    """
    terms = [t.strip() for t in (decision.killer_query_terms or []) if t.strip()]
    record = {
        "description": decision.closest_killer_work or "",
        "query_terms": terms,
        "found": bool(decision.killer_found),
        "residual_uncertainty": decision.residual_uncertainty or "",
    }
    if (decision.killer_found or not terms
            or not (decision.closest_killer_work or "").strip()
            or decision.audit_result != "confirmed"):
        return record
    if not perform_search:
        record["skipped"] = "perform_search_disabled"
        return record
    executions = []
    for term in terms[:4]:
        query_record = save_search_query(
            db, task_id, term, "gap_killer", None, None, audit_round,
            target_gap_id=gap.id, query_family="killer",
            search_policy_version=GAP_SEARCH_POLICY_VERSION,
        )
        executions.append(SearchQueryExecution(
            query_id=query_record.id, query_text=term, intent="gap_killer",
            target_question_id=None, expected_evidence_type=None,
            target_gap_id=gap.id,
        ))
    db.commit()
    try:
        _, _, new_paper_ids = await search_and_save_papers(
            db, state, executions, task_id, audit_round)
    except Exception as exc:
        logger.warning("Gap %s: killer search degraded: %s", gap.id[:8], exc)
        record["search_degraded"] = True
        return record
    record["retrieved_new_paper_count"] = len(new_paper_ids)
    neighbor_ids = {p.id for p in neighbors}
    fresh_ids = [pid for pid in new_paper_ids if pid not in neighbor_ids][:20]
    record["retrieved_paper_ids"] = fresh_ids
    if not fresh_ids:
        return record
    try:
        from app.services import embedding_service

        papers = db.query(Paper).filter(Paper.id.in_(fresh_ids)).all()
        texts = [f"{(p.title or '')} {(p.abstract or '')[:400]}" for p in papers]
        vectors = embedding_service.embed_texts(
            [decision.closest_killer_work] + texts)
        hits = []
        for idx, paper in enumerate(papers):
            sim = embedding_service.cosine_similarity(vectors[0], vectors[idx + 1])
            if sim >= _KILLER_HIT_SIMILARITY:
                hits.append({"paper_id": paper.id, "similarity": round(sim, 3)})
        record["killer_hits"] = hits
    except Exception as exc:
        logger.warning("Gap %s: killer relevance check degraded: %s",
                       gap.id[:8], exc)
        record["killer_hits"] = []
    return record


async def _ensure_neighbor_evidence(db, state, llm, task_id: str, gap: GapCandidate,
                                    neighbors: list[Paper], audit_round: int) -> None:
    """Extract full-text evidence for audit neighbors that lack it (bounded).

    E2E 2026-08-26: one gap's audit recalled 10 candidate papers with 42 pieces
    of abstract-only evidence and zero full-text units — the claim-level
    verdict and NO_FULLTEXT_EVIDENCE both reflected an extraction gap, not the
    literature. Only neighbors without verified evidence go through the
    standard extraction path (PDF download + section extraction, idempotent by
    chunk hash), bounded by settings.audit_neighbor_evidence_max_papers.
    Failures are non-fatal: the audit proceeds with whatever material exists.
    """
    from app.agent.steps.extract_evidence import (
        _download_pdfs, _extract_from_paper_safe,
    )

    limit = max(settings.audit_neighbor_evidence_max_papers, 0)
    if limit == 0:
        return
    pending: list[tuple[Paper, TaskPaper]] = []
    for paper in neighbors[:limit]:
        has_fulltext = db.query(EvidenceUnit.id).filter(
            EvidenceUnit.paper_id == paper.id,
            EvidenceUnit.verification_status.in_(("verified", "upgraded")),
        ).limit(1).first() is not None
        if has_fulltext:
            continue
        tp = db.query(TaskPaper).filter(
            TaskPaper.task_id == task_id,
            TaskPaper.paper_id == paper.id,
        ).first()
        if tp is not None:
            pending.append((paper, tp))
    if not pending:
        return
    logger.info("Gap %s audit: extracting full-text evidence for %d neighbor(s) "
                "(round %d)", gap.id[:8], len(pending), audit_round)
    try:
        pdf_paths = await _download_pdfs(pending, task_id)
        semaphore = asyncio.Semaphore(len(pending))
        tasks_list = [
            _extract_from_paper_safe(
                task_id, paper, tp, pdf_paths.get(paper.id), llm, audit_round, semaphore
            )
            for paper, tp in pending
        ]
        results = await asyncio.gather(*tasks_list, return_exceptions=True)
        extracted = sum(r for r in results if isinstance(r, int))
        fulltext_after = db.query(EvidenceUnit).filter(
            EvidenceUnit.paper_id.in_([p.id for p, _ in pending]),
            EvidenceUnit.verification_status.in_(("verified", "upgraded")),
        ).count()
        paper_repo.save_trace(db, task_id, "audit_neighbor_evidence", "action",
                              output_data={
                                  "gap_id": gap.id,
                                  "round": audit_round,
                                  "papers_attempted": len(pending),
                                  "evidence_extracted": extracted,
                                  "fulltext_evidence_after": fulltext_after,
                              })
        db.commit()
    except Exception as exc:
        logger.warning("Gap %s: neighbor evidence extraction failed (non-fatal): %s",
                       gap.id[:8], exc)


def _format_neighbors(db, neighbors: list[Paper]) -> str:
    if not neighbors:
        return "(No neighboring papers were retrieved; return uncertain.)"
    parts = []
    for paper in neighbors:
        lines = [
            f"- Paper ID: {paper.id}",
            f"  Title: {paper.title}",
            f"  Abstract: {(paper.abstract or '')[:1200]}",
        ]
        for eu in _neighbor_verified_evidence(db, paper.id, limit=2):
            claim = (eu.normalized_claim or "").strip()
            if claim:
                lines.append(f"  Verified full-text evidence ({eu.evidence_type}): "
                             f"{claim[:200]}")
        parts.append("\n".join(lines))
    return "\n".join(parts)


def _median(values):
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


@dataclass
class NPADiagnostics:
    family_coverage: float
    family_stabilities: dict[str, float | None]   # family -> stability@5
    median_family_stability: float | None
    cross_round_stability: float | None
    stability_at_k: dict[int, float | None]        # k -> median family stability @k (diagnostic)
    search_confidence: str
    instable_families: list[str]                   # stability@5 < floor
    # Final-step Jaccard of the cumulative global Top-K (convergence of the
    # multi-query union). This — not family-internal Jaccard — is the stability
    # signal that matters for novelty: variants complement each other, so their
    # pairwise overlap being low is expected and uninformative.
    cumulative_convergence: float | None = None


def _compute_overlap_waterfall(db, gap: GapCandidate, npa_neighbor_ids: list[str]) -> dict:
    """Recall waterfall: where does cross-variant instability get introduced?

    For each query family, compares overlap between its variants at successive
    pipeline stages, so we can tell whether instability is a retrieval property
    (raw already low), a canonicalization artifact, a pre-filter artifact, or a
    re-ranking artifact. Reported as median pairwise Jaccard + OverlapCoefficient
    at @5/@10/@20.

    Stages (paper identity used for overlap):
      raw         = external_paper_id (fallback title-hash) in raw_rank order
      canonical   = title-hash in raw_rank order (dedup across sources)
      post_filter = SearchQueryPaper.paper_id in rank order
      final       = post-filter papers ordered by TaskPaper.final_score desc
    Also reports:
      high_value  = post-filter subset with priority == "high"
      npa_candidate = the gap's final neighbour set (novelty audit actually uses this)

    NOTE: a low raw overlap only licenses "retrieval is paraphrase-sensitive";
    it is NOT, by itself, proof of lexical mismatch (that needs raw identity
    inspection, which this table now enables).
    """
    from app.services.scoring_service import title_hash

    records = db.query(SearchQueryRecord).filter(
        SearchQueryRecord.target_gap_id == gap.id,
    ).all()
    families = sorted({r.query_family for r in records if r.query_family})
    queries_by_family = defaultdict(list)
    for r in records:
        if r.query_family:
            queries_by_family[r.query_family].append(r.id)

    def _median_pairwise(sets_list: list[set]):
        if len(sets_list) < 2:
            return None, None
        jacs, overlaps = [], []
        for i in range(len(sets_list)):
            for j in range(i + 1, len(sets_list)):
                a, b = sets_list[i], sets_list[j]
                inter = len(a & b)
                union = len(a | b)
                jacs.append(inter / union if union else 0.0)
                overlaps.append(inter / min(len(a), len(b)) if min(len(a), len(b)) else 0.0)
        return _median(jacs), _median(overlaps)

    def _raw_ids(query_id, k):
        rows = db.query(SearchRawResult).filter(
            SearchRawResult.query_id == query_id,
        ).order_by(SearchRawResult.raw_rank).limit(k).all()
        ids = []
        for r in rows:
            if r.external_paper_id:
                ids.append(("ext", r.external_paper_id))
            elif r.title:
                ids.append(("hash", title_hash(r.title)))
            else:
                ids.append(("row", r.id))
        return ids

    def _canonical_ids(query_id, k):
        rows = db.query(SearchRawResult).filter(
            SearchRawResult.query_id == query_id,
        ).order_by(SearchRawResult.raw_rank).all()
        seen, ids = set(), []
        for r in rows:
            key = title_hash(r.title) if r.title else (r.canonical_paper_id or r.external_paper_id or r.id)
            if key not in seen:
                seen.add(key)
                ids.append(key)
            if len(ids) >= k:
                break
        return ids

    def _postfilter_ids(query_id, k):
        rows = db.query(SearchQueryPaper.paper_id).filter(
            SearchQueryPaper.query_id == query_id,
        ).order_by(SearchQueryPaper.rank).limit(k).all()
        return [p for (p,) in rows]

    # Pre-load per-query final_score / priority once (task-level, shared across
    # families) so the final / high-value stages don't hit the DB per paper.
    score_cache: dict[str, float] = {}
    priority_cache: dict[str, str] = {}
    tps = db.query(TaskPaper).filter(TaskPaper.task_id == gap.task_id).all()
    for tp in tps:
        score_cache[tp.paper_id] = tp.final_score or 0.0
        priority_cache[tp.paper_id] = tp.priority or ""

    def _final_ids(query_id, k):
        rows = db.query(SearchQueryPaper.paper_id).filter(
            SearchQueryPaper.query_id == query_id,
        ).all()
        pids = [p for (p,) in rows]
        return sorted(pids, key=lambda p: score_cache.get(p, 0.0), reverse=True)[:k]

    def _high_value_ids(query_id, k):
        rows = db.query(SearchQueryPaper.paper_id).filter(
            SearchQueryPaper.query_id == query_id,
        ).order_by(SearchQueryPaper.rank).all()
        hv = []
        for (p,) in rows:
            if priority_cache.get(p) == "high":
                hv.append(p)
            if len(hv) >= k:
                break
        return hv

    out = {}
    for family in families:
        qids = queries_by_family[family]
        fam = {"query_count": len(qids)}
        for stage, getter in (
            ("raw", _raw_ids),
            ("canonical", _canonical_ids),
            ("post_filter", _postfilter_ids),
            ("final", _final_ids),
            ("high_value", _high_value_ids),
        ):
            stage_out = {}
            for k in (5, 10, 20):
                sets = [set(getter(qid, k)) for qid in qids]
                sets = [s for s in sets if s]
                jac, oc = _median_pairwise(sets)
                stage_out[str(k)] = {"jaccard": jac, "overlap_coefficient": oc}
            fam[stage] = stage_out
        out[family] = fam

    out["npa_candidate"] = {
        "neighbor_paper_ids": list(npa_neighbor_ids),
        "neighbor_count": len(npa_neighbor_ids),
    }
    return out


def _rank_retrieved_papers(db, gap: GapCandidate, query_ids: list[str]) -> list[str]:
    """Rank papers retrieved by `query_ids` with the NPA scoring formula.

    Mirrors `select_gap_specific_neighbors`' scoring (hits / families / rank /
    sources / final_score) but returns ordered paper_ids for the RETRIEVED set
    only (no corpus fallback), so the cumulative-stability diagnostic can watch
    the global Top-K change as queries accumulate.
    """
    mappings = db.query(SearchQueryPaper).filter(
        SearchQueryPaper.query_id.in_(query_ids)).all()
    qfam = {r.id: r.query_family for r in db.query(SearchQueryRecord).filter(
        SearchQueryRecord.id.in_(query_ids)).all()}
    stats = defaultdict(lambda: {"hits": 0, "families": set(), "sources": set(),
                                 "best_rank": 10**6})
    for m in mappings:
        s = stats[m.paper_id]
        s["hits"] += 1
        s["families"].add(qfam.get(m.query_id, ""))
        s["sources"].add(m.source)
        s["best_rank"] = min(s["best_rank"], m.rank)
    tps = {tp.paper_id: (tp.final_score or 0.0) for tp in db.query(TaskPaper).filter(
        TaskPaper.task_id == gap.task_id,
        TaskPaper.paper_id.in_(list(stats.keys()))).all()}
    scored = []
    for pid, s in stats.items():
        score = (0.4 * s["hits"] + 0.3 * len(s["families"])
                 + 0.2 / (s["best_rank"] + 1) + 0.1 * len(s["sources"])
                 + 0.1 * tps.get(pid, 0.0))
        scored.append((score, pid))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [pid for _, pid in scored]


def _compute_cumulative_npa_stability(db, gap: GapCandidate, query_ids: list[str],
                                      k: int = _MAX_NEIGHBORS) -> dict:
    """Cumulative NPA Top-K convergence across query variants.

    Family variants have COMPLEMENTARY raw recall, so their pairwise Jaccard is
    the wrong stability signal (low overlap just means they surface different
    papers, which is desirable). Instead we union queries incrementally and track
    how the re-ranked global Top-K changes. A converging Top-K means nearest-
    prior-art selection has stabilised; a drifting Top-K means more variants are
    still needed before the audit can trust its neighbour set.
    """
    records = db.query(SearchQueryRecord).filter(
        SearchQueryRecord.id.in_(query_ids)).order_by(
        SearchQueryRecord.created_at, SearchQueryRecord.id).all()
    ordered = [r.id for r in records]
    curve = []
    prev = None
    for i in range(1, len(ordered) + 1):
        topk = _rank_retrieved_papers(db, gap, ordered[:i])[:k]
        jac = None
        if prev is not None:
            jac = len(set(prev) & set(topk)) / max(1, len(set(prev) | set(topk)))
        curve.append({"n_queries": i, "topk_paper_ids": topk,
                      "jaccard_vs_prev": jac})
        prev = topk
    final_jac = curve[-1]["jaccard_vs_prev"] if len(curve) >= 2 else None
    return {
        "n_queries": len(ordered),
        "convergence_curve": curve,
        "final_topk_paper_ids": prev if prev else [],
        "final_step_jaccard": final_jac,
    }


def _compute_final_score_distribution(db, gap: GapCandidate, query_ids: list[str]) -> dict:
    """final_score distribution over the gap's retrieved papers.

    The high-value overlap diagnostic needs a relevance cutoff, but we do NOT
    pre-commit to a fixed 0.5. Report the distribution first so a threshold can
    be chosen from data, not assumption.
    """
    pids = {m.paper_id for m in db.query(SearchQueryPaper).filter(
        SearchQueryPaper.query_id.in_(query_ids)).all()}
    if not pids:
        return {"count": 0, "scores": [], "histogram": {}}
    tps = db.query(TaskPaper).filter(
        TaskPaper.task_id == gap.task_id,
        TaskPaper.paper_id.in_(pids)).all()
    scores = sorted((tp.final_score or 0.0) for tp in tps)

    def _pct(p):
        if not scores:
            return None
        idx = min(len(scores) - 1, int(round(p * (len(scores) - 1))))
        return scores[idx]

    hist = {}
    for lo in range(10):
        lo_v, hi_v = lo / 10.0, (lo + 1) / 10.0
        cnt = sum(1 for s in scores if lo_v <= s < hi_v)
        if cnt:
            hist[f"{lo_v:.1f}-{hi_v:.1f}"] = cnt
    return {
        "count": len(scores),
        "min": scores[0] if scores else None,
        "max": scores[-1] if scores else None,
        "mean": round(sum(scores) / len(scores), 4) if scores else None,
        "median": _pct(0.5),
        "p25": _pct(0.25),
        "p75": _pct(0.75),
        "p90": _pct(0.9),
        "histogram": hist,
    }


def _compute_npa_diagnostics(db, gap) -> NPADiagnostics:
    """Four-state search confidence + NPA stability diagnostics (P1-1), mechanical.

    States: INSUFFICIENT_OBSERVATION / high / medium / low.
    Driven by (a) query-family coverage, (b) MEDIAN family-internal stability
    (same family, different query variants -> Top-K overlap), and (c) cross-round
    global Top-K stability. The first audit round has no cross-round data and
    returns INSUFFICIENT_OBSERVATION, never low. Thresholds come from settings.

    Also records @5/@10/@20 family stability so the "query wording sensitivity"
    signal can be diagnosed later: if @5 is low but @20 recovers, the instability
    is rank-boundary sensitivity; if both are low, it is deeper. This does NOT by
    itself prove "lexical mismatch" — that needs raw recall/ranking inspection.
    """
    from app.config import settings
    records = db.query(SearchQueryRecord).filter(
        SearchQueryRecord.target_gap_id == gap.id,
    ).all()
    families = sorted({r.query_family for r in records if r.query_family})
    family_coverage = min(1.0, len(families) / 5.0)

    def _query_topk(query_id, k):
        rows = db.query(SearchQueryPaper.paper_id).filter(
            SearchQueryPaper.query_id == query_id,
        ).order_by(SearchQueryPaper.rank).limit(k).all()
        return {p for (p,) in rows}

    queries_by_family = defaultdict(list)
    for r in records:
        if r.query_family:
            queries_by_family[r.query_family].append(r.id)

    def _family_stabilities(k):
        out = {}
        for family in families:
            topks = [t for t in (_query_topk(qid, k) for qid in queries_by_family[family]) if t]
            if len(topks) >= 2:
                jacs = []
                for i in range(len(topks)):
                    for j in range(i + 1, len(topks)):
                        a, b = topks[i], topks[j]
                        jacs.append(len(a & b) / max(1, len(a | b)))
                out[family] = _median(jacs)
            else:
                out[family] = None
        return out

    family_stabilities = _family_stabilities(5)
    valid = [s for s in family_stabilities.values() if s is not None]
    median_family = _median(valid) if valid else None
    instable_families = [f for f, s in family_stabilities.items()
                         if s is not None and s < settings.family_stability_floor]

    comparisons = gap_repo.list_neighbor_comparisons(db, gap.id)
    current_topk = {c.paper_id for c in comparisons[:5]}
    prev_topk = set()
    audits = db.query(GapAudit).filter(
        GapAudit.gap_id == gap.id,
        GapAudit.search_admission_status == "PASS",
    ).order_by(GapAudit.created_at.desc()).limit(2).all()
    if len(audits) >= 2:
        prev_topk = set(json.loads(audits[1].neighbor_paper_ids_json or "[]")[:5])
    cross_round = (len(current_topk & prev_topk) / 5.0) if (prev_topk and current_topk) else None

    stability_at_k = {}
    for k in (5, 10, 20):
        stabs = _family_stabilities(k)
        vals = [s for s in stabs.values() if s is not None]
        stability_at_k[k] = _median(vals) if vals else None

    # Cumulative global Top-K convergence (multi-query union). This replaces
    # family-internal Jaccard as the stability gate: variants complement each
    # other, so their pairwise overlap being low is expected, not a signal of
    # instability. What matters is whether the unioned Top-K converges.
    cumulative = _compute_cumulative_npa_stability(db, gap, [r.id for r in records])
    cumulative_convergence = cumulative["final_step_jaccard"]

    if cross_round is None:
        confidence = "INSUFFICIENT_OBSERVATION"
    elif (cross_round >= settings.npa_stability_high
            and (cumulative_convergence is None
                 or cumulative_convergence >= settings.npa_stability_high)
            and family_coverage >= settings.family_coverage_high):
        confidence = "high"
    elif (cross_round >= settings.npa_stability_medium
            and family_coverage >= settings.family_coverage_medium):
        confidence = "medium"
    else:
        confidence = "low"

    return NPADiagnostics(
        family_coverage=family_coverage,
        family_stabilities=family_stabilities,
        median_family_stability=median_family,
        cross_round_stability=cross_round,
        stability_at_k=stability_at_k,
        search_confidence=confidence,
        instable_families=instable_families,
        cumulative_convergence=cumulative_convergence,
    )


def _record_nearest_prior_art(db, gap: GapCandidate, decision,
                              diagnostics: NPADiagnostics | None = None) -> None:
    """Materialise a surviving gap's nearest-prior-art provenance + P1-1 NPA
    stability and four-state search confidence.

    The closest prior work is the neighbour with the highest judged overlap
    (tie-broken by similarity). The residual gap is prefered from the claim-level
    residual already computed (set subtraction); it falls back to the audit's
    differentiation summary only when atomic claims are absent.
    """
    comparisons = gap_repo.list_neighbor_comparisons(db, gap.id)
    if diagnostics is None:
        diagnostics = _compute_npa_diagnostics(db, gap)
    gap.family_coverage = diagnostics.family_coverage
    gap.npa_stability = diagnostics.median_family_stability
    gap.search_confidence = diagnostics.search_confidence
    # Record @5/@10/@20 stability for later diagnosis/calibration. Query wording
    # sensitivity is real, but its CAUSE (lexical mismatch vs rank-boundary
    # sensitivity) needs this raw signal — never a conclusion from one number.
    paper_repo.save_trace(db, gap.task_id, "gap_npa_stability_diagnostic", "decision",
                          output_data={
                              "gap_id": gap.id,
                              "stability_at_k": diagnostics.stability_at_k,
                              "family_stabilities": diagnostics.family_stabilities,
                              "cross_round_stability": diagnostics.cross_round_stability,
                              "instable_families": diagnostics.instable_families,
                              "search_confidence": diagnostics.search_confidence,
                          })
    if not comparisons:
        gap.nearest_prior_art_paper_id = None
        gap.nearest_prior_art_title = None
        if not gap.residual_gap:
            gap.residual_gap = None
        return
    nearest = max(comparisons, key=lambda c: (c.overlap_ratio or 0.0,
                                              c.similarity_score or 0.0))
    paper = db.get(Paper, nearest.paper_id)
    gap.nearest_prior_art_paper_id = nearest.paper_id
    gap.nearest_prior_art_title = paper.title if paper else None
    # Prefer the claim-level residual (set subtraction); fall back to the audit's
    # differentiation summary only when atomic claims are absent.
    if not gap.residual_gap:
        uncovered = json.loads(nearest.uncovered_claims_json or "[]")
        parts = []
        if decision.differentiation_summary:
            parts.append(decision.differentiation_summary.strip())
        if uncovered:
            parts.append("Nearest prior work does not cover: " + "; ".join(uncovered))
        gap.residual_gap = "\n".join(parts).strip() or None


def _gap_evidence_ids(db, gap_id: str, relation_type: str) -> list[str]:
    return [
        link.evidence_id for link in gap_repo.list_gap_evidence(db, gap_id)
        if link.relation_type == relation_type
    ]


def _validate_audit_decision(decision, gap, neighbors, db) -> tuple[list[str], list[str]]:
    """Enforce the audit contract, naming the offending value on failure.

    Returns a tuple of (dropped_evidence_ids, dropped_comparison_paper_ids). The
    strict-reject vs drop-and-continue distinction is deliberate: audit_result /
    recommended_action are the verdict itself, so an invalid value rejects the
    decision. A comparison bound to an unknown paper is unusable — it either
    attaches a verdict to the wrong work or cites a hallucinated ID — so that
    single comparison is dropped and recorded, and the remaining comparisons
    still carry the verdict. Only when EVERY comparison is unusable is the
    decision rejected (no comparison material left). The evidence_for/against
    lists are supporting annotations, so an unusable ID there is likewise
    dropped rather than discarding an otherwise sound audit. Observed in
    production: a model copied one UUID with a corrupted segment (…-44a2-b4-…
    instead of …-42b4-…) and a complete partially_closed verdict was downgraded
    to uncertain; and a hallucinated neighbor paper ID discarded the whole
    decision, leaving no surviving gap after audit.
    """
    if decision.audit_result not in {"confirmed", "partially_closed", "closed", "uncertain"}:
        raise AuditDecisionInvalid(f"invalid audit_result: {decision.audit_result!r}")
    if decision.recommended_action not in {"continue", "narrow", "more_search", "reject"}:
        raise AuditDecisionInvalid(
            f"invalid recommended_action: {decision.recommended_action!r}")
    valid_paper_ids = {paper.id for paper in neighbors}
    unknown_papers = sorted({item.paper_id for item in decision.comparisons
                             if item.paper_id not in valid_paper_ids})
    if unknown_papers:
        decision.comparisons = [item for item in decision.comparisons
                                if item.paper_id in valid_paper_ids]
        if not decision.comparisons:
            raise AuditDecisionInvalid(
                f"all comparisons cite unknown neighbor papers: {unknown_papers}")
    valid_evidence_ids = {link.evidence_id for link in gap_repo.list_gap_evidence(db, gap.id)}
    dropped = sorted(
        set(decision.evidence_for_gap_ids + decision.evidence_against_gap_ids)
        - valid_evidence_ids)
    if dropped:
        decision.evidence_for_gap_ids = [item for item in decision.evidence_for_gap_ids
                                        if item in valid_evidence_ids]
        decision.evidence_against_gap_ids = [item for item in decision.evidence_against_gap_ids
                                            if item in valid_evidence_ids]
    return dropped, unknown_papers
