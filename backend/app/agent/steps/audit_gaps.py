"""Step: Run a lightweight adversarial audit for evidence-backed gap candidates."""

import json
import logging
from dataclasses import dataclass
from collections import defaultdict

from pydantic import BaseModel, Field

from app.agent.state import ResearchState
from app.agent.steps.generate_queries import SearchQueryExecution
from app.agent.steps.search_papers import search_and_save_papers
from app.agent.steps.mine_gaps import GAP_MINING_POLICY_VERSION
from app.db.models import (EvidenceUnit, GapAudit, GapCandidate, Paper, QuestionEvidenceLink,
                          SearchQueryPaper, SearchQueryRecord, TaskPaper)
from app.db.repositories import gap_repo, paper_repo, task_repo
from app.db.repositories.search_query_repo import save_search_query

logger = logging.getLogger(__name__)

_MAX_NEIGHBORS = 5
# Neighbors whose highest semantic similarity falls below this bound did not
# meaningfully cover the gap's domain, so a "confirmed" verdict resting on such
# neighbors is a retrieval miss rather than a novelty signal.
_MIN_NEIGHBOR_SIMILARITY = 0.3
# A high novelty_confidence combined with a low-similarity neighbor set is the
# contradictory signal that flags retrieval failure (observed: 5 application-
# domain neighbors at max similarity 0.15 yet novelty_confidence 0.95).
_HIGH_NOVELTY = 0.8
# v2 made corpus papers from the same research questions admissible comparison
# material; v3 closes a gap as undecidable when a repeated adversarial round
# brings no new material for the same claim; v4 generates English adversarial
# queries (anchored on mechanism/capability rather than surface keywords) and
# downgrades confirmed verdicts whose neighbors are semantically too distant to
# have ruled prior work out. Like the mining policy version, this is what
# invalidates audits already stamped for a round, so an admission-or-verdict
# rule change must bump it or resumed tasks keep their old verdicts.
GAP_SEARCH_POLICY_VERSION = "gap-search-admission-v4"


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
evidence IDs and must never appear there. Copy every ID character by character."""

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

def build_adversarial_queries(gap: GapCandidate) -> list[AdversarialQuerySpec]:
    specs = [
        AdversarialQuerySpec("exact_gap", " ".join(filter(None, [gap.target_setting, gap.missing_capability, gap.claimed_delta]))),
        AdversarialQuerySpec("alternative_coverage", " ".join(filter(None, [gap.observed_problem, "method evaluation benchmark"]))),
        AdversarialQuerySpec("claim_falsification", gap.falsification_condition or gap.description),
    ]
    seen = set()
    return [spec for spec in specs if spec.query_text and not (spec.query_text.lower() in seen or seen.add(spec.query_text.lower()))]


class GapQueryGenSchema(BaseModel):
    exact_gap: str = Field(description="English query matching the gap's exact claimed delta")
    alternative_coverage: str = Field(description="English query for alternative methods that might already cover the observed problem")
    claim_falsification: str = Field(description="English query targeting the falsification condition / strongest prior work")


_QUERY_GEN_SYSTEM = """You are generating English academic search queries for an adversarial gap audit.
The gap description may be in Chinese or mixed language. Produce queries that would retrieve the
most relevant PRIOR WORK that could already close this gap, focusing on systems- and
mechanism-level work rather than surface application papers.

Rules:
- Output English only.
- Anchor each query on the gap's core mechanism/capability (e.g. memory offloading, quantization,
  gradient checkpointing, distributed training) rather than surface keywords (e.g. adapter, benchmark).
- Translate the gap's concepts into standard English technical vocabulary; do not invent new terms."""

_QUERY_GEN_USER = """Candidate gap:
- Setting: {target_setting}
- Observed problem: {observed_problem}
- Missing capability: {missing_capability}
- Claimed delta: {claimed_delta}
- Falsification condition: {falsification_condition}

Produce the three English search queries."""


async def generate_english_adversarial_queries(llm, gap: GapCandidate) -> list[AdversarialQuerySpec]:
    """Generate English search queries anchored on the gap's mechanism/capability.

    Raw concatenation of the gap's (often Chinese) fields produced queries that
    matched surface keywords (e.g. "adapter") across application domains while
    missing the systems-level prior work (e.g. ZeRO-Offload, QLoRA CPU offloading)
    the audit was meant to find. English, mechanism-anchored queries fix the
    language/recall mismatch; on failure we fall back to raw concatenation.
    """
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
        ], GapQueryGenSchema)
        specs = [
            AdversarialQuerySpec("exact_gap", (gen.exact_gap or "").strip()),
            AdversarialQuerySpec("alternative_coverage", (gen.alternative_coverage or "").strip()),
            AdversarialQuerySpec("claim_falsification", (gen.claim_falsification or "").strip()),
        ]
        specs = [spec for spec in specs if spec.query_text]
        if specs:
            return specs
    except Exception as exc:
        logger.warning("Gap %s: English query generation failed (%s), falling back to raw concatenation", gap.id[:8], exc)
    return build_adversarial_queries(gap)


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
    paper_ids = sorted(retrieved_paper_ids | set(corpus_paper_ids))
    if len(paper_ids) < min_papers: reasons.append("INSUFFICIENT_GAP_SPECIFIC_PAPERS")
    support_papers = {db.get(EvidenceUnit, link.evidence_id).paper_id for link in gap_repo.list_gap_evidence(db, gap.id) if link.relation_type == "suggests" and db.get(EvidenceUnit, link.evidence_id)}
    external = [item for item in paper_ids if item not in support_papers]
    if not external: reasons.append("NO_EXTERNAL_NEIGHBOR")
    if constrained and reasons:
        logger.info("Gap %s admission evaluated under constrained retrieval mode "
                    "(relaxed thresholds: completed>=%d, families>=%d, papers>=%d)",
                    gap.id[:8], min_completed, min_families, min_papers)
    return GapSearchAdmission(gap.id, "PASS" if not reasons else "UNKNOWN", reasons, [item.id for item in queries], [item.id for item in completed], [item.id for item in failed], families, len(sources), paper_ids, external)

def select_gap_specific_neighbors(db, gap, query_ids, limit=_MAX_NEIGHBORS):
    mappings = db.query(SearchQueryPaper).filter(SearchQueryPaper.query_id.in_(query_ids)).all()
    query_family = {item.id: item.query_family for item in db.query(SearchQueryRecord).filter(SearchQueryRecord.id.in_(query_ids)).all()}
    stats = defaultdict(lambda: {"hits": 0, "families": set(), "sources": set(), "best_rank": 10**6})
    for mapping in mappings:
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
    for gap in gaps:
        results.append(await audit_gap_candidate(db, state, llm, task_id, gap, perform_search))
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
    """Run one bounded audit: three query families and at most five neighbors."""
    if gap.task_id != task_id:
        raise ValueError("Gap does not belong to task")

    gap.status = "auditing"
    audit_round = state.current_round + 1
    query_specs = await generate_english_adversarial_queries(llm, gap)
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
        except RuntimeError as exc:
            logger.info("Gap %s search admission deferred: %s", gap.id[:8], exc)
    admission = evaluate_gap_search_admission(db, gap, [item.query_id for item in executions])
    _save_search_admission_trace(db, task_id, admission)
    if admission.status != "PASS":
        gap.status = "auditing"
        gap_repo.create_gap_audit(
            db, gap_id=gap.id, task_id=task_id, adversarial_queries=queries,
            audit_result="uncertain", neighbor_paper_ids=admission.candidate_paper_ids,
            recommended_action="more_search", audit_round=audit_round,
            search_policy_version=GAP_SEARCH_POLICY_VERSION, search_admission_status=admission.status,
            search_admission_reasons=admission.reason_codes, search_query_ids=admission.query_ids,
            audited_claimed_delta=gap.claimed_delta or "",
        )
        db.commit()
        return GapAuditResult(gap.id, "uncertain", "more_search", "")
    neighbors = select_gap_specific_neighbors(db, gap, admission.completed_query_ids)
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
            neighbors=_format_neighbors(neighbors),
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
    for comparison in decision.comparisons:
        gap_repo.create_neighbor_comparison(            db,
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
        )

    action = decision.recommended_action
    if action == "continue" and decision.audit_result == "confirmed":
        gap.status = "surviving"
    elif action == "reject" or decision.audit_result == "closed":
        gap.status = "rejected"
        action = "reject"
    elif action == "narrow" or decision.audit_result == "partially_closed":
        gap.status = "audited"
        action = "narrow"
    else:
        gap.status = "auditing"
        action = "more_search"

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
    )
    paper_repo.save_trace(db, task_id, "audit_gap_candidate", "decision", output_data={
        "gap_id": gap.id,
        "audit_result": decision.audit_result,
        "recommended_action": action,
        "neighbor_count": len(neighbors),
        "query_count": len(queries),
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


def _format_neighbors(neighbors: list[Paper]) -> str:
    if not neighbors:
        return "(No neighboring papers were retrieved; return uncertain.)"
    return "\n".join(
        f"- Paper ID: {paper.id}\n  Title: {paper.title}\n  Abstract: {(paper.abstract or '')[:1200]}"
        for paper in neighbors
    )


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
