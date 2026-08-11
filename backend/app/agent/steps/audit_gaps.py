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
from app.db.models import (EvidenceUnit, GapCandidate, Paper, QuestionEvidenceLink,
                          SearchQueryPaper, SearchQueryRecord, TaskPaper)
from app.db.repositories import gap_repo, paper_repo, task_repo
from app.db.repositories.search_query_repo import save_search_query

logger = logging.getLogger(__name__)

_MAX_NEIGHBORS = 5
# Bumped to v2 when corpus papers from the same research questions became
# admissible comparison material. Like the mining policy version, this is what
# invalidates audits already stamped for a round, so an admission-rule change
# must bump it or resumed tasks keep their old verdicts.
GAP_SEARCH_POLICY_VERSION = "gap-search-admission-v2"


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
    state.surviving_gap_ids = [result.gap_id for result in results if result.audit_result == "confirmed"]
    task_repo.save_state(db, task_id, state)
    db.commit()
    return results

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
    query_specs = build_adversarial_queries(gap)
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
        )
        db.commit()
        return GapAuditResult(gap.id, "uncertain", "more_search", "")
    neighbors = select_gap_specific_neighbors(db, gap, admission.completed_query_ids)
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
        dropped_evidence_ids = _validate_audit_decision(decision, gap, neighbors, db)
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
        )
        paper_repo.save_trace(db, task_id, "gap_audit_invalid_decision", "decision", output_data={
            "gap_id": gap.id, "error": str(err),
            "audit_result": decision.audit_result,
            "recommended_action": decision.recommended_action,
        })
        db.commit()
        return GapAuditResult(gap.id, "uncertain", "more_search", "")
    if dropped_evidence_ids:
        logger.warning("Gap %s: dropped %d unusable evidence ID(s) from the audit "
                       "decision: %s", gap.id[:8], len(dropped_evidence_ids),
                       dropped_evidence_ids)
        paper_repo.save_trace(db, task_id, "gap_audit_dropped_evidence_ids", "decision",
                              output_data={"gap_id": gap.id,
                                           "dropped_evidence_ids": dropped_evidence_ids,
                                           "audit_result": decision.audit_result})
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


def _validate_audit_decision(decision, gap, neighbors, db) -> list[str]:
    """Enforce the audit contract, naming the offending value on failure.

    Returns the evidence IDs that were dropped as unusable. The distinction is
    deliberate: a comparison is bound to one specific neighbour paper, so a wrong
    paper ID would attach a verdict to the wrong work and must invalidate the
    decision. The evidence_for/against lists are supporting annotations — the
    verdict rests on audit_result, remaining_delta and the per-neighbour
    comparisons — so an unusable ID there is dropped and recorded instead of
    discarding an otherwise sound audit. Observed in production: a model copied
    one UUID with a corrupted segment (…-44a2-b4-… instead of …-42b4-…) and a
    complete partially_closed verdict was downgraded to uncertain because of it.
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
        raise AuditDecisionInvalid(
            f"comparison cites unknown neighbor paper(s): {unknown_papers}")
    valid_evidence_ids = {link.evidence_id for link in gap_repo.list_gap_evidence(db, gap.id)}
    dropped = sorted(
        set(decision.evidence_for_gap_ids + decision.evidence_against_gap_ids)
        - valid_evidence_ids)
    if dropped:
        decision.evidence_for_gap_ids = [item for item in decision.evidence_for_gap_ids
                                        if item in valid_evidence_ids]
        decision.evidence_against_gap_ids = [item for item in decision.evidence_against_gap_ids
                                            if item in valid_evidence_ids]
    return dropped
