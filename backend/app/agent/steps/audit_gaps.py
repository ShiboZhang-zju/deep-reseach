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
from app.db.models import EvidenceUnit, GapCandidate, Paper, SearchQueryPaper, SearchQueryRecord, TaskPaper
from app.db.repositories import gap_repo, paper_repo, task_repo
from app.db.repositories.search_query_repo import save_search_query

logger = logging.getLogger(__name__)

_MAX_NEIGHBORS = 5
GAP_SEARCH_POLICY_VERSION = "gap-search-admission-v1"

_AUDIT_SYSTEM = """You are conducting an adversarial research-gap audit.
Your job is to find whether the supplied neighboring papers already close the candidate gap.
Do not infer novelty from absence alone. Return uncertain when the evidence is insufficient.

A gap is:
- confirmed when neighbors do not cover its core claimed delta;
- partially_closed when they cover part of it and a concrete remaining delta exists;
- closed when a neighbor covers the core problem, capability, and evaluation claim;
- uncertain when the supplied evidence cannot decide.

For each neighbor, compare only the supplied gap claims and paper content. Do not invent papers or evidence IDs."""

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
    audit_result: str
    recommended_action: str
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

def evaluate_gap_search_admission(db, gap, query_ids):
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
    if len(completed) < 2: reasons.append("INSUFFICIENT_COMPLETED_QUERIES")
    if len(families) < 2: reasons.append("INSUFFICIENT_QUERY_FAMILIES")
    if len(completed) / max(len(queries), 1) < 0.5: reasons.append("SEARCH_SUCCESS_RATE_TOO_LOW")
    mappings = db.query(SearchQueryPaper).filter(SearchQueryPaper.query_id.in_([item.id for item in completed])).all()
    paper_ids = sorted({item.paper_id for item in mappings})
    sources = {item.source for item in mappings if item.source and item.source != "unknown"}
    if not sources: reasons.append("NO_SUCCESSFUL_SOURCE")
    if len(paper_ids) < 3: reasons.append("INSUFFICIENT_GAP_SPECIFIC_PAPERS")
    support_papers = {db.get(EvidenceUnit, link.evidence_id).paper_id for link in gap_repo.list_gap_evidence(db, gap.id) if link.relation_type == "suggests" and db.get(EvidenceUnit, link.evidence_id)}
    external = [item for item in paper_ids if item not in support_papers]
    if not external: reasons.append("NO_EXTERNAL_NEIGHBOR")
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
    return [paper for _, paper in sorted(ranked, key=lambda item: item[0], reverse=True)[:limit]]

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

    _validate_audit_decision(decision, gap, neighbors, db)
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


def _validate_audit_decision(decision, gap, neighbors, db) -> None:
    if decision.audit_result not in {"confirmed", "partially_closed", "closed", "uncertain"}:
        raise ValueError("Invalid audit result")
    if decision.recommended_action not in {"continue", "narrow", "more_search", "reject"}:
        raise ValueError("Invalid recommended action")
    valid_paper_ids = {paper.id for paper in neighbors}
    if any(item.paper_id not in valid_paper_ids for item in decision.comparisons):
        raise ValueError("Audit contains a comparison for an unknown neighbor paper")
    valid_evidence_ids = {link.evidence_id for link in gap_repo.list_gap_evidence(db, gap.id)}
    if not set(decision.evidence_for_gap_ids + decision.evidence_against_gap_ids).issubset(valid_evidence_ids):
        raise ValueError("Audit contains an unknown evidence ID")
