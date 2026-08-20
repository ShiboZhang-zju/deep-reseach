"""Step: Mine lightweight, evidence-backed research gap candidates."""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field

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

logger = logging.getLogger(__name__)

_SUPPORTED_GAP_TYPES = {"boundary_gap", "missing_evaluation"}
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
# Bumped to v3 when "covered" questions became eligible for mining: the policy
# version is what invalidates gaps already stamped for a round, so a change to
# the admission rules must bump it or the new rules never take effect on a
# resumed task.
GAP_MINING_POLICY_VERSION = "evidence-admission-v4"

# Semantic-dedup threshold: two gap candidates whose fingerprint (observed
# problem + missing capability + claimed delta) embeds to a cosine similarity at
# or above this bound are the same gap reworded. Without this, the same gap
# enters the audit twice under different wording and can be rejected once and
# "confirmed" once — the single most damaging false signal a research-idea
# system can emit.
_GAP_DEDUP_SIMILARITY = 0.85


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
    papers = sorted({item.paper_id for item in supporting})
    spans = sum(_is_fulltext_locatable(item) for item in supporting)
    limiting = [item for item in supporting if item.evidence_type in _LIMITATION_SIGNAL_TYPES]
    reasons = []
    if len(papers) < 2: reasons.append("INSUFFICIENT_INDEPENDENT_PAPERS")
    if not limiting: reasons.append("NO_LIMITATION_SIGNAL")
    if contradicting: reasons.append("UNRESOLVED_VERIFIED_CONTRADICTION")

    # O5(b): full-text is no longer a hard requirement. Determine an evidence
    # tier instead:
    #   A-tier: at least one full-text-locatable span  -> high-confidence gap
    #   B-tier: no full-text, but >= _MIN_ABSTRACT_TIER_PAPERS papers with a
    #           limitation signal -> admissible but flagged partial (needs
    #           downstream audit / follow-up to confirm)
    # Only when neither condition holds do we withhold admission.
    if spans >= 1:
        evidence_tier = "A"
    elif len(papers) >= _MIN_ABSTRACT_TIER_PAPERS and limiting:
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
    return QuestionEvidenceAdmission(
        question.id, status, reasons,
        [item.id for item in supporting], [item.id for item in supporting],
        [item.id for item in limiting], [item.id for item in contradicting],
        papers, spans, evidence_tier,
    )

_GAP_MINING_SYSTEM = """You identify candidate research gaps from supplied paper evidence.
Return only gaps that are supported by the supplied evidence. A gap is not merely a topic with little evidence.

Allowed gap types:
- boundary_gap: an existing method or evaluation has a documented boundary, failure condition, or untested setting.
- missing_evaluation: existing work does not evaluate a concrete, important capability, condition, cost, or risk.

For every gap:
- cite only supplied question IDs and evidence IDs;
- every evidence ID you cite must list one of the gap's question IDs under
  "Linked questions", so the gap stays grounded in the questions it claims;
- cite evidence from at least two different papers (see "Paper ID");
- cite at least one evidence unit whose Type is "limitation" or
  "negative_result": a gap needs a documented shortcoming, not only comparisons;
- state what existing work already covers and the smallest missing capability;
- write a falsifiable condition that would close the gap;
- do not propose a solution or invent paper findings;
- write every gap field in English (target setting, observed problem, existing
  coverage, missing capability, claimed delta, falsification condition). The
  downstream adversarial search and audit run against English academic
  databases, and Chinese gap text degrades their retrieval recall.
Return an empty list when the evidence is insufficient."""

_GAP_MINING_USER = """Research topic: {topic}
Target problem: {target_problem}
Target setting: {target_setting}

Research questions:
{questions}

Evidence (only these IDs may be cited):
{evidence}

Generate at most {max_gaps} evidence-backed candidate gaps."""


def _format_evidence(evidence: list[EvidenceUnit],
                     question_ids_by_evidence: dict[str, list[str]] | None = None) -> str:
    """Render the evidence block.

    Every attribute that downstream validation tests must be visible here,
    otherwise the model is judged on constraints it was never shown and valid
    candidates are silently discarded: `Paper ID` backs the "at least two
    independent papers" gate, `Type` backs the limitation-signal gate, and
    `question_ids_by_evidence` backs the question-grounding gate.
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
        linked = (question_ids_by_evidence or {}).get(item.id)
        if linked:
            entry += "\n  Linked questions: " + ", ".join(linked)
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
) -> tuple[list[EvidenceUnit], dict[str, list[str]]]:
    """Pick a bounded evidence subset to offer the model, and its question links.

    Every downstream gate still has to be satisfiable from what is offered, so
    each question contributes its highest-value units first (limitation signals,
    full-text spans) while a per-paper cap keeps at least two independent papers
    in view. Questions are then interleaved round-robin, so the global cap
    truncates the long tail rather than whole questions.
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
        per_paper_seen: dict[str, int] = {}
        shortlist: list[EvidenceUnit] = []
        for item in candidates:
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

    question_ids_by_evidence: dict[str, list[str]] = {}
    for question_id in sorted(passed_admissions):
        for evidence_id in passed_admissions[question_id].admissible_evidence_ids:
            if evidence_id in selected:
                question_ids_by_evidence.setdefault(evidence_id, []).append(question_id)
    return list(selected.values()), question_ids_by_evidence


async def mine_gap_candidates(
    db,
    state: ResearchState,
    llm,
    task_id: str,
    max_gaps: int = 3,
    input_version: str = "",
) -> list:
    """Create evidence-backed boundary/evaluation gap candidates.

    This MVP step deliberately mines only gap types that can be grounded in
    limitations, negative results, evaluation, and comparison evidence. It does
    not infer that missing evidence alone is a research gap.

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
                    if gap.mining_policy_version == GAP_MINING_POLICY_VERSION
                    and gap.mining_input_version == input_version
                    and gap.status not in {"rejected", "superseded"}]
    else:
        existing = [gap for gap in gap_repo.list_gaps_for_contract(db, task_id, contract.id)
                    if gap.mining_policy_version == GAP_MINING_POLICY_VERSION
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
    # (distinct papers, verified spans, limitation-type evidence), so widening
    # the pool lets that gate do its job instead of pre-filtering on a signal
    # that means the opposite of what mining needs.
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
    evidence, question_ids_by_evidence = select_prompt_evidence(
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
    result = await llm.chat_json([{
        "role": "system", "content": _GAP_MINING_SYSTEM,
    }, {
        "role": "user", "content": _GAP_MINING_USER.format(
            topic=contract.topic, target_problem=contract.target_problem or "(not specified)",
            target_setting=contract.target_setting or "(not specified)", questions=question_text,
            evidence=_format_evidence(evidence, question_ids_by_evidence), max_gaps=max_gaps,
        ),
    }], GapCandidateList)

    # Semantic dedup prep: fingerprint every already-accepted gap (this batch
    # excluded) and every returned candidate, then embed them once. Dedup is
    # non-fatal — if embedding is unavailable we skip it rather than blocking
    # mining — so a duplicate can only slip through when embeddings fail.
    # Compare against every non-rejected gap in the contract, INCLUDING
    # superseded versions: a gap that was narrowed (v1 superseded -> v2) must
    # still block a later candidate that rewords v1's original broad claim,
    # otherwise "reword the rejected broad claim" defeats the lineage. Only
    # genuinely rejected gaps (audit said closed) leave the comparison pool.
    existing_gaps = [g for g in gap_repo.list_gaps_for_contract(
        db, task_id, contract.id, include_superseded=True)
        if g.status != "rejected"]
    existing_fps = [_gap_fingerprint(g.observed_problem, g.missing_capability,
                                     g.claimed_delta) for g in existing_gaps]
    candidate_objs = result.gaps[:max_gaps]
    candidate_fps = [_gap_fingerprint(c.observed_problem, c.missing_capability,
                                      c.claimed_delta) for c in candidate_objs]
    all_vecs = await _embed_fingerprints(existing_fps + candidate_fps)
    dedup_enabled = bool(all_vecs) and len(all_vecs) == len(existing_fps) + len(candidate_fps)
    existing_vecs = all_vecs[:len(existing_fps)]
    candidate_vecs = all_vecs[len(existing_fps):] if dedup_enabled else []
    accepted_vecs = list(existing_vecs)

    created = []
    rejected_candidates = []
    accepted_candidates = []
    for idx, candidate in enumerate(candidate_objs):
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
        if candidate.gap_type not in _SUPPORTED_GAP_TYPES: reason = "UNSUPPORTED_GAP_TYPE"
        elif not candidate_question_ids or not candidate_question_ids.issubset(passed_admissions): reason = "UNKNOWN_QUESTION_ID"
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
        elif candidate.contradicting_evidence_ids: reason = "UNEXPECTED_CONTRADICTING_EVIDENCE"
        elif len({item.paper_id for item in support}) < 2: reason = "INSUFFICIENT_CANDIDATE_PAPER_SUPPORT"
        elif not any(item.evidence_type in _LIMITATION_SIGNAL_TYPES for item in support): reason = "CANDIDATE_LACKS_LIMITATION_SIGNAL"
        if reason:
            rejected_candidates.append({
                "reason": reason,
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
        if dedup_enabled and candidate_vecs[idx]:
            for vec in accepted_vecs:
                if vec and cosine_similarity(candidate_vecs[idx], vec) >= _GAP_DEDUP_SIMILARITY:
                    dup = True
                    break
        if dup:
            rejected_candidates.append({
                "reason": "DUPLICATE_GAP",
                "gap_type": candidate.gap_type,
                "question_ids": candidate.question_ids,
                "cited_evidence_ids": candidate.supporting_evidence_ids,
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
            mining_policy_version=GAP_MINING_POLICY_VERSION,
            mining_input_version=input_version,
        )
        # Only admitted evidence gets linked; an unoffered ID would otherwise
        # create a gap-evidence link pointing at nothing.
        for evidence_id in usable_evidence_ids:
            gap_repo.create_gap_evidence_link(db, gap.id, evidence_id, "suggests", 0.8)
        for evidence_id in candidate.contradicting_evidence_ids:
            gap_repo.create_gap_evidence_link(db, gap.id, evidence_id, "contradicts", 0.8)
        created.append(gap)
        if dedup_enabled and candidate_vecs[idx]:
            accepted_vecs.append(candidate_vecs[idx])
        accepted_candidates.append({
            "gap_id": gap.id, "gap_type": candidate.gap_type,
            "linked_evidence_ids": usable_evidence_ids,
            "dropped_evidence_ids": unoffered_evidence_ids,
        })
    paper_repo.save_trace(db, task_id, "gap_mining_candidates", "decision", output_data={
        "llm_candidate_count": len(result.gaps), "accepted_candidate_count": len(created),
        "accepted_candidates": accepted_candidates,
        "rejected_candidates": rejected_candidates,
    })
    state.active_gap_ids = [gap.id for gap in created]
    paper_repo.save_trace(db, task_id, "mine_gap_candidates", "action", output_data={
        "contract_id": contract.id,
        "candidate_count": len(created),
        "evidence_count": len(evidence),
        "admissible_evidence_count": len(admissible_evidence_ids),
        "supported_gap_types": sorted(_SUPPORTED_GAP_TYPES),
    })
    db.commit()
    return created
