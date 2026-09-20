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
GAP_MINING_POLICY_VERSION = "opportunity-ontology-v7"

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
      cited evidence units (a contradiction needs the two conflicting claims).
    """
    gap_type: str
    definition: str
    discovery_directive: str
    citation_rule: str
    required_evidence_types: frozenset[str] | None = None
    signal_rejection_code: str = ""
    min_usable_units: int = 1
    min_units_rejection_code: str = ""


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
            "degrades, or a setting its evaluation never covered."
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
            "Look for an assumption several papers inherit without verifying "
            "(e.g. 'retrieval similarity ranks future usefulness'). The "
            "opportunity is testing the inverted or untested assumption. A "
            "limitation admission is NOT required for this type."
        ),
        citation_rule=(
            "cite evidence units from at least two independent papers that "
            "share or rely on the same implicit assumption; method and result "
            "units are admissible"
        ),
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
            "that maps onto an unsolved problem in the target setting. Name "
            "the source field and the mapping. A limitation admission is NOT "
            "required for this type."
        ),
        citation_rule=(
            "cite the evidence that documents the source mechanism and the "
            "evidence that shows the target setting lacks it"
        ),
    ),
)

_CONTRACTS_BY_TYPE = {spec.gap_type: spec for spec in _OPPORTUNITY_CONTRACTS}


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

    # --- Opportunity lens fan-out (v7) ---------------------------------------
    # One call per opportunity type. A candidate returned by the wrong lens is
    # rejected as TYPE_LENS_MISMATCH without consuming an audit slot.
    for contract_spec in _OPPORTUNITY_CONTRACTS:
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
                # before the universal paper floor so the rejection names the
                # actual defect (only one side of the conflict cited); for every
                # other type min_usable_units is 1 and this never fires.
                reason = contract_spec.min_units_rejection_code
            elif len({item.paper_id for item in support}) < 2: reason = "INSUFFICIENT_CANDIDATE_PAPER_SUPPORT"
            elif (contract_spec.required_evidence_types is not None
                  and not any(item.evidence_type in contract_spec.required_evidence_types
                              for item in support)):
                # v7 per-type admission contract: the signal this opportunity
                # type is built on must actually be cited. boundary /
                # missing_evaluation demand a documented shortcoming; mechanism
                # / scaling demand a documented phenomenon or tested range.
                reason = contract_spec.signal_rejection_code
            elif (_PLACEHOLDER_OPERAND_RE.search(candidate.testable_hypothesis or "")
                    or _PLACEHOLDER_OPERAND_RE.search(candidate.claimed_delta or "")):
                # P1-1 (task d6f64087): ">X% label noise or <Y parameters" is a
                # hypothesis the mining LLM never finished quantifying. Accepting it
                # lets the placeholder flow through narrowing into intervention and
                # experiment prompts (observed verbatim on the surviving gap).
                reason = "PLACEHOLDER_OPERAND_IN_HYPOTHESIS"
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
