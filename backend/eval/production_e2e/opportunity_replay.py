"""v7 Opportunity Replay — frozen-evidence v6 vs v7 discovery comparison.

Frozen causal question
----------------------
Given the SAME knowledge input (same topic / contract / research questions /
retrieved papers / evidence units / question-evidence links), does the v7
opportunity ontology discover more real research opportunities than v6 —
without a measurable increase in false gaps?

What is frozen (the discovery input, copied verbatim from the v6 task):
    ResearchQuestion (incl. status / importance / coverage)
    EvidenceUnit (incl. verification + span provenance)
    QuestionEvidenceLink
    TaskPaper (per-task paper association)
    ResearchContract

No re-retrieval, no re-extraction, no re-decomposition. NOTE the direction of
the bias this introduces: the frozen pool is the v6 task's FINAL evidence
state, a superset of what v6 mining saw in any single round — so if anything
the comparison is generous to BOTH miners equally, and it never re-queries
the literature for discovery.

What is NOT frozen (shared downstream behaviour, runs live for v7):
    adversarial audit retrieval, phenomenon plans, interventions,
    experiments. The v6 side is NOT re-run — its historical funnel is read
    from the DB.

Deviations from a full production run (declared, by design):
    single mining round (no remediation re-search — the input is frozen by
    definition), one narrowing pass max, no diminishing-return veto loop.
    The v6 side funnel is the full historical production run.

Usage
-----
    cd backend
    # 1) dry-run: read-only, runs v7 admission arithmetic on the frozen pool,
    #    reports the raw material per opportunity lens. Zero writes.
    python -m eval.production_e2e.opportunity_replay \
        --source-run pe2e_fullv2_clean --dry-run

    # 2) real replay (writes replay tasks into the production DB and calls
    #    the LLM + external retrieval for audit):
    python -m eval.production_e2e.opportunity_replay \
        --source-run pe2e_fullv2_clean \
        --topics pe2e-001 pe2e-007 pe2e-013 pe2e-015 pe2e-019 pe2e-024 \
        --run-id v7_replay_pilot

    # 3) repair an existing run: rebuild every derived number from the DB,
    #    no LLM call (use after a failed follow-up sweep):
    python -m eval.production_e2e.opportunity_replay \
        --run-id v7_replay_pilot --recompute-only

Outputs (backend/eval_results/<run_id>/):
    replay_topics.jsonl        one row per topic: frozen counts, v6 funnel,
                               v7 funnel, survival basis, cost, v7-only
                               determination
    replay_report.json         aggregated answer to the five questions, plus
                               q6_survival_basis_2x2, q7 P2 counterfactual and
                               db_consistency
    v7_only_opportunities.jsonl  v7 discoveries with no v6 counterpart, with
                               EMPTY human_prior_art_verdict fields — the
                               blind prior-art audit input
    mech_override_review.jsonl  the subset of the above that entered surviving
                               via the mechanical-novelty override, with its
                               atomic claims and neighbours inlined — the
                               highest-value slice of the blind audit
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from eval.config import DEFAULT_RESULTS_DIR  # noqa: E402
from eval.production_e2e.baseline_direct import load_topics  # noqa: E402

REPLAY_STATUS = "replay_v7"
# Same semantic-overlap bound as the production mining dedup: a v7
# opportunity whose fingerprint embeds >=0.78 close to some v6 gap is the
# same discovery reworded, not a new one.
NOVELTY_SIMILARITY = 0.78

# Pilot topic set: two per stratum, all v6-terminal propose_idea tasks (the
# abstained/abstain topics are excluded so both arms had a complete funnel).
DEFAULT_PILOT_TOPICS = [
    "pe2e-001", "pe2e-007",   # narrow_mature
    "pe2e-013", "pe2e-015",   # emerging_sparse
    "pe2e-019", "pe2e-024",   # broad_ambiguous
]

# --- Survival basis (P1, 2026-10-28) ---------------------------------------
# A surviving gap can reach that state two very different ways, and the
# difference decides whether a v6-vs-v7 surviving-rate gap is attributable to
# the ontology or to the audit gate:
#   self_report          the auditor's own novelty_confidence cleared the
#                        low-novelty floor — the verdict rests on the audit's
#                        search.
#   mechanical_override  the auditor self-reported LOW novelty (<=0.4) and the
#                        claim-coverage matrix overrode it upward
#                        (gap_audit_mechanical_novelty_backed). The verdict
#                        rests on "no covering neighbour among those found" —
#                        which, when external retrieval is rate-limited, can
#                        read search failure as novelty.
#   provisional          budgeted low-novelty provisional survive (no
#                        mechanical backing available).
#   unknown              surviving with no audit row at all (data defect).
SURVIVAL_SELF_REPORT = "self_report"
SURVIVAL_MECHANICAL_OVERRIDE = "mechanical_override"
SURVIVAL_PROVISIONAL = "provisional"
SURVIVAL_UNKNOWN = "unknown"
SURVIVAL_BASES = (
    SURVIVAL_SELF_REPORT, SURVIVAL_MECHANICAL_OVERRIDE,
    SURVIVAL_PROVISIONAL, SURVIVAL_UNKNOWN,
)

_MECH_OVERRIDE_TRACE = "gap_audit_mechanical_novelty_backed"
_PROVISIONAL_TRACE = "gap_audit_low_novelty_provisional_survive"


def _empty_basis_counts() -> dict:
    counts = {"surviving_total": 0}
    counts.update({basis: 0 for basis in SURVIVAL_BASES})
    return counts


def collect_mechanical_overrides(db, task_ids) -> dict[str, dict]:
    """gap_id -> how the audit gate overrode the auditor's own novelty score.

    Both override traces are read: the mechanical-backing override (score
    corrected upward from the coverage matrix) and the budgeted provisional
    survive (no backing available, downstream gate caps the ideas instead).
    """
    from app.db.models import AgentTrace

    overrides: dict[str, dict] = {}
    for task_id in task_ids:
        traces = db.query(AgentTrace).filter(
            AgentTrace.task_id == task_id,
            AgentTrace.step_name.in_([_MECH_OVERRIDE_TRACE, _PROVISIONAL_TRACE]),
        ).order_by(AgentTrace.created_at).all()
        for trace in traces:
            try:
                data = json.loads(trace.output_json)
            except Exception:
                continue
            gap_id = data.get("gap_id")
            if not gap_id:
                continue
            if trace.step_name == _MECH_OVERRIDE_TRACE:
                overrides[gap_id] = {
                    "kind": SURVIVAL_MECHANICAL_OVERRIDE,
                    "mechanical_novelty": data.get("mechanical_novelty"),
                    "audit_self_reported_novelty": data.get("self_reported_novelty"),
                }
            else:
                overrides[gap_id] = {
                    "kind": SURVIVAL_PROVISIONAL,
                    "mechanical_novelty": None,
                    "audit_self_reported_novelty": data.get("novelty_confidence"),
                }
    return overrides


def classify_survival_basis(status: str | None, audit_info: dict | None,
                            override_info: dict | None) -> str | None:
    """Which gate actually let this gap through (None when not surviving)."""
    if status != "surviving":
        return None
    if override_info:
        kind = override_info.get("kind")
        if kind in (SURVIVAL_MECHANICAL_OVERRIDE, SURVIVAL_PROVISIONAL):
            return kind
    novelty = (audit_info or {}).get("novelty_confidence")
    if novelty is not None:
        from app.agent.steps.audit_gaps import _LOW_NOVELTY_CONFIRMED

        if novelty >= _LOW_NOVELTY_CONFIRMED:
            return SURVIVAL_SELF_REPORT
    return SURVIVAL_UNKNOWN


def collect_survival_basis(db, task_id: str, contract_id: str) -> dict:
    """Surviving-gap count split by which gate admitted it (one arm)."""
    from app.db.models import GapAudit
    from app.db.repositories import gap_repo

    gaps = gap_repo.list_gaps_for_contract(
        db, task_id, contract_id, include_superseded=True)
    overrides = collect_mechanical_overrides(db, {task_id})
    counts = _empty_basis_counts()
    for gap in gaps:
        if gap.status != "surviving":
            continue
        counts["surviving_total"] += 1
        audit = db.query(GapAudit).filter(GapAudit.gap_id == gap.id).order_by(
            GapAudit.created_at.desc()).first()
        audit_info = ({"novelty_confidence": audit.novelty_confidence}
                      if audit else None)
        basis = classify_survival_basis(
            gap.status, audit_info, overrides.get(gap.id))
        counts[basis] += 1
    return counts


def build_survival_2x2(records: list[dict]) -> dict:
    """{v6, v7, v7-only} x {self_report, mechanical_override, ...} table.

    This is the P1 de-confounding step. The v7 surviving RATE is only evidence
    for the ontology if it holds up on the self_report column: a gap that
    survives because the auditor's own novelty score cleared the floor is a
    discovery claim, while one that survives because the coverage matrix
    overrode a low self-report is a claim about the audit gate.

    Rates are per CANDIDATE, not per surviving gap. Per-surviving shares
    answer "what fraction of survivors came from each gate", which cannot be
    compared across arms with different surviving counts; per-candidate rates
    decompose the surviving-rate difference into the part the ontology
    contributes and the part the gate contributes.
    """
    v6 = _empty_basis_counts()
    v7 = _empty_basis_counts()
    v7_only = _empty_basis_counts()
    candidates = {"v6": 0, "v7": 0, "v7_only": 0}
    for record in records:
        if not record_is_usable(record) or not record.get("replay_task_id"):
            continue
        for key, target in (("v6_survival_basis", v6), ("v7_survival_basis", v7)):
            basis_counts = record.get(key)
            if not basis_counts:
                continue
            for field, value in basis_counts.items():
                target[field] = target.get(field, 0) + value
        candidates["v6"] += (record.get("v6_funnel") or {}).get("gap_candidates", 0)
        candidates["v7"] += (record.get("v7_funnel") or {}).get("gap_candidates", 0)
        for row in record.get("v7_vs_v6", {}).get("gaps", []):
            if not row.get("is_new_opportunity"):
                continue
            candidates["v7_only"] += 1
            basis = row.get("survival_basis")
            if not basis:
                continue
            v7_only["surviving_total"] += 1
            v7_only[basis] = v7_only.get(basis, 0) + 1

    def _rate(counts: dict, basis: str, arm: str) -> float | None:
        total = candidates.get(arm) or 0
        if not total:
            return None
        return round(counts.get(basis, 0) / total, 3)

    rates = {
        arm: {
            "candidates": candidates[arm],
            "surviving_rate": (round(counts["surviving_total"] / candidates[arm], 3)
                               if candidates[arm] else None),
            "self_report_rate": _rate(counts, SURVIVAL_SELF_REPORT, arm),
            "mechanical_override_rate": _rate(
                counts, SURVIVAL_MECHANICAL_OVERRIDE, arm),
        }
        for arm, counts in (("v6", v6), ("v7", v7), ("v7_only", v7_only))
    }

    # Decomposition of the v7-vs-v6 surviving-rate difference. The ontology
    # claim survives only if the self_report component is positive and
    # non-trivial; a gain concentrated in mechanical_override is the gate.
    decomposition = None
    v6_rates, v7_rates = rates["v6"], rates["v7"]
    if (v6_rates["surviving_rate"] is not None
            and v7_rates["surviving_rate"] is not None):
        delta_total = round(v7_rates["surviving_rate"] - v6_rates["surviving_rate"], 3)
        delta_self = round((v7_rates["self_report_rate"] or 0)
                           - (v6_rates["self_report_rate"] or 0), 3)
        delta_mech = round((v7_rates["mechanical_override_rate"] or 0)
                           - (v6_rates["mechanical_override_rate"] or 0), 3)
        decomposition = {
            "delta_surviving_rate": delta_total,
            "delta_from_self_report": delta_self,
            "delta_from_mechanical_override": delta_mech,
            "mechanical_share_of_gain": (round(delta_mech / delta_total, 3)
                                         if delta_total else None),
        }

    return {
        "note": "surviving gaps split by the gate that admitted them; "
                "mechanical_override = auditor self-reported low novelty and "
                "the claim-coverage matrix overrode it upward. Rates are per "
                "candidate, so they are comparable across arms.",
        "counts": {"v6": v6, "v7": v7, "v7_only": v7_only},
        "rates": rates,
        "decomposition": decomposition,
        "interpretation_rule": "the ontology claim requires "
                               "delta_from_self_report > 0; a gain concentrated "
                               "in delta_from_mechanical_override belongs to the "
                               "audit gate, not to the 7-lens ontology",
    }


def build_p2_counterfactual(db, records: list[dict],
                            floor: float | None = None) -> dict:
    """Surviving funnel once the relevance floor is enforced on override gaps.

    The surviving gaps in the database were admitted under whatever floor was
    active when they were audited (typically 0 — no floor at all). Their stored
    novelty_confidence has since been OVERWRITTEN with the mechanically backed
    score (that is what the override does), so the admission path cannot be
    read off the audit row any more — only the
    `gap_audit_mechanical_novelty_backed` trace still records it. An audit row
    that looks like a self-reported 0.9 may have been a self-reported 0.2 that
    the matrix corrected.

    This recomputes, for the gaps that entered through that path, whether the
    score would still clear `_MECHANICAL_NOVELTY_FLOOR` at `floor`, and reports
    the surviving rate each arm would have had. Gaps admitted on any other path
    (self-report, provisional survive) are untouched by definition: P2 only
    gates the mechanical override.

    Returns {"enabled": False} when the floor is off, because a counterfactual
    of the current behaviour is not informative.
    """
    from app.agent.steps.audit_gaps import (
        _MECHANICAL_NOVELTY_FLOOR,
        _mechanical_novelty_score,
    )
    from app.config import settings
    from app.db.models import AgentTrace, GapCandidate
    from app.db.repositories import gap_repo

    effective_floor = (settings.gap_relevance_min_neighbor
                       if floor is None else floor)
    if effective_floor <= 0:
        return {"enabled": False,
                "note": "relevance floor is 0 — nothing to counterfactual"}

    task_ids = {record.get(key) for record in records
                for key in ("v6_task_id", "replay_task_id")
                if record.get(key)}
    overridden: set[str] = set()
    for trace in db.query(AgentTrace).filter(
            AgentTrace.task_id.in_(task_ids),
            AgentTrace.step_name == _MECH_OVERRIDE_TRACE).all():
        try:
            gap_id = json.loads(trace.output_json).get("gap_id")
        except Exception:
            continue
        if gap_id:
            overridden.add(gap_id)

    arms = {"v6": {"candidates": 0, "surviving_now": 0, "surviving_after": 0},
            "v7": {"candidates": 0, "surviving_now": 0, "surviving_after": 0}}
    dropped: list[dict] = []
    for record in records:
        if not record_is_usable(record):
            continue
        for arm, task_key, contract_key in (
            ("v6", "v6_task_id", None),
            ("v7", "replay_task_id", "replay_contract_id"),
        ):
            task_id = record.get(task_key)
            if not task_id:
                continue
            contract_id = record.get(contract_key) if contract_key else None
            if not contract_id:
                from app.db.models import ResearchContract

                contract = db.query(ResearchContract).filter(
                    ResearchContract.task_id == task_id,
                    ResearchContract.status == "active").first()
                contract_id = contract.id if contract else None
            if not contract_id:
                continue
            gaps = gap_repo.list_gaps_for_contract(
                db, task_id, contract_id, include_superseded=True)
            arms[arm]["candidates"] += len(gaps)
            for gap in gaps:
                if gap.status != "surviving":
                    continue
                arms[arm]["surviving_now"] += 1
                if gap.id not in overridden:
                    arms[arm]["surviving_after"] += 1
                    continue
                score = _mechanical_novelty_score(
                    db, gap, relevance_floor=effective_floor)
                if score is not None and score["novelty"] >= _MECHANICAL_NOVELTY_FLOOR:
                    arms[arm]["surviving_after"] += 1
                else:
                    dropped.append({
                        "topic_id": record.get("topic_id"),
                        "arm": arm,
                        "gap_id": gap.id,
                        "gap_type": gap.gap_type,
                        "novelty_after_floor": (round(score["novelty"], 3)
                                                if score else None),
                        "neighbor_relevance_median": (score["neighbor_relevance_median"]
                                                      if score else None),
                    })

    for arm in arms.values():
        candidates = arm["candidates"] or 1
        arm["surviving_rate_now"] = round(arm["surviving_now"] / candidates, 3)
        arm["surviving_rate_after"] = round(arm["surviving_after"] / candidates, 3)
        arm["rate_delta"] = round(
            arm["surviving_rate_after"] - arm["surviving_rate_now"], 3)

    return {
        "enabled": True,
        "floor": effective_floor,
        "mechanical_floor": _MECHANICAL_NOVELTY_FLOOR,
        "override_traces_found": len(overridden),
        "arms": arms,
        "dropped": dropped,
        "n_dropped": len(dropped),
        "note": "gaps admitted by self-report or provisional survive are "
                "unaffected by construction; only the mechanical override is "
                "gated by the relevance floor. A removed judgment can be a "
                "PARTIAL, so novelty may RISE for some gaps (observed: one gap "
                "0.75 -> 0.88) — that is correct: an irrelevant neighbour's "
                "claim of partial coverage was never evidence, and dropping it "
                "leaves the claim genuinely uncovered.",
    }


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_topic_task_map(source_run: str) -> dict[str, str]:
    """topic_id -> v6 task_id from the frozen source run's predictions."""
    pred_path = Path(DEFAULT_RESULTS_DIR) / source_run / "predictions.jsonl"
    mapping: dict[str, str] = {}
    with open(pred_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("task_id"):
                mapping[row["topic_id"]] = row["task_id"]
    return mapping


def _copy_row(source, overrides: dict) -> dict:
    """Column values of `source` minus primary keys, plus overrides."""
    values = {}
    for column in source.__table__.columns:
        if column.primary_key:
            continue
        values[column.name] = getattr(source, column.name)
    values.update(overrides)
    return values


def collect_funnel(db, task_id: str, contract_id: str) -> dict:
    """Gap/audit/idea funnel of one task (works for both v6 and v7 rows)."""
    from app.db.models import ResearchIdea
    from app.db.repositories import gap_repo

    gaps = gap_repo.list_gaps_for_contract(
        db, task_id, contract_id, include_superseded=True)
    by_type: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for gap in gaps:
        by_type[gap.gap_type] = by_type.get(gap.gap_type, 0) + 1
        by_status[gap.status] = by_status.get(gap.status, 0) + 1
    ideas = db.query(ResearchIdea).filter(ResearchIdea.task_id == task_id).all()
    return {
        "gap_candidates": len(gaps),
        "gaps_by_type": by_type,
        "gaps_by_status": by_status,
        "surviving": by_status.get("surviving", 0),
        "ideas": len(ideas),
        "executable_ideas": sum(1 for idea in ideas
                                if idea.decision == "executable_candidate"),
    }


async def compare_v7_against_v6(db, v7_gaps, v6_gaps) -> dict:
    """Mark each v7 gap as new vs covered by a semantically matching v6 gap.

    Uses the production embedding + the production dedup bound, so "new"
    means exactly what the mining dedup would have refused to merge.
    """
    from app.agent.steps.mine_gaps import _embed_fingerprints, _gap_fingerprint
    from app.services.embedding_service import cosine_similarity

    v7_fps = [_gap_fingerprint(g.observed_problem, g.missing_capability,
                               g.claimed_delta) for g in v7_gaps]
    v6_fps = [_gap_fingerprint(g.observed_problem, g.missing_capability,
                               g.claimed_delta) for g in v6_gaps]
    v6_vecs = await _embed_fingerprints(v6_fps) if v6_fps else []
    method = "embedding"
    if v6_fps and (not v6_vecs or len(v6_vecs) != len(v6_fps)):
        method = "token_overlap_fallback"

    def _tokens(text: str) -> set[str]:
        return {token for token in text.lower().split() if len(token) > 3}

    from app.db.models import GapAudit

    latest_audit_by_gap = {}
    for gap in v7_gaps:
        audit = db.query(GapAudit).filter(GapAudit.gap_id == gap.id).order_by(
            GapAudit.created_at.desc()).first()
        if audit:
            latest_audit_by_gap[gap.id] = {
                "audit_result": audit.audit_result,
                "recommended_action": audit.recommended_action,
                "novelty_confidence": (round(audit.novelty_confidence, 3)
                                       if audit.novelty_confidence is not None else None),
                "search_admission_status": audit.search_admission_status,
            }

    # Pre-registered (2026-09-21): preserve the mechanical-vs-self-reported
    # novelty disagreement on every v7 row. The two signals measure different
    # things — mechanical coverage ("no covering neighbour among those found")
    # vs auditor self-confidence — and neither is ground truth until the blind
    # human adjudication lands. Keep both so the 2x2 against the human verdict
    # can be built without touching the DB again.
    overrides_by_gap = collect_mechanical_overrides(
        db, {gap.task_id for gap in v7_gaps})
    mechanical_by_gap: dict[str, dict] = {}
    for gap_id, info in overrides_by_gap.items():
        if info.get("kind") != SURVIVAL_MECHANICAL_OVERRIDE:
            continue
        mechanical_by_gap[gap_id] = {
            "mechanical_novelty": info.get("mechanical_novelty"),
            "audit_self_reported_novelty": info.get("audit_self_reported_novelty"),
        }

    rows = []
    for gap, fingerprint in zip(v7_gaps, v7_fps):
        best = 0.0
        best_v6_id = None
        if method == "embedding":
            vec = (await _embed_fingerprints([fingerprint]) or [None])[0]
            if vec and v6_vecs:
                for v6_gap, v6_vec in zip(v6_gaps, v6_vecs):
                    if not v6_vec:
                        continue
                    similarity = cosine_similarity(vec, v6_vec)
                    if similarity > best:
                        best, best_v6_id = similarity, v6_gap.id
        else:
            tokens = _tokens(fingerprint)
            for v6_gap, v6_fp in zip(v6_gaps, v6_fps):
                overlap = len(tokens & _tokens(v6_fp))
                union = len(tokens | _tokens(v6_fp)) or 1
                similarity = overlap / union
                if similarity > best:
                    best, best_v6_id = similarity, v6_gap.id
        audit_info = latest_audit_by_gap.get(gap.id)
        override_info = overrides_by_gap.get(gap.id)
        rows.append({
            "gap_id": gap.id,
            "gap_type": gap.gap_type,
            "status": gap.status,
            "claimed_delta": (gap.claimed_delta or "")[:400],
            "max_similarity_to_v6": round(best, 4),
            "closest_v6_gap_id": best_v6_id,
            "is_new_opportunity": best < NOVELTY_SIMILARITY,
            # P1: which gate admitted this gap, so the surviving-rate
            # comparison can be de-confounded from the mechanical-novelty
            # override without re-running anything.
            "survival_basis": classify_survival_basis(
                gap.status, audit_info, override_info),
            **latest_audit_by_gap.get(gap.id, {}),
            **mechanical_by_gap.get(gap.id, {}),
        })
    return {"method": method, "gaps": rows}


async def dry_run_topic(db, topic_id: str, stratum: str, task_id: str) -> dict:
    """Read-only: v7 admission arithmetic over the frozen v6 evidence pool."""
    from app.agent.evidence_relations import MIN_RELATION_RELEVANCE
    from app.agent.steps.mine_gaps import (
        _OPPORTUNITY_CONTRACTS,
        evaluate_gap_mining_admission,
    )
    from app.db.models import EvidenceUnit, QuestionEvidenceLink, ResearchContract, ResearchQuestion

    contract = db.query(ResearchContract).filter(
        ResearchContract.task_id == task_id,
        ResearchContract.status == "active",
    ).first()
    if not contract:
        return {"topic_id": topic_id, "stratum": stratum, "error": "no active contract"}

    questions = db.query(ResearchQuestion).filter(
        ResearchQuestion.task_id == task_id,
        ResearchQuestion.contract_id == contract.id,
    ).all()
    question_ids = [question.id for question in questions]
    links = db.query(QuestionEvidenceLink).filter(
        QuestionEvidenceLink.question_id.in_(question_ids)).all()
    links_by_question: dict[str, list] = {qid: [] for qid in question_ids}
    for link in links:
        links_by_question.setdefault(link.question_id, []).append(link)
    evidence_by_id = {
        item.id: item for item in db.query(EvidenceUnit).filter(
            EvidenceUnit.task_id == task_id,
            EvidenceUnit.id.in_({link.evidence_id for link in links}),
        ).all()}

    admissions = [evaluate_gap_mining_admission(
        question, links_by_question[question.id], evidence_by_id)
        for question in questions]
    passed = [item for item in admissions if item.status == "PASS"]

    # Raw material per lens: which opportunity types have admissible evidence
    # to work with on this frozen pool.
    lens_material = {}
    for spec in _OPPORTUNITY_CONTRACTS:
        if spec.required_evidence_types is None:
            available = sum(len(item.admissible_evidence_ids) for item in passed)
        else:
            available = sum(
                1 for item in passed for evidence_id in item.admissible_evidence_ids
                if evidence_id in evidence_by_id
                and evidence_by_id[evidence_id].evidence_type in spec.required_evidence_types)
        lens_material[spec.gap_type] = available

    return {
        "topic_id": topic_id,
        "stratum": stratum,
        "v6_task_id": task_id,
        "questions_total": len(questions),
        "admission_passed": len(passed),
        "admission_results": [{
            "status": item.status, "reason_codes": item.reason_codes,
            "contradiction_count": len(item.contradicting_evidence_ids),
            "tier": item.evidence_tier,
        } for item in admissions],
        "evidence_linked": len(evidence_by_id),
        "conflict_signal_questions": sum(
            1 for item in admissions if item.contradicting_evidence_ids),
        "v6_funnel": collect_funnel(db, task_id, contract.id),
        "lens_material_estimate": lens_material,
    }


async def replay_topic(db, topic_id: str, stratum: str, v6_task_id: str,
                       run_id: str) -> dict:
    """Copy the frozen input, run the v7 downstream, compare against v6."""
    from app.agent.runner import _apply_gap_competition_selection
    from app.agent.state import ResearchState
    from app.agent.steps.audit_gaps import audit_gap_candidates
    from app.agent.steps.generate_interventions import generate_interventions
    from app.agent.steps.generate_minimal_experiments import generate_minimal_experiments
    from app.agent.steps.generate_phenomenon_plans import generate_phenomenon_plans
    from app.agent.steps.mine_gaps import GAP_MINING_POLICY_VERSION, mine_gap_candidates
    from app.agent.steps.narrow_gaps import narrow_audited_gaps
    from app.db.models import (
        EvidenceUnit,
        QuestionEvidenceLink,
        ResearchContract,
        ResearchQuestion,
        ResearchTask,
        TaskPaper,
    )
    from app.db.repositories import task_repo
    from app.llm.factory import get_llm

    v6_contract = db.query(ResearchContract).filter(
        ResearchContract.task_id == v6_task_id,
        ResearchContract.status == "active",
    ).first()
    if not v6_contract:
        return {"topic_id": topic_id, "error": "no active v6 contract"}

    v6_funnel = collect_funnel(db, v6_task_id, v6_contract.id)

    # --- Copy the frozen discovery input under a new replay task -----------
    v6_task = db.get(ResearchTask, v6_task_id)
    replay_task = ResearchTask(**_copy_row(v6_task, {
        "user_input": f"[v7-replay:{run_id}] {v6_task.user_input}",
        "status": REPLAY_STATUS,
    }))
    db.add(replay_task)
    db.flush()

    replay_contract = ResearchContract(**_copy_row(v6_contract, {
        "task_id": replay_task.id,
    }))
    db.add(replay_contract)
    db.flush()

    questions = db.query(ResearchQuestion).filter(
        ResearchQuestion.task_id == v6_task_id,
        ResearchQuestion.contract_id == v6_contract.id).all()
    question_id_map: dict[str, str] = {}
    for question in questions:
        row = ResearchQuestion(**_copy_row(question, {
            "task_id": replay_task.id, "contract_id": replay_contract.id}))
        db.add(row)
        db.flush()
        question_id_map[question.id] = row.id

    question_ids = list(question_id_map)
    links = db.query(QuestionEvidenceLink).filter(
        QuestionEvidenceLink.question_id.in_(
            [question.id for question in questions])).all()
    evidence_by_v6_id = {
        item.id: item for item in db.query(EvidenceUnit).filter(
            EvidenceUnit.task_id == v6_task_id,
            EvidenceUnit.id.in_({link.evidence_id for link in links}),
        ).all()}
    evidence_id_map: dict[str, str] = {}
    for v6_evidence in evidence_by_v6_id.values():
        row = EvidenceUnit(**_copy_row(v6_evidence, {
            "task_id": replay_task.id}))
        db.add(row)
        db.flush()
        evidence_id_map[v6_evidence.id] = row.id

    for link in links:
        if link.evidence_id not in evidence_id_map:
            continue
        db.add(QuestionEvidenceLink(**_copy_row(link, {
            "question_id": question_id_map[link.question_id],
            "evidence_id": evidence_id_map[link.evidence_id],
        })))

    task_papers = db.query(TaskPaper).filter(TaskPaper.task_id == v6_task_id).all()
    for task_paper in task_papers:
        db.add(TaskPaper(**_copy_row(task_paper, {"task_id": replay_task.id})))
    db.commit()

    frozen_counts = {
        "questions": len(questions),
        "evidence_units": len(evidence_id_map),
        "links": len(links),
        "task_papers": len(task_papers),
    }

    # --- Run the v7 downstream on the frozen input -------------------------
    llm = get_llm()
    state = ResearchState(task_id=replay_task.id, contract_id=replay_contract.id,
                          pipeline_version=2, current_round=1)
    cost: dict = {"policy_version": GAP_MINING_POLICY_VERSION}

    started = time.monotonic()
    mined = await mine_gap_candidates(db, state, llm, replay_task.id)
    cost["mining_seconds"] = round(time.monotonic() - started, 1)
    cost["mined_candidates"] = len(mined)

    audit_ids = _apply_gap_competition_selection(
        db, replay_task.id, mined, [gap.id for gap in mined], False)
    started = time.monotonic()
    audits = await audit_gap_candidates(db, state, llm, replay_task.id,
                                        gap_ids=audit_ids)
    cost["audit_seconds"] = round(time.monotonic() - started, 1)
    cost["audited"] = len(audits)

    state = task_repo.get_state(db, replay_task.id)
    narrowed = await narrow_audited_gaps(db, state, replay_task.id, llm=llm)
    if narrowed:
        cost["narrowed"] = len(narrowed)
        await audit_gap_candidates(db, state, llm, replay_task.id, gap_ids=narrowed)
        state = task_repo.get_state(db, replay_task.id)

    started = time.monotonic()
    await generate_phenomenon_plans(db, state, llm, replay_task.id)
    state = task_repo.get_state(db, replay_task.id)
    interventions = await generate_interventions(db, state, llm, replay_task.id)
    state = task_repo.get_state(db, replay_task.id)
    experiments = await generate_minimal_experiments(db, state, llm, replay_task.id)
    cost["downstream_seconds"] = round(time.monotonic() - started, 1)
    cost["interventions_passed"] = len(getattr(interventions, "passed_intervention_ids", []) or [])
    cost["experiments_ideas"] = len(getattr(experiments, "idea_ids", []) or [])

    # --- Compare -----------------------------------------------------------
    from app.db.repositories import gap_repo
    v7_gaps = gap_repo.list_gaps_for_contract(
        db, replay_task.id, replay_contract.id, include_superseded=True)
    v6_gaps = gap_repo.list_gaps_for_contract(
        db, v6_task_id, v6_contract.id, include_superseded=True)
    comparison = await compare_v7_against_v6(db, v7_gaps, v6_gaps)

    return {
        "topic_id": topic_id,
        "stratum": stratum,
        "v6_task_id": v6_task_id,
        "replay_task_id": replay_task.id,
        "replay_contract_id": replay_contract.id,
        "frozen": frozen_counts,
        "v6_funnel": v6_funnel,
        "v7_funnel": collect_funnel(db, replay_task.id, replay_contract.id),
        "v6_survival_basis": collect_survival_basis(db, v6_task_id, v6_contract.id),
        "v7_survival_basis": collect_survival_basis(
            db, replay_task.id, replay_contract.id),
        "v7_vs_v6": comparison,
        "cost": cost,
    }


def record_is_usable(record: dict) -> bool:
    """A record whose numbers may enter an aggregate.

    Two failure modes must both be excluded. `error` is a topic that never
    produced a replay at all. `audit_pending_error` is subtler and was the
    actual defect in the 2026-09-22 pilot: the follow-up audit sweep failed
    per topic, the record was written back as a copy of its PREVIOUS snapshot,
    and the aggregate silently reported those stale numbers as if the sweep
    had succeeded (report said topics_errored=0 while all 6 topics carried an
    error and the DB had moved on to 23 surviving gaps).
    """
    return not record.get("error") and not record.get("audit_pending_error")


def record_is_stale(record: dict) -> bool:
    """Record whose stored funnel predates a failed follow-up sweep."""
    return bool(record.get("audit_pending_error"))


def collect_db_truth(db, records: list[dict]) -> dict[str, dict]:
    """Ground-truth funnel counts straight from the DB, per topic.

    The report must not be able to disagree with the database: every number in
    it is derived from these rows, so a mismatch means the record snapshot is
    stale (a failed sweep wrote back an old copy) and the report is wrong.
    """
    from app.db.models import ResearchContract
    from app.db.repositories import gap_repo

    truth: dict[str, dict] = {}
    for record in records:
        topic_id = record.get("topic_id")
        if not topic_id:
            continue
        entry: dict = {}
        for arm, task_key, contract_key in (
            ("v6", "v6_task_id", None),
            ("v7", "replay_task_id", "replay_contract_id"),
        ):
            task_id = record.get(task_key)
            if not task_id:
                continue
            contract_id = record.get(contract_key) if contract_key else None
            if not contract_id:
                contract = db.query(ResearchContract).filter(
                    ResearchContract.task_id == task_id,
                    ResearchContract.status == "active").first()
                contract_id = contract.id if contract else None
            if not contract_id:
                continue
            gaps = gap_repo.list_gaps_for_contract(
                db, task_id, contract_id, include_superseded=True)
            entry[arm] = {
                "gap_candidates": len(gaps),
                "surviving": sum(1 for gap in gaps if gap.status == "surviving"),
            }
        truth[topic_id] = entry
    return truth


def check_report_consistency(records: list[dict], db_truth: dict | None) -> list[str]:
    """Every disagreement between a record snapshot and the DB, as strings.

    Returns an empty list when the report is trustworthy. Callers must treat a
    non-empty list as a hard failure — the whole point of P0 is that a stale
    report must never again be written out looking healthy.
    """
    if not db_truth:
        return []
    problems: list[str] = []
    for record in records:
        topic_id = record.get("topic_id")
        truth = db_truth.get(topic_id)
        if not truth:
            continue
        for arm in ("v6", "v7"):
            expected = truth.get(arm)
            if not expected:
                continue
            funnel = record.get(f"{arm}_funnel") or {}
            for key in ("gap_candidates", "surviving"):
                actual = funnel.get(key)
                if actual is None:
                    continue
                if actual != expected[key]:
                    problems.append(
                        f"{topic_id}/{arm}.{key}: record={actual} db={expected[key]}")
    return problems


def summarize(records: list[dict], cluster_meta: dict | None = None,
              db_truth: dict | None = None,
              p2_counterfactual: dict | None = None) -> dict:
    """Aggregate the pre-registered five-question report over per-topic records.

    Reporting protocol (frozen 2026-09-21, BEFORE the full audit finished):

    - Layer separation: this report answers Layer 1 (Opportunity Discovery
      Quality under a full-jury audit). v7 numbers here must NOT be compared
      against v6's budgeted-production surviving counts — v6's historical
      funnel is the terminal state of a budgeted multi-round run, v7's is a
      full-jury single sweep. Layer 2 (Production System Value under a fixed
      identical budget) is a separate future experiment.
    - Q2 (marginal opportunity precision) and Q3 (marginal credible idea
      yield) are filled only after the BLIND human prior-art adjudication of
      v7_only_opportunities.jsonl.
    - Pilot gate (pre-registered): GO iff at least one v7-only opportunity is
      human-true AND yields a credible idea AND the new opportunities are not
      dominated by false-opens. Sample size is a pilot gate, not a per-type
      deletion verdict.

    Only `record_is_usable` records enter any aggregate; stale ones are listed
    by name so a reader can see exactly which topics the numbers exclude.
    """
    usable = [record for record in records if record_is_usable(record)]
    stale = [record.get("topic_id") for record in records if record_is_stale(record)]
    errored = [record.get("topic_id") for record in records
               if record.get("error")]

    new_rows = []
    for record in usable:
        new_rows.extend([row for row in record["v7_vs_v6"]["gaps"]
                         if row["is_new_opportunity"]])
    by_type_new: dict[str, int] = {}
    for row in new_rows:
        by_type_new[row["gap_type"]] = by_type_new.get(row["gap_type"], 0) + 1

    # Per-type funnel: the machine-judgeable columns are computed here (new /
    # audited / inconclusive / surviving / rejected); the human columns
    # (true / false-open / credible idea) stay empty until the blind audit.
    per_type: dict[str, dict] = {}
    for row in new_rows:
        bucket = per_type.setdefault(row["gap_type"], {
            "new": 0, "audited": 0, "inconclusive": 0, "surviving": 0,
            "rejected": 0, "other_status": 0,
            "surviving_self_report": 0, "surviving_mechanical_override": 0,
            "true_gap": None, "false_open": None, "credible_idea": None})
        bucket["new"] += 1
        if row.get("audit_result") or row.get("recommended_action"):
            bucket["audited"] += 1
        status = row.get("status")
        if status == "inconclusive":
            bucket["inconclusive"] += 1
        elif status == "surviving":
            bucket["surviving"] += 1
            basis = row.get("survival_basis")
            if basis == SURVIVAL_SELF_REPORT:
                bucket["surviving_self_report"] += 1
            elif basis == SURVIVAL_MECHANICAL_OVERRIDE:
                bucket["surviving_mechanical_override"] += 1
        elif status == "rejected":
            bucket["rejected"] += 1
        elif status not in (None, "candidate", "auditing"):
            bucket["other_status"] += 1

    def _sum(arm: str, key: str) -> int:
        return sum(record.get(arm, {}).get(key, 0) for record in usable
                   if arm in record)

    total_v7 = _sum("v7_funnel", "gap_candidates")
    audit_cost = {
        "first_sweep_audited": sum(record.get("cost", {}).get("audited", 0)
                                   for record in usable),
        "pending_sweep_audited": sum(record.get("cost", {}).get("audit_pending_count", 0)
                                     for record in usable),
        "mining_llm_calls": sum(
            7 for record in usable),  # 7 lens calls/topic (fixed)
        "mining_seconds": sum(record.get("cost", {}).get("mining_seconds", 0)
                              for record in usable),
        "audit_seconds": (sum(record.get("cost", {}).get("audit_seconds", 0)
                              for record in usable)
                          + sum(record.get("cost", {}).get("audit_pending_seconds", 0)
                                for record in usable)),
    }
    consistency = check_report_consistency(records, db_truth)
    return {
        "generated_at": _utcnow(),
        "reporting_protocol": {
            "pre_registered": "2026-09-21",
            "layer_1_this_report": "Opportunity Discovery Quality (full-jury audit)",
            "layer_2_deferred": "Production System Value (fixed-budget v6 vs v7 policy) — "
                                "separate future experiment; do NOT headline-compare "
                                "v7 full-jury surviving vs v6 budgeted surviving",
            "more_search_rule": "at most one remediation re-audit, then terminal=inconclusive",
            "pilot_gate": "GO iff >=1 v7-only opportunity is human-true AND yields a "
                          "credible idea AND new opportunities are not mostly false-open; "
                          "6 topics decide GO/NO-GO only, not per-type deletion",
        },
        "topics_completed": len(usable),
        "topics_errored": len(errored),
        "topics_stale_excluded": stale,
        "topics_errored_ids": errored,
        "db_consistency": {
            "checked": bool(db_truth),
            "ok": not consistency,
            "mismatches": consistency,
            "note": "record snapshots vs live DB counts; a mismatch means the "
                    "record is stale and the report must not be used",
        },
        "q1_v7_only_recall": {
            "v7_candidates_total": total_v7,
            "v7_only_new": len(new_rows),
            "note": "recall of the search space only — does not by itself prove "
                    "v7 is better, nor is it cluster-adjusted (see cross_type_clusters)",
        },
        "q2_marginal_opportunity_precision": "PENDING blind human adjudication "
                                             "(true gap / adjudicated v7-only)",
        "q3_marginal_credible_idea_yield": "PENDING blind human adjudication "
                                           "(v7-only ∩ human-true ∩ credible idea)",
        "q4_per_type_funnel": per_type,
        "q5_marginal_cost": audit_cost,
        "q6_survival_basis_2x2": build_survival_2x2(records),
        "q7_p2_relevance_floor_counterfactual": (
            p2_counterfactual if p2_counterfactual is not None else "PENDING"),
        "cross_type_clusters": cluster_meta or "PENDING",
        "v6_totals_context": {
            "gap_candidates": _sum("v6_funnel", "gap_candidates"),
            "surviving": _sum("v6_funnel", "surviving"),
            "ideas": _sum("v6_funnel", "ideas"),
            "note": "budgeted-production terminal state — context only, not a "
                    "like-for-like comparator for the v7 full-jury funnel",
        },
        "pending_human_steps": [
            "blind prior-art audit of v7_only_opportunities.jsonl "
            "(human_prior_art_verdict: FULL / PARTIAL / NONE)",
            "record mechanical-vs-self-reported novelty disagreements as a 2x2 "
            "against the human verdict (fields already on each row)",
            "fill q2/q3/q4 human columns, then freeze the report",
        ],
    }


async def compute_cluster_meta(rows: list[dict]) -> dict:
    """Cross-type duplicate clustering over the v7-only opportunities.

    Seven active lenses can narrate the SAME underlying research question
    three different ways (contradiction / boundary_gap / scaling_regime all
    describing "the method breaks at long context"), which inflates raw
    recall. Raw new-count therefore gets cluster-adjusted: opportunities are
    embedded (same production embedding as the dedup) and connected at
    cosine >= 0.78 (the production dedup bound); connected components are
    the underlying opportunity clusters. Pairs in [0.70, 0.78) are reported
    as borderline for human inspection — no new clustering machinery.
    """
    if len(rows) < 2:
        return {"clusters": len(rows), "raw_opportunities": len(rows),
                "borderline_pairs": [], "method": "trivial"}

    from app.agent.steps.mine_gaps import _embed_fingerprints
    from app.services.embedding_service import cosine_similarity

    texts = [row["claimed_delta"] for row in rows]
    vecs = await _embed_fingerprints(texts)
    if not vecs or len(vecs) != len(rows):
        return {"clusters": len(rows), "raw_opportunities": len(rows),
                "borderline_pairs": [], "method": "embedding unavailable — "
                "raw count used, cluster adjustment not computed"}

    # Union-find over similarity edges.
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    borderline = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if not (vecs[i] and vecs[j]):
                continue
            similarity = cosine_similarity(vecs[i], vecs[j])
            if similarity >= NOVELTY_SIMILARITY:
                parent[find(i)] = find(j)
            elif similarity >= 0.70:
                borderline.append({
                    "pair": [rows[i]["gap_id"], rows[j]["gap_id"]],
                    "types": [rows[i]["gap_type"], rows[j]["gap_type"]],
                    "similarity": round(similarity, 3),
                    "deltas": [rows[i]["claimed_delta"][:120],
                               rows[j]["claimed_delta"][:120]],
                })
    clusters = {find(i) for i in range(len(rows))}
    return {
        "raw_opportunities": len(rows),
        "clusters": len(clusters),
        "cluster_compression": round(len(rows) / len(clusters), 2),
        "borderline_pairs": borderline,
        "method": f"production embedding, union-find at cosine>={NOVELTY_SIMILARITY}",
    }


def build_mech_override_review(db, new_rows: list[dict]) -> list[dict]:
    """Prior-art review sheet for v7-only gaps that survived on the override.

    The gaps that enter surviving through the mechanical override are the ones
    whose neighbours were both RELEVANT and judged non-covering. That is a
    defensible novelty claim — but it is still made under a fixed
    `budgeted_audit_neighbors` budget, so it is exactly the set a human has to
    check against real prior art before v7 is credited with a discovery.

    IMPORTANT — the P2 relevance floor does not retroactively rewrite already
    surviving gaps, so this sheet covers every override gap in the DB (11 in the
    pilot) and marks per row whether it would still survive once the floor is
    enforced (8 in the pilot). The `survives_p2_floor: true` rows are the ones
    that decide the v7 claim; the `false` rows are the ones P2 already judges as
    false novelty, kept here so the floor's verdict can itself be checked.

    Everything needed to judge is inlined: the atomic claims (the thing that
    has to be novel), each neighbour's title and gap-relevance (what the audit
    compared them against), and the per-claim judgment. No adjudication is
    performed here — the two human fields arrive empty.
    """
    if db is None:
        return []
    from app.agent.steps.audit_gaps import (
        _MECHANICAL_NOVELTY_FLOOR,
        _mechanical_novelty_score,
    )
    from app.config import settings
    from app.db.models import GapCandidate, Paper
    from app.db.repositories import gap_repo

    floor = settings.gap_relevance_min_neighbor
    rows = []
    for row in new_rows:
        if row.get("survival_basis") != SURVIVAL_MECHANICAL_OVERRIDE:
            continue
        gap_id = row["gap_id"]
        claims = gap_repo.list_atomic_claims(db, gap_id)
        coverages = gap_repo.list_neighbor_claim_coverage(db, gap_id)
        relevance = {item.paper_id: item.relevance_score
                     for item in gap_repo.list_gap_paper_relevance(db, gap_id)}
        neighbours = {}
        for cov in coverages:
            neighbours[cov.neighbor_paper_id] = cov

        gap = db.get(GapCandidate, gap_id)
        post_p2 = _mechanical_novelty_score(db, gap, relevance_floor=floor) if gap else None
        neighbour_rows = []
        for paper_id in sorted(neighbours,
                               key=lambda pid: -(relevance.get(pid) or 0.0)):
            paper = db.get(Paper, paper_id)
            neighbour_rows.append({
                "paper_id": paper_id,
                "title": (paper.title if paper else None) or "(missing paper)",
                "gap_relevance": relevance.get(paper_id),
            })
        rows.append({
            "topic_id": row.get("topic_id"),
            "stratum": row.get("stratum"),
            "gap_id": gap_id,
            "gap_type": row.get("gap_type"),
            "claimed_delta": row.get("claimed_delta"),
            "mechanical_novelty": row.get("mechanical_novelty"),
            "audit_self_reported_novelty": row.get("audit_self_reported_novelty"),
            # Would the same gap still clear the mechanical floor today, with
            # the P2 relevance floor enforced? true = the row that decides the
            # v7 claim.
            "survives_p2_floor": (post_p2 is not None
                                  and post_p2["novelty"] >= _MECHANICAL_NOVELTY_FLOOR),
            "novelty_after_p2_floor": (round(post_p2["novelty"], 3)
                                       if post_p2 else None),
            "atomic_claims": [
                {"claim_index": claim.claim_index, "claim_text": claim.claim_text}
                for claim in claims],
            "n_neighbors": len(neighbour_rows),
            "neighbors": neighbour_rows,
            # Why this row is here, stated in the artifact itself: the audit
            # disagreed with itself and the matrix won.
            "override_reason": "auditor self-reported novelty at or below the "
                               "low-novelty floor; the claim-coverage matrix "
                               "corrected it upward",
            # Filled by a human against real prior art: FULL / PARTIAL / NONE.
            "human_prior_art_verdict": "",
            "human_notes": "",
        })
    return rows


async def write_outputs(run_id: str, records: list[dict], db=None,
                        strict: bool = True) -> dict:
    """Write the four artifacts; return the report.

    `strict` makes a DB/record disagreement fatal. It defaults to True because
    the failure it guards against is silent: the 2026-09-22 pilot wrote a
    report claiming topics_errored=0 and 6 v7-only surviving gaps while the
    database held 23 surviving gaps across the same 6 topics. A report that
    disagrees with the DB is not a report, it is a stale snapshot.
    """
    out_dir = Path(DEFAULT_RESULTS_DIR) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "replay_topics.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    new_rows = []
    for record in records:
        if not record_is_usable(record):
            continue
        for row in record["v7_vs_v6"]["gaps"]:
            if row["is_new_opportunity"]:
                new_rows.append({
                    "topic_id": record["topic_id"],
                    "stratum": record["stratum"],
                    **row,
                })

    cluster_meta = await compute_cluster_meta(new_rows)
    db_truth = collect_db_truth(db, records) if db is not None else None
    p2_counterfactual = (build_p2_counterfactual(db, records)
                         if db is not None else None)
    report = summarize(records, cluster_meta, db_truth=db_truth,
                       p2_counterfactual=p2_counterfactual)

    with open(out_dir / "replay_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    with open(out_dir / "v7_only_opportunities.jsonl", "w", encoding="utf-8") as handle:
        for row in new_rows:
            handle.write(json.dumps({
                **row,
                # The blind prior-art audit fills these; verdict semantics
                # follow the frozen super-audit protocol (FULL / PARTIAL /
                # NONE coverage of the claimed delta by real prior art).
                "human_prior_art_verdict": "",
                "human_notes": "",
            }, ensure_ascii=False) + "\n")

    # The set that decides whether v7's surviving gain is real: gaps the
    # mechanical override let through. Written separately so the reviewer does
    # not have to reconstruct claims and neighbours from the DB.
    review_rows = build_mech_override_review(db, new_rows)
    with open(out_dir / "mech_override_review.jsonl", "w", encoding="utf-8") as handle:
        for row in review_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    mismatches = report["db_consistency"]["mismatches"]
    if mismatches and strict:
        raise SystemExit(
            "report disagrees with the database — refusing to publish a stale "
            "report:\n  " + "\n  ".join(mismatches[:20])
            + f"\n({len(mismatches)} mismatch(es); re-run with --recompute-only "
              "to rebuild the records from the DB)")
    return report


async def recompute_only(db, run_id: str) -> list[dict]:
    """Rebuild every derived number from the DB, without any LLM call.

    The replay's expensive half (mining + audit) is already committed to the
    database; only the record snapshots in replay_topics.jsonl can be stale.
    This mode re-reads the gaps, re-derives the v6 comparison, the survival
    basis and the funnels, and rewrites the three artifacts — so a failed
    follow-up sweep can be repaired for free instead of re-running the pilot.

    Embeddings are still needed for the v6-vs-v7 novelty comparison and the
    cross-type clustering; those are the only external calls, and they are
    cheap relative to mining/audit.
    """
    from app.db.models import ResearchContract
    from app.db.repositories import gap_repo

    out_dir = Path(DEFAULT_RESULTS_DIR) / run_id
    topics_path = out_dir / "replay_topics.jsonl"
    if not topics_path.exists():
        raise SystemExit(f"no existing replay run at {topics_path}")

    old_records = [json.loads(line) for line in open(topics_path, encoding="utf-8")
                   if line.strip()]
    records: list[dict] = []
    for old in old_records:
        record = dict(old)
        # A stale record is exactly the one worth recomputing: drop the error
        # marker only after the numbers have been rebuilt from the DB below.
        record.pop("audit_pending_error", None)
        replay_task_id = record.get("replay_task_id")
        v6_task_id = record.get("v6_task_id")
        if not replay_task_id or not v6_task_id:
            records.append(record)
            continue
        contract_id = record.get("replay_contract_id")
        if not contract_id:
            contract = db.query(ResearchContract).filter(
                ResearchContract.task_id == replay_task_id,
                ResearchContract.status == "active").first()
            contract_id = contract.id if contract else None
        v6_contract = db.query(ResearchContract).filter(
            ResearchContract.task_id == v6_task_id,
            ResearchContract.status == "active").first()
        if not contract_id or not v6_contract:
            record["audit_pending_error"] = "recompute: missing contract"
            records.append(record)
            continue
        record["replay_contract_id"] = contract_id
        v7_gaps = gap_repo.list_gaps_for_contract(
            db, replay_task_id, contract_id, include_superseded=True)
        v6_gaps = gap_repo.list_gaps_for_contract(
            db, v6_task_id, v6_contract.id, include_superseded=True)
        record["v6_funnel"] = collect_funnel(db, v6_task_id, v6_contract.id)
        record["v7_funnel"] = collect_funnel(db, replay_task_id, contract_id)
        record["v6_survival_basis"] = collect_survival_basis(
            db, v6_task_id, v6_contract.id)
        record["v7_survival_basis"] = collect_survival_basis(
            db, replay_task_id, contract_id)
        record["v7_vs_v6"] = await compare_v7_against_v6(db, v7_gaps, v6_gaps)
        record["recomputed_at"] = _utcnow()
        print(f"[recompute] {record.get('topic_id')}: v7 "
              f"{json.dumps(record['v7_funnel']['gaps_by_status'])}", flush=True)
        records.append(record)

    report = await write_outputs(run_id, records, db=db)
    print(f"[recompute] report rewritten to eval_results/{run_id}/ "
          f"(db_consistency ok={report['db_consistency']['ok']})", flush=True)
    return records


async def audit_pending_topics(db, run_id: str) -> list[dict]:
    """Full-audit follow-up for an existing replay run.

    The production budget configuration (GAP_AUDIT_BUDGETED=1,
    MAX_GAPS_TO_DEEP_AUDIT=1) admits only the top-1 candidate per round to
    deep audit — correct for the live pipeline's cost control, but it leaves
    a single-round replay with ~5/6 of its v7 candidates never judged, which
    makes the v6-vs-v7 funnel comparison meaningless (v6's historical funnel
    is the terminal state of a full budgeted multi-round run).

    This mode re-opens the replay tasks created by an earlier run (no
    re-copy, no re-mining) and sends every still-pending gap (candidate /
    auditing) through the production adversarial audit, then re-runs the
    downstream chain (idempotent: already-planned gaps are skipped) and
    recomputes the funnels and the v6 comparison.

    Declared deviation from production semantics: audit admission here is
    full-jury, not budgeted top-1 — the replay measures the QUALITY of the
    discovered opportunities, not the production scheduler's prioritization.
    """
    from app.agent.steps.audit_gaps import audit_gap_candidates
    from app.agent.steps.generate_interventions import generate_interventions
    from app.agent.steps.generate_minimal_experiments import generate_minimal_experiments
    from app.agent.steps.generate_phenomenon_plans import generate_phenomenon_plans
    from app.db.models import GapCandidate
    from app.db.repositories import gap_repo, task_repo
    from app.llm.factory import get_llm

    out_dir = Path(DEFAULT_RESULTS_DIR) / run_id
    topics_path = out_dir / "replay_topics.jsonl"
    if not topics_path.exists():
        raise SystemExit(f"no existing replay run at {topics_path}; run the replay first")
    old_records = [json.loads(line) for line in open(topics_path, encoding="utf-8")]

    llm = get_llm()
    records: list[dict] = []
    for old in old_records:
        if old.get("error") or not old.get("replay_task_id"):
            records.append(old)
            continue
        topic_id = old["topic_id"]
        try:
            replay_task_id = old["replay_task_id"]
            contract_id = old.get("replay_contract_id")
            if not contract_id:
                # Older record without the contract id — resolve it from the task.
                from app.db.models import ResearchContract

                contract = db.query(ResearchContract).filter(
                    ResearchContract.task_id == replay_task_id,
                    ResearchContract.status == "active").first()
                contract_id = contract.id if contract else None
            if not contract_id:
                records.append(old)
                continue
            pending = db.query(GapCandidate).filter(
                GapCandidate.task_id == replay_task_id,
                GapCandidate.contract_id == contract_id,
                GapCandidate.status.in_(["candidate", "auditing"]),
            ).order_by(GapCandidate.created_at).all()
            print(f"[audit-pending] {topic_id}: {len(pending)} pending gap(s)", flush=True)
            cost = dict(old.get("cost", {}))
            if pending:
                # Pre-registered rule (frozen 2026-09-21): a gap that enters
                # this sweep already suspended on more_search gets exactly ONE
                # remediation re-audit; if it is still undecided afterwards it
                # is terminalized as inconclusive. Hard samples must not
                # accumulate unbounded retrieval budget (outcome-dependent
                # compute) just to be forced into confirmed/rejected.
                reaudit_ids = {g.id for g in pending if g.status == "auditing"}
                state = task_repo.get_state(db, replay_task_id)
                started = time.monotonic()
                results = await audit_gap_candidates(
                    db, state, llm, replay_task_id, gap_ids=[g.id for g in pending])
                cost["audit_pending_count"] = len(pending)
                cost["audit_pending_seconds"] = round(time.monotonic() - started, 1)
                undecided = sum(1 for r in results if r and r.recommended_action == "more_search")
                print(f"[audit-pending] {topic_id}: audited={len(results)} "
                      f"undecided(more_search)={undecided}", flush=True)

                if reaudit_ids:
                    from app.db.repositories import paper_repo

                    db.expire_all()
                    still_undecided = db.query(GapCandidate).filter(
                        GapCandidate.id.in_(reaudit_ids),
                        GapCandidate.status == "auditing").all()
                    if still_undecided:
                        for gap in still_undecided:
                            gap.status = "inconclusive"
                        db.commit()
                        paper_repo.save_trace(
                            db, replay_task_id, "audit_pending_finalization", "decision",
                            output_data={
                                "rule": "more_search_reaudit_once_then_inconclusive",
                                "finalized_gap_ids": [g.id for g in still_undecided],
                            })
                        print(f"[audit-pending] {topic_id}: finalized "
                              f"{len(still_undecided)} still-undecided gap(s) "
                              f"as inconclusive", flush=True)

                # Downstream for newly surviving gaps (idempotent per gap).
                state = task_repo.get_state(db, replay_task_id)
                started = time.monotonic()
                await generate_phenomenon_plans(db, state, llm, replay_task_id)
                state = task_repo.get_state(db, replay_task_id)
                interventions = await generate_interventions(db, state, llm, replay_task_id)
                state = task_repo.get_state(db, replay_task_id)
                experiments = await generate_minimal_experiments(db, state, llm, replay_task_id)
                cost["downstream_pending_seconds"] = round(time.monotonic() - started, 1)
                cost["interventions_passed"] = len(
                    getattr(interventions, "passed_intervention_ids", []) or [])
                cost["experiments_ideas"] = len(getattr(experiments, "idea_ids", []) or [])

            v6_contract = db.query(ResearchContract).filter(
                ResearchContract.task_id == old["v6_task_id"],
                ResearchContract.status == "active").first()
            v7_gaps = gap_repo.list_gaps_for_contract(
                db, replay_task_id, contract_id, include_superseded=True)
            v6_gaps = gap_repo.list_gaps_for_contract(
                db, old["v6_task_id"], v6_contract.id, include_superseded=True)
            comparison = await compare_v7_against_v6(db, v7_gaps, v6_gaps)
            record = dict(old)
            record.pop("audit_pending_error", None)
            record["v7_funnel"] = collect_funnel(db, replay_task_id, contract_id)
            record["v6_funnel"] = collect_funnel(
                db, old["v6_task_id"], v6_contract.id)
            record["v6_survival_basis"] = collect_survival_basis(
                db, old["v6_task_id"], v6_contract.id)
            record["v7_survival_basis"] = collect_survival_basis(
                db, replay_task_id, contract_id)
            record["v7_vs_v6"] = comparison
            record["cost"] = cost
            new_count = sum(1 for row in comparison["gaps"] if row["is_new_opportunity"])
            print(f"[audit-pending] {topic_id}: v7 funnel "
                  f"{json.dumps(record['v7_funnel']['gaps_by_status'])} new={new_count}",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            record = dict(old)
            record["audit_pending_error"] = f"{type(exc).__name__}: {exc}"
            print(f"[error] {topic_id}: {record['audit_pending_error']}", flush=True)
        records.append(record)
    # strict=False: a failed sweep must still write the artifacts (so the
    # failure is visible in replay_topics.jsonl), but the report marks those
    # topics stale and excludes them from every aggregate.
    report = await write_outputs(run_id, records, db=db, strict=False)
    print(f"outputs rewritten to eval_results/{run_id}/ "
          f"(stale={report['topics_stale_excluded']}, "
          f"db_consistency ok={report['db_consistency']['ok']})", flush=True)
    return records


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-run", default="pe2e_fullv2_clean",
                        help="frozen v6 run whose predictions map topics to tasks")
    parser.add_argument("--run-id", default="v7_replay_pilot")
    parser.add_argument("--topics", nargs="*", default=DEFAULT_PILOT_TOPICS)
    parser.add_argument("--dry-run", action="store_true",
                        help="read-only admission arithmetic on the frozen pool; no writes, no LLM")
    parser.add_argument("--audit-pending", action="store_true",
                        help="re-open an existing replay run and send every pending gap "
                             "(candidate/auditing) through the full adversarial audit, "
                             "then re-run downstream and rewrite the outputs")
    parser.add_argument("--recompute-only", action="store_true",
                        help="rebuild every derived number (funnels, v6 comparison, "
                             "survival basis, report) from the DB without any LLM call; "
                             "use this to repair a run whose follow-up sweep failed")
    args = parser.parse_args()

    from app.db.session import SessionLocal

    if args.recompute_only:
        session = SessionLocal()
        try:
            await recompute_only(session, args.run_id)
        finally:
            session.close()
        return

    if args.audit_pending:
        session = SessionLocal()
        try:
            await audit_pending_topics(session, args.run_id)
        finally:
            session.close()
        return

    topics = {row["topic_id"]: row["stratum"]
              for row in load_topics()}
    task_map = load_topic_task_map(args.source_run)

    session = SessionLocal()
    records_by_topic: dict[str, dict] = {}
    try:
        # One retry sweep for transient infra failures (network blips, LLM
        # circuit breakers) — a blip must not burn a whole pilot topic.
        queue = list(dict.fromkeys(args.topics))
        for attempt in range(2):
            if not queue:
                break
            if attempt:
                print(f"[retry] waiting 90s before retrying: {queue}", flush=True)
                await asyncio.sleep(90)
            still_failing: list[str] = []
            for topic_id in queue:
                stratum = topics.get(topic_id, "unknown")
                task_id = task_map.get(topic_id)
                if not task_id:
                    records_by_topic[topic_id] = {
                        "topic_id": topic_id, "error": "no task in source run"}
                    continue
                try:
                    if args.dry_run:
                        record = await dry_run_topic(session, topic_id, stratum, task_id)
                        print(f"[dry-run] {topic_id} ({stratum}): "
                              f"{json.dumps(record, ensure_ascii=False)[:400]}")
                    else:
                        print(f"[replay] {topic_id} ({stratum}) starting ...", flush=True)
                        record = await replay_topic(session, topic_id, stratum, task_id, args.run_id)
                        new_count = sum(
                            1 for row in record.get("v7_vs_v6", {}).get("gaps", [])
                            if row["is_new_opportunity"])
                        v7_count = record.get("v7_funnel", {}).get("gap_candidates")
                        print(f"[replay] {topic_id}: v7 candidates={v7_count} new={new_count}",
                              flush=True)
                except Exception as exc:  # noqa: BLE001 — one topic must not burn the run
                    session.rollback()
                    record = {"topic_id": topic_id, "error": f"{type(exc).__name__}: {exc}"}
                    if attempt == 0:
                        still_failing.append(topic_id)
                        print(f"[error] {topic_id}: {record['error']} (will retry)",
                              flush=True)
                    else:
                        print(f"[error] {topic_id}: {record['error']} (final)", flush=True)
                records_by_topic[topic_id] = record
            queue = still_failing
        records = [records_by_topic[topic_id] for topic_id in dict.fromkeys(args.topics)
                   if topic_id in records_by_topic]
        if not args.dry_run:
            report = await write_outputs(args.run_id, records, db=session,
                                         strict=False)
            print(f"outputs written to eval_results/{args.run_id}/ "
                  f"(db_consistency ok={report['db_consistency']['ok']})", flush=True)
        else:
            out_dir = Path(DEFAULT_RESULTS_DIR) / f"{args.run_id}_dryrun"
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "dry_run_report.json", "w", encoding="utf-8") as handle:
                json.dump(records, handle, ensure_ascii=False, indent=2)
            print(f"dry-run report: eval_results/{args.run_id}_dryrun/dry_run_report.json")
    finally:
        session.close()


if __name__ == "__main__":
    # Windows console default is cp936; the report contains non-ASCII text.
    # Applied here rather than at import time so importing this module under
    # pytest does not replace the captured stdout (which breaks capture).
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace")
    asyncio.run(main())
