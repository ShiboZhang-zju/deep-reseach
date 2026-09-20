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

Outputs (backend/eval_results/<run_id>/):
    replay_topics.jsonl        one row per topic: frozen counts, v6 funnel,
                               v7 funnel, cost, v7-only determination
    replay_report.json         aggregated answer to the five questions
    v7_only_opportunities.jsonl  v7 discoveries with no v6 counterpart, with
                               EMPTY human_prior_art_verdict fields — the
                               blind prior-art audit input
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

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

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


async def compare_v7_against_v6(v7_gaps, v6_gaps) -> dict:
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
        rows.append({
            "gap_id": gap.id,
            "gap_type": gap.gap_type,
            "status": gap.status,
            "claimed_delta": (gap.claimed_delta or "")[:400],
            "max_similarity_to_v6": round(best, 4),
            "closest_v6_gap_id": best_v6_id,
            "is_new_opportunity": best < NOVELTY_SIMILARITY,
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
    comparison = await compare_v7_against_v6(v7_gaps, v6_gaps)

    return {
        "topic_id": topic_id,
        "stratum": stratum,
        "v6_task_id": v6_task_id,
        "replay_task_id": replay_task.id,
        "frozen": frozen_counts,
        "v6_funnel": v6_funnel,
        "v7_funnel": collect_funnel(db, replay_task.id, replay_contract.id),
        "v7_vs_v6": comparison,
        "cost": cost,
    }


def summarize(records: list[dict]) -> dict:
    """Aggregate the five replay questions over the per-topic records."""
    new_rows = []
    for record in records:
        if record.get("error"):
            continue
        new_rows.extend([row for row in record["v7_vs_v6"]["gaps"]
                         if row["is_new_opportunity"]])
    by_type_new: dict[str, int] = {}
    for row in new_rows:
        by_type_new[row["gap_type"]] = by_type_new.get(row["gap_type"], 0) + 1

    def _sum(arm: str, key: str) -> int:
        return sum(record.get(arm, {}).get(key, 0) for record in records
                   if not record.get("error") and arm in record)

    return {
        "generated_at": _utcnow(),
        "topics_completed": sum(1 for record in records if not record.get("error")),
        "topics_errored": sum(1 for record in records if record.get("error")),
        "q1_new_opportunity_count": len(new_rows),
        "q3_new_opportunities_by_type": by_type_new,
        "v6_totals": {
            "gap_candidates": _sum("v6_funnel", "gap_candidates"),
            "surviving": _sum("v6_funnel", "surviving"),
            "ideas": _sum("v6_funnel", "ideas"),
            "executable_ideas": _sum("v6_funnel", "executable_ideas"),
        },
        "v7_totals": {
            "gap_candidates": _sum("v7_funnel", "gap_candidates"),
            "surviving": _sum("v7_funnel", "surviving"),
            "ideas": _sum("v7_funnel", "ideas"),
            "executable_ideas": _sum("v7_funnel", "executable_ideas"),
        },
        "cost_totals": {
            "mining_seconds": sum(record.get("cost", {}).get("mining_seconds", 0)
                                  for record in records if not record.get("error")),
            "audit_seconds": sum(record.get("cost", {}).get("audit_seconds", 0)
                                 for record in records if not record.get("error")),
            "audited": sum(record.get("cost", {}).get("audited", 0)
                           for record in records if not record.get("error")),
        },
        # q2 (false-open rate of new opportunities) and the marginal credible
        # opportunity yield are filled after the BLIND prior-art audit of
        # v7_only_opportunities.jsonl — deliberately not auto-judged here.
        "pending_human_steps": [
            "blind prior-art audit of v7_only_opportunities.jsonl "
            "(human_prior_art_verdict: FULL / PARTIAL / NONE)",
            "re-run with --summarize-after-audit once verdicts are filled",
        ],
    }


def write_outputs(run_id: str, records: list[dict]) -> None:
    out_dir = Path(DEFAULT_RESULTS_DIR) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "replay_topics.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(out_dir / "replay_report.json", "w", encoding="utf-8") as handle:
        json.dump(summarize(records), handle, ensure_ascii=False, indent=2)

    with open(out_dir / "v7_only_opportunities.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            if record.get("error"):
                continue
            for row in record["v7_vs_v6"]["gaps"]:
                if not row["is_new_opportunity"]:
                    continue
                handle.write(json.dumps({
                    "topic_id": record["topic_id"],
                    "stratum": record["stratum"],
                    **row,
                    # The blind prior-art audit fills these; verdict semantics
                    # follow the frozen super-audit protocol (FULL / PARTIAL /
                    # NONE coverage of the claimed delta by real prior art).
                    "human_prior_art_verdict": "",
                    "human_notes": "",
                }, ensure_ascii=False) + "\n")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-run", default="pe2e_fullv2_clean",
                        help="frozen v6 run whose predictions map topics to tasks")
    parser.add_argument("--run-id", default="v7_replay_pilot")
    parser.add_argument("--topics", nargs="*", default=DEFAULT_PILOT_TOPICS)
    parser.add_argument("--dry-run", action="store_true",
                        help="read-only admission arithmetic on the frozen pool; no writes, no LLM")
    args = parser.parse_args()

    from app.db.session import SessionLocal

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
            write_outputs(args.run_id, records)
            print(f"outputs written to eval_results/{args.run_id}/", flush=True)
        else:
            out_dir = Path(DEFAULT_RESULTS_DIR) / f"{args.run_id}_dryrun"
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "dry_run_report.json", "w", encoding="utf-8") as handle:
                json.dump(records, handle, ensure_ascii=False, indent=2)
            print(f"dry-run report: eval_results/{args.run_id}_dryrun/dry_run_report.json")
    finally:
        session.close()


if __name__ == "__main__":
    asyncio.run(main())
