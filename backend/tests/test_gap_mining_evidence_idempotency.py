"""Regression: gap-mining evidence-sensitive idempotency.

Reproduces the observed 08005641 failure mode: on that run, mining ran once on
a thin evidence pool (9 questions -> 1 admission PASS), then an O2 remediation
round added 46 evidence units that would have flipped at least 4 more questions
to PASS, but the mining phase was skipped on re-entry because its input_version
only hashed {contract, round, pipeline, policy} — all unchanged by remediation.
The new evidence was never mined.

Fix under test:
- compute_evidence_fingerprint() hashes the CURRENT evidence pool + question
  links (content-based, stable rows), and the runner embeds it in the mining
  input_version;
- mine_gap_candidates(input_version=...) binds its existing-gap short-circuit
  to that same fingerprint, so added evidence re-mines instead of reusing the
  old gaps;
- legacy calls without input_version keep the round-based short-circuit.

Scenario:
1. RQ1 is admissible from the start (2 papers + 1 limitation).
2. RQ2 starts inadmissible (1 paper, no limitation) -> mining only sees RQ1.
3. "Remediation" adds one more paper + one limitation to RQ2 -> fingerprint
   changes -> mining re-runs -> admission now includes RQ2 -> a new gap is
   created that the old run could never have surfaced.
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def temp_db():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    from alembic.config import Config
    from alembic import command
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    config = Config()
    config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    config.set_main_option("script_location",
                          os.path.join(os.path.dirname(__file__), "..", "alembic_migrations"))
    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{db_path}")
    yield sessionmaker(bind=engine)
    engine.dispose()
    try:
        os.unlink(db_path)
    except PermissionError:
        pass


class ScriptedGapLLM:
    """Returns one gap per call, for the question at a scripted index.

    Parses the mining prompt to get the passed question IDs and the evidence
    each question links to, then emits a candidate for the scripted question.
    This lets the test say "mining now surfaces RQ2" without depending on a
    real model.
    """

    def __init__(self, picks):
        self.picks = picks  # list of question index per call
        self.calls = 0

    async def chat_json(self, messages, schema):
        from app.schemas.schemas import GapCandidateList, GapCandidateSchema

        content = messages[1]["content"]
        questions = [ln.split("Question ID: ", 1)[1].strip()
                     for ln in content.splitlines() if "Question ID: " in ln]
        # Question text (used to make each candidate's semantic fingerprint
        # distinct so the gap-dedup gate does not collapse them into one).
        question_text = {}
        current_q = None
        for ln in content.splitlines():
            if "Question ID: " in ln:
                current_q = ln.split("Question ID: ", 1)[1].strip()
            elif ln.strip().startswith("Question: ") and current_q:
                question_text[current_q] = ln.strip().split("Question: ", 1)[1]
        # Parse "Linked questions:" lines to map evidence -> its questions.
        evidence_by_q = {}
        current_ev = None
        for ln in content.splitlines():
            if ln.startswith("- Evidence ID: "):
                current_ev = ln.split(": ", 1)[1].strip()
            elif "Linked questions: " in ln and current_ev:
                for q in ln.split("Linked questions: ", 1)[1].split(","):
                    evidence_by_q.setdefault(q.strip(), []).append(current_ev)
                current_ev = None
        pick = self.picks[self.calls % len(self.picks)]
        self.calls += 1
        if pick >= len(questions):
            return GapCandidateList(gaps=[])
        qid = questions[pick]
        ev = evidence_by_q.get(qid, [])
        qtext = question_text.get(qid, qid[:8])
        return GapCandidateList(gaps=[GapCandidateSchema(
            gap_type="boundary_gap",
            description=f"Untested boundary in: {qtext}.",
            target_setting=f"Boundary of {qtext}",
            observed_problem=f"The {qtext} boundary is not evaluated by existing work.",
            existing_coverage="Existing work evaluates the standard setting only.",
            missing_capability=f"Boundary-condition evaluation for {qtext}.",
            claimed_delta=f"Measure the untested {qtext} boundary explicitly.",
            testable_hypothesis=f"The {qtext} boundary diverges from the standard setting.",
            falsification_condition="A neighbour already evaluates this exact boundary.",
            question_ids=[qid],
            supporting_evidence_ids=ev,
        )])


def _seed_task(db):
    from app.db.models import (EvidenceUnit, Paper, QuestionEvidenceLink,
                               ResearchContract, ResearchQuestion, ResearchTask,
                               TaskPaper)
    task = ResearchTask(user_input="agent memory")
    db.add(task)
    db.flush()
    contract = ResearchContract(task_id=task.id, topic="Agent Memory", status="active",
                                version=1, input_hash="contract-v1")
    db.add(contract)
    db.flush()
    return task, contract


def _seed_question(db, task, contract, text, importance):
    from app.db.models import ResearchQuestion
    q = ResearchQuestion(task_id=task.id, contract_id=contract.id, question=text,
                         question_type="failure", importance=importance,
                         status="partially_covered")
    db.add(q)
    db.flush()
    return q


def _add_evidence(db, task, question, paper_title, evidence_type, fulltext=True):
    from app.db.models import EvidenceUnit, Paper, QuestionEvidenceLink, TaskPaper
    paper = Paper(title=paper_title, abstract="Memory evaluation")
    db.add(paper)
    db.flush()
    db.add(TaskPaper(task_id=task.id, paper_id=paper.id, discovered_round=1,
                     priority="high", final_score=0.9))
    unit = EvidenceUnit(
        task_id=task.id, paper_id=paper.id, evidence_type=evidence_type,
        normalized_claim=f"Claim {paper_title} {evidence_type}",
        original_span="Span for " + paper_title if fulltext else None,
        section="Limitations", page_number=4, page_start=4, page_end=4,
        span_start=10, span_end=66, source_chunk_hash=f"chunk-{paper_title}",
        verification_status="verified" if fulltext else "abstract_only",
        extraction_confidence=0.9,
    )
    db.add(unit)
    db.flush()
    db.add(QuestionEvidenceLink(question_id=question.id, evidence_id=unit.id,
                                relation_type="supports", relevance_score=0.9))
    return paper, unit


def _last_trace(db, task_id, step_name):
    from app.db.models import AgentTrace
    return db.query(AgentTrace).filter(
        AgentTrace.task_id == task_id, AgentTrace.step_name == step_name,
    ).order_by(AgentTrace.created_at.desc()).first()


def _trace_question_ids(trace):
    import json
    if trace is None or not trace.output_json:
        return []
    data = json.loads(trace.output_json)
    return data.get("passed_question_ids", [])


@pytest.mark.asyncio
async def test_evidence_sensitive_remine_surfaces_blocked_question(temp_db):
    from app.agent.state import ResearchState
    from app.agent.steps.mine_gaps import (compute_evidence_fingerprint,
                                           mine_gap_candidates)
    from app.db.models import GapCandidate

    db = temp_db()
    task, contract = _seed_task(db)
    rq1 = _seed_question(db, task, contract, "Does memory survive state changes?",
                         importance=0.9)
    rq2 = _seed_question(db, task, contract, "Is retention measured under a fixed budget?",
                         importance=0.8)
    db.commit()

    # RQ1 admissible from the start: 2 independent papers + 1 limitation.
    _add_evidence(db, task, rq1, "Paper A1", "limitation")
    _add_evidence(db, task, rq1, "Paper A2", "comparison")
    # RQ2 inadmissible: 1 paper, no limitation signal.
    _add_evidence(db, task, rq2, "Paper B1", "method")
    db.commit()

    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=1)
    fp_before = compute_evidence_fingerprint(db, task.id)
    assert fp_before  # non-empty

    # First mining pass: only RQ1 passes admission.
    gaps1 = await mine_gap_candidates(db, state, ScriptedGapLLM(picks=[0]),
                                      task.id, input_version=fp_before)
    assert len(gaps1) == 1
    assert gaps1[0].mining_input_version == fp_before
    trace1 = _last_trace(db, task.id, "gap_mining_admission")
    assert set(_trace_question_ids(trace1)) == {rq1.id}, \
        f"expected only RQ1 admitted, got {_trace_question_ids(trace1)}"

    # --- "Remediation": RQ2 gains a second paper + a limitation signal. ---
    _add_evidence(db, task, rq2, "Paper B2", "limitation")
    db.commit()

    fp_after = compute_evidence_fingerprint(db, task.id)
    assert fp_after != fp_before, "fingerprint must change when evidence is added"
    assert compute_evidence_fingerprint(db, task.id) == fp_after, \
        "fingerprint must be stable for the same evidence pool"

    # Second mining pass with the NEW fingerprint: RQ2 must now be admitted.
    state2 = ResearchState(task_id=task.id, contract_id=contract.id, current_round=1)
    gaps2 = await mine_gap_candidates(db, state2, ScriptedGapLLM(picks=[1]),
                                      task.id, input_version=fp_after)
    trace2 = _last_trace(db, task.id, "gap_mining_admission")
    passed = _trace_question_ids(trace2)
    assert rq2.id in passed, f"RQ2 should be admitted after remediation, got {passed}"
    assert len(gaps2) >= 1
    assert any(g.mining_input_version == fp_after for g in gaps2)
    # The new gap is grounded in RQ2, not a reworded RQ1.
    import json
    rq2_gap = [g for g in gaps2
               if rq2.id in json.loads(g.question_ids_json or "[]")]
    assert rq2_gap, "expected a new gap tied to RQ2 after remediation"
    db.close()


@pytest.mark.asyncio
async def test_short_circuit_binds_to_input_version(temp_db):
    from app.agent.state import ResearchState
    from app.agent.steps.mine_gaps import compute_evidence_fingerprint, mine_gap_candidates

    db = temp_db()
    task, contract = _seed_task(db)
    rq1 = _seed_question(db, task, contract, "Does memory survive state changes?",
                         importance=0.9)
    db.commit()
    _add_evidence(db, task, rq1, "Paper A1", "limitation")
    _add_evidence(db, task, rq1, "Paper A2", "comparison")
    db.commit()

    fp = compute_evidence_fingerprint(db, task.id)
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=1)
    llm = ScriptedGapLLM(picks=[0])
    gaps1 = await mine_gap_candidates(db, state, llm, task.id, input_version=fp)
    assert len(gaps1) == 1
    calls_after_first = llm.calls

    # Same fingerprint -> short-circuit returns the existing gap WITHOUT
    # calling the model again.
    gaps2 = await mine_gap_candidates(db, state, llm, task.id, input_version=fp)
    assert llm.calls == calls_after_first, "same input_version must short-circuit"
    assert len(gaps2) == 1
    assert gaps2[0].id == gaps1[0].id

    # A different fingerprint -> re-mines (calls the model again).
    _add_evidence(db, task, rq1, "Paper A3", "comparison")
    db.commit()
    fp2 = compute_evidence_fingerprint(db, task.id)
    assert fp2 != fp
    await mine_gap_candidates(db, state, llm, task.id, input_version=fp2)
    assert llm.calls > calls_after_first, "changed fingerprint must re-mine"
    db.close()
