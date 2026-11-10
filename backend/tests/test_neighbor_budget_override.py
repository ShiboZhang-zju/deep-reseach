"""Neighbour-budget experiment switch (2026-11-03).

The audit decides novelty against `budgeted_audit_neighbors` papers. Measured on
the fixed-budget pilot's 37 reviewed gaps, the reviewer's own killer paper sits
at rank (n=71 in-corpus killers):

    k=1 14.1% | k=2 28.2% | k=3 36.6% | k=5 53.5% | k=8 60.6% | k=10 63.4%

so at the production budget (3) about two thirds of the falsifying literature is
invisible to the judgement. The experiment needs the neighbour count to be the
ONLY variable that moves.

These tests pin three properties:

  1. the override applies to the neighbour count and NOTHING else (queries,
     candidate cap, full-text cap, killer queries all unchanged) — otherwise the
     comparison moves several budgets at once and stops being an experiment;
  2. it is resolved at call time, so one process reports what it actually used;
  3. the trace records the effective limit, whether the cap bound, and how many
     selected neighbours fell below the relevance floor, so "the larger arm did
     worse" can be separated from "its extra neighbours were noise".
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# === Override semantics ====================================================

def test_override_zero_keeps_the_configured_budget(monkeypatch):
    from app.config import settings
    from app.agent.steps.audit_gaps import _audit_budget

    monkeypatch.setattr(settings, "gap_audit_budgeted", True)
    monkeypatch.setattr(settings, "budgeted_audit_neighbors", 3)
    monkeypatch.setattr(settings, "audit_neighbors_override", 0)
    assert _audit_budget()["neighbors"] == 3


def test_override_raises_only_the_neighbour_count(monkeypatch):
    """Single-variable guarantee: everything else must be byte-identical."""
    from app.config import settings
    from app.agent.steps.audit_gaps import _audit_budget

    monkeypatch.setattr(settings, "gap_audit_budgeted", True)
    monkeypatch.setattr(settings, "budgeted_audit_neighbors", 3)
    monkeypatch.setattr(settings, "audit_neighbors_override", 0)
    control = dict(_audit_budget())

    monkeypatch.setattr(settings, "audit_neighbors_override", 8)
    treated = dict(_audit_budget())

    assert treated.pop("neighbors") == 8
    control.pop("neighbors")
    assert treated == control, (
        "the override must move ONLY the neighbour count; queries / candidate "
        "cap / full-text cap / killer queries / sources are not part of the "
        "treatment")


def test_override_negative_is_ignored(monkeypatch):
    """A negative value must not collapse the budget to zero."""
    from app.config import settings
    from app.agent.steps.audit_gaps import _audit_budget

    monkeypatch.setattr(settings, "gap_audit_budgeted", True)
    monkeypatch.setattr(settings, "budgeted_audit_neighbors", 3)
    monkeypatch.setattr(settings, "audit_neighbors_override", -5)
    assert _audit_budget()["neighbors"] == 3


def test_override_is_resolved_at_call_time(monkeypatch):
    """Same process, two values — the budget must follow, not cache."""
    from app.config import settings
    from app.agent.steps.audit_gaps import effective_neighbor_budget

    monkeypatch.setattr(settings, "audit_neighbors_override", 0)
    assert effective_neighbor_budget(3) == 3
    monkeypatch.setattr(settings, "audit_neighbors_override", 8)
    assert effective_neighbor_budget(3) == 8
    monkeypatch.setattr(settings, "audit_neighbors_override", 0)
    assert effective_neighbor_budget(3) == 3


def test_override_applies_in_legacy_mode_too(monkeypatch):
    """Legacy (non-budgeted) mode uses _MAX_NEIGHBORS; it must be overridable.

    The experiment may run on either mode, and silently ignoring the switch in
    one of them would produce an arm that looks treated but is not.
    """
    from app.config import settings
    from app.agent.steps.audit_gaps import _audit_budget, _MAX_NEIGHBORS

    monkeypatch.setattr(settings, "gap_audit_budgeted", False)
    monkeypatch.setattr(settings, "audit_neighbors_override", 0)
    assert _audit_budget()["neighbors"] == _MAX_NEIGHBORS
    monkeypatch.setattr(settings, "audit_neighbors_override", 8)
    assert _audit_budget()["neighbors"] == 8


# === Why 8, not 5 or 10 ====================================================

def test_documented_dose_response_matches_the_measured_curve():
    """Pin the measurement that justifies k=8.

    This is not a behavioural assertion — it is a guard against someone
    changing the default arm to a value the data does not support, or
    "remembering" the curve as monotone when it bends hard after k=8.
    """
    curve = {1: 0.141, 2: 0.282, 3: 0.366, 5: 0.535,
             6: 0.563, 8: 0.606, 10: 0.634, 15: 0.690, 20: 0.761}
    assert curve[3] == pytest.approx(0.366, abs=0.001)
    assert curve[8] == pytest.approx(0.606, abs=0.001)
    # The 3 -> 5 step is the single largest gain.
    steps = {k: round(curve[k] - curve[p], 4)
             for p, k in zip([1, 2, 3, 5, 6, 8, 10, 15], [2, 3, 5, 6, 8, 10, 15, 20])}
    assert max(steps, key=steps.get) == 5
    # And the curve bends: 8 -> 10 buys much less than 5 -> 8.
    assert (curve[10] - curve[8]) < (curve[8] - curve[5])


# === Instrumentation ======================================================

def _seed(db):
    from app.agent.steps.mine_gaps import GAP_MINING_POLICY_VERSION
    from app.db.models import (GapCandidate, ResearchContract, ResearchTask,
                              TaskPaper, Paper)

    task = ResearchTask(user_input="neighbour budget", status="auditing_gaps")
    db.add(task)
    db.flush()
    contract = ResearchContract(task_id=task.id, topic="t",
                                input_hash="h-nb", version=1)
    db.add(contract)
    db.flush()
    gap = GapCandidate(
        task_id=task.id, contract_id=contract.id, gap_type="boundary_gap",
        description="d", question_ids_json="[]", supporting_evidence_ids_json="[]",
        contradicting_evidence_ids_json="[]", mining_round=1, status="candidate",
        target_setting="x", observed_problem="y", existing_coverage="z",
        missing_capability="m", claimed_delta="c", testable_hypothesis="h",
        falsification_condition="f", provenance_status="complete",
        mining_policy_version=GAP_MINING_POLICY_VERSION,
    )
    db.add(gap)
    papers = []
    for i in range(10):
        p = Paper(title=f"p{i}", abstract="a", normalized_title=f"p{i}")
        db.add(p)
        db.flush()
        db.add(TaskPaper(task_id=task.id, paper_id=p.id, discovered_round=1,
                         final_score=0.5))
        papers.append(p)
    db.commit()
    return task, gap, papers


@pytest.fixture
def temp_db():
    import tempfile
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    from alembic.config import Config as AlembicConfig
    from alembic import command as alembic_command
    cfg = AlembicConfig()
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    cfg.set_main_option("script_location",
                        os.path.join(os.path.dirname(__file__), "..",
                                     "alembic_migrations"))
    alembic_command.upgrade(cfg, "head")
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    engine = create_engine(f"sqlite:///{db_path}")
    yield sessionmaker(bind=engine)
    engine.dispose()
    try:
        os.unlink(db_path)
    except PermissionError:
        pass


def _state(task, gap):
    from app.agent.state import ResearchState
    return ResearchState(task_id=task.id, contract_id=gap.contract_id,
                         current_round=1)


async def test_neighbor_budget_trace_records_effective_limit_and_noise(temp_db,
                                                                 monkeypatch):
    """The trace is the only way to attribute a null result later."""
    from app.config import settings
    from app.agent.steps import audit_gaps
    from app.db.repositories import paper_repo
    from app.db.models import AgentTrace

    # Pin every premise this test depends on instead of inheriting process
    # state. This test failed in the full suite while passing in isolation
    # because another test (or a previous run) left the no-audit ablation
    # switch on: `_ablate_audit` short-circuits BEFORE the neighbour-budget
    # trace, so an inherited flag silently turns this into a different test.
    monkeypatch.setattr(settings, "gap_audit_disabled_ablation", False)
    monkeypatch.setattr(settings, "gap_audit_budgeted", True)
    monkeypatch.setattr(settings, "budgeted_audit_neighbors", 3)
    monkeypatch.setattr(settings, "audit_neighbors_override", 8)
    monkeypatch.setattr(settings, "gap_relevance_min_neighbor", 0.3)

    db = temp_db()
    task, gap, papers = _seed(db)

    # Fake the pool/admission so the audit runs without any network.
    fake_admission = audit_gaps.GapSearchAdmission(
        gap.id, "PASS", [], [], [], [], ["exact_gap"], 1,
        [p.id for p in papers], [p.id for p in papers])

    def _fake_neighbors(db_, gap_, query_ids, limit=5, candidate_paper_ids=None):
        return papers[:limit]

    monkeypatch.setattr(audit_gaps, "select_gap_specific_neighbors", _fake_neighbors)
    # Half the neighbours score below the floor: the noise signal that must be
    # visible, not inferred.
    for i, p in enumerate(papers[:8]):
        db.add(audit_gaps.GapPaperRelevance(
            gap_id=gap.id, paper_id=p.id, task_id=task.id,
            relevance_score=(0.1 if i < 4 else 0.6), problem_overlap="yes",
            mechanism_overlap="no", evaluation_overlap="no",
            claim_overlap="no", rationale="r",
            scoring_version=settings.gap_relevance_scoring_version))
    db.commit()

    from app.agent.steps.generate_queries import SearchQueryExecution
    monkeypatch.setattr(audit_gaps, "save_search_query",
                        lambda *a, **k: type("R", (), {"id": "q1"})())
    async def _noop(*a, **k): return None
    monkeypatch.setattr(audit_gaps, "search_and_save_papers", _noop)
    async def _queries(*a, **k):
        return [audit_gaps.AdversarialQuerySpec(family="exact_gap",
                                                query_text="q")]
    monkeypatch.setattr(audit_gaps, "generate_english_adversarial_queries", _queries)
    monkeypatch.setattr(audit_gaps, "evaluate_gap_search_admission",
                        lambda *a, **k: fake_admission)
    monkeypatch.setattr(audit_gaps, "recall_corpus_neighbors",
                        lambda *a, **k: ([], {"enabled": True}))
    async def _score(db_, llm_, gap_, ids, task_id_):
        return []
    monkeypatch.setattr(audit_gaps, "score_all_gap_candidates", _score)
    # Stop right after the neighbour-budget trace: the audit decision is the
    # first llm.chat_json call, and `llm=None` makes it raise. Only the
    # post-trace stage is stubbed, so the trace assertion remains meaningful.
    from types import SimpleNamespace

    class _NoLLM:
        async def chat_json(self, *a, **k):
            raise RuntimeError("stop after neighbour-budget trace")

    monkeypatch.setattr(audit_gaps, "_format_neighbors", lambda *a, **k: "")

    state = _state(task, gap)
    # pytest.ini uses asyncio_mode=auto, so this test can simply await. The
    # earlier `run_until_complete(get_event_loop())` form relied on a loop this
    # process may not own.
    # The audit is expected to stop at the decision call (no LLM); what matters
    # is that everything before it — including the neighbour-budget trace — ran.
    with pytest.raises(Exception):
        await audit_gaps.audit_gap_candidate(db, state, _NoLLM(), task.id, gap,
                                             perform_search=False)

    traces = db.query(AgentTrace).filter(
        AgentTrace.step_name == "gap_audit_neighbor_budget").all()
    assert traces, "the neighbour-budget trace must always be written"
    import json as _json
    payload = _json.loads(traces[-1].output_json)
    assert payload["effective_limit"] == 8
    assert payload["override"] == 8
    assert payload["selected_count"] == 8
    assert payload["cap_bound"] is True
    assert payload["selected_below_floor"] == 4, (
        "below-floor neighbours must be counted so 'more neighbours' cannot be "
        "confused with 'noisier neighbours'")
    db.close()
