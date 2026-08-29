"""Research validity regression cases (run7 / run8 / run10).

The three E2E runs each exposed a distinct scientific-validity failure that
the pipeline MUST NOT regress into. These cases pin the corresponding gates:

  run7 (task d6f64087): Self/External LABEL effect claimed as
      self-preference over one's own generations — claim/manipulation
      construct mismatch.

  run8 (task 03e0f59a): temporal benchmark score decay renamed
      "contamination velocity" and confirmed at novelty 0.9 with
      zero fulltext evidence, unstable retrieval, and a vacuous killer
      search — observed-signal/claimed-construct mismatch PLUS a strong
      verdict on a retrieval basis that could not have found prior work.

  run10 review: schema-compliant control declarations ("control for model
      capability") with no matched arm in the steps — compliance theater;
      and more_search loops that stop learning but keep buying audit rounds.

Run this file whenever prompts, models, or audit policies change. The thing
being guarded is NOT "the code does not error" but "the system does not
regress into a generator that packages wrong scientific hypotheses
convincingly."
"""
import json
import os
import tempfile

import pytest

from app.agent.steps.audit_gaps import GapAuditDecisionSchema
from app.agent.steps.generate_minimal_experiments import (
    ConstructIdentificationSchema,
    MinimalExperimentSchema,
    _construct_gate_verdict,
    _control_implementation_check,
)


@pytest.fixture()
def temp_db():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    config = Config()
    config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    config.set_main_option("script_location",
                           os.path.join(os.path.dirname(__file__), "..",
                                        "alembic_migrations"))
    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{db_path}")
    yield sessionmaker(bind=engine)
    engine.dispose()
    try:
        os.unlink(db_path)
    except PermissionError:
        pass


# --------------------------------------------------------------------------
# Shared schema builders
# --------------------------------------------------------------------------

def _ci(**overrides) -> ConstructIdentificationSchema:
    base = dict(
        observed_variable="Per-arm regression rate and pass@1 delta",
        claimed_construct="Functional-regression risk of stylistic corrections",
        identification_assumptions=["Execution oracle flags regressions reliably"],
        major_confounders=[],
        required_control_or_oracle="Execution engine verifier plus human adjudication",
    )
    base.update(overrides)
    return ConstructIdentificationSchema(**base)


def _plan(ci: ConstructIdentificationSchema, **overrides) -> MinimalExperimentSchema:
    base = dict(
        title="Diff-Risk Classification for Self-Correction Filtering",
        idea_method="We hypothesize that filtering stylistic corrections reduces regressions.",
        summary="Compare filtered vs unfiltered self-correction on HumanEval.",
        hypothesis="Filtering stylistic-only corrections reduces functional regression rate.",
        core_factor="correction-type filter",
        core_operation="toggle_filter",
        core_contrast="logical_only_vs_apply_all",
        expected_signature="Regression rate drops in the logical-only arm, unchanged pass@1.",
        mechanism_being_tested="Stylistic corrections carry functional regression risk.",
        dataset="HumanEval with injected stylistic noise",
        baselines="apply-all corrections; heuristic filter",
        metrics="pass@1 delta and regression rate across 164 problems",
        model_spec="Qwen2.5-Coder-7B-Instruct",
        dataset_provenance="Inject stylistic edits verified function-preserving",
        oracle="Python execution engine plus 10% human adjudication",
        statistical_analysis="Paired t-test on per-problem pass/fail deltas (p<0.05).",
        resource_budget="CPU-only, <2 hours",
        scenario_atoms=["verifier"],
        controls=["no-correction", "apply-all"],
        steps=["Build dataset", "Run corrections", "Evaluate"],
        success_condition="Regression rate drops by >5% relative",
        falsification_condition="No significant difference",
        risks="Synthetic noise may not reflect real stylistic edits",
        construct_identification=ci,
    )
    base.update(overrides)
    return MinimalExperimentSchema(**base)


# --------------------------------------------------------------------------
# run7: label attribution effect != self-preference
# --------------------------------------------------------------------------

def test_run7_label_effect_is_not_self_preference():
    """A label-swap-only design measures attribution bias, NOT self-preference.

    The run7 failure: an experiment that toggled the DISPLAYED source
    attribution label claimed to measure self-preference over one's own
    generations. The label effect is real, but self-preference requires
    comparing actual self-generated content against matched external
    content — the design's confounders (content quality differences,
    familiarity) are exactly what a label swap cannot separate.
    """
    run7 = _ci(
        observed_variable="Acceptance rate difference between Self-labeled "
                          "and External-labeled outputs",
        claimed_construct="Self-preference over one's own generations",
        identification_assumptions=[
            "Label-induced acceptance differences reflect preference for "
            "self-generated content"],
        major_confounders=[
            "content quality differences between self and external outputs",
            "position and familiarity effects"],
        required_control_or_oracle="",
    )
    assert _construct_gate_verdict(run7) == "UNCONTROLLED_CONFOUNDER"

    # The same design WITH a matched-content control identifies the
    # construct: quality-matched self vs external pairs under both labels.
    fixed = run7.model_copy(update={
        "required_control_or_oracle":
            "Quality-matched self/external content pairs under a label swap"})
    assert _construct_gate_verdict(fixed) is None


# --------------------------------------------------------------------------
# run8: temporal performance decay != contamination velocity
# --------------------------------------------------------------------------

def test_run8_temporal_decay_is_not_contamination_velocity():
    """Run8's surviving gap: score decay on recency gaps claimed as
    contamination velocity. Capability growth, knowledge recency, and item
    drift produce the same observed signature, so the slope measures none
    of them in particular."""
    run8 = _ci(
        observed_variable="Regression slope of benchmark accuracy on "
                          "item-publication-to-training-cutoff time gap",
        claimed_construct="Data contamination accumulation rate",
        identification_assumptions=[
            "Score decay over recency gaps is contamination-driven"],
        major_confounders=[
            "model capability growth over the same period",
            "knowledge recency (post-cutoff knowledge the items require)",
            "benchmark item difficulty and construction drift"],
        required_control_or_oracle="",
    )
    assert _construct_gate_verdict(run8) == "UNCONTROLLED_CONFOUNDER"

    # The user-specified rescue: controlled exposure identifies the
    # construct (injected dose x checkpoints vs matched clean control).
    rescued = run8.model_copy(update={
        "observed_variable": "Score difference between exposed and matched "
                             "clean-control items at each dose and checkpoint",
        "required_control_or_oracle":
            "Matched clean-control benchmark items plus injected "
            "contamination dose ladder",
    })
    assert _construct_gate_verdict(rescued) is None


def test_run10_control_declaration_must_be_implemented(monkeypatch):
    """Schema compliance != scientific validity: naming a control the design
    does not contain is compliance theater. "control for model capability"
    with no matched arm anywhere in the steps must be withheld."""
    import app.services.embedding_service as emb

    # Semantic route also finds no implementation (orthogonal embeddings).
    monkeypatch.setattr(emb, "embed_texts",
                        lambda texts: [[float(i == 0), 1.0]
                                       for i in range(len(texts))])
    monkeypatch.setattr(emb, "cosine_similarity", lambda a, b: 0.0)

    declared_not_implemented = _ci(
        major_confounders=["capability growth"],
        required_control_or_oracle="control for model capability",
    )
    plan = _plan(declared_not_implemented,
                 steps=["Compute temporal score slope per benchmark",
                        "Fit Kalman filter to score series"],
                 controls=["same decoding budget"])
    code = _control_implementation_check(plan, declared_not_implemented)
    assert code == "CONTROL_NOT_IMPLEMENTED"

    # The same confounder with the control actually present in the steps
    # passes: "capability" and "growth"/"matched" appear in the design.
    implemented = _ci(
        major_confounders=["capability growth"],
        required_control_or_oracle="matched model capability arms",
    )
    plan_ok = _plan(implemented,
                    steps=["Evaluate both arms on matched model capability",
                           "Compare regression rates"],
                    controls=["matched capability arms"])
    assert _control_implementation_check(plan_ok, implemented) is None

    # No confounders declared: nothing to cross-check.
    clean = _ci(major_confounders=[],
                required_control_or_oracle="")
    assert _control_implementation_check(_plan(clean), clean) is None


def test_run10_control_check_fails_closed_without_embeddings(monkeypatch):
    """Without embeddings the token check decides alone — a declaration that
    shares no vocabulary with the design is still withheld (fail-closed:
    the demotion keeps the idea visible, it does not delete it)."""
    import app.services.embedding_service as emb

    def boom(texts):
        raise RuntimeError("embedding service down")

    monkeypatch.setattr(emb, "embed_texts", boom)
    ci = _ci(major_confounders=["difficulty drift"],
             required_control_or_oracle="difficulty matched split")
    plan = _plan(ci, steps=["Run all items", "Average scores"],
                 controls=["same budget"])
    assert _control_implementation_check(plan, ci) == "CONTROL_NOT_IMPLEMENTED"


# --------------------------------------------------------------------------
# run10: unstable retrieval must reject strong verdicts / stop dead loops
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run10_unstable_retrieval_rejects_strong_verdict(temp_db, monkeypatch):
    """Run8 stamped confirmed/0.9 while stability@20 was 0.055 with zero
    fulltext evidence and a 2-paper killer recall. The verdict ceiling must
    downgrade exactly that shape to uncertain/more_search (first violation)
    or weak-survive <= 0.6 (budget exhausted). Full integration shape lives
    in test_audit_gaps; this case pins the run8 verdict itself."""
    from app.agent.steps.audit_gaps import _apply_audit_verdict_ceiling

    db = temp_db()
    # The run8 shape: no verified fulltext evidence in the pool.
    from app.db.models import Paper

    neighbors = [Paper(title=f"Off-topic neighbor {i}",
                       abstract="Converged but irrelevant.", citation_count=1)
                 for i in range(5)]
    db.add_all(neighbors)
    db.commit()

    decision = GapAuditDecisionSchema(
        audit_result="confirmed", recommended_action="continue",
        remaining_delta="The audit could not rule out longitudinal contamination "
                        "studies.",
        novelty_confidence=0.9, audit_confidence=0.85,
        killer_query_terms=["contamination velocity benchmark"],
        closest_killer_work="A longitudinal study tracking benchmark validity decay",
        killer_found=False, residual_uncertainty="temporal leakage not searched")
    killer_work = {"description": decision.closest_killer_work,
                   "query_terms": decision.killer_query_terms,
                   "found": False, "retrieved_new_paper_count": 2,
                   "killer_hits": []}

    from app.db.models import GapCandidate
    gap = GapCandidate(
        task_id="t-run10-reg", contract_id="c1", gap_type="capability",
        description="Benchmarks lack contamination velocity measurement",
        observed_problem="Benchmarks lack contamination velocity measurement",
        claimed_delta="A longitudinal contamination velocity framework",
        status="auditing")
    db.add(gap)
    db.commit()

    failure_codes: list[str] = []
    _apply_audit_verdict_ceiling(db, "t-run10-reg", gap, decision,
                                 failure_codes, killer_work, neighbors)
    assert decision.audit_result != "confirmed" or (
        decision.novelty_confidence <= 0.6)
    assert failure_codes  # the ceiling fired, not passed silently
    db.close()


def test_run10_diminishing_return_stop():
    """A more_search loop whose verdict stays flat while the neighbors barely
    move must stop instead of buying another audit round (run10 spent ~2.5h
    in exactly this loop)."""
    from app.agent.steps.audit_gaps import _diminishing_return_stop
    from app.db.models import GapAudit, GapCandidate
    from app.db.models import Paper

    previous = GapAudit(
        gap_id="g1", task_id="t1", audit_result="uncertain",
        recommended_action="more_search", novelty_confidence=0.5,
        audit_confidence=0.6, audited_claimed_delta="A longitudinal framework",
        neighbor_paper_ids_json=json.dumps(["p1", "p2", "p3", "p4", "p5"]),
        search_admission_status="PASS")
    gap = GapCandidate(task_id="t1", contract_id="c1", gap_type="capability",
                       observed_problem="x", claimed_delta="A longitudinal framework",
                       status="auditing")
    decision = GapAuditDecisionSchema(
        audit_result="uncertain", recommended_action="more_search",
        remaining_delta="", novelty_confidence=0.5, audit_confidence=0.6)
    neighbors = [Paper(id=f"p{i}", title=f"N{i}", abstract="a")
                 for i in range(1, 7)]  # 5 of 6 shared -> Jaccard 0.83

    meta = _diminishing_return_stop(previous, gap, decision, neighbors)
    assert meta is not None
    assert meta["neighbor_jaccard"] >= 0.5
    assert meta["novelty"] <= meta["prev_novelty"] + 0.1

    # Verdict improving -> keep searching.
    improving = decision.model_copy(update={"novelty_confidence": 0.75})
    assert _diminishing_return_stop(previous, gap, improving, neighbors) is None

    # Fresh neighbors (low Jaccard) -> keep searching.
    fresh = [Paper(id=f"q{i}", title=f"M{i}", abstract="a") for i in range(1, 6)]
    assert _diminishing_return_stop(previous, gap, decision, fresh) is None

    # Claim rewritten by narrowing -> the re-audit is warranted.
    narrowed = GapCandidate(task_id="t1", contract_id="c1", gap_type="capability",
                            observed_problem="x",
                            claimed_delta="A narrowed longitudinal framework",
                            status="auditing")
    assert _diminishing_return_stop(previous, narrowed, decision, neighbors) is None
