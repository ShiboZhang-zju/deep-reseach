"""Report-integrity tests for the v7 opportunity replay.

The bug these guard against (measured on the 2026-09-22 pilot): the follow-up
audit sweep failed per topic, each record was written back as a copy of its
PREVIOUS snapshot, and the aggregate reported `topics_errored: 0` with 6
v7-only surviving gaps — while the database held 23 surviving gaps across the
same 6 topics. Every number in that report was stale, and nothing in the
artifact said so.

The rules pinned here:
1. A record carrying `audit_pending_error` is STALE: excluded from every
   aggregate and named in the report, never silently counted as healthy.
2. The report must agree with the database. A disagreement is a hard failure,
   not a footnote — a report that contradicts the DB is a stale snapshot.
3. Surviving gaps are split by the gate that admitted them, and the v6-vs-v7
   surviving-rate difference is decomposed per candidate, so an ontology gain
   can be told apart from a mechanical-novelty-gate gain.
"""

import json

import pytest

from eval.production_e2e import opportunity_replay as replay


def _gap_row(gap_id, gap_type="boundary_gap", status="surviving",
             is_new=True, basis=None, audit_result="confirmed"):
    return {
        "gap_id": gap_id,
        "gap_type": gap_type,
        "status": status,
        "claimed_delta": f"delta for {gap_id}",
        "max_similarity_to_v6": 0.1,
        "closest_v6_gap_id": None,
        "is_new_opportunity": is_new,
        "survival_basis": basis,
        "audit_result": audit_result,
        "recommended_action": "continue",
    }


def _record(topic_id, v6_candidates=2, v6_surviving=1, v7_candidates=4,
            v7_surviving=2, gaps=None, v6_basis=None, v7_basis=None,
            stale_error=None, error=None):
    record = {
        "topic_id": topic_id,
        "stratum": "narrow_mature",
        "v6_task_id": f"v6-{topic_id}",
        "replay_task_id": f"v7-{topic_id}",
        "replay_contract_id": f"c-{topic_id}",
        "v6_funnel": {"gap_candidates": v6_candidates, "surviving": v6_surviving,
                      "ideas": 1},
        "v7_funnel": {"gap_candidates": v7_candidates, "surviving": v7_surviving,
                      "ideas": 1},
        "v6_survival_basis": v6_basis or replay._empty_basis_counts(),
        "v7_survival_basis": v7_basis or replay._empty_basis_counts(),
        "v7_vs_v6": {"method": "embedding", "gaps": gaps or []},
        "cost": {},
    }
    if stale_error:
        record["audit_pending_error"] = stale_error
    if error:
        record["error"] = error
    return record


# --- rule 1: stale records are excluded and named -------------------------

def test_stale_record_is_excluded_from_aggregates_and_named():
    healthy = _record("t1", gaps=[_gap_row("g1")])
    stale = _record("t2", v7_candidates=9, v7_surviving=9,
                    gaps=[_gap_row("g2"), _gap_row("g3")],
                    stale_error="NameError: name 'db' is not defined")

    report = replay.summarize([healthy, stale])

    assert report["topics_completed"] == 1
    assert report["topics_stale_excluded"] == ["t2"]
    # The stale topic's 9 candidates and 2 new gaps must not leak in.
    assert report["q1_v7_only_recall"]["v7_candidates_total"] == 4
    assert report["q1_v7_only_recall"]["v7_only_new"] == 1


def test_errored_and_stale_are_reported_separately():
    report = replay.summarize([
        _record("t1"),
        _record("t2", error="RuntimeError: circuit-open"),
        _record("t3", stale_error="ReadError: connection reset"),
    ])

    assert report["topics_completed"] == 1
    assert report["topics_errored"] == 1
    assert report["topics_errored_ids"] == ["t2"]
    assert report["topics_stale_excluded"] == ["t3"]


def test_all_stale_run_reports_zero_completed_not_a_healthy_run():
    report = replay.summarize([
        _record("t1", stale_error="boom"),
        _record("t2", stale_error="boom"),
    ])

    assert report["topics_completed"] == 0
    assert report["q1_v7_only_recall"]["v7_only_new"] == 0
    assert len(report["topics_stale_excluded"]) == 2


# --- rule 2: the report must agree with the DB ----------------------------

def test_db_mismatch_is_reported_as_a_mismatch():
    records = [_record("t1", v7_candidates=4, v7_surviving=2)]
    db_truth = {"t1": {"v6": {"gap_candidates": 2, "surviving": 1},
                       "v7": {"gap_candidates": 4, "surviving": 7}}}

    report = replay.summarize(records, db_truth=db_truth)

    assert report["db_consistency"]["checked"] is True
    assert report["db_consistency"]["ok"] is False
    assert any("t1/v7.surviving" in item
               for item in report["db_consistency"]["mismatches"])


def test_matching_db_truth_is_clean():
    records = [_record("t1", v7_candidates=4, v7_surviving=2)]
    db_truth = {"t1": {"v6": {"gap_candidates": 2, "surviving": 1},
                       "v7": {"gap_candidates": 4, "surviving": 2}}}

    report = replay.summarize(records, db_truth=db_truth)

    assert report["db_consistency"]["ok"] is True
    assert report["db_consistency"]["mismatches"] == []


def test_consistency_check_is_skipped_without_db_truth():
    report = replay.summarize([_record("t1")])

    assert report["db_consistency"]["checked"] is False
    assert report["db_consistency"]["ok"] is True


def test_write_outputs_refuses_to_publish_a_stale_report(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "DEFAULT_RESULTS_DIR", str(tmp_path))
    records = [_record("t1", v7_candidates=4, v7_surviving=2)]
    db_truth = {"t1": {"v7": {"gap_candidates": 4, "surviving": 7}}}

    class _FakeDb:
        pass

    monkeypatch.setattr(replay, "collect_db_truth",
                        lambda db, recs: db_truth)
    monkeypatch.setattr(replay, "build_p2_counterfactual",
                        lambda db, recs, floor=None: {"enabled": False})
    monkeypatch.setattr(replay, "build_mech_override_review", lambda db, rows: [])

    with pytest.raises(SystemExit) as excinfo:
        import asyncio
        asyncio.run(replay.write_outputs("run", records, db=_FakeDb(), strict=True))

    assert "disagrees with the database" in str(excinfo.value)


def test_write_outputs_publishes_when_strict_is_off(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "DEFAULT_RESULTS_DIR", str(tmp_path))
    records = [_record("t1", v7_candidates=4, v7_surviving=2)]
    monkeypatch.setattr(replay, "collect_db_truth",
                        lambda db, recs: {"t1": {"v7": {"gap_candidates": 4,
                                                       "surviving": 7}}})
    monkeypatch.setattr(replay, "build_p2_counterfactual",
                        lambda db, recs, floor=None: {"enabled": False})
    monkeypatch.setattr(replay, "build_mech_override_review", lambda db, rows: [])

    import asyncio
    report = asyncio.run(replay.write_outputs(
        "run", records, db=object(), strict=False))

    assert report["db_consistency"]["ok"] is False
    assert (tmp_path / "run" / "replay_report.json").exists()


# --- the override review sheet -------------------------------------------

def _patch_review_sources(monkeypatch, claims=(), coverages=(), relevances=(),
                          papers=None, gap=None):
    import app.db.repositories.gap_repo as gap_repo

    monkeypatch.setattr(gap_repo, "list_atomic_claims", lambda db, gap_id: list(claims))
    monkeypatch.setattr(gap_repo, "list_neighbor_claim_coverage",
                        lambda db, gap_id: list(coverages))
    monkeypatch.setattr(gap_repo, "list_gap_paper_relevance",
                        lambda db, gap_id: list(relevances))
    monkeypatch.setattr(
        "app.agent.steps.audit_gaps._mechanical_novelty_score",
        lambda db, g, relevance_floor=None: {"novelty": 1.0})


def test_review_sheet_covers_only_override_gaps(monkeypatch):
    from types import SimpleNamespace

    new_rows = [
        _gap_row("g-mech", basis=replay.SURVIVAL_MECHANICAL_OVERRIDE),
        _gap_row("g-self", basis=replay.SURVIVAL_SELF_REPORT),
        _gap_row("g-none", basis=None),
    ]
    claims = [SimpleNamespace(claim_index=0, claim_text="C0")]
    cov = [SimpleNamespace(neighbor_paper_id="p1", claim_id="c0", coverage="NONE")]
    rel = [SimpleNamespace(paper_id="p1", relevance_score=0.68)]

    class _Db(_FakeDb):
        def query(self, model):
            return _Rows([])

    db = _Db()
    db.get = lambda model, key: SimpleNamespace(title="A relevant paper")
    _patch_review_sources(monkeypatch, claims=claims, coverages=cov,
                          relevances=rel)

    rows = replay.build_mech_override_review(db, new_rows)

    assert len(rows) == 1
    assert rows[0]["gap_id"] == "g-mech"
    assert rows[0]["survives_p2_floor"] is True
    assert rows[0]["atomic_claims"][0]["claim_text"] == "C0"
    assert rows[0]["neighbors"][0]["title"] == "A relevant paper"
    assert rows[0]["neighbors"][0]["gap_relevance"] == 0.68
    # Two empty fields for the human — never a machine guess.
    assert rows[0]["human_prior_art_verdict"] == ""
    assert rows[0]["human_notes"] == ""


def test_review_sheet_marks_gaps_the_floor_already_drops(monkeypatch):
    """The sheet keeps the P2-failing rows but flags them, so the floor's own
    verdict stays checkable instead of silently disappearing."""
    from app.config import settings

    from types import SimpleNamespace

    monkeypatch.setattr(settings, "gap_relevance_min_neighbor", 0.3)
    new_rows = [_gap_row("g-weak", basis=replay.SURVIVAL_MECHANICAL_OVERRIDE)]
    _patch_review_sources(
        monkeypatch,
        claims=[SimpleNamespace(claim_index=0, claim_text="C0")],
        coverages=[],
        relevances=[])
    monkeypatch.setattr(
        "app.agent.steps.audit_gaps._mechanical_novelty_score",
        lambda db, g, relevance_floor=None: None)  # no usable neighbours left

    db = _FakeDb()
    db.get = lambda model, key: None
    rows = replay.build_mech_override_review(db, new_rows)

    assert len(rows) == 1
    assert rows[0]["survives_p2_floor"] is False
    assert rows[0]["novelty_after_p2_floor"] is None


def test_review_sheet_is_empty_without_a_db():
    assert replay.build_mech_override_review(
        None, [_gap_row("g1", basis=replay.SURVIVAL_MECHANICAL_OVERRIDE)]) == []


# --- rule 3: survival basis and the de-confounded decomposition -----------

def test_classify_survival_basis_distinguishes_gates():
    assert replay.classify_survival_basis("surviving", {"novelty_confidence": 0.8},
                                          None) == replay.SURVIVAL_SELF_REPORT
    assert replay.classify_survival_basis(
        "surviving", {"novelty_confidence": 0.2},
        {"kind": replay.SURVIVAL_MECHANICAL_OVERRIDE}
    ) == replay.SURVIVAL_MECHANICAL_OVERRIDE
    assert replay.classify_survival_basis(
        "surviving", {"novelty_confidence": 0.2},
        {"kind": replay.SURVIVAL_PROVISIONAL}) == replay.SURVIVAL_PROVISIONAL
    # Surviving with no audit row at all is a data defect, not a verdict.
    assert replay.classify_survival_basis("surviving", None,
                                          None) == replay.SURVIVAL_UNKNOWN
    # Non-surviving gaps have no basis.
    assert replay.classify_survival_basis("rejected", {"novelty_confidence": 0.9},
                                          None) is None


def test_mechanical_override_beats_a_high_self_report():
    """The override trace is authoritative: it only exists when the gate fired."""
    basis = replay.classify_survival_basis(
        "surviving", {"novelty_confidence": 0.9},
        {"kind": replay.SURVIVAL_MECHANICAL_OVERRIDE})
    assert basis == replay.SURVIVAL_MECHANICAL_OVERRIDE


def test_decomposition_attributes_a_gate_driven_gain_to_the_gate():
    """v7 doubles its surviving rate, but only via mechanical overrides."""
    v6_basis = replay._empty_basis_counts()
    v6_basis.update({"surviving_total": 2, "self_report": 2})
    v7_basis = replay._empty_basis_counts()
    v7_basis.update({"surviving_total": 4, "self_report": 2,
                     "mechanical_override": 2})
    records = [_record("t1", v6_candidates=10, v6_surviving=2,
                       v7_candidates=10, v7_surviving=4,
                       v6_basis=v6_basis, v7_basis=v7_basis)]

    table = replay.build_survival_2x2(records)
    decomposition = table["decomposition"]

    assert decomposition["delta_surviving_rate"] == pytest.approx(0.2)
    assert decomposition["delta_from_self_report"] == pytest.approx(0.0)
    assert decomposition["delta_from_mechanical_override"] == pytest.approx(0.2)
    assert decomposition["mechanical_share_of_gain"] == pytest.approx(1.0)


def test_decomposition_credits_the_ontology_when_self_report_rises():
    v6_basis = replay._empty_basis_counts()
    v6_basis.update({"surviving_total": 1, "self_report": 1})
    v7_basis = replay._empty_basis_counts()
    v7_basis.update({"surviving_total": 3, "self_report": 3})
    records = [_record("t1", v6_candidates=10, v6_surviving=1,
                       v7_candidates=10, v7_surviving=3,
                       v6_basis=v6_basis, v7_basis=v7_basis)]

    decomposition = replay.build_survival_2x2(records)["decomposition"]

    assert decomposition["delta_from_self_report"] == pytest.approx(0.2)
    assert decomposition["delta_from_mechanical_override"] == pytest.approx(0.0)
    assert decomposition["mechanical_share_of_gain"] == pytest.approx(0.0)


def test_rates_are_per_candidate_not_per_survivor():
    """A per-survivor share cannot be compared across arms with different
    surviving counts; the rates must divide by candidates."""
    v6_basis = replay._empty_basis_counts()
    v6_basis.update({"surviving_total": 1, "self_report": 1})
    v7_basis = replay._empty_basis_counts()
    v7_basis.update({"surviving_total": 5, "self_report": 5})
    records = [_record("t1", v6_candidates=10, v6_surviving=1,
                       v7_candidates=50, v7_surviving=5,
                       v6_basis=v6_basis, v7_basis=v7_basis)]

    rates = replay.build_survival_2x2(records)["rates"]

    assert rates["v6"]["self_report_rate"] == pytest.approx(0.1)
    assert rates["v7"]["self_report_rate"] == pytest.approx(0.1)
    # Same per-survivor share (100%), very different per-candidate rates.
    assert rates["v6"]["surviving_rate"] == pytest.approx(0.1)
    assert rates["v7"]["surviving_rate"] == pytest.approx(0.1)


def test_v7_only_arm_counts_only_new_opportunities():
    gaps = [
        _gap_row("g1", is_new=True, basis=replay.SURVIVAL_SELF_REPORT),
        _gap_row("g2", is_new=True, basis=replay.SURVIVAL_MECHANICAL_OVERRIDE),
        _gap_row("g3", is_new=False, basis=replay.SURVIVAL_SELF_REPORT),
    ]
    records = [_record("t1", gaps=gaps)]

    table = replay.build_survival_2x2(records)

    assert table["counts"]["v7_only"]["surviving_total"] == 2
    assert table["counts"]["v7_only"]["self_report"] == 1
    assert table["counts"]["v7_only"]["mechanical_override"] == 1
    assert table["rates"]["v7_only"]["candidates"] == 2


def test_stale_records_do_not_enter_the_2x2():
    v7_basis = replay._empty_basis_counts()
    v7_basis.update({"surviving_total": 9, "mechanical_override": 9})
    records = [_record("t1", v7_basis=v7_basis, stale_error="boom")]

    table = replay.build_survival_2x2(records)

    assert table["counts"]["v7"]["surviving_total"] == 0


def test_per_type_funnel_splits_surviving_by_basis():
    gaps = [
        _gap_row("g1", gap_type="contradiction",
                 basis=replay.SURVIVAL_SELF_REPORT),
        _gap_row("g2", gap_type="contradiction",
                 basis=replay.SURVIVAL_MECHANICAL_OVERRIDE),
    ]
    report = replay.summarize([_record("t1", gaps=gaps)])

    bucket = report["q4_per_type_funnel"]["contradiction"]
    assert bucket["surviving"] == 2
    assert bucket["surviving_self_report"] == 1
    assert bucket["surviving_mechanical_override"] == 1


def test_report_is_json_serialisable():
    report = replay.summarize([_record("t1", gaps=[_gap_row("g1")])])
    json.dumps(report, ensure_ascii=False)


# --- P2 counterfactual: what the funnel becomes with the floor enforced ---

def _fake_gap(gap_id, status="surviving", gap_type="boundary_gap"):
    from types import SimpleNamespace
    return SimpleNamespace(id=gap_id, status=status, gap_type=gap_type)


def _fake_trace(task_id, gap_id):
    from types import SimpleNamespace
    return SimpleNamespace(task_id=task_id,
                           output_json=json.dumps({"gap_id": gap_id}))


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *args, **kwargs):
        return self

    def all(self):
        return self._rows

    def first(self):
        return self._rows[0] if self._rows else None


class _FakeDb:
    def __init__(self, traces=(), contracts=()):
        self._traces = list(traces)
        self._contracts = list(contracts)

    def query(self, model):
        from app.db.models import AgentTrace, ResearchContract

        if model is AgentTrace:
            return _Rows(self._traces)
        if model is ResearchContract:
            return _Rows(self._contracts)
        return _Rows([])


def _cf_record(topic_id="t1"):
    return {"topic_id": topic_id, "v6_task_id": "v6task",
            "replay_task_id": "v7task", "replay_contract_id": "c-v7"}


def _patch_gaps(monkeypatch, gaps_by_task):
    import app.db.repositories.gap_repo as gap_repo

    monkeypatch.setattr(
        gap_repo, "list_gaps_for_contract",
        lambda db, task_id, contract_id, include_superseded=True:
            gaps_by_task.get(task_id, []))


def test_counterfactual_disabled_when_floor_is_off(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "gap_relevance_min_neighbor", 0.0)
    result = replay.build_p2_counterfactual(None, [_cf_record()])

    assert result["enabled"] is False


def test_counterfactual_leaves_non_override_gaps_alone(monkeypatch):
    """A gap with no override trace was admitted by self-report; P2 cannot
    touch it, so the rate must not move."""
    from types import SimpleNamespace

    from app.config import settings

    monkeypatch.setattr(settings, "gap_relevance_min_neighbor", 0.3)
    _patch_gaps(monkeypatch, {"v6task": [_fake_gap("g-v6")],
                              "v7task": [_fake_gap("g-v7")]})
    # The v6 record carries no contract id, so it is resolved through the DB.
    db = _FakeDb(contracts=[SimpleNamespace(id="c-v6", task_id="v6task")])

    result = replay.build_p2_counterfactual(db, [_cf_record()])

    assert result["n_dropped"] == 0
    assert result["arms"]["v6"]["surviving_rate_now"] == 1.0
    assert result["arms"]["v6"]["surviving_rate_after"] == 1.0
    assert result["arms"]["v7"]["surviving_rate_after"] == 1.0


def test_counterfactual_drops_an_override_that_fails_the_floor(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "gap_relevance_min_neighbor", 0.3)
    _patch_gaps(monkeypatch, {"v6task": [], "v7task": [_fake_gap("g-weak")]})
    db = _FakeDb(traces=[_fake_trace("v7task", "g-weak")])
    monkeypatch.setattr(
        "app.agent.steps.audit_gaps._mechanical_novelty_score",
        lambda db, gap, relevance_floor=None: {
            "novelty": 0.17, "neighbor_relevance_median": 0.0})

    result = replay.build_p2_counterfactual(db, [_cf_record()])

    assert result["n_dropped"] == 1
    assert result["dropped"][0]["gap_id"] == "g-weak"
    assert result["dropped"][0]["novelty_after_floor"] == 0.17
    assert result["arms"]["v7"]["surviving_now"] == 1
    assert result["arms"]["v7"]["surviving_after"] == 0
    assert result["arms"]["v7"]["rate_delta"] == pytest.approx(-1.0)


def test_counterfactual_keeps_an_override_that_still_clears_the_floor(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "gap_relevance_min_neighbor", 0.3)
    _patch_gaps(monkeypatch, {"v6task": [], "v7task": [_fake_gap("g-strong")]})
    db = _FakeDb(traces=[_fake_trace("v7task", "g-strong")])
    monkeypatch.setattr(
        "app.agent.steps.audit_gaps._mechanical_novelty_score",
        lambda db, gap, relevance_floor=None: {
            "novelty": 1.0, "neighbor_relevance_median": 0.68})

    result = replay.build_p2_counterfactual(db, [_cf_record()])

    assert result["n_dropped"] == 0
    assert result["arms"]["v7"]["surviving_rate_after"] == 1.0


def test_counterfactual_reports_pending_when_not_computed():
    report = replay.summarize([_record("t1")])

    assert report["q7_p2_relevance_floor_counterfactual"] == "PENDING"
