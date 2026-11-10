"""Corpus-wide prior-art recall for the gap audit (2026-11-02).

Motivating measurement (fixed-budget pilot, 37 reviewed gaps): of the 82 killer
papers the independent reviewer named that the task corpus ALREADY held, only
4 (4.9%) were ever shown to the audit as neighbours. The largest single loss
bucket was "in the corpus, never admitted to any pool" — 55 papers (67.1%).

These tests pin the three properties that make the fix safe to run:

  1. recall is CORPUS-ONLY and soft (disclosed counts, threshold in the diag);
  2. the gap's own supporting papers are excluded (a claim cannot be falsified
     by the evidence that established it);
  3. embeddings failing must degrade to "no recall + a reason", never to an
     exception and never to a silent empty list indistinguishable from
     "nothing is close".
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


def _seed(db, papers, support_ids=()):
    """papers: list of (title, abstract). support_ids: indices used as evidence."""
    from app.agent.steps.mine_gaps import GAP_MINING_POLICY_VERSION
    from app.db.models import (EvidenceUnit, GapCandidate, GapEvidenceLink, Paper,
                              ResearchContract, ResearchTask, TaskPaper)

    task = ResearchTask(user_input="recall topic", status="auditing_gaps")
    db.add(task)
    db.flush()
    contract = ResearchContract(task_id=task.id, topic="t",
                                input_hash="h-recall", version=1)
    db.add(contract)
    db.flush()

    paper_rows = []
    for i, (title, abstract) in enumerate(papers):
        paper = Paper(title=title, abstract=abstract,
                      normalized_title=title.lower())
        db.add(paper)
        db.flush()
        db.add(TaskPaper(task_id=task.id, paper_id=paper.id,
                         discovered_round=1, final_score=0.5))
        paper_rows.append(paper)
    db.flush()

    gap = GapCandidate(
        task_id=task.id, contract_id=contract.id, gap_type="boundary_gap",
        description="d", question_ids_json="[]", supporting_evidence_ids_json="[]",
        contradicting_evidence_ids_json="[]", mining_round=1, status="candidate",
        target_setting="long-context QA",
        observed_problem="accuracy degrades past 32k tokens",
        existing_coverage="evaluated to 32k",
        missing_capability="no measurement beyond 32k tokens",
        claimed_delta="Accuracy beyond 32k tokens has not been measured.",
        testable_hypothesis="accuracy drops >10pp beyond 32k",
        falsification_condition="a paper measures beyond 32k",
        provenance_status="complete",
        mining_policy_version=GAP_MINING_POLICY_VERSION,
    )
    db.add(gap)
    db.flush()

    for idx in support_ids:
        unit = EvidenceUnit(
            task_id=task.id, paper_id=paper_rows[idx].id,
            evidence_type="limitation", verification_status="verified",
            normalized_claim="degradation observed past 32k",
            original_span="span", source_chunk_hash=f"h{idx}", page_number=1,
            span_start=0, span_end=5, section="conclusion",
        )
        db.add(unit)
        db.flush()
        db.add(GapEvidenceLink(gap_id=gap.id, evidence_id=unit.id,
                               relation_type="supports", relevance_score=0.9))
    db.commit()
    return task, gap, paper_rows


def _fake_embedder(mapping):
    """Return an embed_texts replacement keyed by text prefix."""
    def _embed(texts):
        out = []
        for text in texts:
            vec = None
            for prefix, v in mapping.items():
                if text.startswith(prefix) or prefix in text:
                    vec = v
                    break
            out.append(vec or [0.0, 1.0])
        return out
    return _embed


# === Core behaviour ========================================================

def test_recall_returns_corpus_papers_closest_to_the_claim(temp_db, monkeypatch):
    from app.config import settings
    from app.agent.steps import corpus_recall

    db = temp_db()
    task, gap, papers = _seed(db, [
        ("Lost in the Middle: long-context degradation", "position bias"),
        ("Cooking with cast iron", "recipes for pans"),
    ])
    # Claim vector is [1,0]; the first paper matches, the second is orthogonal.
    monkeypatch.setattr(corpus_recall.embedding_service, "embed_texts",
                        _fake_embedder({"Accuracy beyond": [1.0, 0.0],
                                        "Lost in the Middle": [0.95, 0.31]}))
    monkeypatch.setattr(settings, "audit_corpus_recall_enabled", True)
    monkeypatch.setattr(settings, "audit_corpus_recall_threshold", 0.5)

    ids, diag = corpus_recall.recall_corpus_neighbors(db, gap, task.id)

    assert ids == [papers[0].id]
    assert diag["semantic_filtered"] is True
    assert diag["threshold"] == 0.5
    assert diag["corpus_count"] == 2
    assert diag["filtered_count"] >= 1
    # Diagnostic must always carry the kept/filtered split: a caller that shows
    # only `kept` reproduces the original "reviewer saw nothing" bug.
    assert {"kept_count", "filtered_count", "candidate_count"} <= set(diag)
    db.close()


def test_recall_excludes_the_gaps_own_supporting_papers(temp_db, monkeypatch):
    """A claim cannot be falsified by the evidence that established it."""
    from app.config import settings
    from app.agent.steps import corpus_recall

    db = temp_db()
    task, gap, papers = _seed(db, [
        ("Source paper that established the limitation", "degradation past 32k"),
        ("Independent prior art on the same claim", "also studies 32k"),
    ], support_ids=[0])
    monkeypatch.setattr(corpus_recall.embedding_service, "embed_texts",
                        _fake_embedder({"Accuracy beyond": [1.0, 0.0],
                                        "Source paper": [0.99, 0.01],
                                        "Independent prior": [0.97, 0.05]}))
    monkeypatch.setattr(settings, "audit_corpus_recall_enabled", True)
    monkeypatch.setattr(settings, "audit_corpus_recall_threshold", 0.5)

    ids, diag = corpus_recall.recall_corpus_neighbors(db, gap, task.id)

    assert papers[0].id not in ids, "support paper must never be recalled"
    assert ids == [papers[1].id]
    assert diag["excluded_support_papers"] == 1
    db.close()


# === Degradation contract ==================================================

def test_embedding_failure_degrades_with_a_reason(temp_db, monkeypatch):
    """Infra failure is not evidence about the claim: return nothing, say why."""
    from app.config import settings
    from app.agent.steps import corpus_recall

    db = temp_db()
    task, gap, _ = _seed(db, [("Some paper", "abstract")])
    monkeypatch.setattr(settings, "audit_corpus_recall_enabled", True)

    def _boom(texts):
        raise RuntimeError("embedding endpoint down")

    monkeypatch.setattr(corpus_recall.embedding_service, "embed_texts", _boom)

    ids, diag = corpus_recall.recall_corpus_neighbors(db, gap, task.id)

    assert ids == []
    assert diag["semantic_filtered"] is False
    assert "embedding_failed" in diag["degraded"]
    db.close()


def test_disabled_recall_is_distinguishable_from_empty_recall(temp_db, monkeypatch):
    """A disabled pass and a pass that found nothing must not look alike."""
    from app.config import settings
    from app.agent.steps import corpus_recall

    db = temp_db()
    task, gap, _ = _seed(db, [("Some paper", "abstract")])
    monkeypatch.setattr(settings, "audit_corpus_recall_enabled", False)

    ids, diag = corpus_recall.recall_corpus_neighbors(db, gap, task.id)

    assert ids == []
    assert diag["degraded"] == "disabled"
    assert diag["enabled"] is False
    db.close()


def test_empty_corpus_is_reported_as_such(temp_db, monkeypatch):
    from app.config import settings
    from app.agent.steps import corpus_recall

    db = temp_db()
    task, gap, _ = _seed(db, [])
    monkeypatch.setattr(settings, "audit_corpus_recall_enabled", True)

    ids, diag = corpus_recall.recall_corpus_neighbors(db, gap, task.id)

    assert ids == []
    assert diag["degraded"] == "empty_corpus"
    db.close()


# === Pool union ============================================================

async def test_recall_hits_are_unioned_into_the_audit_candidate_pool(
        temp_db, monkeypatch):
    """The 67.1% bucket: a corpus killer must be able to reach the audit.

    Guards the wiring end to end: with an adversarial search that recalls only
    one paper, a corpus-recall hit must still appear in the pool handed to the
    gap-relevance screen and the neighbour selector. Recall that never enters
    the pool changes nothing, and that was the whole defect.
    """
    from app.config import settings
    from app.agent.steps import audit_gaps

    db = temp_db()
    task, gap, papers = _seed(db, [
        ("Unrelated survey of cooking", "cast iron recipes"),
        ("Corpus killer: exact claim covered", "measures beyond 32k directly"),
    ])

    # One adversarial query, completing successfully, recalling only paper 0 —
    # so paper 1 can ONLY reach the pool through corpus recall.
    from app.agent.steps.generate_queries import SearchQueryExecution

    def _fake_save_search_query(db_, task_id, query_text, intent, *a, **k):
        from types import SimpleNamespace
        return SimpleNamespace(id=f"q-{query_text[:6]}")

    async def _fake_search_and_save(db_, state_, executions, task_id, round_, **k):
        return None

    async def _fake_generate_queries(db_, llm_, gap_):
        return [audit_gaps.AdversarialQuerySpec(family="exact_gap",
                                                query_text="beyond 32k measurement")]

    monkeypatch.setattr(audit_gaps, "save_search_query", _fake_save_search_query)
    monkeypatch.setattr(audit_gaps, "search_and_save_papers", _fake_search_and_save)
    monkeypatch.setattr(audit_gaps, "generate_english_adversarial_queries",
                        _fake_generate_queries)
    monkeypatch.setattr(audit_gaps, "recall_corpus_neighbors",
                        lambda db_, gap_, task_id_: ([papers[1].id], {
                            "kept_count": 1, "semantic_filtered": True,
                            "threshold": 0.45, "corpus_count": 2,
                            "candidate_count": 2, "returned_count": 1,
                            "enabled": True, "max_similarity": 0.9}))

    # Admission passed with an EMPTY candidate list on purpose: the pool must be
    # built entirely from corpus recall, which is exactly the 67.1% bucket.
    def _fake_admission(db_, gap_, query_ids):
        return audit_gaps.GapSearchAdmission(
            gap_.id, "PASS", [], list(query_ids), list(query_ids), [], ["exact_gap"],
            1, [], [])

    monkeypatch.setattr(audit_gaps, "evaluate_gap_search_admission",
                        _fake_admission)
    monkeypatch.setattr(settings, "audit_screen_before_cap", True)

    seen_pools = []

    async def _fake_score(db_, llm_, gap_, paper_ids, task_id_):
        seen_pools.append(list(paper_ids))
        return [(pid, 0.5, object()) for pid in paper_ids]

    monkeypatch.setattr(audit_gaps, "score_all_gap_candidates", _fake_score)
    monkeypatch.setattr(audit_gaps, "select_gap_specific_neighbors",
                        lambda *a, **k: [])

    await audit_gaps.audit_gap_candidate(
        db, _state(task), None, task.id, gap, perform_search=False)

    assert seen_pools, "the pool must reach the relevance screen"
    assert papers[1].id in seen_pools[0], (
        "a corpus-recall hit must enter the audit candidate pool; this is the "
        "67.1% loss bucket")
    db.close()


def _state(task):
    from app.agent.state import ResearchState
    from app.db.models import ResearchContract
    contract_id = None
    return ResearchState(task_id=task.id, contract_id=contract_id, current_round=1)


async def test_recall_trace_is_written_even_when_recall_returns_nothing(
        temp_db, monkeypatch):
    """A never-run recall and a run-that-found-nothing must be distinguishable."""
    from app.agent.steps import corpus_recall

    db = temp_db()
    task, gap, _ = _seed(db, [("Unrelated", "nothing in common")])
    monkeypatch.setattr(corpus_recall.embedding_service, "embed_texts",
                        _fake_embedder({"Accuracy beyond": [1.0, 0.0],
                                        "Unrelated": [0.0, 1.0]}))

    from app.config import settings
    monkeypatch.setattr(settings, "audit_corpus_recall_enabled", True)
    monkeypatch.setattr(settings, "audit_corpus_recall_threshold", 0.9)

    ids, diag = corpus_recall.recall_corpus_neighbors(db, gap, task.id)
    assert ids == []
    assert diag["semantic_filtered"] is True
    assert diag["kept_count"] == 0
    assert diag.get("degraded") is None
    db.close()
