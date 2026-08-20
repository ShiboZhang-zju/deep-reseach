"""Regression: gap-specific prior-art relevance screening (0027).

Reproduces the fd688ba6 false-novelty failure mode: audit-recalled papers were
stored with final_score=NULL, and the neighbor selector then ranked a broad RAG
survey ahead of a directly relevant abstention paper on raw query hits — so
direct prior art never reached the NPA pool and a false broad gap survived.

Under test:
- score_gap_papers: cheap title/abstract gap-specific scoring, LLM emits only
  qualitative labels + claim ids; relevance_score is aggregated by code.
- select_gap_specific_neighbors: gap relevance is the primary screening signal;
  a directly relevant paper must not be crowded out of the Top-M by a survey.

Scenario mirrors fd688ba6: a gap about "RAG lacks evaluation of refusing when
evidence is insufficient", with BioRefusalAudit (direct prior art, low hits)
vs "A Systematic Literature Review of Retrieval-Augmented Generation" (survey,
high hits). After screening, BioRefusalAudit must enter the NPA pool; the
survey must not crowd it out.
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


class FakeGapRelLLM:
    """Emits qualitative overlap labels matching a paper->label map.

    claim_overlap is 'yes' for direct prior art, 'partial' for an adjacent
    paper, 'no' for a survey — the exact signal that must survive aggregation
    and change the neighbor ranking.
    """

    def __init__(self, labels):
        self.labels = labels  # {paper_id: dict(...)}
        self.calls = 0

    async def chat_json(self, messages, schema):
        from app.agent.steps.gap_relevance import GapPaperRelevanceList, GapPaperRelevanceSchema
        self.calls += 1
        papers = []
        for pid, lab in self.labels.items():
            papers.append(GapPaperRelevanceSchema(
                paper_id=pid,
                problem_overlap=lab.get("problem_overlap", "no"),
                mechanism_overlap=lab.get("mechanism_overlap", "no"),
                evaluation_overlap=lab.get("evaluation_overlap", "no"),
                claim_overlap=lab.get("claim_overlap", "no"),
                addresses_claim_ids=lab.get("addresses_claim_ids", []),
                rationale=lab.get("rationale", ""),
            ))
        return GapPaperRelevanceList(papers=papers)


def _seed_gap(db, task_id):
    from app.db.models import GapCandidate, GapAtomicClaim
    gap = GapCandidate(
        id="gap-rel-test", task_id=task_id, contract_id="contract-1",
        status="auditing", gap_type="missing_evaluation",
        description="RAG lacks a dedicated refusal-under-insufficiency evaluation.",
        target_setting="RAG when retrieved evidence is insufficient",
        observed_problem="Systems do not refuse when evidence is missing",
        existing_coverage="HaluEval covers generated hallucinations",
        missing_capability="Dedicated refusal-under-insufficiency evaluation",
        claimed_delta="RAG lacks a dedicated evaluation of refusing when "
                      "retrieved evidence is insufficient.",
    )
    db.add(gap)
    db.flush()
    db.add(GapAtomicClaim(id="claim-1", task_id=task_id, gap_id=gap.id,
                          claim_index=0,
                          claim_text="A benchmark that measures refusal accuracy "
                                     "when evidence is absent."))
    db.add(GapAtomicClaim(id="claim-2", task_id=task_id, gap_id=gap.id,
                          claim_index=1,
                          claim_text="A protocol distinguishing confident "
                                     "hallucination from correct abstention."))
    db.flush()
    return gap


def _add_paper(db, paper_id, title, abstract, hits=0, families=0, rank=99, final_score=None):
    from app.db.models import Paper, TaskPaper, SearchQueryPaper, SearchQueryRecord
    db.add(Paper(id=paper_id, title=title, abstract=abstract))
    if final_score is not None:
        db.add(TaskPaper(task_id="task-1", paper_id=paper_id, discovered_round=2,
                         final_score=final_score))
    for i in range(hits):
        qid = f"q-{paper_id}-{i}"
        db.add(SearchQueryRecord(id=qid, task_id="task-1",
                                 query_text=f"query {i} for {paper_id}",
                                 normalized_query_text=f"q{i}-{paper_id}",
                                 intent="gap_mechanism",
                                 round_number=2, target_gap_id="gap-rel-test",
                                 query_family=f"fam-{i % max(families, 1)}"))
        db.add(SearchQueryPaper(query_id=qid, paper_id=paper_id, rank=rank,
                                source="openalex"))
    db.flush()


@pytest.mark.asyncio
async def test_direct_prior_art_survives_survey_in_neighbor_pool(temp_db):
    from app.agent.steps.gap_relevance import score_gap_papers
    from app.agent.steps.audit_gaps import select_gap_specific_neighbors

    db = temp_db()
    gap = _seed_gap(db, "task-1")
    # Direct prior art: low hits but exactly on-gap.
    _add_paper(db, "bio-refusal", "BioRefusalAudit: Audit of Refusal under Missing Evidence",
               "We evaluate whether biomedical RAG systems refuse when retrieved "
               "evidence is insufficient, measuring refusal accuracy on unanswerable queries.",
               hits=2, families=1, rank=0, final_score=None)
    # Broad survey: high hits, generic.
    _add_paper(db, "rag-survey", "A Systematic Literature Review of Retrieval-Augmented Generation",
               "We review the field of retrieval-augmented generation, taxonomizing "
               "retrieval strategies and generation methods across many tasks.",
               hits=8, families=5, rank=1, final_score=0.81)
    # Adjacent paper: partial overlap.
    _add_paper(db, "adjacent", "Multi-Hop Reasoning Evaluation in RAG Systems",
               "We evaluate multi-hop reasoning quality of RAG systems over "
               "multi-hop QA benchmarks.",
               hits=5, families=4, rank=2, final_score=0.7)
    db.commit()

    llm = FakeGapRelLLM({
        "bio-refusal": {"problem_overlap": "yes", "mechanism_overlap": "partial",
                        "evaluation_overlap": "yes", "claim_overlap": "yes",
                        "addresses_claim_ids": ["claim-1", "claim-2"],
                        "rationale": "Directly evaluates refusal under missing evidence."},
        "rag-survey": {"problem_overlap": "no", "mechanism_overlap": "no",
                       "evaluation_overlap": "no", "claim_overlap": "no",
                       "addresses_claim_ids": [],
                       "rationale": "Generic survey, no refusal evaluation."},
        "adjacent": {"problem_overlap": "partial", "mechanism_overlap": "no",
                     "evaluation_overlap": "partial", "claim_overlap": "partial",
                     "addresses_claim_ids": [],
                     "rationale": "Evaluates multi-hop QA but not refusal under insufficiency."},
    })

    scored = await score_gap_papers(db, llm, gap, ["bio-refusal", "rag-survey", "adjacent"],
                                    task_id="task-1")
    scored_by_id = {pid: score for pid, score, _ in scored}
    # Direct prior art must rank above the broad survey.
    assert scored_by_id["bio-refusal"] > scored_by_id["rag-survey"], \
        f"direct prior art {scored_by_id} must outscore survey"
    assert scored_by_id["bio-refusal"] > scored_by_id["adjacent"], scored_by_id
    assert scored_by_id["adjacent"] > scored_by_id["rag-survey"], scored_by_id

    # Selection: query_ids over the audit pool; with gap relevance active, the
    # direct prior art must enter the Top-K even though the survey has more hits.
    query_ids = ["q-bio-refusal-0", "q-bio-refusal-1"] + \
                [f"q-rag-survey-{i}" for i in range(8)] + \
                [f"q-adjacent-{i}" for i in range(5)]
    neighbors = select_gap_specific_neighbors(db, gap, query_ids, limit=2)
    nbr_ids = [p.id for p in neighbors]
    assert "bio-refusal" in nbr_ids, \
        f"direct prior art must enter NPA pool, got {nbr_ids}"
    assert "rag-survey" not in nbr_ids, \
        f"generic survey must not crowd the pool, got {nbr_ids}"
    db.close()


@pytest.mark.asyncio
async def test_survey_outranks_prior_art_without_screening(temp_db):
    """Without gap relevance, the legacy formula lets the survey win — proving
    the screen is what changes the outcome (the fd688ba6 failure mode)."""
    from app.agent.steps.audit_gaps import select_gap_specific_neighbors

    db = temp_db()
    gap = _seed_gap(db, "task-1")
    _add_paper(db, "bio-refusal", "BioRefusalAudit",
               "Refusal under missing evidence evaluation.",
               hits=2, families=1, rank=0, final_score=None)
    _add_paper(db, "rag-survey", "A Systematic Literature Review of RAG",
               "Broad review of RAG.",
               hits=8, families=5, rank=1, final_score=0.81)
    db.commit()

    query_ids = ["q-bio-refusal-0", "q-bio-refusal-1"] + \
                [f"q-rag-survey-{i}" for i in range(8)]
    # No GapPaperRelevance rows exist -> gate inactive -> legacy formula wins.
    neighbors = select_gap_specific_neighbors(db, gap, query_ids, limit=1)
    assert neighbors[0].id == "rag-survey", \
        f"legacy formula should prefer the survey, got {neighbors[0].id}"
    db.close()


def test_aggregate_relevance_weights():
    from app.agent.steps.gap_relevance import _aggregate_relevance, GapPaperRelevanceSchema

    def rel(**kw):
        return _aggregate_relevance(GapPaperRelevanceSchema(
            paper_id="p", problem_overlap=kw.get("problem", "no"),
            mechanism_overlap=kw.get("mechanism", "no"),
            evaluation_overlap=kw.get("evaluation", "no"),
            claim_overlap=kw.get("claim", "no"),
            addresses_claim_ids=[], rationale="",
        ))

    from app.config import settings
    cw, pw, ew = (settings.gap_relevance_claim_weight,
                  settings.gap_relevance_problem_weight,
                  settings.gap_relevance_evaluation_weight)
    # all no -> 0
    assert rel() == 0.0
    # claim yes only
    assert rel(claim="yes") == pytest.approx(cw)
    # problem yes only
    assert rel(problem="yes") == pytest.approx(pw)
    # evaluation partial only (0.5)
    assert rel(evaluation="partial") == pytest.approx(0.5 * ew)
    # full yes
    assert rel(claim="yes", problem="yes", evaluation="yes") == pytest.approx(cw + pw + ew)
