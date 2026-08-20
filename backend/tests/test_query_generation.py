"""Tests for the two-step query-generation contract (P1-1 Phase 3I).

Locks: (1) a family's canonical intent is fixed first, (2) variants are validated
for semantic invariance with regenerate-on-drift, (3) <2 valid variants marks
QUERY_GENERATION_INVALID (not SEARCH_UNSTABLE), (4) drifted variants never enter
the query set.
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    config = Config()
    config.set_main_option("sqlalchemy.url", f"sqlite:///{path}")
    config.set_main_option("script_location",
                          os.path.join(os.path.dirname(__file__), "..", "alembic_migrations"))
    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{path}")
    yield sessionmaker(bind=engine)
    engine.dispose()
    try:
        os.unlink(path)
    except PermissionError:
        pass


def _intent(family="exact_gap", n_variants=3):
    from app.agent.steps.audit_gaps import IntentWithVariantsSchema
    return IntentWithVariantsSchema(
        family=family,
        problem="self-correction is evaluated on sparse test suites",
        mechanism="sparse tests cannot distinguish accidental from robust correctness",
        intervention="sparse-to-dense test evaluation",
        evaluation_setting="HumanEval-style code benchmarks",
        task_scope="LLM code generation self-correction",
        variants=[f"variant {i} about sparse-to-dense correctness" for i in range(n_variants)],
    )


def _seed_gap(db):
    from app.agent.steps.mine_gaps import GAP_MINING_POLICY_VERSION
    from app.db.models import GapCandidate, ResearchContract, ResearchTask
    task = ResearchTask(user_input="code gen", status="auditing_gaps")
    db.add(task)
    db.flush()
    contract = ResearchContract(task_id=task.id, topic="Code Gen", status="active",
                                version=1, input_hash="v1")
    db.add(contract)
    db.flush()
    gap = GapCandidate(
        task_id=task.id, contract_id=contract.id, gap_type="boundary_gap",
        description="self-correction eval", target_setting="code gen",
        observed_problem="accidental correctness", existing_coverage="generic",
        missing_capability="dense hidden tests", claimed_delta="sparse-to-dense eval",
        testable_hypothesis="h", falsification_condition="f",
        status="auditing", mining_policy_version=GAP_MINING_POLICY_VERSION,
    )
    db.add(gap)
    db.commit()
    return task, gap


class IntentLLM:
    def __init__(self, intents):
        self.intents = intents

    async def chat_json(self, messages, schema):
        from app.agent.steps.audit_gaps import GapQueryGenList, RegenerateVariantSchema
        if schema is GapQueryGenList:
            return GapQueryGenList(intents=self.intents)
        if schema is RegenerateVariantSchema:
            return RegenerateVariantSchema(variant="regenerated faithful paraphrase")
        raise ValueError(f"unexpected schema {schema}")


@pytest.mark.asyncio
async def test_two_step_generation_returns_variants_with_index(temp_db, monkeypatch):
    import app.agent.steps.audit_gaps as ag
    from app.db.repositories import paper_repo

    async def _always_invariant(canonical_text, variant, threshold):
        return True
    monkeypatch.setattr(ag, "_variant_is_invariant", _always_invariant)

    db = temp_db()
    task, gap = _seed_gap(db)
    llm = IntentLLM([_intent()])
    specs = await ag.generate_english_adversarial_queries(db, llm, gap)

    assert len(specs) == 3
    assert specs[0].variant_index == 0
    assert specs[1].variant_index == 1
    assert specs[2].variant_index == 2
    assert all(s.family == "exact_gap" for s in specs)
    db.close()


@pytest.mark.asyncio
async def test_drifted_variant_is_regenerated_then_kept(temp_db, monkeypatch):
    import app.agent.steps.audit_gaps as ag

    calls = {"n": 0}

    async def _invariant_first_fails(canonical_text, variant, threshold):
        # First variant is drifted -> fail once, then regenerated variant passes.
        calls["n"] += 1
        return "variant 0" not in variant

    monkeypatch.setattr(ag, "_variant_is_invariant", _invariant_first_fails)

    db = temp_db()
    task, gap = _seed_gap(db)
    llm = IntentLLM([_intent()])
    specs = await ag.generate_english_adversarial_queries(db, llm, gap)

    # All 3 survive (drifted variant regenerated to "regenerated faithful paraphrase").
    assert len(specs) == 3
    assert "regenerated faithful paraphrase" in [s.query_text for s in specs]
    db.close()


@pytest.mark.asyncio
async def test_under_two_valid_variants_marks_query_generation_invalid(temp_db, monkeypatch):
    import app.agent.steps.audit_gaps as ag

    async def _always_drifted(canonical_text, variant, threshold):
        return False
    async def _regenerate_empty(llm, intent, bad_variant):
        return ""  # regenerate fails -> variant dropped
    monkeypatch.setattr(ag, "_variant_is_invariant", _always_drifted)
    monkeypatch.setattr(ag, "_regenerate_variant", _regenerate_empty)

    db = temp_db()
    task, gap = _seed_gap(db)
    llm = IntentLLM([_intent(n_variants=3)])
    specs = await ag.generate_english_adversarial_queries(db, llm, gap)

    # All variants drifted and could not be regenerated -> no valid family, so the
    # query set is EMPTY (never an invalid family masquerading as queries).
    assert len(specs) == 0
    db.close()


@pytest.mark.asyncio
async def test_invalid_family_name_is_skipped(temp_db, monkeypatch):
    import app.agent.steps.audit_gaps as ag

    async def _always_invariant(canonical_text, variant, threshold):
        return True
    monkeypatch.setattr(ag, "_variant_is_invariant", _always_invariant)

    db = temp_db()
    task, gap = _seed_gap(db)
    llm = IntentLLM([_intent(family="not_a_real_family")])
    specs = await ag.generate_english_adversarial_queries(db, llm, gap)

    assert specs == []
    db.close()


@pytest.mark.asyncio
async def test_synonym_family_prompt_demands_term_of_art_diversity(temp_db):
    """fd688ba6 RETRIEVAL MISS: known prior art (UAEval4RAG, Evidence Sufficiency
    Benchmark, SURE-RAG) was never recalled because the queries used only the gap's
    own wording while the literature names the same concepts differently. The
    synonym-family prompt must instruct each variant to anchor on a DIFFERENT
    standard term of art, with the renaming pattern spelled out."""
    import app.agent.steps.audit_gaps as ag

    captured = {}

    class CapturingLLM:
        async def chat_json(self, messages, schema):
            captured["system"] = messages[0]["content"]
            from app.agent.steps.audit_gaps import GapQueryGenList
            return GapQueryGenList(intents=[])

    db = temp_db()
    task, gap = _seed_gap(db)
    await ag.generate_english_adversarial_queries(db, CapturingLLM(), gap)

    system = captured["system"]
    normalized = " ".join(system.split())
    # The synonym family must enumerate its distinct terms of art FIRST into the
    # structured `standard_terms` field, then anchor exactly one variant per term —
    # the explicit enumeration is what keeps variants from concentrating on one term.
    assert "standard_terms" in normalized
    assert "never two variants on the same term" in normalized
    assert "terms of art" in normalized.lower()
    # The fd688ba6 renaming pattern must be spelled out so the LLM enumerates
    # the actual literature vocabulary instead of echoing the gap's wording.
    for term in ("abstention", "unanswerability", "selective prediction", "sufficiency"):
        assert term in normalized, f"terminology example '{term}' missing from prompt"
    db.close()


def test_intent_schema_carries_standard_terms():
    """The structured term enumeration must survive as a schema field — it is
    what forces term-first thinking during generation."""
    from app.agent.steps.audit_gaps import IntentWithVariantsSchema

    intent = IntentWithVariantsSchema(
        family="synonym", problem="ppp", mechanism="mmm", intervention="iii",
        evaluation_setting="eee", task_scope="ttt",
        standard_terms=["abstention", "unanswerability", "evidence sufficiency"],
        variants=["a query", "b query", "c query"])
    assert intent.standard_terms == ["abstention", "unanswerability", "evidence sufficiency"]


def test_cover_standard_terms_appends_queries_for_uncovered_terms():
    from app.agent.steps.audit_gaps import IntentWithVariantsSchema, _cover_standard_terms

    intent = IntentWithVariantsSchema(
        family="synonym", problem="ppp", mechanism="mmm", intervention="iii",
        evaluation_setting="eee", task_scope="LLM code generation self-correction",
        standard_terms=["abstention", "unanswerability", "evidence sufficiency"],
        variants=["Evaluating abstention in RAG", "RAG refusal under missing evidence"])
    out = _cover_standard_terms(intent, ["Evaluating abstention in RAG",
                                         "RAG refusal under missing evidence"])
    # Covered term unchanged; uncovered terms each get a term-anchored query.
    assert len(out) == 4
    assert any("unanswerability" in q for q in out[2:])
    assert any("evidence sufficiency" in q for q in out[2:])
    assert all(q.endswith("LLM code generation self-correction") for q in out[2:])


def test_cover_standard_terms_noop_when_covered_or_empty():
    from app.agent.steps.audit_gaps import IntentWithVariantsSchema, _cover_standard_terms

    intent = IntentWithVariantsSchema(
        family="synonym", problem="ppp", mechanism="mmm", intervention="iii",
        evaluation_setting="eee", task_scope="ttt",
        standard_terms=["abstention"],
        variants=["Evaluating abstention in RAG", "abstention and refusal metrics"])
    variants = ["Evaluating abstention in RAG", "abstention and refusal metrics"]
    assert _cover_standard_terms(intent, variants) == variants

    intent.standard_terms = []
    assert _cover_standard_terms(intent, variants) == variants


@pytest.mark.asyncio
async def test_synonym_uncovered_terms_get_constructed_queries(temp_db, monkeypatch):
    """fd688ba6 replays showed the LLM enumerates standard terms but drifts most
    variants back to the gap's own wording — enumerated-but-unused terms are
    retrieval blind spots. The generator must deterministically append a
    term-anchored query for every uncovered enumerated term."""
    import app.agent.steps.audit_gaps as ag

    async def _always_invariant(canonical_text, variant, threshold):
        return True
    monkeypatch.setattr(ag, "_variant_is_invariant", _always_invariant)

    db = temp_db()
    task, gap = _seed_gap(db)
    intent = ag.IntentWithVariantsSchema(
        family="synonym",
        problem="self-correction is evaluated on sparse test suites",
        mechanism="sparse tests cannot distinguish accidental from robust correctness",
        intervention="sparse-to-dense test evaluation",
        evaluation_setting="HumanEval-style code benchmarks",
        task_scope="LLM code generation self-correction",
        standard_terms=["abstention", "unanswerability", "selective prediction"],
        variants=["query about abstention behaviour",
                  "query about abstention metrics",
                  "generic refusal accuracy query"],
    )
    llm = IntentLLM([intent])
    specs = await ag.generate_english_adversarial_queries(db, llm, gap)

    texts = [s.query_text for s in specs]
    assert any("unanswerability" in t for t in texts), \
        "uncovered enumerated term must get a constructed query"
    assert any("selective prediction" in t for t in texts)
    # Covered term ("abstention") gets no duplicate construction.
    assert len([t for t in texts if "abstention" in t]) == 2
    db.close()
