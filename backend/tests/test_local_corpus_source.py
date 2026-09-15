"""LocalCorpusSource regression tests.

Pins the retrieval behaviour the super audit depends on. The source is easy to
get subtly wrong -- two earlier implementations of it were unusable and the
tests below encode exactly why:

1. A plain `OR` match ordered by citation_count let one incidental word admit a
   paper, so three different queries all returned the same mega-cited surveys.
2. Admitting papers freely from the full-text tier had the same effect for a
   different reason: long survey papers own the most chunks, so free matching
   over 5k chunks favoured length over topicality.

Both are caught by `test_different_queries_return_different_papers`, which is
the property that actually matters for auditing.
"""

import asyncio

import pytest

from app.paper_sources.local_corpus import LocalCorpusSource, _terms


class TestTermExtraction:
    def test_drops_stopwords_and_short_tokens(self):
        terms = _terms("The use of a model for the evaluation of systems")
        assert "the" not in terms
        assert "of" not in terms
        assert "model" not in terms, "domain-generic words are stopworded"
        assert "evaluation" in terms

    def test_deduplicates_preserving_order(self):
        terms = _terms("diffusion sampling diffusion efficiency sampling")
        assert terms == ["diffusion", "sampling", "efficiency"]

    def test_caps_term_count(self):
        long_query = " ".join(f"distinctterm{i}" for i in range(30))
        assert len(_terms(long_query, max_terms=8)) == 8

    def test_pure_stopword_query_yields_nothing(self):
        assert _terms("the a of and or") == []


class TestDegenerateQueries:
    """A query with no usable terms must not scan the corpus."""

    @pytest.mark.parametrize("query", ["", "   ", "a the of", "x", "12 34"])
    def test_returns_empty(self, query):
        hits = asyncio.run(LocalCorpusSource().search(query, limit=5))
        assert hits == []


@pytest.fixture
def corpus_db(tmp_path, monkeypatch):
    """Seed a throwaway corpus and point the source at it.

    `conftest.py` pins DATABASE_URL to an empty temp database, so a test that
    relied on the real corpus would silently skip forever (and a skip is
    indistinguishable from a pass in CI). Seeding our own rows keeps these
    assertions meaningful and independent of whatever the developer's local
    corpus happens to contain.
    """
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    db_path = tmp_path / "corpus.db"
    url = "sqlite:///" + str(db_path).replace("\\", "/")

    # Build the schema with Alembic, not Base.metadata.create_all: the models
    # have drifted ahead of create_all (e.g. `papers.is_oa` arrives via a later
    # migration), so create_all produces a table missing columns the ORM writes.
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")

    from app.db import session as session_mod
    from app.db.models import Paper, PaperChunk, EvidenceUnit

    engine = create_engine(url, connect_args={"check_same_thread": False})
    db = sessionmaker(bind=engine)()
    # LocalCorpusSource reads `SessionLocal`, which is bound to the engine built
    # at import time from settings.database_url. Setting DATABASE_URL via env
    # here would be too late, so rebind the module globals directly.
    monkeypatch.setattr(session_mod, "engine", engine)
    monkeypatch.setattr(session_mod, "SessionLocal", sessionmaker(bind=engine))
    try:
        papers = {
            "p_diff": Paper(id="p_diff", title="Accelerating Diffusion Sampling with Optimized Time Steps",
                            abstract="We reduce the number of diffusion sampling steps while "
                                     "preserving image quality via optimized time step schedules.",
                            year=2024, citation_count=31, is_oa=True),
            "p_rag": Paper(id="p_rag", title="Evidence Grounding for Retrieval Augmented Generation",
                           abstract="A retrieval augmented generation framework that verifies "
                                    "evidence grounding to suppress unsupported claims.",
                           year=2025, citation_count=7, is_oa=True),
            "p_noise": Paper(id="p_noise", title="A Survey of Everything: Diffusion, Retrieval and Beyond",
                             abstract="A broad survey touching diffusion sampling, retrieval "
                                      "augmented generation, evidence grounding and more.",
                             year=2023, citation_count=9000, is_oa=True),
            "p_other": Paper(id="p_other", title="Protein Folding With Graph Networks",
                             abstract="We predict protein structure using graph neural networks.",
                             year=2021, citation_count=500, is_oa=True),
        }
        db.add_all(papers.values())
        db.add_all([
            PaperChunk(paper_id="p_rag", chunk_index=0, section="conclusion", chunk_type="text",
                       text="Our evidence grounding check rejects claims that the retrieved "
                            "context does not support, improving faithfulness.",
                       has_pdf=True),
            PaperChunk(paper_id="p_other", chunk_index=0, section="method", chunk_type="text",
                       text="Graph networks are trained on residue distance maps.",
                       has_pdf=True),
        ])
        db.add(EvidenceUnit(task_id="t1", paper_id="p_rag", evidence_type="finding",
                            normalized_claim="evidence grounding reduces unsupported claims",
                            original_span="grounding verification lowered hallucination rate",
                            section="results"))
        db.commit()
    finally:
        db.close()
    return db_path


class TestRetrievalAgainstSeededCorpus:
    """Retrieval behaviour against a known corpus, so assertions can be exact."""

    def test_finds_the_relevant_paper(self, corpus_db):
        hits = asyncio.run(LocalCorpusSource().search(
            "evidence grounding retrieval augmented generation", limit=5))
        titles = [h.title for h in hits]
        assert any("Evidence Grounding" in t for t in titles), titles

    def test_returns_match_metadata_for_the_review_sheet(self, corpus_db):
        hits = asyncio.run(LocalCorpusSource().search(
            "evidence grounding retrieval augmented generation", limit=5))
        assert hits, "seeded corpus should match"
        for h in hits:
            assert h.source == "local_corpus"
            assert h.raw_data["match_type"] in ("full_text", "evidence", "metadata")

    def test_respects_limit(self, corpus_db):
        hits = asyncio.run(LocalCorpusSource().search(
            "diffusion sampling evidence retrieval grounding", limit=2))
        assert len(hits) <= 2

    def test_unrelated_paper_is_not_admitted(self, corpus_db):
        """Precision: a single incidental word must not pull in an off-topic paper."""
        hits = asyncio.run(LocalCorpusSource().search(
            "evidence grounding retrieval augmented generation", limit=10))
        assert not any(h.title.startswith("Protein Folding") for h in hits), \
            "off-topic paper admitted — the match threshold is too loose"

    def test_different_queries_return_different_papers(self, corpus_db):
        """The core anti-regression: retrieval must be query-sensitive.

        Both earlier broken implementations failed here — every query came back
        with the same handful of high-citation surveys.

        `p_noise` is deliberately seeded as a paper that mentions everything, so
        it is EXPECTED to appear for both queries; the property under test is
        that it does not displace the genuinely relevant paper from the top spot.
        (Ranking it above the topical papers is exactly the failure mode that
        made the first two implementations unusable.)
        """
        src = LocalCorpusSource()
        a = asyncio.run(src.search("diffusion sampling acceleration time steps", limit=5))
        b = asyncio.run(src.search(
            "evidence grounding retrieval augmented generation", limit=5))
        assert a and b, "both queries should match the seeded corpus"
        assert "Diffusion Sampling" in a[0].title, \
            f"omnibus survey outranked the topical paper: {[h.title for h in a]}"
        assert "Evidence Grounding" in b[0].title, \
            f"omnibus survey outranked the topical paper: {[h.title for h in b]}"
        # And they must not return an identical result set.
        assert [h.title for h in a] != [h.title for h in b]

    def test_full_text_tier_upgrades_an_already_selected_paper(self, corpus_db):
        """The chunk tier may enrich a hit but must never admit a new paper."""
        hits = asyncio.run(LocalCorpusSource().search(
            "evidence grounding retrieval augmented generation unsupported claims",
            limit=5))
        rag = [h for h in hits if "Evidence Grounding" in h.title]
        assert rag, "expected the grounding paper"
        assert rag[0].raw_data["match_type"] == "full_text"
        assert rag[0].raw_data.get("snippet"), "a full-text hit should carry a snippet"
        # The off-topic paper has a chunk, but no query-term overlap worth
        # admitting it.
        assert not any(h.title.startswith("Protein Folding") for h in hits)
