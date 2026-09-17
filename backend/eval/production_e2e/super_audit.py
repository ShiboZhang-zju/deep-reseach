"""production_e2e_v1 — Independent Evaluation Procedure (design points 3 & 4).

Terminology note (frozen): this is an *independent evaluation procedure*, NOT an
independent retrieval corpus — OpenAlex / Semantic Scholar / arXiv overlap with
the production data sources. What makes it independent enough for v1:
  - a DIFFERENT prompt template (wider query families: exact / paraphrase /
    adjacent-domain / survey-style),
  - a fresh multi-source retrieval pass,
  - human prior-art adjudication of the machine-found candidates.
The production Gap Audit must NOT be re-run and called independent validation:
it would share query family, retrieval sources, model preference and retrieval
blind spots with the system under test — both could miss the same killer paper.
Citation snowballing is deferred until the pilot shows the miss rate.

Blind review (design point 4): the human review sheet exposes ONLY
research topic / claim / candidate prior art. System identity (A/B/C),
target_type, topic stratum, V2 internal scores and production audit verdicts
are hidden behind random submission_ids; mapping.json restores identities
only after review. Order is shuffled.

Usage:
    cd backend
    python -m eval.production_e2e.super_audit --run-id pe2e_v1_audit \
        --systems ../eval_results/pe2e_v1_direct ../eval_results/pe2e_v1_retellm \
                  ../eval_results/pe2e_v1_fullv2
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from eval.common import CallStats, EvalRun
from eval.config import DEFAULT_RESULTS_DIR, build_run_config

BENCHMARK = "production_e2e"
MODE = "super_audit"
CANDIDATES_PER_TARGET = 12
PER_QUERY_PER_SOURCE = 5

SUPER_AUDIT_SYSTEM_PROMPT = (
    "You are an adversarial literature auditor. Given a research claim, generate "
    "diverse search queries to find papers that may ALREADY have implemented or "
    "covered it. You are deliberately broader than a novelty checker."
)


class AuditQueries(BaseModel):
    queries: list[str] = Field(min_length=4, max_length=8)


SUPER_AUDIT_USER_PROMPT = """Research topic: {topic}

Claim to adversarially audit (does prior art already cover this?):
\"\"\"{claim}\"\"\"

Generate 4-8 search queries covering ALL of these angles:
1. exact implementations of the same mechanism for the same purpose;
2. close method neighbours (same technique, different framing);
3. adjacent domains where this mechanism may have been published first;
4. survey / benchmark style queries that would list the relevant prior work.

Queries must be standalone search strings (no context needed). Do not repeat
the same wording across queries."""


def _iter_targets(run_dirs: list[Path]):
    """Yield (system, topic_id, topic, claim, target_type) audit targets.

    - every proposed idea from every system (target_type=idea)
    - every surviving gap of full_v2 (target_type=gap) — the direct object of
      the false-open-gap metric.
    """
    for run_dir in run_dirs:
        config_path = run_dir / "config.json"
        system = "unknown"
        if config_path.exists():
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            system = str(cfg.get("system") or "unknown")
        pred_path = run_dir / "predictions.jsonl"
        if pred_path.exists():
            for line in pred_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("parse_status") not in (None, "ok"):
                    continue
                if rec.get("protocol_flag"):
                    continue  # protocol violations are not audited outcomes
                if rec.get("decision") == "propose_idea" and rec.get("idea"):
                    idea = rec["idea"]
                    claim = idea.get("research_question") or idea.get("title") or ""
                    yield system, rec["topic_id"], rec.get("topic", ""), str(claim), "idea"
        gaps_path = run_dir / "gaps_export.jsonl"
        if gaps_path.exists():
            for line in gaps_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                for gap in rec.get("gaps") or []:
                    if gap.get("status") == "surviving" and gap.get("claimed_delta"):
                        yield "full_v2", rec["topic_id"], "", str(gap["claimed_delta"]), "gap"


async def gen_queries(llm, topic: str, claim: str, stats: CallStats) -> AuditQueries:
    from eval.common import chat_json

    return await chat_json(
        llm,
        [
            {"role": "system", "content": SUPER_AUDIT_SYSTEM_PROMPT},
            {"role": "user", "content": SUPER_AUDIT_USER_PROMPT.format(
                topic=topic or "(unspecified)", claim=claim)},
        ],
        AuditQueries,
        temperature=0.0,   # audit-side determinism; NOT the generation temperature
        stats=stats,
    )


# Entries that are demonstrably not research papers, so showing them to a
# reviewer as "prior art to check" is simply wrong. Deliberately narrow: it must
# never drop a paper that could plausibly be a killer paper, because a dropped
# candidate is invisible and turns into a false gap.
_NON_PAPER_TITLE = re.compile(
    r'^(proceedings of|front matter|table of contents|author index|'
    r'subject index|erratum|corrigendum|retraction|'
    r'conference program|workshop program|keynote abstract|'
    r'editorial board|call for papers)\b', re.I)


def _seed_openalex_ids(topic_id: str, claim: str, limit: int = 4,
                       prior_candidates: list[dict] | None = None) -> list[str]:
    """Pick citation-snowball seeds: papers the audited system itself retrieved.

    Seeds are the papers the audited topic actually drew on, resolved to
    OpenAlex ids. Ranking by raw citation count was measured to be actively
    harmful: the corpus's most-cited paper is "R: A Language and Environment for
    Statistical Computing" (353k citations), and restricting ITS citers to a
    topic term returned phylogenetics and ecology papers.

    Two resolution paths, because neither is sufficient alone:

    * `prior_candidates` — the local-corpus hits already recorded for this
      submission. This is the most faithful seed set: it is literally the
      literature the audited system was reading. Only usable on a re-run that
      already has a first pass.
    * title/keyword match against the topic's own papers — the fallback for a
      cold run, where no prior candidate list exists yet.

    Note `topic_id` here is the EVAL sample id (pe2e-014), not a
    `research_tasks.id` UUID; audit artifacts carry no link to the task table,
    which is why seeds cannot be looked up through `task_papers`.

    Forward citations (who cites it) are the useful direction for killing a
    claim, since a killer paper must postdate the work it challenges.
    """
    from sqlalchemy import text
    from app.db.session import SessionLocal

    terms = _snowball_terms(claim)
    db = SessionLocal()
    try:
        ids: list[str] = []

        # Path 1: titles already surfaced by the local corpus for this submission.
        #
        # Ordered by CLAIM OVERLAP, not citation count. Measured why: the
        # local-corpus candidate list is itself ranked by citations, so its top
        # entries are broad surveys ("A review of uncertainty quantification in
        # deep learning", 2,989 citers). A survey's citers are the whole field,
        # and the claim's specific terms (`entropy-adaptive`, `diffusion-lm`)
        # appear in ZERO of their titles. A paper that is precisely about the
        # claim is what makes the topic restriction bite — with it,
        # `title.search:catastrophic` returned 3 directly relevant works.
        titles = [((c.get("title") or "").strip())
                  for c in (prior_candidates or [])
                  if c.get("source") == "local_corpus" and c.get("title")]
        titles = [t for t in titles if t]
        if titles:
            claim_terms = set(_snowball_terms(claim, 20))
            ranked = sorted(
                titles,
                key=lambda t: -len(claim_terms & set(_snowball_terms(t, 20))),
            )
            ranked = ranked[:40]
            params = {f"t{i}": " ".join(t.lower().split())
                      for i, t in enumerate(ranked)}
            ph = ", ".join(f":t{i}" for i in range(len(ranked)))
            rows = db.execute(text(f"""
                SELECT openalex_id FROM papers
                WHERE LOWER(TRIM(title)) IN ({ph})
                  AND openalex_id IS NOT NULL AND openalex_id != ''
                ORDER BY citation_count DESC LIMIT :lim
            """), {**params, "lim": limit}).fetchall()
            ids = [r[0] for r in rows]
        if ids:
            # Dedupe while preserving order: title matching can hit several rows
            # for the same work, and repeating a seed multiplies the same query.
            seen: set[str] = set()
            uniq = []
            for i in ids:
                if i not in seen:
                    seen.add(i)
                    uniq.append(i)
            return uniq

        # Path 2 (cold run): the topic's papers whose titles share a claim term.
        if terms:
            like = " OR ".join(f"LOWER(p.title) LIKE :t{i}" for i in range(len(terms)))
            params = {f"t{i}": f"%{t}%" for i, t in enumerate(terms)}
            params["lim"] = limit
            rows = db.execute(text(f"""
                SELECT p.openalex_id FROM papers p
                JOIN task_papers tp ON tp.paper_id = p.id
                WHERE p.openalex_id IS NOT NULL AND p.openalex_id != ''
                  AND ({like})
                ORDER BY p.citation_count DESC LIMIT :lim
            """), params).fetchall()
            ids = [r[0] for r in rows]
            if ids:
                return ids

        # Last resort: any paper whose title mentions a claim term at all.
        if terms:
            for t in terms[:3]:
                row = db.execute(text("""
                    SELECT openalex_id FROM papers
                    WHERE openalex_id IS NOT NULL AND openalex_id != ''
                      AND LOWER(title) LIKE :t
                    ORDER BY citation_count DESC LIMIT :lim
                """), {"t": f"%{t}%", "lim": limit}).fetchall()
                ids.extend(r[0] for r in row)
                if len(ids) >= limit:
                    break
        return ids[:limit]
    finally:
        db.close()


def _snowball_terms(text: str, max_terms: int = 6) -> list[str]:
    from app.paper_sources.citation_snowball import _topic_terms
    return _topic_terms(text, max_terms)


async def search_candidates(queries: list[str], claim: str = "",
                            topic_id: str = "",
                            prior_candidates: list[dict] | None = None) -> list[dict]:
    """Multi-source retrieval with per-source failure isolation."""
    from app.paper_sources.arxiv import ArxivSource
    from app.paper_sources.citation_snowball import CitationSnowballSource
    from app.paper_sources.local_corpus import LocalCorpusSource
    from app.paper_sources.openalex import OpenAlexSource
    from app.paper_sources.semantic_scholar import SemanticScholarSource

    # `local_corpus` searches papers we already downloaded and indexed, including
    # 5k full-text chunks. It is the one source that keeps the audit multi-source
    # when the public APIs are down (on 2026-09-15 S2 and arXiv were fully
    # rate-limited, leaving OpenAlex alone), and it can surface prior art that
    # production itself retrieved -- the most likely place for a killer paper to
    # hide, and invisible to a public-API-only audit.
    #
    # `citation_snowball` attacks the opposite weakness: keyword search returns
    # whatever shares a few words with the claim (39% of the 2026-09-16
    # candidates shared NO content word with it). Snowball constrains by a real
    # citation edge first, then by topic, so its hits are later work that
    # actually builds on the audited topic.
    sources = [("openalex", OpenAlexSource()),
               ("local_corpus", LocalCorpusSource()),
               ("semantic_scholar", SemanticScholarSource()),
               ("arxiv", ArxivSource())]

    # citation_snowball is implemented and working, but is OFF by default because
    # it cannot yet do what it was added for. Measured across four tuning
    # attempts on the real records: a single narrow claim term returned 0 hits
    # (the seeds are broad surveys whose citers never use the claim's exact
    # compounds in a title), a single generic term returned 12 irrelevant ones
    # ("gradients" produced MARL policy-gradient papers for an orthogonal-LoRA
    # claim), and 2-term conjunctions returned 0-1.
    #
    # The blocker is the seed set, not the query. The audit records carry no
    # `task_id` link to `research_tasks`, so seeds cannot be taken from the
    # papers production actually read for this topic; they fall back to
    # title-matching, which surfaces broad surveys ("Multi-task Learning Using
    # Uncertainty to Weigh Losses", "Counterfactual Multi-Agent Policy
    # Gradients"). A broad survey's citers are its entire field, so within that
    # set no claim term can discriminate — the citation edge guarantees
    # "cites the seed" and nothing about "asks the same question".
    #
    # Left wired but disabled so the next run can enable it once records carry
    # task_id. Enabled it would only add latency and API load for no recall.
    if args.enable_snowball:
        seeds = _seed_openalex_ids(topic_id, claim,
                                   prior_candidates=prior_candidates)
        if seeds:
            sources.insert(1, ("citation_snowball",
                               CitationSnowballSource(seed_ids=seeds, topic=claim)))
            print(f"    [super_audit] snowball seeds: {len(seeds)}")
    seen_titles: set[str] = set()
    # Per-source buckets, then round-robin. Concatenating per source and
    # truncating at CANDIDATES_PER_TARGET made the first healthy source eat the
    # entire quota, collapsing "multi-source" into one source — observed
    # 2026-09-15 when 12/12 candidates came from OpenAlex. Multi-source
    # independence is a property of the CAP, so the cap must be filled evenly.
    per_source: dict[str, list[dict]] = {name: [] for name, _ in sources}
    for source_name, source in sources:
        for query in queries:
            try:
                papers = await source.search(query, limit=PER_QUERY_PER_SOURCE)
            except Exception as exc:  # source-level isolation: 429s are expected
                print(f"    [super_audit] {source_name} failed on "
                      f"{query[:40]!r}: {type(exc).__name__}: {exc}")
                continue
            for paper in papers:
                title = (getattr(paper, "title", None) or "").strip()
                if not title:
                    continue
                key = " ".join(title.lower().split())
                if key in seen_titles:
                    continue
                seen_titles.add(key)
                raw = getattr(paper, "raw_data", None) or {}
                per_source[source_name].append({
                    "title": title,
                    "year": getattr(paper, "year", None),
                    "venue": getattr(paper, "venue", None) or None,
                    "url": getattr(paper, "url", None) or None,
                    "source": source_name,
                    "query": query,
                    # Our own papers.id when the hit came from the local corpus.
                    # Without it a candidate cannot be joined back to `papers`,
                    # so full-text backfill has no way to find the PDF to parse.
                    "local_paper_id": raw.get("local_paper_id"),
                    # For snowball hits: which of our papers it cites. This is
                    # the strongest relevance signal the sheet can show — the
                    # candidate demonstrably builds on the audited topic, which
                    # no keyword match can establish.
                    "cites_seed": raw.get("cites_seed"),
                    # local_corpus can quote the sentence that matched; that is
                    # what lets a reviewer adjudicate FULL/PARTIAL/NONE instead of
                    # guessing from a title. Empty for the public sources.
                    "match_type": raw.get("match_type"),
                    "snippet": raw.get("snippet"),
                    "section": raw.get("section"),
                })
    # Allocate the cap per SOURCE rather than filling it first-come-first-served.
    #
    # Plain round-robin over source order silently starved the later sources:
    # openalex and arxiv each returned 6+ hits, so the first two laps filled all
    # 12 slots and citation_snowball contributed NOTHING even though it had
    # returned 6 directly relevant works (measured: "Regularizing Deep Multi-Task
    # Networks using Orthogonal Gradients" for an orthogonal-gradient claim).
    # A source's position in the list must not decide whether it is heard.
    #
    # Leftover slots are redistributed because some sources find nothing (S2 is
    # rate-limited most of the time), and unused quota should not be wasted.
    candidates: list[dict] = []
    per_source_quota = max(1, CANDIDATES_PER_TARGET // len(sources))
    taken = {name: 0 for name, _ in sources}

    for source_name, _ in sources:
        for cand in per_source[source_name]:
            if len(candidates) >= CANDIDATES_PER_TARGET:
                break
            if taken[source_name] >= per_source_quota:
                break
            taken[source_name] += 1
            candidates.append(cand)

    # Fill remaining slots from any source with leftovers, round-robin so no
    # single source dominates the spill.
    if len(candidates) < CANDIDATES_PER_TARGET:
        leftover = {name: per_source[name][taken[name]:] for name, _ in sources}
        cursor = 0
        while len(candidates) < CANDIDATES_PER_TARGET:
            progressed = False
            for source_name, _ in sources:
                bucket = leftover[source_name]
                if cursor < len(bucket):
                    candidates.append(bucket[cursor])
                    progressed = True
                    if len(candidates) >= CANDIDATES_PER_TARGET:
                        break
            if not progressed:
                break
            cursor += 1
    return candidates


def filter_candidates_by_relevance(candidates: list[dict], claim: str) -> list[dict]:
    """Drop only CLEARLY non-academic junk, never "off-topic but plausibly related".

    Two attempts at a vocabulary-overlap gate were measured against the real
    2026-09-16 records and both were rejected:

    * requiring >=2 shared content words dropped 630/895 candidates (70%) and
      left 20 of 91 targets with NOTHING, because a claim is an experimental
      design description ("minimal experiment measuring gradient norms to
      isolate the optimization state") while a title is a topic phrase
      ("Gradient-Based Importance Smoothing for Dynamic Rank Allocation").
      Their vocabularies legitimately differ, so low overlap does not mean
      irrelevant. That gate discarded highly relevant papers.
    * requiring >=1 dropped 367/895 (41%) and still removed relevant work.
      A gate that fires on 41-100% of all samples is not filtering noise.

    The deeper reason to abandon the approach: lexical overlap cannot adjudicate
    topical relevance here, and a wrong drop is worse than a wrong keep. A
    spurious candidate costs the reviewer one "NONE"; a dropped one is
    invisible, and if it was the actual killer paper the audit reports a false
    gap — precisely the error the audit exists to detect. Recall must win.

    What remains is a narrow, defensible filter for entries that are not
    research papers at all (proceedings front-matter, errata, indices). Topical
    judgement is left to the human reviewer and, where available, citation
    signal.
    """
    out = []
    for cand in candidates:
        title = (cand.get("title") or "").strip()
        if not title:
            continue                      # nothing to show a reviewer anyway
        if _NON_PAPER_TITLE.search(title):
            continue
        out.append(cand)
    return out


def blind_sheet_header() -> list[str]:
    """The sheet's front matter, including the candidate-relevance disclosure.

    Kept as ONE function because it previously existed as two copies — the
    generator's and the sheet rebuilder's — and the disclosure was added to only
    one, so a `--rebuild-sheet` silently produced a sheet WITHOUT the limitation
    notice. The reviewer's only warning about candidate noise must not depend on
    which code path wrote the file.
    """
    return [
        "# Super Audit — BLIND human review sheet",
        "",
        "Review each submission WITHOUT knowing which system produced it.",
        "Fill every field; identities are restored automatically afterwards.",
        "",
        "## Known limitation: candidate lists are NOT relevance-filtered",
        "",
        "Candidates come from keyword search over public APIs plus a local corpus "
        "search that DOES apply a relevance gate. Measured on this run's records, "
        "39% of the keyword-sourced candidates share NO content word with the "
        "claim they are meant to test (a claim about MoE visual salience drew "
        "\"Obesity and Cancer\" and \"IoT Network Security Threat Detection\").",
        "",
        "Two consequences for your verdicts:",
        "",
        "1. A candidate whose TITLE is clearly off-topic is noise, not evidence. "
        "Mark it NONE, and if a submission's list is mostly such candidates say "
        "so in `notes` — an all-noise list means the audit FAILED TO FIND prior "
        "art, which is a different fact from `the prior art does not exist`.",
        "2. Do NOT treat a list of irrelevant candidates as evidence IN FAVOUR of "
        "a submission's novelty. Importing that noise into the verdict is the "
        "single largest way this evaluation can be wrong.",
        "",
        "An automatic relevance filter was attempted and rejected: at the "
        "thresholds tested it discarded 41-70% of candidates and removed "
        "relevant work (e.g. \"Gradient-Based Importance Smoothing for Dynamic "
        "Rank Allocation\" for a claim about measuring gradient norms). Recall "
        "won because a wrongly dropped candidate is invisible and fabricates a "
        "false gap, whereas a spurious one costs one NONE verdict.",
        "",
        "Each submission states its own candidate-list quality, so you can weigh "
        "a `false-open: no` verdict against how good the search actually was.",
        "",
        "",
    ]


def blind_review_section(submission_id: str, topic: str, claim: str,
                         candidates: list[dict], query_failure: str | None = None) -> str:
    """Blind sheet: NO system identity, NO target_type, NO internal scores."""
    local_n = sum(1 for c in candidates if c.get("source") == "local_corpus")
    snippet_n = sum(1 for c in candidates if c.get("snippet"))
    lines = [
        f"## Submission {submission_id}",
        "",
        f"Research topic: {topic or '(unspecified)'}",
        "",
        f"Claim under audit: {claim}",
        "",
        "For each candidate below, mark whether it already covers the claim:",
        "FULL (implements the same mechanism for the same purpose) / "
        "PARTIAL (overlapping but not the same) / NONE.",
        "",
        # Surface how trustworthy the candidate LIST is, per submission. The
        # reviewer cannot see the retrieval process, so without this they have no
        # way to distinguish "searched well, found nothing" from "searched
        # badly". local_corpus hits passed a relevance gate (quote-verified);
        # keyword hits did not, so a list made only of the latter deserves less
        # weight when concluding that no prior art exists.
        f"Candidate list quality: {len(candidates)} candidates "
        f"({local_n} relevance-gated, {snippet_n} with a quoted passage). "
        f"Remaining are unfiltered keyword matches — judge their titles "
        f"critically.",
        "",
    ]
    if query_failure:
        # A target we could not even query is NOT evidence of novelty. Say so
        # explicitly, otherwise an empty candidate list reads as "we searched
        # and prior art genuinely does not exist".
        lines.append(
            f"- (RETRIEVAL FAILED for this submission: {query_failure})")
        lines.append(
            "- Do NOT read the empty list as absence of prior art. Judge from "
            "your own knowledge and mark the verdict accordingly.")
        lines.append("")
    elif not candidates:
        lines.append("- (no candidates found — control sample: judge from your "
                     "own knowledge)")
    for idx, cand in enumerate(candidates, 1):
        # Do NOT fall back to `cand['source']` here. It leaked "openalex" /
        # "local_corpus" into the sheet, which breaks the blind protocol twice
        # over: it reveals the retrieval path, and since local-corpus hits carry
        # quoted passages the pairing lets a reviewer infer which system's
        # candidate set they are looking at.
        lines.append(
            f"- [ ] {idx}. {cand['title']} ({cand.get('year') or 'n.d.'}"
            f"{', ' + str(cand['venue']) if cand.get('venue') else ''}"
            f") — FULL / PARTIAL / NONE: ____")
        # Quote the matched passage when the local corpus supplied one. A title
        # alone often cannot settle whether prior art implements the SAME
        # mechanism for the SAME purpose; the sentence can.
        if cand.get("snippet"):
            snippet = " ".join(str(cand["snippet"]).split())
            where = cand.get("section") or cand.get("match_type") or "local corpus"
            lines.append(f"      - matched ({where}): \"{snippet[:300]}\"")
    lines += [
        "",
        "Overall: does prior art already cover this claim (false-open)? [ ] yes [ ] no",
        "Novelty of the claim vs the nearest prior art (1-5): ____",
        "Feasibility of testing this claim as a research direction (1-5): ____",
        "Credible as a research idea/direction? [ ] yes [ ] no",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


async def _run(args) -> None:
    from app.llm.factory import get_llm

    run_dirs = [Path(p) for p in args.systems]
    for run_dir in run_dirs:
        if not (run_dir / "predictions.jsonl").exists():
            raise FileNotFoundError(f"{run_dir} has no predictions.jsonl")

    results_dir = Path(args.output_dir) if args.output_dir else DEFAULT_RESULTS_DIR
    run = EvalRun(results_dir, run_id=args.run_id, run_prefix="pe2e_audit")
    cfg = build_run_config(
        benchmark=BENCHMARK, task="super_audit", mode=MODE, split="topics_v1",
        sample_count=len(run_dirs), seed=None, limit=None,
        extra={
            "systems": [str(p.resolve()) for p in run_dirs],
            "query_generation_temperature": 0.0,
            "candidates_per_target": CANDIDATES_PER_TARGET,
            "independence": "independent evaluation procedure (NOT an independent "
                            "retrieval corpus: sources overlap with production); "
                            "different prompt template + fresh multi-source "
                            "retrieval + human prior-art adjudication; production "
                            "query family NOT reused",
            "citation_snowball": "deferred until pilot shows the miss rate",
            "blind_review": {
                "shuffle_seed": args.shuffle_seed,
                "hidden_fields": ["system", "target_type", "stratum",
                                  "internal scores", "production audit verdicts"],
                "protocol_flagged_samples": "excluded from audit",
            },
        },
    )
    run.write_config(cfg, overwrite=True)

    llm = get_llm()
    targets = list(_iter_targets(run_dirs))
    # Blind protocol: shuffle order, hide identities behind submission_ids.
    random.Random(args.shuffle_seed).shuffle(targets)
    submissions = []
    for system, topic_id, topic, claim, target_type in targets:
        submissions.append({
            "submission_id": uuid.uuid4().hex[:12],
            "system": system,
            "topic_id": topic_id,
            "target_type": target_type,
            "claim": claim,
            "_topic": topic,
        })
    candidates_path = run.dir / "candidate_killer_papers.jsonl"
    template_path = run.dir / "human_verdicts.jsonl"
    review_md = blind_sheet_header()

    # ---- resume: re-run only the targets whose retrieval failed ------------
    # The first pass (2026-09-16) left 21/91 targets with `query_failure` set and
    # therefore zero candidates. Their blind verdicts are meaningless and would
    # silently dilute the false-open-gap rate, but re-running all 91 to repair 21
    # wastes hours. So: keep every successful record verbatim, re-run only the
    # failures, and rebuild the review artifacts from the union.
    #
    # Identity is preserved deliberately. submission_id is what ties the blind
    # sheet to submission_mapping.json and to any verdict already keyed to it, so
    # a re-run must reuse the ORIGINAL id rather than minting a fresh uuid. The
    # two runs' random ids differ, so records are matched on a natural key.
    #
    # That key MUST include the claim. (system, topic_id, target_type) is NOT
    # unique: one topic produces several ideas, so keying on the triple collapsed
    # them onto a single record and reported "1 kept, 90 to re-run" instead of
    # "70 kept, 21 to re-run" — silently re-running 69 good targets.
    # Prior candidates by target, available whether or not this is a resume: on a
    # cold run it is empty and the snowball falls back to title matching, on a
    # re-run it supplies the real literature the target drew on.
    prior_by_target: dict[tuple, list[dict]] = {}
    for _rec in _load_prior_records(candidates_path):
        prior_by_target[_target_key(_rec)] = _rec.get("candidate_papers") or []

    kept: list[dict] = []
    if args.resume:
        prior = _load_prior_records(candidates_path)
        if not prior:
            raise SystemExit(
                f"[super_audit] --resume needs an existing {candidates_path}")
        prior_by_key = {}
        for rec in prior:
            prior_by_key.setdefault(_target_key(rec), []).append(rec)
        to_rerun = []
        for sub in submissions:
            bucket = prior_by_key.get(_target_key(sub)) or []
            # Prefer a successful record for this exact target; otherwise reuse a
            # failed one's identity so its verdict stays addressable.
            rec = next((r for r in bucket if not r.get("query_failure")), None)
            if rec is not None:
                kept.append(rec)               # retrieval succeeded, keep as-is
                continue
            if bucket:
                sub["submission_id"] = bucket[0]["submission_id"]  # keep identity
            to_rerun.append(sub)
        print(f"[super_audit] --resume: {len(kept)} kept, {len(to_rerun)} to re-run "
              f"(of {len(submissions)} targets)")
        submissions = to_rerun
        # Do NOT unlink here. Deleting up front and then dying (or being killed)
        # destroys the previous results with nothing written in their place —
        # which is exactly what happened on 2026-09-16. Instead the rewrite is
        # atomic-per-file at the end, from `kept` + the new records.
    if not submissions:
        print("[super_audit] nothing to re-run")

    print(f"[super_audit] running {len(submissions)} target(s) "
          f"({sum(1 for s in submissions if s['target_type'] == 'idea')} ideas, "
          f"{sum(1 for s in submissions if s['target_type'] == 'gap')} surviving gaps), "
          f"shuffle seed={args.shuffle_seed}")

    # Under --resume the existing file already holds `kept`; new records are
    # APPENDED so an interruption leaves the previous results intact instead of
    # having to be rebuilt from a deleted file.
    fresh: list[dict] = []
    for idx, sub in enumerate(submissions):
        stats = CallStats()
        print(f"[{idx+1}/{len(submissions)}] {sub['submission_id']}")
        failure = None
        try:
            queries = await gen_queries(llm, sub["_topic"], sub["claim"], stats)
            # Pass the submission's previously-recorded local-corpus hits so the
            # snowball seeds on the literature this target actually drew on.
            candidates = await search_candidates(
                queries.queries, claim=sub["claim"], topic_id=sub["topic_id"],
                prior_candidates=prior_by_target.get(_target_key(sub)))
            # Gate AFTER merging so it applies to every source uniformly. The
            # public APIs rank by their own relevance model, which happily
            # returns a cancer-epidemiology review for a claim about MoE
            # interpretability; without this the sheet is 39% noise and a
            # reviewer's "NONE" on that noise reads as "no prior art exists".
            before = len(candidates)
            candidates = filter_candidates_by_relevance(candidates, sub["claim"])
            if before != len(candidates):
                print(f"    relevance: {before} -> {len(candidates)} candidates")
        except Exception as exc:  # per-target isolation
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            candidates = []
            queries = []
            # Record why, so a target that produced no candidates is
            # distinguishable from one where the search genuinely found
            # nothing. Auditing a claim we could not even query is NOT
            # evidence of novelty.
            failure = f"{type(exc).__name__}: {exc}"[:500]
        # NOTE: never construct AuditQueries() on the failure path. Its
        # min_length=4 constraint exists to validate LLM output; feeding it an
        # empty list raises ValidationError INSIDE the except block, which
        # escapes the per-target isolation and kills the whole run (observed
        # 2026-09-15: 3 of 91 targets done, then the process died outright).
        # Store the PLAIN list of query strings. This used to be
        # `sub["queries"] = queries` where `queries` held an AuditQueries model,
        # so json.dumps fell through to repr() and wrote the Python literal
        # "queries=['...']" — a string json.loads cannot read back. Every
        # downstream consumer of the recorded queries was silently broken.
        if isinstance(queries, list):
            sub["queries"] = queries                    # failure path: already []
        else:
            sub["queries"] = list(queries.queries)      # success path: pydantic model
        sub["query_failure"] = failure
        sub["candidate_papers"] = candidates
        sub["llm"] = stats.as_dict()
        fresh.append(sub)
        with candidates_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(sub, ensure_ascii=False, default=str) + "\n")
        review_md.append(blind_review_section(
            sub["submission_id"], sub["_topic"], sub["claim"], candidates, failure))
        # Fill-in template: reviewer edits the nulls, keyed by submission_id only.
        with template_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "submission_id": sub["submission_id"],
                "false_open": None,
                "novelty": None,
                "feasibility": None,
                "credible": None,
                "notes": "",
            }, ensure_ascii=False) + "\n")

    # `candidates_path` now holds kept + fresh (kept was never removed and fresh
    # was appended). Only the REVIEW ARTIFACTS are rebuilt from scratch, since a
    # resumed run starts with the header and must present every target again.
    if kept:
        for rec in kept:
            review_md.append(blind_review_section(
                rec["submission_id"], rec.get("_topic") or rec.get("topic_id") or "",
                rec.get("claim") or "", rec.get("candidate_papers") or [],
                rec.get("query_failure")))
        with template_path.open("a", encoding="utf-8") as fh:
            for rec in kept:
                fh.write(json.dumps({
                    "submission_id": rec["submission_id"],
                    "false_open": None,
                    "novelty": None,
                    "feasibility": None,
                    "credible": None,
                    "notes": "",
                }, ensure_ascii=False) + "\n")
        print(f"[super_audit] folded {len(kept)} previously-successful record(s) "
              f"back into the review artifacts")

    # Identity mapping: DO NOT share with the reviewer before review is done.
    all_records = _load_prior_records(candidates_path)
    mapping = {
        "shuffle_seed": args.shuffle_seed,
        "resumed": bool(args.resume),
        "submissions": [
            {k: rec.get(k) for k in ("submission_id", "system", "topic_id",
                                     "target_type", "claim")}
            for rec in all_records
        ],
    }
    (run.dir / "submission_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    (run.dir / "human_review_blind.md").write_text("\n".join(review_md), encoding="utf-8")

    total = len(all_records)
    failed = sum(1 for rec in all_records if rec.get("query_failure"))
    print(f"[super_audit] run dir: {run.dir}")
    print(f"[super_audit] audit targets on record: {total} "
          f"({failed} with retrieval failure)")
    print("[super_audit] next: fill human_verdicts.jsonl from human_review_blind.md "
          "(keyed by submission_id), then run evaluate.py --audit-dir")


def _target_key(rec: dict) -> tuple:
    """Natural key identifying an audit target across runs.

    MUST include the claim: one topic yields several ideas, so keying on
    (system, topic_id, target_type) alone maps them all onto a single entry.
    Using that triple made a resume report "1 kept, 90 to re-run" instead of
    "70 kept, 21 to re-run", silently re-running 69 targets that had already
    succeeded.
    """
    return (rec.get("system"), rec.get("topic_id"),
            rec.get("target_type"), (rec.get("claim") or "").strip())


def _load_prior_records(path: Path) -> list[dict]:
    """Read existing audit records, tolerating a partially-written last line.

    A crash mid-append can leave a truncated JSON object; skipping it is right
    because such a target has no usable candidates anyway and will be re-run.
    """
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="production_e2e super audit")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--systems", nargs="+", required=True,
                        help="run dirs of the three systems (predictions.jsonl required)")
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", action="store_true",
                        help="keep targets whose retrieval succeeded and re-run only "
                             "those that failed; preserves submission_ids")
    parser.add_argument("--enable-snowball", action="store_true",
                        help="also query citation_snowball. Off by default: seeds "
                             "cannot be resolved to the papers production read "
                             "(audit records carry no task_id), so it currently "
                             "returns no relevant candidates.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
