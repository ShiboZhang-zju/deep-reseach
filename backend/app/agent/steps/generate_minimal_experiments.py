"""Step: Turn gate-approved interventions into minimal decisive experiments."""

import json
import logging
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from app.agent.state import ResearchState
from app.config import settings
from app.db.models import GapCandidate, GapPhenomenonPlan, InterventionCandidate
from app.db.repositories import paper_repo

logger = logging.getLogger(__name__)

_MIN_EXPERIMENT_SYSTEM = """You design a minimal decisive experiment for a research intervention.
The experiment must test the stated mechanism before proposing a full paper evaluation.
Use only supplied paper IDs as related work. Specify a falsifiable hypothesis, controls, metrics,
success condition, and failure condition.

For dataset and baselines, be concrete enough to actually run:
- Prefer a specific real, well-known public dataset or baseline (e.g. from the supplied
  evidence, or widely-used ones such as Alpaca, FLAN, OpenOrca, GLUE, ImageNet).
- Never fabricate a specific-looking name that does not exist; if you are not certain,
  name a real widely-known candidate and mark it as "candidate (verify availability)".
- Do not fall back to a vague placeholder like "a standard instruction-tuning corpus";
  a vague placeholder cannot be executed.
- SCENARIO MATCH: the dataset (or data-generation procedure) MUST exercise every
  supplied scenario atom. If no public dataset contains those elements, construct a
  small synthetic dataset that instantiates them instead of reaching for an unrelated
  benchmark. State which atoms are covered in `risks`.
- The output is a research contract, not prose. Fill model_spec, dataset_provenance,
  oracle, statistical_analysis, resource_budget, and scenario_atoms explicitly.
- An LLM may assist labeling, but it MUST NOT be the sole correctness oracle.
- Keep model_spec consistent with the gap's parameter/compute scope.
- TITLE: name the concrete mechanism under test (e.g. "Diff-Risk Classification for
  Self-Correction Filtering"), NOT the gap topic. Never prefix with "Minimal
  Experiment:" or similar — sibling ideas must be distinguishable by title alone.
- IDEA_METHOD: 3-6 sentences describing the method from the IDEA's perspective —
  hypothesis -> method -> expected outcome. Do NOT restate the intervention's
  engineering description; drop implementation details a reader of the idea does
  not need to judge the contribution.
- IDEA_CONTRIBUTION: 1-2 sentences stating what THIS experiment would establish
  as a novel contribution if it succeeds — a claim about the phenomenon or
  mechanism (what we would know afterwards that we do not know now). Do NOT
  restate the intervention's measurable outcome; that only names the metric,
  not the knowledge gain.
- STATISTICS: ratio-style paired metrics (pass rate, regression rate, accuracy,
  success rate, proportions) must be analyzed with McNemar's test, Wilcoxon
  signed-rank, a permutation test, or bootstrap confidence intervals — NEVER a
  plain (paired) t-test on proportions. Continuous metrics (latency, edit
  distance) may use t-tests.
- MODEL SCOPE: honor the gap's compute/parameter scope verbatim — if the gap
  targets small language models (<7B / SLM), the model_spec must stay within
  that bound.
"""

# Bumped when the experiment-generation RULES change so a resumed task
# invalidates previously stamped generate_minimal_experiments PhaseRuns and
# re-runs the phase instead of replaying its old (idea_ids=[]) output.
# v1 (2026-08-27): rejection-aware feedback retry (SCENARIO_MISMATCH fixes).
# v2 (2026-08-27): hypothesis-cluster gate (P2-A) — interventions of the same
# gap that test the SAME experimental hypothesis are clustered by one LLM
# call; each cluster yields ONE idea + ONE experiment, with the variant
# mechanisms folded into the experiment baselines as ablation arms instead of
# surfacing as "independent" ideas (task 23ec8f20: two tier-A interventions
# produced two executable_candidates testing the identical hypothesis
# "apply only logical fixes vs apply all").
# v3 (2026-08-27): idea-level novelty quick-check (P2-C) — retrieval against
# the cluster's hypothesis + primary mechanism; METHOD_ALREADY_PUBLISHED demotion.
# v4 (2026-08-27): idea metadata differentiation (P2-B) — titles name the concrete
# mechanism (prefix-free), idea_method replaces verbatim intervention copying,
# motivations quote evidence claims verbatim, and ideas from a previous
# generation round are soft-deleted (superseded) when the phase re-runs.
# v5 (2026-08-27): plan-consistency gates — STATISTICAL_TEST_MISMATCH (ratio
# metrics analyzed by plain t-test with no non-parametric alternative) and
# generalized MODEL_SCOPE_CONFLICT (numeric cap or SLM scope from the gap's
# target_setting vs the plan's model_spec).
# v6 (2026-08-27): fixes from the v5 E2E run — "SLMs" plural now matches the
# scope keyword (target_setting "using SLMs" previously escaped the 7B cap
# while plans used Llama-3-8B); the generation prompt now states the
# statistics rule (McNemar/Wilcoxon/permutation/bootstrap for ratio metrics)
# and model-scope rule up front so the feedback retry is not the model's
# first exposure to them (v5 run: both retries rewrote t-test as t-test).
# v7 (2026-08-28, task d6f64087): rejection feedback is now actionable — the
# failure strings carry the violating parameter count vs. the gap's cap and
# the offending oracle text (a bare "MODEL_SCOPE_CONFLICT" code name left the
# retry rewriting blind and failing twice on the same example); the retry
# budget is 3 attempts; and clusters that still fail every attempt are
# persisted as research_direction_only ideas (full rejected plan in the
# trace) instead of vanishing — a 90-minute run previously ended with the
# user seeing only "abstained" while two near-complete plans evaporated.
# v8 (2026-08-28, task db7e1adc): boundary-study exemption — gaps whose delta
# is itself a capacity range ("from <10B ... to >70B") may compare models on
# both sides; range bounds are no longer extracted as compute caps. P2-a:
# idea_contribution (LLM-authored knowledge-gain claim) replaces the verbatim
# copy of intervention.measurable_outcome in expected_contribution. P2-b:
# related_paper_ids are re-selected from the novelty check's mechanism-relevant
# papers instead of inheriting the gap audit's full neighbour set.
EXPERIMENT_GENERATION_POLICY_VERSION = "experiment-consistency-v8"

# P0-3: rejection-aware rewrite attempts per cluster. Two was too few once the
# feedback became specific — one attempt burns on absorbing the feedback, the
# model earns a second real rewrite. Bounded to keep the phase cheap.
_PLAN_GENERATION_ATTEMPTS = 3

# P2-b: upper bound on mechanism-relevant papers re-selected onto an idea —
# related work metadata stays readable instead of inheriting the full audit
# neighbour set.
_IDEA_RELATED_PAPER_CAP = 8

# Initial calibration point; revisit against historical task annotations (P3)
# before changing — cluster merges REDUCE idea counts, so false merges are the
# risk to control. Degradation on low confidence is deliberately conservative.
_CLUSTER_MERGE_MIN_CONFIDENCE = 0.7

_MIN_EXPERIMENT_USER = """Gap:
- Observed problem: {observed_problem}
- Remaining delta: {remaining_delta}
- Hypothesis: {gap_hypothesis}
- Expected scenario atoms (all must be exercised): {scenario_atoms}

Phenomenon validation plan:
- Phenomenon: {phenomenon}
- Mechanism under test: {mechanism_under_test}
- Comparator: {comparator}
- Oracle experiment: {oracle_experiment}
- Kill criterion: {kill_criterion}
- Measurement: {measurement}

Intervention (primary mechanism of this idea):
- Failure mechanism: {failure_mechanism}
- Proposed intervention: {proposed_intervention}
- Intermediate effect: {intermediate_effect}
- Measurable outcome: {measurable_outcome}
{variant_section}
Related paper IDs: {paper_ids}
Resource constraints: GPU={gpu_available}; max GPU hours={max_gpu_hours}; runtime minutes={max_runtime_minutes}
"""


class HypothesisClusterSchema(BaseModel):
    """One cluster = one falsifiable hypothesis shared by >=1 interventions."""
    hypothesis: str = Field(min_length=10, description="The single experimental hypothesis this cluster tests (what is measured, falsifiable)")
    primary_intervention_id: str = Field(description="ID of the intervention whose mechanism is most direct for testing the hypothesis")
    variant_intervention_ids: list[str] = Field(default_factory=list)
    differentiation_rationale: str = Field(min_length=10, description="Why the variants test the SAME hypothesis through DIFFERENT mechanisms")
    confidence: float = Field(ge=0.0, le=1.0)


class HypothesisClusterListSchema(BaseModel):
    clusters: list[HypothesisClusterSchema] = Field(default_factory=list)


class IdeaNoveltyQueriesSchema(BaseModel):
    """Adversarial queries for the idea-level novelty quick-check (P2-C)."""
    queries: list[str] = Field(min_length=1, description="Up to 3 self-contained English keyword queries")


class IdeaNoveltyVerdictSchema(BaseModel):
    """Verdict on whether the cluster's specific method is already implemented.

    already_implemented=true requires one listed paper to directly implement the
    SAME core mechanism (same technique for the same purpose); related-but-
    different methods (different technique, different purpose, or only a
    component of the proposal) must yield false.

    P2-b (2026-08-28): mechanism_relevant_paper_ids lists the examined papers
    genuinely relevant to the idea's mechanism — prior art worth citing or
    related work the experiment must compare against. These re-select the
    idea's related_paper_ids, which previously inherited the gap audit's full
    neighbor set (gap-relevant, not necessarily mechanism-relevant).
    """
    already_implemented: bool
    evidence_paper_id: str | None = None
    mechanism_relevant_paper_ids: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=5)


_HYPOTHESIS_CLUSTER_SYSTEM = """You group research interventions by the experimental HYPOTHESIS they test, not by their mechanisms.
Two interventions belong to the same cluster when ONE decisive experiment could confirm or refute both — they test the
same falsifiable claim about "what is measured and what effect is expected", even if the way they achieve it ("how")
differs. Keep interventions in separate clusters when they would need different experimental setups, different
dependent variables, or different falsification conditions.
Rules:
- every supplied intervention ID must appear exactly once (as primary or as a variant);
- a single intervention is a valid cluster (empty variant list);
- do NOT merge merely because interventions address the same research gap or share motivation; merge ONLY when their
  hypotheses state the same testable claim;
- confidence is your certainty that the merged interventions really test the same hypothesis (1.0 = certain, below 0.7
  means you are guessing).
Output in English."""

_HYPOTHESIS_CLUSTER_USER = """Phenomenon under test: {phenomenon}
Kill criterion: {kill_criterion}

Interventions:
{interventions}

Group these interventions into hypothesis clusters."""


def _format_variant_section(variants: list[InterventionCandidate]) -> str:
    """Render variant mechanisms for the experiment prompt: the cluster gate
    folds them into the SAME experiment as ablation arms instead of letting
    each surface as an "independent" idea."""
    if not variants:
        return ""
    lines = ["", "Mechanism variants (test the SAME hypothesis via different mechanisms): fold EACH into the",
             "experiment baselines/controls as its own ablation arm, so one decisive experiment adjudicates",
             "the whole cluster; report per-arm results separately:"]
    for v in variants:
        mech = (v.failure_mechanism or "(unspecified)")[:220]
        prop = (v.proposed_intervention or "(unspecified)")[:320]
        lines.append(f"- Variant [{v.id[:8]}] failure mechanism: {mech} | intervention: {prop}")
    return "\n".join(lines) + "\n"


async def _cluster_interventions_by_hypothesis(llm, phenomenon, gap_interventions):
    """Group one gap's passed interventions by the hypothesis they test.

    Returns a list of cluster dicts {primary, variants, hypothesis, rationale,
    confidence}, or None when the gate must degrade (LLM output unparseable,
    intervention IDs not covered exactly once, empty cluster list) — the caller
    then falls back to one-cluster-per-intervention (the pre-v2 behaviour) and
    logs a trace. Degradation keeps the gate fail-open for INFRASTRUCTURE
    failures while the merge decision itself stays an explicit LLM judgement
    with a confidence floor.
    """
    if len(gap_interventions) <= 1:
        return [{"primary": gap_interventions[0], "variants": [],
                 "hypothesis": "", "rationale": "", "confidence": 1.0}]

    lines = []
    for iv in gap_interventions:
        lines.append(
            f"- Intervention ID: {iv.id}\n"
            f"  Failure mechanism: {iv.failure_mechanism or '(unspecified)'}\n"
            f"  Proposed intervention: {iv.proposed_intervention or '(unspecified)'}\n"
            f"  Measurable outcome: {iv.measurable_outcome or '(unspecified)'}"
        )
    try:
        result = await llm.chat_json([
            {"role": "system", "content": _HYPOTHESIS_CLUSTER_SYSTEM},
            {"role": "user", "content": _HYPOTHESIS_CLUSTER_USER.format(
                phenomenon=(phenomenon.phenomenon if phenomenon else "(not specified)"),
                kill_criterion=(phenomenon.kill_criterion if phenomenon else "(not specified)"),
                interventions="\n\n".join(lines),
            )},
        ], HypothesisClusterListSchema)
    except Exception as exc:
        logger.warning("hypothesis-cluster gate LLM call failed (%s); degrading to per-intervention clusters", exc)
        return None
    if not result or not result.clusters:
        return None

    id_set = {iv.id for iv in gap_interventions}
    by_id = {iv.id: iv for iv in gap_interventions}
    seen: set[str] = set()
    for c in result.clusters:
        if c.primary_intervention_id not in id_set or c.primary_intervention_id in seen:
            return None
        seen.add(c.primary_intervention_id)
        for vid in c.variant_intervention_ids:
            if vid not in id_set or vid in seen:
                return None
            seen.add(vid)
    if seen != id_set:
        return None

    clusters = []
    for c in result.clusters:
        if c.variant_intervention_ids and c.confidence < _CLUSTER_MERGE_MIN_CONFIDENCE:
            # Low-confidence merge: split back into singletons rather than
            # risking a false merge (merges reduce idea output; the
            # conservative direction is no-merge).
            clusters.append({"primary": by_id[c.primary_intervention_id], "variants": [],
                             "hypothesis": "", "rationale": "low-confidence merge split",
                             "confidence": c.confidence})
            for vid in c.variant_intervention_ids:
                clusters.append({"primary": by_id[vid], "variants": [],
                                 "hypothesis": "", "rationale": "low-confidence merge split",
                                 "confidence": c.confidence})
        else:
            clusters.append({"primary": by_id[c.primary_intervention_id],
                             "variants": [by_id[vid] for vid in c.variant_intervention_ids],
                             "hypothesis": c.hypothesis,
                             "rationale": c.differentiation_rationale,
                             "confidence": c.confidence})
    return clusters


_IDEA_NOVELTY_QUERIES_SYSTEM = """You generate adversarial literature-search queries to check whether a proposed research
method has ALREADY been directly implemented in prior work. Output up to 3 short English queries:
1) exact-method: names the concrete technique (e.g. "diff-based regression risk classifier for code correction filtering");
2) method-neighbour: close variants of the technique under other terminology;
3) adjacent-domain: casts the technique into a neighbouring field that would plausibly have published it first
   (e.g. correction filtering -> automated program repair patch filtering; evaluation gating -> test selection).
Queries are self-contained keyword phrases: no quotes, no boolean operators, no field prefixes."""


_IDEA_NOVELTY_VERDICT_SYSTEM = """You judge whether a proposed research method has ALREADY been directly implemented by prior work.
already_implemented=true ONLY when one listed paper directly implements the SAME core mechanism — same technique
used for the same purpose. Related-but-different work (different technique, different purpose, only one component of
the proposal, or a survey) must yield false. If true, set evidence_paper_id to the implementing paper's id exactly
as listed. Be conservative: uncertainty means false.
Additionally, set mechanism_relevant_paper_ids to the ids (exactly as listed) of papers that are genuinely relevant
to the idea's mechanism — prior art a reader must know, or related work the experiment should compare against.
Exclude papers that only share the broad topic without touching the mechanism. Leave it empty when none qualify."""


async def _check_idea_novelty(db, state, llm, task_id, gap, cluster_hypothesis,
                               intervention, existing_paper_ids):
    """P2-C idea-level novelty quick-check.

    The gap audit validated the GAP's novelty (the mechanism is missing from the
    literature), not the specific method's (someone may already have built this
    exact technique). This check retrieves against the cluster's hypothesis +
    primary mechanism and asks whether prior art directly implements it.

    Returns (verdict, matched_paper_id, mechanism_relevant_paper_ids) where
    verdict is one of:
    - "disabled":           check turned off by configuration
    - "already_implemented": demote the idea (METHOD_ALREADY_PUBLISHED) — the
                            LLM identified a listed paper implementing the method
    - "passed":             no direct implementation found among retrieved papers
    - "passed_no_results":  retrieval succeeded but returned nothing (novel by
                            absence of evidence, not proof — recorded as passed)
    - "degraded":           infrastructure failure (LLM/search). NEVER demotes:
                            an external-source outage is not evidence about the
                            idea (distinct from a successful search that found
                            prior art).

    P2-b: mechanism_relevant_paper_ids is non-empty only on a successful
    verdict — the examined papers the verdict LLM judged genuinely relevant to
    the idea's mechanism. The caller re-selects the idea's related_paper_ids
    from them; every degraded path returns [] so the idea keeps its
    gap-audit-neighbour fallback instead of losing related work entirely.
    """
    if not settings.idea_novelty_check_enabled:
        return "disabled", None, []
    from app.agent.steps.generate_queries import SearchQueryExecution
    from app.agent.steps.search_papers import search_and_save_papers
    from app.db.models import Paper, SearchQueryPaper
    from app.db.repositories.search_query_repo import save_search_query

    queries: list[str] = []
    try:
        spec = await llm.chat_json([
            {"role": "system", "content": _IDEA_NOVELTY_QUERIES_SYSTEM},
            {"role": "user", "content": json.dumps({
                "hypothesis": cluster_hypothesis or "",
                "primary_mechanism": intervention.proposed_intervention or "",
                "failure_mechanism": intervention.failure_mechanism or "",
            }, ensure_ascii=False)},
        ], IdeaNoveltyQueriesSchema)
        queries = [q.strip() for q in (spec.queries if spec else []) if q and q.strip()]
    except Exception as exc:
        logger.warning("Task %s: idea novelty query generation degraded: %s", task_id[:8], exc)
        paper_repo.save_trace(db, task_id, "idea_novelty_check", "decision", output_data={
            "gap_id": gap.id, "intervention_id": intervention.id,
            "verdict": "degraded", "stage": "query_generation", "error": str(exc)[:200],
        })
        return "degraded", None, []
    if not queries:
        paper_repo.save_trace(db, task_id, "idea_novelty_check", "decision", output_data={
            "gap_id": gap.id, "intervention_id": intervention.id,
            "verdict": "degraded", "stage": "query_generation", "error": "no queries returned",
        })
        return "degraded", None, []

    round_num = state.current_round
    executions = []
    for idx, text in enumerate(queries):
        record = save_search_query(
            db, task_id, text, f"idea_novelty_{idx + 1}", None, None, round_num,
            target_gap_id=gap.id, query_family=f"idea_novelty_{idx + 1}",
            search_policy_version=EXPERIMENT_GENERATION_POLICY_VERSION,
        )
        executions.append(SearchQueryExecution(
            query_id=record.id, query_text=text, intent=f"idea_novelty_{idx + 1}",
            target_question_id=None, expected_evidence_type=None, target_gap_id=gap.id,
        ))
    db.commit()
    try:
        await search_and_save_papers(db, state, executions, task_id, round_num)
    except Exception as exc:
        logger.warning("Task %s: idea novelty retrieval degraded: %s", task_id[:8], exc)
        paper_repo.save_trace(db, task_id, "idea_novelty_check", "decision", output_data={
            "gap_id": gap.id, "intervention_id": intervention.id, "queries": queries,
            "verdict": "degraded", "stage": "retrieval", "error": str(exc)[:200],
        })
        return "degraded", None, []

    # Top-k papers by rank across the novelty queries (dedup across queries).
    top_rows = (
        db.query(SearchQueryPaper, Paper)
        .join(Paper, SearchQueryPaper.paper_id == Paper.id)
        .filter(SearchQueryPaper.query_id.in_([e.query_id for e in executions]))
        .order_by(SearchQueryPaper.rank)
        .limit(settings.idea_novelty_check_top_k * 3)
        .all()
    )
    seen_ids: set[str] = set()
    top_papers: list[Paper] = []
    for _sqp, paper in top_rows:
        if paper.id in seen_ids:
            continue
        seen_ids.add(paper.id)
        top_papers.append(paper)
        if len(top_papers) >= settings.idea_novelty_check_top_k:
            break
    if not top_papers:
        paper_repo.save_trace(db, task_id, "idea_novelty_check", "decision", output_data={
            "gap_id": gap.id, "intervention_id": intervention.id, "queries": queries,
            "verdict": "passed_no_results",
        })
        return "passed_no_results", None, []

    paper_list = [{
        "id": p.id, "title": (p.title or "")[:150], "abstract": (p.abstract or "")[:500],
    } for p in top_papers]
    try:
        verdict = await llm.chat_json([
            {"role": "system", "content": _IDEA_NOVELTY_VERDICT_SYSTEM},
            {"role": "user", "content": json.dumps({
                "hypothesis": cluster_hypothesis or "",
                "proposed_method": intervention.proposed_intervention or "",
                "papers": paper_list,
            }, ensure_ascii=False)},
        ], IdeaNoveltyVerdictSchema)
    except Exception as exc:
        logger.warning("Task %s: idea novelty verdict degraded: %s", task_id[:8], exc)
        paper_repo.save_trace(db, task_id, "idea_novelty_check", "decision", output_data={
            "gap_id": gap.id, "intervention_id": intervention.id, "queries": queries,
            "verdict": "degraded", "stage": "verdict", "error": str(exc)[:200],
        })
        return "degraded", None, []
    if verdict is None:
        paper_repo.save_trace(db, task_id, "idea_novelty_check", "decision", output_data={
            "gap_id": gap.id, "intervention_id": intervention.id, "queries": queries,
            "verdict": "degraded", "stage": "verdict", "error": "no verdict returned",
        })
        return "degraded", None, []

    matched = None
    if verdict.already_implemented and verdict.evidence_paper_id and verdict.evidence_paper_id in seen_ids:
        matched = verdict.evidence_paper_id
    final_verdict = "already_implemented" if matched else "passed"
    # P2-b: papers the verdict LLM judged genuinely mechanism-relevant. Invalid
    # IDs (hallucinated) are dropped; a bounded cap keeps idea metadata small.
    relevant_ids = [pid for pid in verdict.mechanism_relevant_paper_ids
                    if pid in seen_ids][:_IDEA_RELATED_PAPER_CAP]
    paper_repo.save_trace(db, task_id, "idea_novelty_check", "decision", output_data={
        "gap_id": gap.id, "intervention_id": intervention.id, "queries": queries,
        "verdict": final_verdict, "matched_paper_id": matched,
        "rationale": (verdict.rationale or "")[:400],
        "checked_paper_ids": [p.id for p in top_papers],
        "mechanism_relevant_paper_ids": relevant_ids,
    })
    return final_verdict, matched, relevant_ids


class MinimalExperimentSchema(BaseModel):
    # P2-B: title names the concrete mechanism (no "Minimal Experiment:" prefix
    # — that prefix restates the gap topic and made sibling ideas read as
    # duplicates); idea_method is the idea-level method sketch (hypothesis ->
    # method -> expected outcome), distinct from the intervention's engineering
    # description which used to be copied verbatim into method_sketch.
    title: str = Field(min_length=5)
    idea_method: str = ""
    idea_contribution: str = ""
    summary: str = Field(min_length=10)
    hypothesis: str = Field(min_length=10)
    dataset: str
    baselines: str
    metrics: str
    model_spec: str = ""
    dataset_provenance: str = ""
    oracle: str = ""
    statistical_analysis: str = ""
    resource_budget: str = ""
    scenario_atoms: list[str] = Field(default_factory=list)
    controls: list[str] = Field(default_factory=list)
    steps: list[str] = Field(min_length=2)
    success_condition: str = Field(min_length=5)
    falsification_condition: str = Field(min_length=5)
    risks: str = ""


@dataclass
class MinimalExperimentResult:
    idea_ids: list[str]
    experiment_ids: list[str]
    direction_only_idea_ids: list[str] = field(default_factory=list)

    def to_phase_payload(self) -> dict:
        return {
            "idea_ids": self.idea_ids,
            "experiment_ids": self.experiment_ids,
            "direction_only_idea_ids": self.direction_only_idea_ids,
        }


def _derive_scenario_atoms(gap: GapCandidate, audit, phenomenon: GapPhenomenonPlan | None) -> list[str]:
    """Extract a small, deterministic set of scenario atoms from the gap claim."""
    text = " ".join(
        (getattr(gap, field, "") or "") for field in
        ("target_setting", "observed_problem", "missing_capability", "claimed_delta", "testable_hypothesis", "residual_gap")
    )
    if audit:
        text += " " + (getattr(audit, "remaining_delta", "") or "")
    if phenomenon:
        text += " " + (phenomenon.mechanism_under_test or "")
    lowered = text.lower()
    patterns = [
        ("python", ("python",)), ("java", ("java",)), ("javascript", ("javascript", "js")),
        ("cross-language", ("cross-language", "cross language", "multilingual")),
        ("false negative", ("false negative", "漏检", "漏报")),
        ("false positive", ("false positive", "误检", "误报")),
        ("logic error", ("logic error", "逻辑错误")),
        ("state change", ("state change", "state-change", "状态变化")),
        ("preference reversal", ("preference reversal", "偏好反转")),
        ("retrieval noise", ("retrieval noise", "检索噪声")),
        ("conflict resolution", ("conflict resolution", "冲突解决")),
        ("verifier", ("verifier", "验证器")),
    ]
    atoms = [canonical for canonical, variants in patterns if any(item in lowered for item in variants)]
    if phenomenon and phenomenon.measurement and not atoms:
        atoms.append(phenomenon.measurement.strip()[:80])
    return atoms[:6]


def _text_has_atom(text: str, atom: str) -> bool:
    normalized = re.sub(r"[-_]+", " ", atom.lower()).strip()
    haystack = re.sub(r"[-_]+", " ", text.lower())
    return normalized in haystack or any(token in haystack for token in normalized.split() if len(token) >= 5)


def _validate_experiment_plan(
    plan: MinimalExperimentSchema,
    *,
    phenomenon: GapPhenomenonPlan | None = None,
    expected_atoms: list[str] | None = None,
    gap_text: str = "",
) -> list[str]:
    """Return deterministic failures before exposing an Idea as executable."""
    failures: list[str] = []
    for field in (
        "dataset", "baselines", "metrics", "model_spec", "dataset_provenance",
        "oracle", "statistical_analysis", "resource_budget", "success_condition",
        "falsification_condition",
    ):
        if not getattr(plan, field, "").strip():
            failures.append(f"MISSING_{field.upper()}")
    if len(plan.steps) < 2:
        failures.append("INSUFFICIENT_STEPS")
    if not plan.controls:
        failures.append("MISSING_CONTROLS")
    if phenomenon is None:
        failures.append("MISSING_PHENOMENON_PLAN")
    else:
        for field in ("mechanism_under_test", "comparator", "oracle_experiment", "kill_criterion", "measurement"):
            if not (getattr(phenomenon, field, "") or "").strip():
                failures.append("INCOMPLETE_PHENOMENON_PLAN")
                break
    atoms = expected_atoms or []
    if atoms and not plan.scenario_atoms:
        failures.append("MISSING_SCENARIO_ATOMS")
    plan_text = " ".join([
        plan.dataset, plan.dataset_provenance, plan.model_spec, plan.oracle,
        plan.statistical_analysis, plan.risks, " ".join(plan.steps),
    ])
    for atom in atoms:
        if not _text_has_atom(plan_text, atom):
            failures.append(f"SCENARIO_MISMATCH:{atom}")
    failures.extend(_check_model_scope(gap_text, plan.model_spec))
    failures.extend(_check_statistical_method(plan))
    oracle_text = plan.oracle.lower()
    if "llm" in oracle_text and not any(term in oracle_text for term in ("execution", "test", "static", "formal", "human", "hidden")):
        failures.append(
            "LLM_ONLY_ORACLE: the oracle relies on an LLM's own judgement "
            f"(oracle = \"{(plan.oracle or '')[:120]}\") — ground it in "
            "execution outcomes, hidden test cases, formal verification or "
            "human labels instead")
    return list(dict.fromkeys(failures))


# Ratio-style metrics measured as paired binary outcomes (each sample
# passes/fails, each correction regresses or not) need McNemar / Wilcoxon /
# bootstrap / exact tests; a plain t-test on proportions is the documented
# misuse (E2E 2026-08-26 review: "regression rate" compared with paired t-test).
_RATIO_METRIC_TERMS = (
    "pass rate", "pass@", "accuracy", "error rate", "failure rate",
    "regression rate", "success rate", "proportion", "percentage of",
)
_NONPARAMETRIC_TERMS = (
    "mcnemar", "wilcoxon", "bootstrap", "permutation test", "exact test",
    "binomial", "sign test", "confidence interval via bootstrap",
)


def _check_statistical_method(plan: MinimalExperimentSchema) -> list[str]:
    """STATISTICAL_TEST_MISMATCH: ratio-style paired metrics analyzed with a
    plain t-test and no non-parametric / resampling alternative named."""
    stats_text = (plan.statistical_analysis or "").lower()
    metrics_text = (plan.metrics or "").lower()
    has_ttest = any(term in stats_text for term in ("t-test", "t test", "student's t", "paired t"))
    has_ratio_metric = any(term in metrics_text for term in _RATIO_METRIC_TERMS)
    has_nonparametric = any(term in stats_text for term in _NONPARAMETRIC_TERMS)
    if has_ttest and has_ratio_metric and not has_nonparametric:
        return ["STATISTICAL_TEST_MISMATCH"]
    return []


# P1-2 (task d6f64087): role nouns that can carry a size cap. A gap like
# "reward models (<3B)" bounds the RM, not the RLHF policy LLM — an RLHF
# experiment legitimately pairs a <=3B RM with a larger policy model. The
# scope check must bind each numeric parameter count in model_spec to the
# role it belongs to instead of flagging every large number in the text.
_SCOPE_ROLES = (
    "reward model", "reward models", "policy model", "policy models",
    "verifier", "judge", "critic", "ranker", "scorer", "detector",
    "generator", "student", "teacher", "agent", "assistant",
)
_SCOPE_ROLE_RES = tuple(
    (role, re.compile(rf"\b{re.escape(role)}s?\b"))
    for role in _SCOPE_ROLES
)


def _role_in_window(text: str, start: int, roles: tuple | None = None) -> str | None:
    """Return the closest role keyword appearing in the window [start-70, start)."""
    window = text[max(0, start - 70):start]
    best: str | None = None
    best_pos = -1
    for role, role_re in (roles or _SCOPE_ROLE_RES):
        m = role_re.search(window)
        if m and m.end() > best_pos:
            best = role
            best_pos = m.end()
    return best


def _check_model_scope(gap_text: str, model_spec: str) -> list[str]:
    """MODEL_SCOPE_CONFLICT (generalized): the gap bounds the model size — an
    explicit '<N b' style cap, or a 'small language model / SLM' scope — but the
    plan's model_spec names a strictly larger checkpoint. E2E 2026-08-26 review:
    scope '<7B' with experiments on Llama-3-8B. Conservative by design: only a
    numeric parameter count above the extracted cap (or the SLM default of 7B)
    triggers; vague wording never does.
    P0-3 (task d6f64087): the failure string now names the violating parameter
    count AND the cap so the rejection-aware retry can tell the model exactly
    what to fix — a bare code name left it rewriting blind and failing twice
    on the same "Llama-3-8B distilled" example.
    P1-2 (task d6f64087): role-aware binding. When the cap's surrounding text
    names a role ("reward models (<3B)"), only parameter counts bound to that
    same role in model_spec are checked; numbers attached to a DIFFERENT role
    (e.g. an 8B policy LLM in an RLHF setup) are exempt. No role anywhere
    keeps the old global behaviour (conservative).
    Boundary-study exemption (task db7e1adc): a gap whose delta IS a capacity
    boundary — "Extends robustness evaluation from <10B parameter models to
    >70B parameter models" — legitimately compares models on both sides of
    that boundary; the "<10B" there is a range bound, not a compute cap. Such
    from→to constructions are skipped during cap extraction (and disable the
    SLM keyword fallback, which would otherwise re-impose a 7B cap from the
    boundary's "small model" phrasing)."""
    text = (gap_text or "").lower()
    spec = (model_spec or "").lower()
    if not spec:
        return []
    cap: float | None = None
    cap_role: str | None = None
    boundary_study = False
    cap_re = re.compile(
        r"(?:<|<=|≤|under|below|smaller than|less than|up to)\s*(\d+(?:\.\d+)?)\s*b\b"
    )
    for m in cap_re.finditer(text):
        # A "<NB ... to >NB" continuation marks the match as one side of a
        # studied range, not a cap: "from <10B ... to >70B". The "to NB" must
        # directly continue the match — another size number in between means
        # the match is a real cap and the range belongs to later text.
        tail = text[m.end():m.end() + 70]
        r = re.search(
            r"\bto\s*(?:over|more than|above|at least|>+\s*)?\s*\d+(?:\.\d+)?\s*b\b",
            tail,
        )
        if r and not re.search(r"\b\d+(?:\.\d+)?\s*b\b", tail[:r.start()]):
            boundary_study = True
            continue
        cap = float(m.group(1))
        cap_role = _role_in_window(text, m.start())
        break
    if cap is None and not boundary_study:
        slm = re.search(r"\b(?:small language models?|small models?|slms?)\b", text)
        if slm:
            cap = 7.0
            cap_role = _role_in_window(text, slm.start())
    if cap is None:
        return []
    for num_m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*b\b", spec):
        num = float(num_m.group(1))
        if num <= cap:
            continue
        num_role = _role_in_window(spec, num_m.start())
        if cap_role and num_role and num_role != cap_role:
            # The oversized checkpoint belongs to a different role than the
            # capped one (e.g. policy LLM vs the <=3B reward model) — exempt.
            continue
        return [
            f"MODEL_SCOPE_CONFLICT: model_spec cites {num:g}B at \"{spec[max(0, num_m.start()-40):num_m.end() + 10].strip()}\""
            f" but the gap's scope caps {cap_role or 'models'} at {cap:g}B"
            f" — every model serving that role must stay at or below {cap:g}B parameters"
        ]
    return []


def _normalize_idea_title(title: str) -> str:
    """P2-B: strip the legacy "Minimal Experiment:" style prefixes even when the
    LLM still emits them despite the prompt constraint — the prefix restates the
    gap topic and makes sibling ideas read as duplicates."""
    text = (title or "").strip()
    for prefix in ("Minimal Experiment:", "Minimal experiment:", "minimal experiment:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
            break
    return text or "Untitled experiment"


def _load_gap_evidence_claims(db, gap_id: str, limit: int = 2) -> list[str]:
    """P2-B: quote the gap's supporting evidence claims verbatim so the idea's
    motivation cites the actual text (previously it only said "3 traceable
    evidence units", which a reader could not verify)."""
    from app.db.models import EvidenceUnit, GapEvidenceLink
    rows = (
        db.query(EvidenceUnit.normalized_claim)
        .join(GapEvidenceLink, GapEvidenceLink.evidence_id == EvidenceUnit.id)
        .filter(GapEvidenceLink.gap_id == gap_id)
        .order_by(GapEvidenceLink.relevance_score.desc().nullslast())
        .limit(limit)
        .all()
    )
    return [(r[0] or "").strip() for r in rows if (r[0] or "").strip()]


def _build_idea_motivation(gap: GapCandidate, intervention: InterventionCandidate,
                           evidence_claims: list[str] | None = None) -> str:
    """Compose a per-idea motivation that differs across interventions.

    Two interventions over the same gap used to share gap.observed_problem
    verbatim, surfacing as duplicate-looking ideas. Fold in the intervention's
    distinct failure mechanism so each idea states why *this* direction matters.
    E2E 2026-08-26: also lead with the intervention type, so two ideas diverge
    from the very first line instead of only in the trailing mechanism note.
    P2-B: quote the top evidence claims verbatim so the motivation is
    independently checkable against the evidence store.
    """
    observed = (gap.observed_problem or "").strip()
    mechanism = (intervention.failure_mechanism or "").strip()
    itype = (intervention.intervention_type or "").strip()
    if not observed:
        return mechanism or "(motivation not specified)"
    lead = f"{itype}: " if itype else ""
    parts = [f"{lead}{observed}"]
    if evidence_claims:
        quoted = "\n".join(f"- \"{c[:280]}\"" for c in evidence_claims[:2])
        parts.append(f"Supporting evidence (verbatim):\n{quoted}")
    if mechanism:
        parts.append(f"This direction specifically targets the failure mechanism: {mechanism}.")
    return "\n\n".join(parts)


async def generate_minimal_experiments(db, state: ResearchState, llm, task_id: str) -> MinimalExperimentResult:
    """Persist conditional ideas and small falsifiable experiments from passed interventions."""
    from app.db.models import ResearchContract

    contract = db.get(ResearchContract, state.contract_id) if state.contract_id else None
    if not contract:
        return MinimalExperimentResult([], [])

    interventions = db.query(InterventionCandidate).filter(
        InterventionCandidate.task_id == task_id,
        InterventionCandidate.contract_id == contract.id,
        InterventionCandidate.status == "passed",
    ).all()
    # P2-B: when the phase RE-RUNS (policy bump / resume), ideas from the
    # previous generation round are the old rules' output — soft-delete them so
    # the task surfaces exactly one coherent generation instead of new + stale
    # side by side (task 23ec8f20: v1 and v2/v3 ideas coexisted in the UI).
    # The API exposes history via include_superseded=true. If this run produces
    # zero new ideas, the honest state is "zero active ideas under the current
    # rules", which is exactly the allowed-zero-candidates semantics.
    if interventions:
        from app.db.models import ResearchIdea
        stale = db.query(ResearchIdea).filter(
            ResearchIdea.task_id == task_id,
            ResearchIdea.idea_status == "active",
        ).all()
        for stale_idea in stale:
            stale_idea.idea_status = "superseded"
        if stale:
            db.commit()
            paper_repo.save_trace(db, task_id, "supersede_stale_ideas", "decision", output_data={
                "superseded_idea_ids": [s.id for s in stale],
                "policy_version": EXPERIMENT_GENERATION_POLICY_VERSION,
            })
            logger.info("Task %s: superseded %d stale idea(s) from previous generation round",
                        task_id[:8], len(stale))
    idea_ids = []
    experiment_ids = []
    # P0-3 layer 2: rejected clusters persisted as research_direction_only
    # ideas — surfaced to the user, never counted as executable output.
    direction_only_idea_ids = []
    seen_titles: set[str] = set()

    # P2-A hypothesis-cluster gate: interventions of the SAME gap are grouped
    # by the experimental hypothesis they test BEFORE any experiment/idea is
    # generated. Each cluster yields one idea + one experiment; the variant
    # mechanisms ride along as ablation arms in the experiment baselines. This
    # is the structural fix for "two Ideas package the same contribution as
    # independent work" (task 23ec8f20: both tier-A interventions tested
    # 'apply only logical fixes vs apply all' and both surfaced as
    # executable_candidates). Degradation (LLM failure / ID mismatch) keeps
    # the pre-v2 one-idea-per-intervention behaviour and leaves a trace.
    interventions_by_gap: dict[str, list[InterventionCandidate]] = {}
    for intervention in interventions:
        interventions_by_gap.setdefault(intervention.gap_id, []).append(intervention)

    for gap_id, gap_interventions in interventions_by_gap.items():
        gap = db.get(GapCandidate, gap_id)
        if not gap:
            continue
        audit = gap_repo.list_gap_audits(db, gap.id)[-1] if gap_repo.list_gap_audits(db, gap.id) else None
        paper_ids = json.loads(audit.neighbor_paper_ids_json or "[]") if audit else []
        phenomenon = db.query(GapPhenomenonPlan).filter(
            GapPhenomenonPlan.gap_id == gap.id,
            GapPhenomenonPlan.task_id == task_id,
        ).order_by(GapPhenomenonPlan.created_at.desc()).first()
        expected_atoms = _derive_scenario_atoms(gap, audit, phenomenon)
        # target_setting carries the model/compute scope (e.g. "<7B", "SLM") that
        # MODEL_SCOPE_CONFLICT checks against the plan's model_spec.
        gap_text = " ".join([
            gap.target_setting or "", gap.observed_problem or "",
            gap.claimed_delta or "", gap.testable_hypothesis or "",
            gap.residual_gap or "", (audit.remaining_delta if audit else "") or "",
        ])

        clusters = await _cluster_interventions_by_hypothesis(llm, phenomenon, gap_interventions)
        paper_repo.save_trace(db, task_id, "hypothesis_cluster", "decision", output_data={
            "gap_id": gap_id,
            "clustered": clusters is not None,
            "clusters": [
                {
                    "hypothesis": c["hypothesis"],
                    "primary_intervention_id": c["primary"].id,
                    "variant_intervention_ids": [v.id for v in c["variants"]],
                    "rationale": c["rationale"],
                    "confidence": c["confidence"],
                }
                for c in (clusters or [])
            ] if clusters else
            [{"primary_intervention_id": iv.id, "variant_intervention_ids": []} for iv in gap_interventions],
        })
        if not clusters:
            clusters = [{"primary": iv, "variants": [], "hypothesis": "",
                         "rationale": "", "confidence": 1.0} for iv in gap_interventions]

        for cluster in clusters:
            intervention = cluster["primary"]
            variants = cluster["variants"]
            variant_section = _format_variant_section(variants)

            # Feedback retry (task 23ec8f20, 2026-08-27): both tier-A interventions were
            # rejected with SCENARIO_MISMATCH:verifier and NO retry — the gate is right to
            # demand the gap's scenario in the experiment text, but the model deserves one
            # rejection-aware rewrite (the atom matching is literal substring, so
            # "verification feedback" does NOT satisfy atom "verifier"). The retry feeds
            # the failure codes back verbatim; the gate itself is NOT relaxed.
            plan = None
            plan_failures: list[str] = []
            feedback = ""
            for _attempt in range(_PLAN_GENERATION_ATTEMPTS):
                user_content = _MIN_EXPERIMENT_USER.format(
                    observed_problem=gap.observed_problem or "(not specified)",
                    remaining_delta=(audit.remaining_delta if audit else gap.claimed_delta) or "(not specified)",
                    gap_hypothesis=gap.testable_hypothesis or "(not specified)",
                    scenario_atoms=expected_atoms or ["(not derived)"],
                    phenomenon=phenomenon.phenomenon if phenomenon else "(missing)",
                    mechanism_under_test=phenomenon.mechanism_under_test if phenomenon else "(missing)",
                    comparator=phenomenon.comparator if phenomenon else "(missing)",
                    oracle_experiment=phenomenon.oracle_experiment if phenomenon else "(missing)",
                    kill_criterion=phenomenon.kill_criterion if phenomenon else "(missing)",
                    measurement=phenomenon.measurement if phenomenon else "(missing)",
                    failure_mechanism=intervention.failure_mechanism,
                    proposed_intervention=intervention.proposed_intervention,
                    intermediate_effect=intervention.intermediate_effect,
                    measurable_outcome=intervention.measurable_outcome,
                    variant_section=variant_section,
                    paper_ids=paper_ids,
                    gpu_available=contract.gpu_available,
                    max_gpu_hours=contract.max_gpu_hours,
                    max_runtime_minutes=contract.max_runtime_minutes,
                )
                if feedback:
                    user_content += feedback
                plan = await llm.chat_json([
                    {"role": "system", "content": _MIN_EXPERIMENT_SYSTEM},
                    {"role": "user", "content": user_content},
                ], MinimalExperimentSchema)
                if plan is None:
                    plan_failures = ["NO_PLAN_GENERATED"]
                    break
                plan_failures = _validate_experiment_plan(
                    plan, phenomenon=phenomenon, expected_atoms=expected_atoms, gap_text=gap_text
                )
                if not plan_failures:
                    break
                feedback = (
                    "\n\nYour previous plan was REJECTED by the quality gate:\n- "
                    + "\n- ".join(plan_failures)
                    + "\n\nFix EVERY listed failure exactly as stated (codes name the offending "
                    "value and the bound). Rewrite the plan so that: (1) every scenario atom listed "
                    "above appears VERBATIM as a literal word in the dataset/model_spec/oracle/steps/"
                    "statistical_analysis/risks text (e.g. atom \"verifier\" requires the word "
                    "\"verifier\", not \"verification\"), (2) every required field is non-empty with "
                    ">=2 steps and explicit controls, and (3) the experiment genuinely exercises "
                    "the gap's scenario rather than merely naming it in scenario_atoms."
                )
            if plan_failures:
                # P0-3 layer 2 + P1-3 (task d6f64087): a cluster that burned its
                # retries no longer evaporates. The FULL rejected plan is traced
                # (previously a 100-char summary made post-hoc debugging
                # impossible — the role of the violating 8B checkpoint in the
                # rejected model_spec could not be verified), and a
                # research_direction_only idea is persisted so the user sees
                # the near-complete direction plus exactly which gate withheld
                # it, instead of a bare "abstained" terminal state.
                rejected_plan_full = {}
                if plan is not None:
                    rejected_plan_full = {
                        "title": plan.title,
                        "summary": plan.summary,
                        "dataset": plan.dataset,
                        "dataset_provenance": plan.dataset_provenance,
                        "model_spec": plan.model_spec,
                        "oracle": plan.oracle,
                        "metrics": plan.metrics,
                        "statistical_analysis": plan.statistical_analysis,
                        "resource_budget": plan.resource_budget,
                        "success_condition": plan.success_condition,
                        "falsification_condition": plan.falsification_condition,
                        "scenario_atoms": plan.scenario_atoms,
                        "controls": plan.controls,
                        "risks": plan.risks,
                        "steps": plan.steps,
                    }
                paper_repo.save_trace(db, task_id, "idea_quality_gate", "decision", output_data={
                    "gap_id": gap.id,
                    "intervention_id": intervention.id,
                    "variant_intervention_ids": [v.id for v in variants],
                    "status": "rejected",
                    "reason_codes": plan_failures,
                    "retry_attempted": bool(feedback),
                    "rejected_plan_summary": {
                        "dataset": (plan.dataset or "")[:100] if plan else "",
                        "model_spec": (plan.model_spec or "")[:100] if plan else "",
                        "oracle": (plan.oracle or "")[:100] if plan else "",
                        "steps": [s[:100] for s in (plan.steps or [])[:3]] if plan else [],
                    },
                    "rejected_plan_full": rejected_plan_full,
                })
                if plan is not None and contract is not None:
                    tier = getattr(intervention, "confidence_tier", "C") or "C"
                    fallback_title = _normalize_idea_title(
                        plan.title or (intervention.proposed_intervention or "")[:80])
                    if fallback_title in seen_titles:
                        mech = (intervention.failure_mechanism or "").strip()
                        fallback_title = f"{fallback_title} — {mech[:40]}" if mech else f"{fallback_title} (2)"
                    seen_titles.add(fallback_title)
                    motivation = _build_idea_motivation(
                        gap, intervention,
                        evidence_claims=_load_gap_evidence_claims(db, gap.id))
                    motivation += (
                        "\n\n[research_direction_only] The experiment plan for this "
                        "direction was drafted but rejected by the consistency gate "
                        "after all retries. Blocking reasons: "
                        + "; ".join(plan_failures[:6])
                        + ". The full draft plan is preserved in the "
                        "idea_quality_gate trace (rejected_plan_full)."
                    )
                    direction_idea = paper_repo.save_idea(db, task_id, {
                        "title": fallback_title,
                        "description": plan.summary or "(draft plan rejected by consistency gate)",
                        "motivation": motivation,
                        "method_sketch": (plan.idea_method or "").strip() or intervention.proposed_intervention,
                        "expected_contribution": (plan.idea_contribution or "").strip() or intervention.measurable_outcome,
                        "related_paper_ids_json": json.dumps(paper_ids, ensure_ascii=False),
                        "contract_id": contract.id,
                        "gap_id": gap.id,
                        "intervention_id": intervention.id,
                        "pipeline_version": state.pipeline_version,
                        "decision": "research_direction_only",
                        "score_status": "unscored",
                        "quality_reason_codes_json": json.dumps(
                            ["PLAN_REJECTED_BY_CONSISTENCY_GATE"] + plan_failures[:8],
                            ensure_ascii=False),
                        "confidence_tier": tier,
                    })
                    direction_only_idea_ids.append(direction_idea.id)
                    logger.info("Task %s: persisted rejected cluster as "
                                "research_direction_only idea %s",
                                task_id[:8], direction_idea.id[:8])
                logger.warning("Task %s: experiment plan rejected by quality gate: %s", task_id[:8], plan_failures)
                continue

            # De-duplicate titles: two interventions over the same gap can collapse
            # to an identical experiment title, which surfaces as duplicate ideas in
            # the UI. Append a short mechanism disambiguator when a title repeats.
            final_title = _normalize_idea_title(plan.title)
            if final_title in seen_titles:
                mech = (intervention.failure_mechanism or "").strip()
                suffix = mech[:40] if mech else "variant"
                final_title = f"{final_title} — {suffix}"
                while final_title in seen_titles:
                    final_title = f"{final_title} (2)"
            seen_titles.add(final_title)
            tier = getattr(intervention, "confidence_tier", "C") or "C"
            quality_reason_codes = []
            if tier != "A":
                quality_reason_codes.append("INTERVENTION_NOT_TIER_A")
                for gate_name in ("evidence_gate", "novelty_gate", "feasibility_gate"):
                    gate_value = getattr(intervention, gate_name, "UNKNOWN") or "UNKNOWN"
                    if gate_value != "PASS":
                        quality_reason_codes.append(f"{gate_name.upper()}_{gate_value}")
            if not phenomenon:
                quality_reason_codes.append("MISSING_PHENOMENON_PLAN")
            motivation = _build_idea_motivation(
                gap, intervention,
                evidence_claims=_load_gap_evidence_claims(db, gap.id))
            if variants:
                # Cluster annotation: the variant mechanisms live on in this idea's
                # experiment as ablation arms — surface that on the idea so the
                # user sees ONE idea with N-1 folded mechanisms, not N lookalikes.
                motivation += (
                    "\n\nIncludes " + str(len(variants)) + " mechanism variant(s) folded into the same"
                    " experiment as ablation arms (same hypothesis, different mechanism): "
                    + "; ".join(((v.failure_mechanism or "")[:80]) for v in variants)
                )
            idea = paper_repo.save_idea(db, task_id, {
                "title": final_title,
                "description": plan.summary,
                "motivation": motivation,
                # P2-B: prefer the LLM's idea-level method sketch; fall back to
                # the intervention text only when the model omitted the field.
                "method_sketch": (plan.idea_method or "").strip() or intervention.proposed_intervention,
                # P2-a (2026-08-28): the contribution states what the experiment
                # would establish (knowledge gain), distinct from the
                # intervention's measurable outcome (metric name) which used to
                # be copied here verbatim.
                "expected_contribution": (plan.idea_contribution or "").strip() or intervention.measurable_outcome,
                "related_paper_ids_json": json.dumps(paper_ids, ensure_ascii=False),
                "contract_id": contract.id,
                "gap_id": gap.id,
                "intervention_id": intervention.id,
                "pipeline_version": state.pipeline_version,
                "decision": "conditional_review",
                "score_status": "unscored",
                "quality_reason_codes_json": json.dumps(quality_reason_codes, ensure_ascii=False),
                # O1: inherit the graded confidence tier from the backing
                # intervention so the user sees A/B ranked directions.
                "confidence_tier": tier,
            })
            # Score the idea so its quality is quantifiable instead of leaving
            # novelty/feasibility/... as NULL. Reuse the V1 scoring prompt; even the
            # weak fallback model gives directional signal, and the graded tier keeps
            # the O1 A/B/C ranking on top of it.
            try:
                from app.agent.steps.generate_ideas import _score_idea
                score = await _score_idea(db, state, llm, idea)
                idea_score_val = (
                    0.20 * score.novelty + 0.20 * score.feasibility + 0.20 * score.significance +
                    0.20 * score.evidence_support + 0.10 * score.differentiation +
                    0.05 * score.experimentability + 0.05 * score.potential_impact
                )
                final_score = idea_score_val - 0.08 * score.risk
                decision = "executable_candidate" if tier == "A" else "conditional_review"
                if decision != "executable_candidate":
                    quality_reason_codes.append("EVIDENCE_OR_FEASIBILITY_REVIEW_REQUIRED")
                # P2-C idea-level novelty quick-check: the gap audit validated
                # the GAP's novelty, not the specific method's. Retrieve against
                # the cluster's hypothesis + primary mechanism; a direct prior-
                # art implementation demotes the idea to conditional_review.
                # Infrastructure failures degrade to a trace and never demote.
                novelty_verdict, matched_paper_id, mechanism_relevant_ids = await _check_idea_novelty(
                    db, state, llm, task_id, gap,
                    cluster["hypothesis"], intervention, paper_ids)
                if novelty_verdict == "already_implemented":
                    decision = "conditional_review"
                    quality_reason_codes.append("METHOD_ALREADY_PUBLISHED")
                # P2-b: re-select related work by mechanism relevance — the
                # novelty check retrieved papers against the IDEA's mechanism,
                # so its relevance-filtered ids beat the gap-audit neighbour set
                # (gap-relevant, not necessarily mechanism-relevant). Degraded
                # paths return [] and keep the fallback neighbours.
                if mechanism_relevant_ids:
                    paper_ids = list(mechanism_relevant_ids)
                    if matched_paper_id and matched_paper_id not in paper_ids:
                        paper_ids.append(matched_paper_id)
                    idea.related_paper_ids_json = json.dumps(paper_ids, ensure_ascii=False)
                elif matched_paper_id and matched_paper_id not in paper_ids:
                    paper_ids.append(matched_paper_id)
                    idea.related_paper_ids_json = json.dumps(paper_ids, ensure_ascii=False)
                idea.quality_reason_codes_json = json.dumps(quality_reason_codes, ensure_ascii=False)
                paper_repo.update_idea_scores(db, idea.id, score.model_dump(), final_score, decision)
                paper_repo.save_trace(db, task_id, "idea_quality_gate", "decision", output_data={
                    "idea_id": idea.id,
                    "gap_id": gap.id,
                    "intervention_id": intervention.id,
                    "variant_intervention_ids": [v.id for v in variants],
                    "cluster_hypothesis": cluster["hypothesis"] or None,
                    "status": decision,
                    "confidence_tier": tier,
                    "reason_codes": quality_reason_codes,
                })
            except Exception as e:
                idea.decision = "conditional_review"
                idea.score_status = "failed"
                idea.score_error = f"{type(e).__name__}: {str(e)[:300]}"
                paper_repo.save_trace(db, task_id, "idea_quality_gate", "decision", output_data={
                    "idea_id": idea.id,
                    "gap_id": gap.id,
                    "intervention_id": intervention.id,
                    "variant_intervention_ids": [v.id for v in variants],
                    "status": "rejected",
                    "reason_codes": ["IDEA_SCORE_FAILED"],
                    "error": f"{type(e).__name__}: {str(e)[:300]}",
                })
                logger.warning("Task %s: idea scoring failed; withholding executable Idea '%s': %s",
                               task_id[:8], idea.title[:40], e)
                continue
            steps = [
                *plan.steps,
                f"Success condition: {plan.success_condition}",
                f"Falsification condition: {plan.falsification_condition}",
                *[f"Control: {control}" for control in plan.controls],
            ]
            experiment = paper_repo.save_experiment(db, task_id, idea.id, {
                "hypothesis": plan.hypothesis,
                "dataset": plan.dataset,
                "baselines": plan.baselines,
                "metrics": plan.metrics,
                "model_spec": plan.model_spec,
                "dataset_provenance": plan.dataset_provenance,
                "oracle": plan.oracle,
                "statistical_analysis": plan.statistical_analysis,
                "resource_budget": plan.resource_budget,
                "scenario_atoms_json": json.dumps(plan.scenario_atoms, ensure_ascii=False),
                "steps_markdown": "\n".join(f"{index + 1}. {step}" for index, step in enumerate(steps)),
                "steps_json": json.dumps({
                    "steps": plan.steps,
                    "controls": plan.controls,
                    "success_condition": plan.success_condition,
                    "falsification_condition": plan.falsification_condition,
                }, ensure_ascii=False),
                "risks": plan.risks,
            })
            if decision == "executable_candidate":
                idea_ids.append(idea.id)
            experiment_ids.append(experiment.id)

    paper_repo.save_trace(db, task_id, "generate_minimal_experiments", "action", output_data={
        "idea_count": len(idea_ids),
        "experiment_count": len(experiment_ids),
    })
    db.commit()
    return MinimalExperimentResult(idea_ids, experiment_ids, direction_only_idea_ids)


from app.db.repositories import gap_repo
