"""Adjudication rubric v2 — separating ADJACENCY from COVERAGE.

Why v1 is being replaced (diagnosis, not preference)
----------------------------------------------------
v1's PARTIAL reads:

    "prior work covers PART of the delta, covers it in a different setting, or
     the delta is a re-parameterisation of known work"

The middle clause is satisfied by ANY paper that works on the same topic in
some other setting. Every real research topic has hundreds of such papers, so:

    NONE  -> requires ZERO adjacent work anywhere in the retrieved set
    PARTIAL -> the default for everything else

Measured consequence: across 34 blind-adjudicated gaps (main run + out-of-sample)
the distribution was NONE=0 / PARTIAL=32 / FULL=2. A calibration probe settled
it: a physically absurd claim with no adjacent work (LLM inference under lunar
gravity) came back NONE; a realistic claim about training-script language came
back PARTIAL. The instrument could produce NONE only in a literature vacuum.

So the metric T = NONE/(NONE+PARTIAL+FULL) was structurally 0 in any populated
field. That is an instrument defect, not a property of the gaps. See
GAP_SUPPLY_VALIDATION.md §4.

What v2 changes
---------------
1. The unit of judgement is the DELTA (the specific assertion), never the topic.
2. "Different setting" is no longer sufficient for PARTIAL. It counts only if
   the same assertion plausibly transfers (same mechanism, adjacent scale).
3. Same-topic-but-different-mechanism work is named ADJACENCY and explicitly
   maps to NONE, not PARTIAL.
4. Two diagnostics are recorded per verdict so the fix is auditable:
       delta_established   : no | partly | yes
       adjacent_work_exists: bool
   A verdict of NONE with adjacent_work_exists=true is the exact case v1 could
   not express; it is now visible in the data instead of being inferred.

What v2 must NOT become
-----------------------
The opposite degeneracy. If v2 simply returned NONE everywhere it would be as
uninformative as v1's PARTIAL-everywhere, and it would also be dishonest. The
rubric therefore carries explicit anti-default instructions and the
pre-registration includes REVERSE PROBES (claims whose delta is genuinely
established) that must come back FULL/PARTIAL. Those probes are committed
before any v2 run; see RUBRIC_V2_PREREG.md.

Versioning: v1 lives on in `adjudicate_pilot.RUBRIC` and is untouched, so every
historical number stays reproducible and the v1-vs-v2 contrast is measurable.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RUBRIC_VERSION = "v2.1-no-absence-credit-2026-11-09"

RUBRIC_V2 = """You are an independent prior-art reviewer for a research-gap
adjudication. You will be shown ONE claimed research gap (or ONE research
idea) extracted from a literature review, together with evidence claims the
pipeline used and search results you retrieved yourself.

Your job: decide whether the SPECIFIC CLAIMED DELTA is already established by
prior work. The delta is the assertion in the CLAIMED DELTA field — a specific
mechanism, causal attribution, or effect. It is NOT the topic.

Three distinct situations. Keep them apart:

  (A) ADJACENCY. Prior work studies the same topic, task, or model family, but
      does not test the specific assertion. This is extremely common and is NOT
      coverage.
  (B) PARTIAL COVERAGE. Prior work tests the same assertion, or a necessary
      sub-claim of it, under conditions where the result plausibly transfers
      (same mechanism, comparable data scale/domain).
  (C) ESTABLISHMENT. Prior work already demonstrates the claimed delta in a
      comparable setting.

Verdict rubric for GAPS (use exactly one):
- NONE: no prior work establishes the claimed delta. Situations of type (A)
  alone do NOT make this PARTIAL. A researcher could run the proposed study and
  report a genuinely new result.
- PARTIAL: prior work establishes a substantial part of the claimed delta,
  i.e. type (B). The residual gap is real but narrow: the claim needs
  narrowing, not abandonment.
- FULL: prior work already establishes the claimed delta, i.e. type (C). The
  gap is a false open.

Verdict rubric for IDEAS (use exactly one):
- fund: the experiment is worth running; the contribution would be new and the
  cost is proportionate.
- maybe: worth considering but the novelty or the cost is questionable.
- no: not worth running; already done, or not a real contribution.

Decision procedure (follow it in order):
1. State the delta in one sentence, in your notes, before judging.
2. Ask: does any retrieved work TEST THAT ASSERTION (not merely the topic)?
   If none does, the verdict is NONE — record adjacent_work_exists=true if the
   topic itself is busy, but the verdict stays NONE.
3. If some work tests it, ask whether the result establishes it (FULL) or only
   a substantial part / a plausibly transferring neighbour (PARTIAL).
4. Fill `delta_established` (no / partly / yes) so your reasoning is auditable.

Rules:
- Base your verdict on the SEARCH RESULTS, not on the pipeline's own verdict.
- Do NOT default to NONE. If prior work demonstrates the delta, that is FULL.
  If it demonstrates a substantial part, that is PARTIAL. A reviewer who
  answers NONE to everything is as useless as one who answers PARTIAL to
  everything, and both are wrong.
- Do NOT default to PARTIAL. "There is related work" is not partial coverage;
  only work that tests the same assertion counts.
- NO CREDIT FOR WEAKENING AN ABSENCE CLAIM. Many gaps are phrased partly as an
  absence: "the literature reports no such comparison", "no one measures X for
  Y", "first to quantify Z". If prior work merely makes that absence statement
  less absolute — because it measures something adjacent, in another setting, or
  only one of the conjoined elements — the delta itself is still untested, and
  the verdict is NONE. Example: if the delta is "pipeline A beats baselines B
  and C by >10pp on 2 benchmarks" and no retrieved work performs that head-to-
  head, the verdict is NONE even though benchmark infrastructure and separate
  baseline evaluations exist. Adjacency + a weakened absence claim is still
  adjacency.
- PARTIAL is reserved for: retrieved work tests the SAME assertion (or a
  necessary sub-claim whose result transfers) and finds the claimed effect,
  leaving only a narrower residual. If you find yourself writing "no retrieved
  work directly measures X, but related work exists", that is NONE, not
  PARTIAL.
- If the search results are irrelevant or empty, say so in notes and flag
  `search_useful: false`; then reason from your own knowledge and lower your
  confidence.
- Quantify your confidence (high / medium / low) and name the retrieved work
  (by its number) that drove the verdict.
"""


class _VerdictBaseV2(BaseModel):
    anon_id: str
    unit_type: str
    verdict: str
    confidence: Literal["high", "medium", "low"]
    notes: str
    # --- diagnostics that make the v1->v2 change auditable -------------
    delta_stated: str = Field(
        default="",
        description="The claimed delta restated in one sentence, before judging.")
    delta_established: Literal["no", "partly", "yes"] = "no"
    adjacent_work_exists: bool = False
    residual_novelty: str = ""
    search_useful: bool = True
    key_prior_work: list[str] = Field(default_factory=list)
    blinded_context: bool = True


class GapVerdictV2(_VerdictBaseV2):
    verdict: Literal["NONE", "PARTIAL", "FULL"]


class IdeaVerdictV2(_VerdictBaseV2):
    verdict: Literal["fund", "maybe", "no"]
