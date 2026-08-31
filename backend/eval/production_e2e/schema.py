"""production_e2e_v1 — shared output contract for all three systems.

Fairness rule (frozen in README): every system submits AT MOST ONE final idea
per topic, or explicitly abstains. The schema is identical across systems;
the only variable is information / workflow.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class FinalIdea(BaseModel):
    """The single final idea a system submits for one topic."""

    title: str = Field(min_length=4, description="specific, mechanism-named title")
    research_question: str = Field(
        min_length=10,
        description="the single falsifiable claim / hypothesis this idea would test")
    method_sketch: str = Field(
        min_length=20,
        description="how the method works: mechanism, inputs/outputs, key components")
    experiment_outline: str = Field(
        min_length=20,
        description="minimal experiment design: dataset/baseline/metric/protocol sketch")
    supporting_rationale: str = Field(
        default="",
        description="why this idea matters and what prior knowledge it builds on")


class E2EDecision(BaseModel):
    """One system's final decision on one topic: propose exactly one idea, or abstain."""

    decision: Literal["propose_idea", "abstain"]
    idea: FinalIdea | None = None
    abstain_reason: str = ""

    @model_validator(mode="after")
    def _check_consistency(self) -> "E2EDecision":
        if self.decision == "propose_idea" and self.idea is None:
            raise ValueError("decision=propose_idea requires a non-null idea")
        if self.decision == "abstain" and self.idea is not None:
            raise ValueError("decision=abstain must not carry an idea")
        if self.decision == "abstain" and not self.abstain_reason.strip():
            raise ValueError("decision=abstain requires an abstain_reason")
        return self


# ---------------------------------------------------------------------------
# Prediction record shape persisted to predictions.jsonl (one per topic).
# Reuses the eval-harness-v1 contract: attempts.jsonl keeps every attempt
# (cost accounting); predictions.jsonl keeps the single final success.
# ---------------------------------------------------------------------------

SYSTEM_NAMES = ("direct_llm", "retrieval_llm", "full_v2")

TERMINAL_STATUSES = (
    "waiting_for_user_review",
    "more_research_required",
    "insufficient_evidence",
    "abstained",
    "done",
    "failed",
    "stopped",
)


def build_prediction_record(
    *,
    topic_id: str,
    stratum: str,
    topic: str,
    system: str,
    decision: E2EDecision,
    extra: dict | None = None,
) -> dict:
    """Normalize one topic's final outcome into the shared prediction record."""
    record = {
        "sample_id": topic_id,          # run_samples/resume contract key
        "topic_id": topic_id,
        "stratum": stratum,
        "topic": topic,
        "system": system,
        "decision": decision.decision,
        "idea": decision.idea.model_dump() if decision.idea else None,
        "abstain_reason": decision.abstain_reason or None,
    }
    if extra:
        record.update(extra)
    return record
