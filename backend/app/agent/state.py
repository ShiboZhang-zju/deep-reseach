"""Research State data structure.

Phase 1.5 refactor:
- Added contract_id, active_question_ids, clarification_questions
- Separated clarification questions from research questions
- Added current_phase and pipeline_version for stage orchestration
- Backward compatible: from_json ignores unknown fields, old fields preserved
"""

from dataclasses import dataclass, field, asdict
import json


@dataclass
class ResearchState:
    """Mutable state persisted across agent rounds.

    Phase 1.5 changes:
    - contract_id: Links to the active ResearchContract in DB
    - active_question_ids: IDs of active ResearchQuestions (source of truth is DB table)
    - clarification_questions: Questions asked during clarification (was research_questions)
    - current_phase: Current pipeline phase for stage orchestration
    - pipeline_version: Schema version for forward compatibility
    - terminal_reason: Why the pipeline terminated (distinct from stop_reason for search)
    - research_questions: KEPT for backward compat, but new code should use DB table + active_question_ids
    """
    task_id: str = ""
    user_input: str = ""
    normalized_topic: str = ""
    keywords: list[str] = field(default_factory=list)

    # Phase 1.5: Clarification questions (was conflated with research_questions)
    clarification_questions: list[str] = field(default_factory=list)

    # Phase 1.5: Active research question IDs from DB
    contract_id: str | None = None
    active_question_ids: list[str] = field(default_factory=list)

    # Phase 3A: Gap control plane state
    active_gap_ids: list[str] = field(default_factory=list)
    surviving_gap_ids: list[str] = field(default_factory=list)

    # O2: Targeted remediation — tracks how many extra directed-search rounds
    # were spent trying to unblock a failed pipeline gate, keyed by reason code.
    # Bounded by settings.max_remediation_attempts to avoid infinite loops.
    remediation_attempts: dict = field(default_factory=dict)

    # Pipeline orchestration
    current_phase: str = "pending"
    pipeline_version: int = 2
    terminal_reason: str = ""

    # Search loop state
    current_round: int = 0
    used_queries: list[str] = field(default_factory=list)
    knowledge_gaps: list[str] = field(default_factory=list)
    collected_paper_ids: list[str] = field(default_factory=list)
    high_priority_paper_ids: list[str] = field(default_factory=list)
    medium_priority_paper_ids: list[str] = field(default_factory=list)
    low_priority_paper_ids: list[str] = field(default_factory=list)
    round_summaries: list[str] = field(default_factory=list)

    # User interaction
    selected_idea_ids: list[str] = field(default_factory=list)
    user_feedback: str = ""
    stop_reason: str = ""

    # DEPRECATED: Use clarification_questions + DB ResearchQuestion table instead
    # Kept for backward compatibility — old code may still read this
    research_questions: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "ResearchState":
        if not json_str:
            return cls()
        data = json.loads(json_str)
        # Filter to known fields only (backward compat: ignores unknown keys)
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}

        # Phase 1.5: Migrate old research_questions to clarification_questions
        # if clarification_questions is empty but research_questions has data
        if not known.get("clarification_questions") and known.get("research_questions"):
            known["clarification_questions"] = known["research_questions"]

        return cls(**known)
