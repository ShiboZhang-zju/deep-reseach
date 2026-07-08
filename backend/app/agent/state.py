"""Research State data structure."""

from dataclasses import dataclass, field, asdict
import json


@dataclass
class ResearchState:
    """Mutable state persisted across agent rounds."""
    task_id: str = ""
    user_input: str = ""
    normalized_topic: str = ""
    keywords: list[str] = field(default_factory=list)
    research_questions: list[str] = field(default_factory=list)
    current_round: int = 0
    used_queries: list[str] = field(default_factory=list)
    knowledge_gaps: list[str] = field(default_factory=list)
    collected_paper_ids: list[str] = field(default_factory=list)
    high_priority_paper_ids: list[str] = field(default_factory=list)
    medium_priority_paper_ids: list[str] = field(default_factory=list)
    low_priority_paper_ids: list[str] = field(default_factory=list)
    round_summaries: list[str] = field(default_factory=list)
    selected_idea_ids: list[str] = field(default_factory=list)
    user_feedback: str = ""
    stop_reason: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "ResearchState":
        if not json_str:
            return cls()
        data = json.loads(json_str)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
