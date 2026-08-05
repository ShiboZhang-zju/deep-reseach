"""Steps package — modular agent steps.

Each step is an independent module that can be unit-tested in isolation.
"""

from app.agent.steps.clarify_topic import clarify_topic
from app.agent.steps.generate_queries import generate_queries
from app.agent.steps.search_papers import search_and_save_papers
from app.agent.steps.score_papers import score_papers
from app.agent.steps.summarize_round import summarize_round
from app.agent.steps.build_clusters import build_paper_clusters
from app.agent.steps.generate_report import generate_report
from app.agent.steps.generate_ideas import generate_and_score_ideas, _score_idea
from app.agent.steps.generate_experiment import generate_experiments
from app.agent.steps.analyze_papers import analyze_papers
# Phase 1: New steps
from app.agent.steps.build_contract import build_research_contract
from app.agent.steps.decompose_research_space import decompose_research_space
# Phase 2: New steps
from app.agent.steps.extract_evidence import extract_evidence_units
from app.agent.steps.update_coverage import update_coverage_matrix
from app.agent.steps.mine_gaps import mine_gap_candidates
from app.agent.steps.audit_gaps import audit_gap_candidates
from app.agent.steps.generate_interventions import generate_interventions
from app.agent.steps.generate_minimal_experiments import generate_minimal_experiments
from app.agent.steps.generate_landscape_brief import generate_landscape_brief
# O2: Targeted remediation
from app.agent.steps.targeted_research import (
    run_targeted_research_round,
    can_remediate,
    REMEDIABLE_REASONS,
)

__all__ = [
    "clarify_topic",
    "generate_queries",
    "search_and_save_papers",
    "score_papers",
    "summarize_round",
    "build_paper_clusters",
    "generate_report",
    "generate_and_score_ideas",
    "generate_experiments",
    "_score_idea",
    "analyze_papers",
    # Phase 1
    "build_research_contract",
    "decompose_research_space",
    # Phase 2
    "extract_evidence_units",
    "update_coverage_matrix",
    "mine_gap_candidates",
    "audit_gap_candidates",
    "generate_interventions",
    "generate_minimal_experiments",
    "generate_landscape_brief",
    # O2: Targeted remediation
    "run_targeted_research_round",
    "can_remediate",
    "REMEDIABLE_REASONS",
]
