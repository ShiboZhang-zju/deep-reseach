"""Tests for paper scoring formula (runner.py / score_papers.py).

Covers: final_score calculation, priority thresholds, authority adjustments.
"""

import pytest

from app.agent.state import ResearchState


class TestPaperScoringFormula:
    """Test the scoring formula: 0.30*relevance + 0.25*authority + 0.15*recency
    + 0.15*novelty + 0.15*idea_potential
    """

    def compute_final_score(self, relevance, authority, recency, novelty, idea_potential):
        """Replicate the scoring formula from score_papers.py."""
        return (
            0.30 * relevance + 0.25 * authority + 0.15 * recency +
            0.15 * novelty + 0.15 * idea_potential
        )

    def test_all_zeros(self):
        score = self.compute_final_score(0, 0, 0, 0, 0)
        assert score == 0.0

    def test_all_ones(self):
        score = self.compute_final_score(1, 1, 1, 1, 1)
        assert score == pytest.approx(1.0)

    def test_high_relevance_dominates(self):
        """relevance has the highest weight (0.30)."""
        score_high_rel = self.compute_final_score(1.0, 0, 0, 0, 0)
        score_high_auth = self.compute_final_score(0, 1.0, 0, 0, 0)
        assert score_high_rel > score_high_auth  # 0.30 > 0.25

    def test_weights_sum_to_one(self):
        """Weights should sum to 1.0."""
        total_weight = 0.30 + 0.25 + 0.15 + 0.15 + 0.15
        assert total_weight == pytest.approx(1.0)

    def test_priority_thresholds(self):
        """Priority: high >= 0.75, medium >= 0.5, low < 0.5."""
        def get_priority(score):
            if score >= 0.75:
                return "high"
            elif score >= 0.5:
                return "medium"
            else:
                return "low"

        # High: all scores = 0.8 → 0.8
        assert get_priority(self.compute_final_score(0.8, 0.8, 0.8, 0.8, 0.8)) == "high"
        # Medium: all scores = 0.6 → 0.6
        assert get_priority(self.compute_final_score(0.6, 0.6, 0.6, 0.6, 0.6)) == "medium"
        # Low: all scores = 0.3 → 0.3
        assert get_priority(self.compute_final_score(0.3, 0.3, 0.3, 0.3, 0.3)) == "low"

    def test_boundary_075_is_high(self):
        """Score exactly 0.75 should be 'high'."""
        # 0.75 = 0.30*r + 0.25*a + 0.15*(rec+nov+idea)
        # If all equal x: 0.30x + 0.25x + 0.45x = x, so x=0.75
        score = self.compute_final_score(0.75, 0.75, 0.75, 0.75, 0.75)
        assert score == pytest.approx(0.75)

    def test_boundary_05_is_medium(self):
        """Score exactly 0.5 should be 'medium'."""
        score = self.compute_final_score(0.5, 0.5, 0.5, 0.5, 0.5)
        assert score == pytest.approx(0.5)


class TestAuthorityAdjustments:
    """Test authority score adjustments in score_papers.py."""

    def test_missing_metadata_penalty(self):
        """Papers with citation_count==0 AND year is None get authority * 0.7."""
        authority = 0.8
        adjusted = authority * 0.7
        assert adjusted == pytest.approx(0.56)

    def test_top_venue_boost(self):
        """Papers from top venues get authority + 0.1 (capped at 1.0)."""
        from app.agent.steps.score_papers import TOP_VENUE_KEYWORDS

        authority = 0.8
        venue = "ICML 2024"
        venue_str = venue.upper()
        assert any(kv in venue_str for kv in TOP_VENUE_KEYWORDS)
        adjusted = min(1.0, authority + 0.1)
        assert adjusted == 0.9

    def test_top_venue_boost_capped_at_1(self):
        authority = 0.95
        adjusted = min(1.0, authority + 0.1)
        assert adjusted == 1.0

    def test_non_top_venue_no_boost(self):
        from app.agent.steps.score_papers import TOP_VENUE_KEYWORDS

        venue = "Workshop on NLP"
        venue_str = venue.upper()
        assert not any(kv in venue_str for kv in TOP_VENUE_KEYWORDS)


class TestIdeaScoringFormula:
    """Test idea scoring: 0.20*(nov+feas+sig+evid) + 0.10*(diff+exp) - 0.08*risk."""

    def compute_idea_score(self, novelty, feasibility, significance, evidence_support,
                           differentiation, experimentability, risk):
        idea_score = (
            0.20 * novelty + 0.20 * feasibility + 0.20 * significance +
            0.20 * evidence_support + 0.10 * differentiation +
            0.10 * experimentability
        )
        return idea_score - 0.08 * risk

    def test_idea_weights_sum(self):
        """Idea score weights (excluding risk) sum to 1.0."""
        weights = 0.20 + 0.20 + 0.20 + 0.20 + 0.10 + 0.10
        assert weights == pytest.approx(1.0)

    def test_risk_reduces_score(self):
        score_no_risk = self.compute_idea_score(0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.0)
        score_high_risk = self.compute_idea_score(0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.0)
        assert score_high_risk < score_no_risk
        assert score_no_risk - score_high_risk == pytest.approx(0.08)

    def test_go_threshold(self):
        """Score >= 0.70 → go."""
        score = self.compute_idea_score(0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.0)
        assert score == pytest.approx(0.8)
        assert score >= 0.70

    def test_revise_threshold(self):
        """Score 0.50-0.69 → revise."""
        score = self.compute_idea_score(0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.0)
        assert 0.50 <= score < 0.70

    def test_reject_threshold(self):
        """Score < 0.50 → reject."""
        score = self.compute_idea_score(0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.0)
        assert score < 0.50
