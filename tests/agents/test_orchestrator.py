"""
tests/agents/test_orchestrator.py
----------------------------------
Unit tests for agents/orchestrator.py fan-out execution.
"""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from agents.forensic_agent import ForensicResult
from agents.orchestrator import (
    AnnouncementEvent,
    GameContext,
    ScorecardResult,
    XGBoostResult,
    run_analysis,
    should_run_phase2,
)
from agents.sentiment_auditor import SentimentResult


@pytest.mark.not_live
def test_should_run_phase2():
    scorecard_healthy = ScorecardResult(l1_state="Healthy", composite_score=0.85)
    scorecard_watch = ScorecardResult(l1_state="Watch", composite_score=0.45)
    scorecard_at_risk = ScorecardResult(l1_state="At Risk", composite_score=0.25)

    assert not should_run_phase2(scorecard_healthy)
    assert should_run_phase2(scorecard_watch)
    assert should_run_phase2(scorecard_at_risk)


@pytest.mark.not_live
def test_run_analysis_fan_out():
    """
    Verify that run_analysis triggers both Forensic and Auditor concurrently
    and combines outputs.
    """
    ctx = GameContext(
        appid=100,
        game_name="Test Game",
        snapshot_date=date(2026, 1, 1),
        ea_age_days=300,
        scorecard=ScorecardResult(l1_state="Watch", composite_score=0.40),
        xgboost=XGBoostResult(ml_eligible=True, p_distressed=0.6, is_distressed=1),
        recent_announcements=[
            AnnouncementEvent(
                event_type=14,
                title="Major Update",
                body_stripped="Added new maps and features",
                word_count=150,
                posted_at=date(2025, 12, 1),
            )
        ],
        recent_reviews=[{"text": "Good game", "voted_up": True}],
    )

    mock_forensic_res = ForensicResult(
        appid=100,
        snapshot_date="2026-01-01",
        update_substance_score=7.5,
        fake_heartbeat_flag=0,
        momentum="consistent_progress",
        event_state_mismatch=0,
        reasoning="Good update content",
    )

    mock_auditor_res = SentimentResult(
        appid=100,
        snapshot_date="2026-01-01",
        theme_clusters=[],
        sentiment_shift="stable",
        sentiment_alignment="aligned",
        key_concerns=[],
        auditor_summary="Reviews positive.",
    )

    mock_critic_res = MagicMock()
    mock_critic_res.error = None
    mock_critic_res.signal_alignment = "aligned"
    mock_critic_res.consumer_verdict = "Consumer verdict text"
    mock_critic_res.developer_brief = "Developer brief text"
    mock_critic_res.confidence_note = None

    with patch("agents.orchestrator.run_forensic_agent",
               return_value=mock_forensic_res) as mock_forensic, \
         patch("agents.orchestrator.run_sentiment_auditor",
               return_value=mock_auditor_res) as mock_auditor, \
         patch("agents.orchestrator.run_critic_agent",
               return_value=mock_critic_res) as mock_critic:

        res = run_analysis(ctx)

        assert res.phase2_triggered
        assert res.forensic_ran
        assert res.auditor_ran
        assert res.critic_ran

        mock_forensic.assert_called_once()
        mock_auditor.assert_called_once()
        mock_critic.assert_called_once()

        assert res.update_substance_score == 7.5
        assert res.sentiment_shift == "stable"
        assert res.consumer_verdict == "Consumer verdict text"
