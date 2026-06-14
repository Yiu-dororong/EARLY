"""
tests/agents/test_critic_agent.py
-----------------------------------
Tests for the Critic Agent — primarily signal_alignment computation
(deterministic) and verdict quality (DeepEval G-Eval).

Run:
    pytest tests/agents/test_critic_agent.py -v
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_groq import ChatGroq

from agents.critic_agent import CriticResult, compute_signal_alignment, run_critic_agent

pytestmark = pytest.mark.live

# ---------------------------------------------------------------------------
# Shared base kwargs for run_critic_agent
# ---------------------------------------------------------------------------

BASE_KWARGS = dict(
    appid=1, game_name="Test Game", snapshot_date="2026-01-01",
    ea_age_days=400, l1_state="Watch", l1_composite_score=0.45,
    update_health=0.30, player_retention=0.55, dev_engagement=0.38,
    sentiment=0.61, price_market=0.50, p_distressed=0.58,
    is_distressed=1, ml_eligible=True,
)


# ---------------------------------------------------------------------------
# Deterministic: compute_signal_alignment
# (no LLM — tests the logic node directly)
# ---------------------------------------------------------------------------

def test_alignment_conflicted_when_forensic_mismatch():
    """event_state_mismatch=1 → conflicted, regardless of auditor."""
    from agents.critic_agent import CriticState
    state: CriticState = {
        **{k: None for k in CriticState.__annotations__},
        "forensic_ran": True, "event_state_mismatch": 1,
        "auditor_ran": True, "sentiment_alignment": "aligned",
    }
    assert compute_signal_alignment(state) == "conflicted"


def test_alignment_conflicted_when_auditor_conflicts():
    """sentiment_alignment=conflicted → conflicted, even if forensic is fine."""
    from agents.critic_agent import CriticState
    state: CriticState = {
        **{k: None for k in CriticState.__annotations__},
        "forensic_ran": True, "event_state_mismatch": 0,
        "auditor_ran": True, "sentiment_alignment": "conflicted",
    }
    assert compute_signal_alignment(state) == "conflicted"


def test_alignment_aligned_when_both_ran_no_conflict():
    """Both agents ran, no conflicts → aligned."""
    from agents.critic_agent import CriticState
    state: CriticState = {
        **{k: None for k in CriticState.__annotations__},
        "forensic_ran": True, "event_state_mismatch": 0,
        "auditor_ran": True, "sentiment_alignment": "aligned",
    }
    assert compute_signal_alignment(state) == "aligned"


def test_alignment_partial_when_only_one_ran():
    """Only forensic ran (auditor skipped) → partial."""
    from agents.critic_agent import CriticState
    state: CriticState = {
        **{k: None for k in CriticState.__annotations__},
        "forensic_ran": True, "event_state_mismatch": 0,
        "auditor_ran": False, "sentiment_alignment": None,
    }
    assert compute_signal_alignment(state) == "partial"


def test_alignment_partial_when_neither_ran():
    from agents.critic_agent import CriticState
    state: CriticState = {
        **{k: None for k in CriticState.__annotations__},
        "forensic_ran": False, "event_state_mismatch": None,
        "auditor_ran": False, "sentiment_alignment": None,
    }
    assert compute_signal_alignment(state) == "partial"


# ---------------------------------------------------------------------------
# Live LLM tests — verdict quality
# ---------------------------------------------------------------------------

def test_conflicted_verdict_mentions_discrepancy():
    """
    DeepEval G-Eval: when signal_alignment=conflicted, consumer_verdict
    must communicate the discrepancy in plain language — not just give a
    generic "be cautious" warning without explaining why signals disagree.
    """
    result = run_critic_agent(
        **BASE_KWARGS,
        forensic_ran=True, update_substance_score=2.0, fake_heartbeat_flag=1,
        momentum="hollow_pattern", event_state_mismatch=1,
        forensic_reasoning="Type-14 major update posted, body contains zero changelog content.",
        auditor_ran=True, sentiment_shift="declining", sentiment_alignment="conflicted",
        key_concerns=["No response to bug reports", "Promised features undelivered"],
        auditor_summary=(
            "Reviews describe a game that looks active from its announcement "
            "cadence but is perceived by players as abandoned — dev posted a "
            "'major update' that contained no actual content."
        ),
    )
    assert result.success, result.error
    assert result.signal_alignment == "conflicted"

    metric = GEval(
        name="ConflictCommunication",
        model=ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0),
        criteria=(
            "The verdict must explain WHY there is uncertainty — specifically that "
            "the game's official activity signals (recent announcements) don't match "
            "what players are actually reporting. It should NOT merely say 'be cautious' "
            "without explaining the discrepancy. It should NOT mention internal metric "
            "names like 'signal_alignment', 'p_distressed', 'l1_state', or 'triangulation'. "
            "A verdict that only reports the negative score without noting that the "
            "activity signals may be misleading should FAIL."
        ),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.7,
    )
    test_case = LLMTestCase(
        input="signal_alignment=conflicted, hollow announcement vs declining reviews",
        actual_output=result.consumer_verdict or "",
    )
    assert_test(test_case, [metric])


def test_verdict_no_internal_metric_names():
    """
    DeepEval G-Eval: neither verdict should expose internal metric names
    or system vocabulary to the end user.
    """
    result = run_critic_agent(
        **BASE_KWARGS,
        forensic_ran=True, update_substance_score=5.5, fake_heartbeat_flag=0,
        momentum="single_update", event_state_mismatch=0,
        forensic_reasoning="Moderate changelog with some specifics.",
        auditor_ran=True, sentiment_shift="stable", sentiment_alignment="aligned",
        key_concerns=["Performance issues on older hardware"],
        auditor_summary="Reviews are generally positive with some hardware complaints.",
    )
    assert result.success, result.error

    forbidden_terms = [
        "p_distressed", "l1_state", "signal_alignment", "triangulation",
        "composite_score", "ml_eligible", "EARLY", "XGBoost", "SHAP",
    ]

    full_output = f"{result.consumer_verdict or ''}\n{result.developer_brief or ''}"
    found = [t for t in forbidden_terms if t.lower() in full_output.lower()]
    assert not found, f"Internal metric names found in verdict: {found}"


def test_aligned_healthy_signal_verdict_is_positive():
    """
    When all signals agree (aligned) on a healthy game, consumer_verdict
    should be reassuring, not hedging unnecessarily.
    """
    result = run_critic_agent(
        appid=10, game_name="Thriving Game", snapshot_date="2026-01-01",
        ea_age_days=200, l1_state="Watch", l1_composite_score=0.42,
        update_health=0.72, player_retention=0.68, dev_engagement=0.71,
        sentiment=0.80, price_market=0.55, p_distressed=0.22,
        is_distressed=0, ml_eligible=True,
        forensic_ran=True, update_substance_score=8.0, fake_heartbeat_flag=0,
        momentum="consistent_progress", event_state_mismatch=0,
        forensic_reasoning="Detailed changelog, multiple system improvements.",
        auditor_ran=True, sentiment_shift="improving", sentiment_alignment="aligned",
        key_concerns=[],
        auditor_summary="Reviews are increasingly positive, players note active development.",
    )
    assert result.success, result.error
    assert result.signal_alignment == "aligned"

    metric = GEval(
        name="PositiveTone",
        model=ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0),
        criteria=(
            "When all signals are positive and aligned, the verdict should be "
            "encouraging and reassuring. It should NOT hedge excessively or add "
            "unnecessary caveats. A verdict that says 'proceed with caution despite "
            "all positive signals' should FAIL. The verdict should communicate "
            "that the evidence suggests healthy, active development."
        ),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.7,
    )
    test_case = LLMTestCase(
        input="All signals aligned positive, low distress risk",
        actual_output=result.consumer_verdict or "",
    )
    assert_test(test_case, [metric])


def test_developer_brief_ends_with_action():
    """
    DeepEval G-Eval: developer_brief must end with a concrete actionable
    direction — not a vague platitude.
    """
    result = run_critic_agent(
        **BASE_KWARGS,
        forensic_ran=True, update_substance_score=3.0, fake_heartbeat_flag=1,
        momentum="declining", event_state_mismatch=1,
        forensic_reasoning="Announcement title implies build, body is pure hype text.",
        auditor_ran=True, sentiment_shift="declining", sentiment_alignment="conflicted",
        key_concerns=["No actual build shipped despite announcements", "Forum silence"],
        auditor_summary="Players are skeptical of announcements without accompanying builds.",
    )
    assert result.success, result.error

    metric = GEval(
        name="ActionableEnding",
        model=ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0),
        criteria=(
            "The developer brief must end with a specific, concrete action the "
            "developer can take — e.g. 'ship a small but real patch this week to "
            "rebuild credibility' or 'post a detailed roadmap update explaining the "
            "delay'. Vague endings like 'focus on communication' or 'keep up the "
            "good work' without specific guidance should FAIL."
        ),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.7,
    )
    test_case = LLMTestCase(
        input="conflicted signals, fake heartbeat, declining reviews",
        actual_output=result.developer_brief or "",
    )
    assert_test(test_case, [metric])
