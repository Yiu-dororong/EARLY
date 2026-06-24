"""
tests/agents/test_sentiment_auditor.py
---------------------------------------
Tests for the Sentiment Auditor — primarily the sentiment_alignment
triangulation field and edge cases.

Run:
    pytest tests/agents/test_sentiment_auditor.py -v
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from agents.sentiment_auditor import run_sentiment_auditor
from tests.agents.eval_llm import DeepEvalGroqAdapter
from tests.agents.fixtures import (
    ALIGNED_REVIEWS_OLDER,
    ALIGNED_REVIEWS_RECENT,
    CONFLICTING_REVIEWS_OLDER,
    CONFLICTING_REVIEWS_RECENT,
)


pytestmark = pytest.mark.live


# ---------------------------------------------------------------------------
# Deterministic / fast-path tests (no LLM)
# ---------------------------------------------------------------------------

@pytest.mark.not_live
def test_no_reviews_returns_insufficient_data():
    """Zero reviews → skip LLM, return insufficient_data for both shift fields."""
    result = run_sentiment_auditor(
        appid=1, game_name="Test", snapshot_date="2026-01-01",
        review_score_at_T=0.7, review_score_last_90d=None,
        review_count_at_T=0, recent_reviews=[], older_reviews=[],
        l1_state="Watch",
    )
    assert result.sentiment_shift == "insufficient_data"
    assert result.sentiment_alignment == "insufficient_data"
    assert result.theme_clusters == []
    assert result.error is None


# ---------------------------------------------------------------------------
# Live LLM tests
# ---------------------------------------------------------------------------

def test_conflicting_reviews_flagged():
    """
    Reviews describing abandonment/dev-silence for a game classified as
    Healthy should produce sentiment_alignment='conflicted'.
    This is the abandoned scenario from the review side.
    """
    result = run_sentiment_auditor(
        appid=2, game_name="Abandoned Game", snapshot_date="2026-01-01",
        review_score_at_T=0.64, review_score_last_90d=0.52,
        review_count_at_T=312,
        recent_reviews=CONFLICTING_REVIEWS_RECENT,
        older_reviews=CONFLICTING_REVIEWS_OLDER,
        l1_state="Healthy",
    )
    assert result.success, result.error
    assert result.sentiment_alignment == "conflicted", (
        f"Expected conflicted for abandonment reviews vs Healthy state, "
        f"got {result.sentiment_alignment}. Summary: {result.auditor_summary}"
    )
    assert result.sentiment_shift in ("declining", "mixed")
    assert len(result.key_concerns or []) > 0


def test_aligned_reviews_not_flagged():
    """Positive, active-dev reviews for a Healthy game should be 'aligned'."""
    result = run_sentiment_auditor(
        appid=3, game_name="Good Game", snapshot_date="2026-01-01",
        review_score_at_T=0.85, review_score_last_90d=0.87,
        review_count_at_T=650,
        recent_reviews=ALIGNED_REVIEWS_RECENT,
        older_reviews=ALIGNED_REVIEWS_OLDER,
        l1_state="Healthy",
    )
    assert result.success, result.error
    assert result.sentiment_alignment == "aligned", (
        f"Expected aligned for positive reviews matching Healthy state, "
        f"got {result.sentiment_alignment}"
    )


def test_auditor_summary_mentions_conflict():
    """
    DeepEval G-Eval: when sentiment_alignment is 'conflicted', the
    auditor_summary should explicitly describe the discrepancy — not just
    say "reviews are negative" without connecting to the l1_state context.
    """
    result = run_sentiment_auditor(
        appid=4, game_name="Abandoned Game", snapshot_date="2026-01-01",
        review_score_at_T=0.64, review_score_last_90d=0.52,
        review_count_at_T=312,
        recent_reviews=CONFLICTING_REVIEWS_RECENT,
        older_reviews=CONFLICTING_REVIEWS_OLDER,
        l1_state="Healthy",
    )
    assert result.success, result.error

    if result.sentiment_alignment != "conflicted":
        pytest.skip("sentiment_alignment not conflicted — "
                    "conflict wording test not applicable")

    metric = GEval(
        name="ConflictArticulation",
        model=DeepEvalGroqAdapter(model_name="openai/gpt-oss-120b",
                                  temperature=0.0),
        criteria=(
            "The summary must explicitly state that player reviews CONTRADICT or "
            "CONFLICT with the stated health classification. It should describe "
            "what the classification implies (active/healthy development) vs what "
            "players are actually reporting (no updates, dev silence, abandonment). "
            "A summary that merely lists negative themes without connecting them to "
            "the health classification mismatch should FAIL."
        ),
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        threshold=0.7,
    )

    test_case = LLMTestCase(
        input="l1_state=Healthy, recent reviews describing abandonment",
        actual_output=result.auditor_summary or "",
    )
    assert_test(test_case, [metric])


def test_key_concerns_are_actionable():
    """
    DeepEval G-Eval: key_concerns should be specific, actionable developer
    pain points — not vague summaries ("players are unhappy") or meta-commentary
    ("the game has issues").
    """
    result = run_sentiment_auditor(
        appid=5, game_name="Abandoned Game", snapshot_date="2026-01-01",
        review_score_at_T=0.64, review_score_last_90d=0.52,
        review_count_at_T=312,
        recent_reviews=CONFLICTING_REVIEWS_RECENT,
        older_reviews=CONFLICTING_REVIEWS_OLDER,
        l1_state="At Risk",
    )
    assert result.success, result.error
    concerns = result.key_concerns or []
    assert len(concerns) > 0, "Expected at least one key concern for negative reviews"

    metric = GEval(
        name="ConcernSpecificity",
        model=DeepEvalGroqAdapter(model_name="openai/gpt-oss-120b",
                                  temperature=0.0),
        criteria=(
            "Each concern should identify a SPECIFIC, ACTIONABLE issue a developer "
            "can address — e.g. 'No response to bug reports in forum' or "
            "'Promised roadmap features undelivered for 12+ months'. "
            "Vague concerns like 'players are unhappy' or 'game has bugs' should FAIL. "
            "Each concern should be under 15 words."
        ),
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        threshold=0.7,
    )

    test_case = LLMTestCase(
        input="Negative reviews describing specific developer failures",
        actual_output="\n".join(f"- {c}" for c in concerns),
    )
    assert_test(test_case, [metric])
