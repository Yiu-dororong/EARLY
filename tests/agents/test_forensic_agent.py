"""
tests/agents/test_forensic_agent.py
-------------------------------------
Tests for the Forensic Agent, including the hollow major update mismatch case.

Run:
    pytest tests/agents/test_forensic_agent.py -v

Requires GROQ_API_KEY — these make real LLM calls. Mark as slow/live if you
want to exclude from fast CI runs:
    pytest -m "not live"
"""

import pytest

from agents.forensic_agent import run_forensic_agent
from tests.agents.fixtures import (
    EMPTY_ANNOUNCEMENTS,
    HOTFIX_SERIES,
    HOLLOW_ANNOUNCEMENTS,
    SUBSTANTIVE_ANNOUNCEMENTS,
)

pytestmark = pytest.mark.live  # requires GROQ_API_KEY


# ---------------------------------------------------------------------------
# Fast-path tests (no LLM call)
# ---------------------------------------------------------------------------

@pytest.mark.not_live
def test_empty_announcement_fast_path():
    """Empty body + zero word count should skip the LLM entirely."""
    result = run_forensic_agent(
        appid=1, game_name="Test", snapshot_date="2026-01-01",
        ea_age_days=100, days_since_last_build_update=999,
        announcements=EMPTY_ANNOUNCEMENTS,
    )
    assert result.success
    assert result.update_substance_score == 0.0
    assert result.fake_heartbeat_flag == 1
    assert result.momentum == "single_update"


# ---------------------------------------------------------------------------
# Live LLM tests
# ---------------------------------------------------------------------------

def test_hollow_update_event_state_mismatch():
    """
    Core triangulation test: a type-14 "Major update" announcement with zero
    build content, preceded by another hollow type-13 post. The agent should
    flag event_state_mismatch=1 — the announcement TYPE implies a build,
    but the CONTENT does not support it.
    """
    result = run_forensic_agent(
        appid=2, game_name="Hollow Update Game", snapshot_date="2026-01-01",
        ea_age_days=959, days_since_last_build_update=200,
        announcements=HOLLOW_ANNOUNCEMENTS,
    )
    assert result.success, result.error
    assert result.event_state_mismatch == 1, (
        f"Expected mismatch=1 for hollow type-14 post, got reasoning: {result.reasoning}"
    )
    assert result.update_substance_score < 4.0
    assert result.momentum in ("hollow_pattern", "declining")


def test_substantive_update_no_mismatch():
    """A real changelog with version numbers and specifics should score high
    and report no event/content mismatch."""
    result = run_forensic_agent(
        appid=3, game_name="Test Game", snapshot_date="2026-01-01",
        ea_age_days=280, days_since_last_build_update=3,
        announcements=SUBSTANTIVE_ANNOUNCEMENTS,
    )
    assert result.success, result.error
    assert result.update_substance_score >= 6.0
    assert result.event_state_mismatch == 0
    assert result.fake_heartbeat_flag == 0


def test_hotfix_series_consistent_progress():
    """Multiple small hotfixes in sequence should read as consistent_progress
    momentum, even though individual posts are short."""
    result = run_forensic_agent(
        appid=4, game_name="Iterating Game", snapshot_date="2026-01-01",
        ea_age_days=400, days_since_last_build_update=2,
        announcements=HOTFIX_SERIES,
    )
    assert result.success, result.error
    assert result.momentum == "consistent_progress", (
        f"Expected consistent_progress for hotfix series, got {result.momentum}: {result.reasoning}"
    )
    assert result.event_state_mismatch == 0
