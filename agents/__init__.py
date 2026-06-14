"""
EARLY — Phase 2 LangGraph Agents

Agents:
  forensic_agent   — update substance scoring (Groq meta-llama/llama-4-scout-17b-16e-instruct)
  sentiment_auditor — review theme clustering (Groq meta-llama/llama-4-scout-17b-16e-instruct)
  critic_agent     — narrative synthesis (Groq llama-3.3-70b-versatile)
  orchestrator     — coordinates all three, implements trigger logic

Typical usage:
    from agents.orchestrator import run_analysis, GameContext, ScorecardResult, XGBoostResult, BuildEvent

    ctx = GameContext(...)
    result = run_analysis(ctx)
"""

from agents.orchestrator import (
    run_analysis,
    GameContext,
    ScorecardResult,
    XGBoostResult,
    AnnouncementEvent,
    AnalysisResult,
    should_run_phase2,
)

__all__ = [
    "run_analysis",
    "GameContext",
    "ScorecardResult",
    "XGBoostResult",
    "AnnouncementEvent",
    "AnalysisResult",
    "should_run_phase2",
]
