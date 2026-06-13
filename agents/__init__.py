"""
EARLY — Phase 2 LangGraph Agents

Agents:
  forensic_agent   — update substance scoring (Groq llama-3.3-70b-versatile)
  sentiment_auditor — review theme clustering (Groq llama-3.1-8b-instant)
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
    BuildEvent,
    AnalysisResult,
    should_run_phase2,
)

__all__ = [
    "run_analysis",
    "GameContext",
    "ScorecardResult",
    "XGBoostResult",
    "BuildEvent",
    "AnalysisResult",
    "should_run_phase2",
]
