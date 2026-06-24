"""
EARLY — Phase 2 LangGraph Agents

Agents:
  forensic_agent   — update substance scoring
                    (Groq qwen/qwen3.6-27b)
  sentiment_auditor — review theme clustering
                    (Groq qwen/qwen3.6-27b)
  critic_agent     — narrative synthesis (openai/gpt-oss-120b)
  orchestrator     — coordinates all three, implements trigger logic

Typical usage:
    from agents.orchestrator import run_analysis,
                                    GameContext,
                                    ScorecardResult,
                                    XGBoostResult,
                                    BuildEvent

    ctx = GameContext(...)
    result = run_analysis(ctx)
"""

from agents.orchestrator import (
    AnalysisResult,
    AnnouncementEvent,
    GameContext,
    ScorecardResult,
    XGBoostResult,
    run_analysis,
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
