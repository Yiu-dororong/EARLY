"""
agents/critic_agent.py
EARLY — Critic Agent (Phase 2, Layer 2)

Synthesizes all signals into consumer_verdict and developer_brief.
Two LLM calls, each traced as a separate Langfuse generation span.

Model: Groq llama-3.3-70b-versatile
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages


class CriticState(TypedDict):
    messages: Annotated[list, add_messages]
    appid: int
    game_name: str
    snapshot_date: str
    ea_age_days: int
    l1_state: str
    l1_composite_score: float
    update_health: float | None
    player_retention: float | None
    dev_engagement: float | None
    sentiment: float | None
    price_market: float | None
    p_distressed: float | None
    is_distressed: int | None
    ml_eligible: bool
    forensic_ran: bool
    update_substance_score: float | None
    fake_heartbeat_flag: int | None
    forensic_reasoning: str | None
    auditor_ran: bool
    theme_clusters: list[dict] | None
    sentiment_shift: str | None
    key_concerns: list[str] | None
    auditor_summary: str | None
    trace: Any | None
    consumer_verdict: str | None
    developer_brief: str | None
    confidence_note: str | None
    error: str | None


CONSUMER_SYSTEM = """You are writing a risk assessment for a Steam player considering
buying or continuing to play an Early Access game. Direct, honest, non-alarmist.
2–4 sentences max. Lead with the most actionable signal. Do NOT mention model scores,
numbers, or EARLY's internal metrics. Translate signals into plain language."""

DEVELOPER_SYSTEM = """You are writing a brief for the developer of an Early Access game.
Respectful, specific, action-oriented. 3–5 sentences. Identify the 1–2 most important
signals. Integrate player concerns from the Sentiment Auditor if present. End with one
concrete actionable direction. Do NOT mention model names or ML scores."""


def _fmt(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "N/A"


def _context(state: CriticState) -> str:
    parts = [
        f"Game: {state['game_name']} (appid {state['appid']})",
        f"Snapshot: {state['snapshot_date']} | EA age: {state['ea_age_days']} days",
        "",
        f"Scorecard: {state['l1_state']} (composite {state['l1_composite_score']:.3f})",
        f"  Update Health: {_fmt(state.get('update_health'))}  Player Retention: {_fmt(state.get('player_retention'))}",
        f"  Dev Engagement: {_fmt(state.get('dev_engagement'))}  Sentiment: {_fmt(state.get('sentiment'))}  Price/Market: {_fmt(state.get('price_market'))}",
        "",
        f"ML: {'eligible' if state['ml_eligible'] else 'not eligible'}  P(distressed): {_fmt(state.get('p_distressed'))}",
    ]
    if state.get("forensic_ran"):
        parts += ["", f"Forensic: substance={state.get('update_substance_score')}/10  fake_heartbeat={state.get('fake_heartbeat_flag')}",
                  f"  {state.get('forensic_reasoning', '')}"]
    else:
        parts += ["", "Forensic: did not run — no build update in last 30 days"]

    if state.get("auditor_ran"):
        parts += ["", f"Sentiment shift: {state.get('sentiment_shift')}",
                  f"Key concerns: {'; '.join(state.get('key_concerns') or []) or 'none'}",
                  f"Summary: {state.get('auditor_summary', '')}"]
    else:
        parts += ["", "Sentiment Auditor: did not run"]

    return "\n".join(parts)


def _get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set")
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, max_tokens=512, api_key=api_key)


def _llm_call(system: str, prompt: str, span_name: str, trace: Any) -> tuple[str | None, str | None]:
    """Run one LLM call with a Langfuse span. Returns (content, error)."""
    llm = _get_llm()
    try:
        from utils.langfuse_client import generation_span
        ctx = generation_span(trace, name=span_name, model="llama-3.3-70b-versatile", input_data=prompt)
    except Exception:
        ctx = nullcontext(None)

    try:
        with ctx as span:
            response: AIMessage = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
            content = response.content.strip()
            if span and hasattr(span, "set_output"):
                span.set_output(content)
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    span.set_usage(input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"))
            return content, None
    except Exception as e:
        return None, f"{span_name} failed: {type(e).__name__}: {e}"


def write_consumer_verdict(state: CriticState) -> dict:
    content, error = _llm_call(
        CONSUMER_SYSTEM,
        f"{_context(state)}\n\nWrite the consumer verdict now.",
        "critic_consumer",
        state.get("trace"),
    )
    return {"consumer_verdict": content, "error": error}


def write_developer_brief(state: CriticState) -> dict:
    if state.get("error"):
        return {}
    content, error = _llm_call(
        DEVELOPER_SYSTEM,
        f"{_context(state)}\n\nWrite the developer brief now.",
        "critic_developer",
        state.get("trace"),
    )
    return {"developer_brief": content, "error": error}


def add_confidence_note(state: CriticState) -> dict:
    notes = []
    if not state.get("ml_eligible"):
        notes.append("Score based on rule-based scorecard only — fewer than 50 reviews.")
    if not state.get("forensic_ran"):
        notes.append("No build update in last 30 days at snapshot time.")
    if state.get("fake_heartbeat_flag") == 1:
        notes.append("Most recent update flagged as minimal content.")
    return {"confidence_note": " | ".join(notes) if notes else None}


def _build_graph() -> StateGraph:
    g = StateGraph(CriticState)
    g.add_node("write_consumer_verdict", write_consumer_verdict)
    g.add_node("write_developer_brief", write_developer_brief)
    g.add_node("add_confidence_note", add_confidence_note)
    g.set_entry_point("write_consumer_verdict")
    g.add_edge("write_consumer_verdict", "write_developer_brief")
    g.add_edge("write_developer_brief", "add_confidence_note")
    g.add_edge("add_confidence_note", END)
    return g

_compiled_graph = None
def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_graph().compile()
    return _compiled_graph


@dataclass
class CriticResult:
    appid: int
    snapshot_date: str
    consumer_verdict: str | None
    developer_brief: str | None
    confidence_note: str | None
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None and self.consumer_verdict is not None


def run_critic_agent(
    appid: int, game_name: str, snapshot_date: str, ea_age_days: int,
    l1_state: str, l1_composite_score: float,
    update_health: float | None = None, player_retention: float | None = None,
    dev_engagement: float | None = None, sentiment: float | None = None,
    price_market: float | None = None, p_distressed: float | None = None,
    is_distressed: int | None = None, ml_eligible: bool = False,
    forensic_ran: bool = False, update_substance_score: float | None = None,
    fake_heartbeat_flag: int | None = None, forensic_reasoning: str | None = None,
    auditor_ran: bool = False, theme_clusters: list[dict] | None = None,
    sentiment_shift: str | None = None, key_concerns: list[str] | None = None,
    auditor_summary: str | None = None, trace: Any | None = None,
) -> CriticResult:
    initial: CriticState = {
        "messages": [], "appid": appid, "game_name": game_name,
        "snapshot_date": snapshot_date, "ea_age_days": ea_age_days,
        "l1_state": l1_state, "l1_composite_score": l1_composite_score,
        "update_health": update_health, "player_retention": player_retention,
        "dev_engagement": dev_engagement, "sentiment": sentiment,
        "price_market": price_market, "p_distressed": p_distressed,
        "is_distressed": is_distressed, "ml_eligible": ml_eligible,
        "forensic_ran": forensic_ran, "update_substance_score": update_substance_score,
        "fake_heartbeat_flag": fake_heartbeat_flag, "forensic_reasoning": forensic_reasoning,
        "auditor_ran": auditor_ran, "theme_clusters": theme_clusters,
        "sentiment_shift": sentiment_shift, "key_concerns": key_concerns,
        "auditor_summary": auditor_summary, "trace": trace,
        "consumer_verdict": None, "developer_brief": None,
        "confidence_note": None, "error": None,
    }
    final = get_graph().invoke(initial)
    return CriticResult(
        appid=appid, snapshot_date=snapshot_date,
        consumer_verdict=final.get("consumer_verdict"),
        developer_brief=final.get("developer_brief"),
        confidence_note=final.get("confidence_note"),
        error=final.get("error"),
    )