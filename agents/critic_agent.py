"""
agents/critic_agent.py
EARLY — Critic Agent (Phase 2, Layer 2)

Synthesizes ML + Forensic + Auditor signals into:
  - signal_alignment   ("aligned" | "conflicted" | "partial")
  - consumer_verdict
  - developer_brief
  - confidence_note

signal_alignment is computed deterministically (not by the LLM) from the
three independent signals, BEFORE either verdict is written. Both verdicts
are then told the alignment verdict explicitly, so they communicate it
consistently rather than each agent guessing independently.

Triangulation logic (signal_alignment):
  - If forensic.event_state_mismatch == 1            → "conflicted"
  - If auditor.sentiment_alignment == "conflicted"    → "conflicted"
  - If forensic and auditor both ran and both report
    no conflict                                       → "aligned"
  - If forensic or auditor did not run                → "partial"

Model: Cerebras zai-glm-4.7
Tracing: Langfuse generation span
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from langchain_cerebras import ChatCerebras
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from agents.prompts import CRITIC_CONSUMER_SYSTEM, CRITIC_DEVELOPER_SYSTEM
from agents.states import CriticState


MODEL_NAME = "zai-glm-4.7"

def _fmt(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "N/A"


# ---------------------------------------------------------------------------
# Deterministic triangulation
# ---------------------------------------------------------------------------

def compute_signal_alignment(state: CriticState) -> str:
    """
    Decide signal_alignment BEFORE either verdict is written, from the
    structured (non-LLM-prose) outputs of Forensic and Auditor.
    """
    forensic_ran = state.get("forensic_ran", False)
    auditor_ran  = state.get("auditor_ran", False)

    if not forensic_ran and not auditor_ran:
        return "partial"

    conflicts = []
    if forensic_ran and state.get("event_state_mismatch") == 1:
        conflicts.append("forensic")
    if auditor_ran and state.get("sentiment_alignment") == "conflicted":
        conflicts.append("auditor")

    if conflicts:
        return "conflicted"

    if forensic_ran and auditor_ran:
        return "aligned"

    # Only one of the two ran, and it reported no conflict
    return "partial"


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _context(state: CriticState) -> str:
    parts = [
        f"Game: {state['game_name']} (appid {state['appid']})",
        f"Snapshot: {state['snapshot_date']} | EA age: {state['ea_age_days']} days",
        "",
        f"SIGNAL ALIGNMENT: {state.get('signal_alignment', 'partial')}",
        "",
        f"Scorecard: {state['l1_state']} (composite {state['l1_composite_score']:.3f})",
        (
            f"  Update Health: {_fmt(state.get('update_health'))}  "
            f"Player Retention: {_fmt(state.get('player_retention'))}"
        ),
        (
            f"  Dev Engagement: {_fmt(state.get('dev_engagement'))}  "
            f"Sentiment: {_fmt(state.get('sentiment'))}  "
            f"Price/Market: {_fmt(state.get('price_market'))}"
        ),
        "",
        (
            f"ML: {'eligible' if state['ml_eligible'] else 'not eligible'}  "
            f"P(distressed): {_fmt(state.get('p_distressed'))}"
        ),
    ]

    if state.get("forensic_ran"):
        mismatch = state.get("event_state_mismatch")
        parts += [
            "",
            f"Forensic: substance={state.get('update_substance_score')}/10  "
            f"fake_heartbeat={state.get('fake_heartbeat_flag')}  "
            f"momentum={state.get('momentum')}  "
            f"event_content_mismatch={mismatch}",
            f"  {state.get('forensic_reasoning', '')}",
        ]
        if mismatch == 1:
            parts.append(
                "  >> NOTE: announcement type implies a build update, but content "
                "does not support it — activity signal may be misleading."
            )
    else:
        parts += ["", "Forensic: did not run — no announcements in last 60 days"]

    if state.get("auditor_ran"):
        alignment = state.get("sentiment_alignment")
        parts += [
            "",
            f"Sentiment shift: {state.get('sentiment_shift')}  "
            f"alignment_with_l1_state={alignment}",
            f"Key concerns: {'; '.join(state.get('key_concerns') or []) or 'none'}",
            f"Summary: {state.get('auditor_summary', '')}",
        ]
        if alignment == "conflicted":
            parts.append(
                f"  >> NOTE: player sentiment CONFLICTS with "
                f"l1_state={state['l1_state']} "
                f"— see summary above for the specific discrepancy."
            )
    else:
        parts += ["", "Sentiment Auditor: did not run"]

    return "\n".join(parts)


def _get_llm() -> ChatCerebras:
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        raise OSError("CEREBRAS_API_KEY not set")
    return ChatCerebras(model=MODEL_NAME,
                        temperature=0.3,
                        max_tokens=3000,
                        api_key=api_key)


def _llm_call(
        system: str,
        prompt: str,
        config: RunnableConfig
        ) -> tuple[str | None, str | None]:
    llm = _get_llm()
    try:
        response: AIMessage = llm.invoke([SystemMessage(content=system),
                                          HumanMessage(content=prompt)],
                                          config=config)
        return response.content.strip(), None
    except Exception as e:
        return None, f"LLM call failed: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def determine_alignment(state: CriticState) -> dict:
    return {"signal_alignment": compute_signal_alignment(state)}


def write_consumer_verdict(state: CriticState, config: RunnableConfig) -> dict:
    content, error = _llm_call(
        CRITIC_CONSUMER_SYSTEM,
        f"{_context(state)}\n\nWrite the consumer verdict now.",
        config=config
    )
    return {"consumer_verdict": content, "error_msg": error}


def write_developer_brief(state: CriticState, config: RunnableConfig) -> dict:
    if state.get("error_msg"):
        return {}
    content, error = _llm_call(
        CRITIC_DEVELOPER_SYSTEM,
        f"{_context(state)}\n\nWrite the developer brief now.",
        config=config
    )
    return {"developer_brief": content, "error_msg": error}


def add_confidence_note(state: CriticState) -> dict:
    notes = []
    if not state.get("ml_eligible"):
        notes.append("Score based on rule-based scorecard only "
                     "— fewer than 50 reviews.")
    if not state.get("forensic_ran"):
        notes.append("No announcements in last 60 days at snapshot time.")
    if state.get("fake_heartbeat_flag") == 1:
        notes.append("Most recent update flagged as minimal content.")
    if state.get("signal_alignment") == "conflicted":
        notes.append("Independent signals disagree — see verdict for details.")
    return {"confidence_note": " | ".join(notes) if notes else None}


def _build_graph() -> StateGraph:
    g = StateGraph(CriticState)
    g.add_node("determine_alignment", determine_alignment)
    g.add_node("write_consumer_verdict", write_consumer_verdict)
    g.add_node("write_developer_brief", write_developer_brief)
    g.add_node("add_confidence_note", add_confidence_note)
    g.set_entry_point("determine_alignment")
    g.add_edge("determine_alignment", "write_consumer_verdict")
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
    signal_alignment: str | None
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
    fake_heartbeat_flag: int | None = None, momentum: str | None = None,
    event_state_mismatch: int | None = None, forensic_reasoning: str | None = None,
    auditor_ran: bool = False, theme_clusters: list[dict] | None = None,
    sentiment_shift: str | None = None, sentiment_alignment: str | None = None,
    key_concerns: list[str] | None = None,
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
        "fake_heartbeat_flag": fake_heartbeat_flag, "momentum": momentum,
        "event_state_mismatch": event_state_mismatch,
        "forensic_reasoning": forensic_reasoning,
        "auditor_ran": auditor_ran, "theme_clusters": theme_clusters,
        "sentiment_shift": sentiment_shift, "sentiment_alignment": sentiment_alignment,
        "key_concerns": key_concerns,
        "auditor_summary": auditor_summary,
        "signal_alignment": None,
        "consumer_verdict": None, "developer_brief": None,
        "confidence_note": None, "error_msg": None,
    }
    config = {"callbacks": [trace]} if trace else {}
    final = get_graph().invoke(initial, config=config)
    return CriticResult(
        appid=appid, snapshot_date=snapshot_date,
        signal_alignment=final.get("signal_alignment"),
        consumer_verdict=final.get("consumer_verdict"),
        developer_brief=final.get("developer_brief"),
        confidence_note=final.get("confidence_note"),
        error=final.get("error_msg"),
    )
