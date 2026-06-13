"""
agents/forensic_agent.py
EARLY — Forensic Agent (Phase 2, Layer 2)

Analyzes the most recent build update (type 12/13) within 30 days of snapshot
to produce:
  - update_substance_score  (0–10)
  - fake_heartbeat_flag     (0/1)
  - reasoning               (str)

Model: Groq llama-3.3-70b-versatile
Tracing: Langfuse generation span (optional, no-op if disabled)
"""

from __future__ import annotations

import json
import os
import re
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from utils.langfuse_client import generation_span


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ForensicState(TypedDict):
    messages: Annotated[list, add_messages]
    appid: int
    game_name: str
    snapshot_date: str
    event_type: int
    announcement_title: str
    announcement_body_stripped: str
    word_count: int
    ea_age_days: int
    days_since_last_build_update: int
    trace: Any | None
    update_substance_score: float | None
    fake_heartbeat_flag: int | None
    reasoning: str | None
    error: str | None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the Forensic Agent for EARLY, a system that predicts
whether Steam Early Access games will be abandoned before reaching 1.0 release.

Your job is to assess the *substance* of a single build update announcement.
You are NOT judging writing quality or marketing language. You are auditing
whether the update contains real, meaningful development work.

SCORING RUBRIC — update_substance_score (0 to 10):
  0–2 : Empty heartbeat. No concrete content.
  3–4 : Minimal. One vague fix or non-specific mention.
  5–6 : Moderate. A few concrete changes, but shallow.
  7–8 : Solid. Multiple specific, concrete changes with detail.
  9–10: Exceptional. Detailed, comprehensive, technically specific.

FAKE HEARTBEAT FLAG:
  1 if the update is a deliberate minimal post to reset Steam visibility with
    no real development substance. 0 otherwise.

RULES:
- Score on content quality and specificity, NOT word count alone.
- Do not penalize non-English changelogs.
- Type 12 = major build, type 13 = minor build.
- Empty body after stripping: score = 0, fake_heartbeat_flag = 1.

OUTPUT FORMAT — JSON only, no markdown fences:
{
  "update_substance_score": <float 0.0–10.0>,
  "fake_heartbeat_flag": <0 or 1>,
  "reasoning": "<1–2 sentences>"
}"""


def _build_user_prompt(state: ForensicState) -> str:
    event_label = "Major build update (type 12)" if state["event_type"] == 12 else "Minor build update (type 13)"
    body = state["announcement_body_stripped"].strip() or "[empty body]"
    return f"""Game: {state['game_name']} (appid {state['appid']})
Snapshot date: {state['snapshot_date']}
EA age at snapshot: {state['ea_age_days']} days
Days since last build update: {state['days_since_last_build_update']}
Update type: {event_label}
Word count (pre-computed): {state['word_count']}

--- ANNOUNCEMENT TITLE ---
{state['announcement_title']}

--- ANNOUNCEMENT BODY (BBCode stripped) ---
{body}
---

Assess this update and return JSON only."""


def _get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set")
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, max_tokens=256, api_key=api_key)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def assess_update(state: ForensicState) -> dict:
    body = state.get("announcement_body_stripped", "").strip()
    wc   = state.get("word_count", 0)

    if not body and wc == 0:
        return {
            "update_substance_score": 0.0,
            "fake_heartbeat_flag": 1,
            "reasoning": "Empty body after BBCode strip — no content to assess.",
            "error": None,
        }

    llm    = _get_llm()
    prompt = _build_user_prompt(state)
    messages_in = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]

    # Langfuse span — graceful no-op if unavailable
    trace = state.get("trace")
    try:
        ctx = generation_span(trace, name="forensic_llm", model="llama-3.3-70b-versatile", input_data=prompt)
    except Exception:
        ctx = nullcontext(None)

    try:
        with ctx as span:
            response: AIMessage = llm.invoke(messages_in)
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.content.strip(), flags=re.MULTILINE).strip()
            parsed = json.loads(raw)

            if span and hasattr(span, "set_output"):
                span.set_output(raw)
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    span.set_usage(input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"))

            score  = max(0.0, min(10.0, float(parsed["update_substance_score"])))
            return {
                "update_substance_score": score,
                "fake_heartbeat_flag": int(bool(parsed["fake_heartbeat_flag"])),
                "reasoning": str(parsed.get("reasoning", "")),
                "error": None,
                "messages": [response],
            }

    except json.JSONDecodeError as e:
        return {"update_substance_score": None, "fake_heartbeat_flag": None, "reasoning": None,
                "error": f"JSON parse error: {e}"}
    except Exception as e:
        return {"update_substance_score": None, "fake_heartbeat_flag": None, "reasoning": None,
                "error": f"LLM call failed: {type(e).__name__}: {e}"}


def validate_output(state: ForensicState) -> dict:
    if state.get("error"):
        return {}
    score = state["update_substance_score"]
    flag  = state["fake_heartbeat_flag"]
    if score is not None and score < 4.0 and state.get("word_count", 0) < 20:
        flag = 1
    return {"fake_heartbeat_flag": flag}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def _build_graph() -> StateGraph:
    g = StateGraph(ForensicState)
    g.add_node("assess_update", assess_update)
    g.add_node("validate_output", validate_output)
    g.set_entry_point("assess_update")
    g.add_edge("assess_update", "validate_output")
    g.add_edge("validate_output", END)
    return g

_compiled_graph = None
def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_graph().compile()
    return _compiled_graph


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ForensicResult:
    appid: int
    snapshot_date: str
    update_substance_score: float | None
    fake_heartbeat_flag: int | None
    reasoning: str | None
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None and self.update_substance_score is not None


def run_forensic_agent(
    appid: int, game_name: str, snapshot_date: str, event_type: int,
    announcement_title: str, announcement_body_stripped: str, word_count: int,
    ea_age_days: int, days_since_last_build_update: int, trace: Any | None = None,
) -> ForensicResult:
    initial: ForensicState = {
        "messages": [], "appid": appid, "game_name": game_name,
        "snapshot_date": snapshot_date, "event_type": event_type,
        "announcement_title": announcement_title,
        "announcement_body_stripped": announcement_body_stripped,
        "word_count": word_count, "ea_age_days": ea_age_days,
        "days_since_last_build_update": days_since_last_build_update,
        "trace": trace, "update_substance_score": None,
        "fake_heartbeat_flag": None, "reasoning": None, "error": None,
    }
    final = get_graph().invoke(initial)
    return ForensicResult(
        appid=appid, snapshot_date=snapshot_date,
        update_substance_score=final.get("update_substance_score"),
        fake_heartbeat_flag=final.get("fake_heartbeat_flag"),
        reasoning=final.get("reasoning"), error=final.get("error"),
    )