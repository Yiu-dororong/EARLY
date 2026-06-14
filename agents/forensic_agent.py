"""
agents/forensic_agent.py
EARLY — Forensic Agent (Phase 2, Layer 2)

Analyzes the last 3 announcements within 60 days of snapshot to produce:
  - update_substance_score  (0–10)  — substance of the MOST RECENT post
  - fake_heartbeat_flag     (0/1)   — most recent post is a hollow heartbeat
  - momentum                (str)   — pattern across all posts in the window
  - event_state_mismatch    (0/1)   — TRIANGULATION: does the post content
                                       contradict what the event type/recency
                                       implies about development activity?
  - reasoning               (str)

Triangulation note:
  Steam event types (12/13/14) are announcement categories, not proof a
  build shipped. A game can post "Major Update" announcements with zero
  development content (see: "Never Mourn" case). event_state_mismatch=1
  flags exactly this — the ML model sees "recent event → looks active",
  but the actual content says otherwise. This is the signal the Critic
  Agent uses to override or soften the ML-derived l1_state.

Model: Groq llama-3.3-70b-versatile
Tracing: Langfuse generation span (optional, no-op if disabled)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig

# Lookback window — see design note in module docstring
MAX_EVENTS_CONSIDERED = 3
LOOKBACK_DAYS         = 60
MAX_BODY_CHARS        = 600   # per-event truncation to bound total prompt size


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AnnouncementInput(TypedDict):
    event_type: int
    title: str
    body_stripped: str
    word_count: int
    days_ago: int


class ForensicState(TypedDict):
    messages: Annotated[list, add_messages]
    appid: int
    game_name: str
    snapshot_date: str
    ea_age_days: int
    days_since_last_build_update: int
    announcements: list[AnnouncementInput]   # most recent first, up to MAX_EVENTS_CONSIDERED
    update_substance_score: float | None
    fake_heartbeat_flag: int | None
    momentum: str | None
    event_state_mismatch: int | None
    reasoning: str | None
    error: str | None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the Forensic Agent for EARLY, a system that predicts
whether Steam Early Access games will be abandoned before reaching 1.0 release.

You will be shown the last few announcements posted by a developer, most
recent first. Your job has TWO parts:

PART 1 — SUBSTANCE of the MOST RECENT announcement (index 1):
  update_substance_score (0 to 10):
    0–2 : Empty heartbeat. No concrete content.
    3–4 : Minimal. One vague fix or non-specific mention.
    5–6 : Moderate. A few concrete changes, but shallow.
    7–8 : Solid. Multiple specific, concrete changes with detail.
    9–10: Exceptional. Detailed, comprehensive, technically specific.

  fake_heartbeat_flag: 1 if announcement #1 is a deliberate minimal post with
  no real development substance (resets visibility, says nothing). 0 otherwise.

PART 2 — MOMENTUM across ALL announcements shown:
  momentum: one of
    "consistent_progress" — multiple posts each with real content, suggests
                            active iteration (even if individually small —
                            several hotfixes in sequence is a GOOD sign)
    "single_update"       — only one post available, can't assess pattern
    "declining"           — earlier posts had substance, recent ones don't
    "hollow_pattern"       — most/all posts in the window are low-substance
                            announcements regardless of type

PART 3 — TRIANGULATION (event_state_mismatch):
  Steam event types (12=minor build, 13=regular update, 14=major update) are
  ANNOUNCEMENT CATEGORIES — they do NOT prove a build was actually shipped.
  A developer can post a "Major Update" announcement that is pure marketing
  with zero development content.

  event_state_mismatch = 1 if:
    - The event type suggests a build update (12/13/14) BUT the body content
      contains no actual build/patch evidence (no version numbers, no
      changelog, no specific technical changes) — i.e. the announcement
      TYPE implies activity that the CONTENT does not support.
  event_state_mismatch = 0 if:
    - The content is consistent with its event type (a build-type post that
      actually describes build changes), OR
    - The post is honestly framed as non-build news (community update,
      roadmap discussion) without claiming to be a build.

RULES:
- Score on content quality and specificity, NOT word count alone.
- Do not penalize non-English changelogs — score what you can read.
- Empty body after stripping: score = 0, fake_heartbeat_flag = 1.
- If only 1 announcement provided: momentum = "single_update".

OUTPUT FORMAT — JSON only, no markdown fences:
{
  "update_substance_score": <float 0.0-10.0>,
  "fake_heartbeat_flag": <0 or 1>,
  "momentum": "<consistent_progress|single_update|declining|hollow_pattern>",
  "event_state_mismatch": <0 or 1>,
  "reasoning": "<2-3 sentences covering substance, momentum, and mismatch>"
}"""


def _event_label(event_type: int) -> str:
    return {12: "Minor build update", 13: "Regular update", 14: "Major update"}.get(
        event_type, f"Unknown type {event_type}"
    )


def _build_user_prompt(state: ForensicState) -> str:
    parts = [
        f"Game: {state['game_name']} (appid {state['appid']})",
        f"Snapshot date: {state['snapshot_date']}",
        f"EA age at snapshot: {state['ea_age_days']} days",
        f"Days since last build-type event: {state['days_since_last_build_update']}",
        "",
        f"Showing {len(state['announcements'])} most recent announcements "
        f"(within {LOOKBACK_DAYS} days), most recent first:",
    ]

    for i, ann in enumerate(state["announcements"], 1):
        body = (ann["body_stripped"] or "").strip()[:MAX_BODY_CHARS] or "[empty body]"
        parts += [
            "",
            f"--- ANNOUNCEMENT #{i} ({ann['days_ago']} days ago) ---",
            f"Type: {_event_label(ann['event_type'])} (type {ann['event_type']})",
            f"Word count: {ann['word_count']}",
            f"Title: {ann['title']}",
            f"Body: {body}",
        ]

    parts += ["", "---", "Assess and return JSON only."]
    return "\n".join(parts)


def _get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set")
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, max_tokens=350, api_key=api_key)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def assess_updates(state: ForensicState, config: RunnableConfig) -> dict:
    announcements = state.get("announcements", [])

    # Fast-path: no announcements at all (shouldn't normally reach here —
    # orchestrator gates on at least one event existing)
    if not announcements:
        return {
            "update_substance_score": 0.0,
            "fake_heartbeat_flag": 1,
            "momentum": "hollow_pattern",
            "event_state_mismatch": 0,
            "reasoning": "No announcements found in the lookback window.",
            "error": None,
        }

    # Fast-path: single announcement with empty body
    if len(announcements) == 1:
        body = (announcements[0]["body_stripped"] or "").strip()
        wc   = announcements[0]["word_count"]
        if not body and wc == 0:
            mismatch = 1 if announcements[0]["event_type"] in (12, 13, 14) else 0
            return {
                "update_substance_score": 0.0,
                "fake_heartbeat_flag": 1,
                "momentum": "single_update",
                "event_state_mismatch": mismatch,
                "reasoning": "Empty body after BBCode strip — no content to assess.",
                "error": None,
            }

    llm    = _get_llm()
    prompt = _build_user_prompt(state)
    messages_in = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]

    try:
        response: AIMessage = llm.invoke(messages_in, config=config)
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.content.strip(), flags=re.MULTILINE).strip()
        parsed = json.loads(raw)

        score = max(0.0, min(10.0, float(parsed["update_substance_score"])))
        momentum = parsed.get("momentum", "single_update")
        if momentum not in ("consistent_progress", "single_update", "declining", "hollow_pattern"):
            momentum = "single_update"

        return {
            "update_substance_score": score,
            "fake_heartbeat_flag": int(bool(parsed["fake_heartbeat_flag"])),
            "momentum": momentum,
            "event_state_mismatch": int(bool(parsed.get("event_state_mismatch", 0))),
            "reasoning": str(parsed.get("reasoning", "")),
            "error": None,
            "messages": [response],
        }

    except json.JSONDecodeError as e:
        return {"update_substance_score": None, "fake_heartbeat_flag": None, "momentum": None,
                "event_state_mismatch": None, "reasoning": None, "error": f"JSON parse error: {e}"}
    except Exception as e:
        return {"update_substance_score": None, "fake_heartbeat_flag": None, "momentum": None,
                "event_state_mismatch": None, "reasoning": None,
                "error": f"LLM call failed: {type(e).__name__}: {e}"}


def validate_output(state: ForensicState) -> dict:
    if state.get("error"):
        return {}
    score = state["update_substance_score"]
    flag  = state["fake_heartbeat_flag"]
    most_recent = state["announcements"][0] if state.get("announcements") else None

    # Secondary heuristic: very low score + very short most-recent post → force flag
    if most_recent and score is not None and score < 4.0 and most_recent["word_count"] < 20:
        flag = 1

    return {"fake_heartbeat_flag": flag}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def _build_graph() -> StateGraph:
    g = StateGraph(ForensicState)
    g.add_node("assess_updates", assess_updates)
    g.add_node("validate_output", validate_output)
    g.set_entry_point("assess_updates")
    g.add_edge("assess_updates", "validate_output")
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
    momentum: str | None
    event_state_mismatch: int | None
    reasoning: str | None
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None and self.update_substance_score is not None


def run_forensic_agent(
    appid: int,
    game_name: str,
    snapshot_date: str,
    ea_age_days: int,
    days_since_last_build_update: int,
    announcements: list[AnnouncementInput],
    trace: Any | None = None,
) -> ForensicResult:
    """
    announcements: most recent first, already filtered to the last
    MAX_EVENTS_CONSIDERED events within LOOKBACK_DAYS by the orchestrator.
    """
    initial: ForensicState = {
        "messages": [], "appid": appid, "game_name": game_name,
        "snapshot_date": snapshot_date, "ea_age_days": ea_age_days,
        "days_since_last_build_update": days_since_last_build_update,
        "announcements": announcements[:MAX_EVENTS_CONSIDERED],
        "update_substance_score": None, "fake_heartbeat_flag": None,
        "momentum": None, "event_state_mismatch": None,
        "reasoning": None, "error": None,
    }
    config = {"callbacks": [trace]} if trace else {}
    final = get_graph().invoke(initial, config=config)
    return ForensicResult(
        appid=appid, snapshot_date=snapshot_date,
        update_substance_score=final.get("update_substance_score"),
        fake_heartbeat_flag=final.get("fake_heartbeat_flag"),
        momentum=final.get("momentum"),
        event_state_mismatch=final.get("event_state_mismatch"),
        reasoning=final.get("reasoning"), error=final.get("error"),
    )