"""
agents/sentiment_auditor.py
EARLY — Sentiment Auditor (Phase 2, Layer 2)

Clusters recent Steam reviews into thematic signals.
Only runs when ml_eligible = True.

Triangulation addition:
  Given the ML-derived l1_state, does player sentiment AGREE or CONFLICT
  with it? E.g. ML says "Healthy" but reviews say "no real updates in
  months, content drought" → sentiment_alignment="conflicted". This is
  the review-side half of the triangulation the Critic Agent synthesizes.

Model: Groq meta-llama/llama-4-scout-17b-16e-instruct
Tracing: Langfuse generation span
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableConfig

from agents.states import SentimentState, SentimentOutputModel
from agents.prompts import AUDITOR_SYSTEM_PROMPT

MAX_RECENT_REVIEWS = 25
MAX_OLDER_REVIEWS  = 15
MAX_REVIEW_CHARS   = 300

MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

def _fmt_reviews(reviews: list[dict], label: str) -> str:
    if not reviews:
        return f"[No {label} reviews available]"
    lines = []
    for i, r in enumerate(reviews[:MAX_RECENT_REVIEWS], 1):
        text = (r.get("text") or "")[:MAX_REVIEW_CHARS]
        pol  = "👍" if r.get("voted_up") else "👎"
        if text:
            lines.append(f"{i}. {pol} {text}")
    return "\n".join(lines) or f"[No {label} reviews with text]"


def _build_prompt(state: SentimentState) -> str:
    recent = state.get("recent_reviews", [])
    older  = state.get("older_reviews", [])
    l1     = state.get("l1_state") or "unknown"
    return f"""Game: {state['game_name']} (appid {state['appid']})
Snapshot: {state['snapshot_date']} | Reviews: {state['review_count_at_T']}
Overall score: {state['review_score_at_T']:.1%} | Last 90d: {f"{state['review_score_last_90d']:.1%}" if state.get('review_score_last_90d') else 'N/A'}

Current ML classification (l1_state): {l1}
Check whether player sentiment below AGREES or CONFLICTS with "{l1}".

--- RECENT (last 90d, {len(recent)} shown) ---
{_fmt_reviews(recent, 'recent')}

--- OLDER (90-180d, {len(older[:MAX_OLDER_REVIEWS])} shown) ---
{_fmt_reviews(older, 'older')}

Return JSON only."""


def _get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set")
    return ChatGroq(model=MODEL_NAME, temperature=0.0, max_tokens=1024, api_key=api_key)


def check_eligibility(state: SentimentState) -> dict:
    total = len(state.get("recent_reviews", [])) + len(state.get("older_reviews", []))
    if total == 0:
        return {
            "theme_clusters": [], "sentiment_shift": "insufficient_data",
            "sentiment_alignment": "insufficient_data",
            "key_concerns": [], "auditor_summary": "No reviews available.", "error_msg": None,
        }
    return {}


def should_skip(state: SentimentState) -> str:
    return "end" if state.get("auditor_summary") is not None else "analyse"


def analyse_sentiment(state: SentimentState, config: RunnableConfig) -> dict:
    llm    = _get_llm().with_structured_output(SentimentOutputModel, method="json_mode")
    prompt = _build_prompt(state)
    msgs   = [SystemMessage(content=AUDITOR_SYSTEM_PROMPT), HumanMessage(content=prompt)]

    try:
        parsed = llm.invoke(msgs, config=config)
        if not parsed:
            raise ValueError("Model failed to return structured output.")

        alignment = parsed.sentiment_alignment
        if alignment not in ("aligned", "conflicted", "insufficient_data"):
            alignment = "insufficient_data"

        # Safely extract dicts for the graph state (handles Pydantic v1 vs v2)
        clusters = [c.model_dump() if hasattr(c, 'model_dump') else c.dict() for c in parsed.theme_clusters]

        return {
            "theme_clusters": clusters,
            "sentiment_shift": parsed.sentiment_shift or "insufficient_data",
            "sentiment_alignment": alignment,
            "key_concerns": parsed.key_concerns or [],
            "auditor_summary": parsed.auditor_summary or "",
            "error_msg": None,
        }

    except Exception as e:
        return {"theme_clusters": None, "sentiment_shift": None, "sentiment_alignment": None,
                "key_concerns": None, "auditor_summary": None,
                "error_msg": f"LLM call failed: {type(e).__name__}: {e}"}


def _build_graph() -> StateGraph:
    g = StateGraph(SentimentState)
    g.add_node("check_eligibility", check_eligibility)
    g.add_node("analyse_sentiment", analyse_sentiment)
    g.set_entry_point("check_eligibility")
    g.add_conditional_edges("check_eligibility", should_skip, {"end": END, "analyse": "analyse_sentiment"})
    g.add_edge("analyse_sentiment", END)
    return g

_compiled_graph = None
def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_graph().compile()
    return _compiled_graph


@dataclass
class SentimentResult:
    appid: int
    snapshot_date: str
    theme_clusters: list[dict] | None
    sentiment_shift: str | None
    sentiment_alignment: str | None
    key_concerns: list[str] | None
    auditor_summary: str | None
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None and self.auditor_summary is not None


def run_sentiment_auditor(
    appid: int, game_name: str, snapshot_date: str,
    review_score_at_T: float, review_score_last_90d: float | None,
    review_count_at_T: int, recent_reviews: list[dict], older_reviews: list[dict],
    l1_state: str | None = None,
    trace: Any | None = None,
) -> SentimentResult:
    initial: SentimentState = {
        "messages": [], "appid": appid, "game_name": game_name,
        "snapshot_date": snapshot_date, "review_score_at_T": review_score_at_T,
        "review_score_last_90d": review_score_last_90d,
        "review_count_at_T": review_count_at_T,
        "recent_reviews": recent_reviews or [], "older_reviews": older_reviews or [],
        "l1_state": l1_state,
        "theme_clusters": None, "sentiment_shift": None,
        "sentiment_alignment": None,
        "key_concerns": None, "auditor_summary": None, "error_msg": None,
    }
    config = {"callbacks": [trace]} if trace else {}
    final = get_graph().invoke(initial, config=config)
    return SentimentResult(
        appid=appid, snapshot_date=snapshot_date,
        theme_clusters=final.get("theme_clusters"),
        sentiment_shift=final.get("sentiment_shift"),
        sentiment_alignment=final.get("sentiment_alignment"),
        key_concerns=final.get("key_concerns"),
        auditor_summary=final.get("auditor_summary"),
        error=final.get("error_msg"),
    )