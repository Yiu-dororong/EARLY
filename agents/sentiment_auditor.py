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

Model: Groq llama-3.1-8b-instant
Tracing: Langfuse generation span
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

MAX_RECENT_REVIEWS = 25
MAX_OLDER_REVIEWS  = 15
MAX_REVIEW_CHARS   = 300


class SentimentState(TypedDict):
    messages: Annotated[list, add_messages]
    appid: int
    game_name: str
    snapshot_date: str
    review_score_at_T: float
    review_score_last_90d: float | None
    review_count_at_T: int
    recent_reviews: list[dict]
    older_reviews: list[dict]
    l1_state: str | None              # NEW — for triangulation
    trace: Any | None
    theme_clusters: list[dict] | None
    sentiment_shift: str | None
    sentiment_alignment: str | None   # NEW
    key_concerns: list[str] | None
    auditor_summary: str | None
    error: str | None


SYSTEM_PROMPT = """You are the Sentiment Auditor for EARLY, a Steam Early Access
health prediction system. Analyze player reviews and identify structural sentiment
patterns for two consumers: the Critic Agent and the developer dashboard.

You will be told the ML model's current health classification (l1_state) for
this game: "Healthy", "Watch", or "At Risk". Your job includes checking whether
player sentiment AGREES with that classification or CONTRADICTS it.

Produce:
1. theme_clusters (3-6): [{
     "theme": str, "valence": "positive"|"negative"|"mixed",
     "frequency": "high"|"medium"|"low",
     "representative_quote": "<verbatim fragment under 60 chars or null>"
   }]
2. sentiment_shift: "improving"|"declining"|"stable"|"mixed"|"insufficient_data"
3. sentiment_alignment: does review sentiment AGREE with the stated l1_state?
   "aligned"     — reviews are consistent with l1_state (e.g. l1_state=Healthy
                   and reviews are generally positive/neutral about development)
   "conflicted"  — reviews materially CONTRADICT l1_state (e.g. l1_state=Healthy
                   but reviews describe abandonment, no updates, dev silence —
                   OR l1_state=At Risk but reviews describe an actively engaged,
                   responsive developer)
   "insufficient_data" — too few reviews to judge
4. key_concerns: up to 3 plain-English developer pain points (each under 15 words)
5. auditor_summary: 2-3 sentences for the Critic Agent. If sentiment_alignment is
   "conflicted", explicitly state the conflict (what l1_state implies vs what
   reviews say) — this is the most important thing to surface.

representative_quote must come verbatim from provided reviews (under 60 chars) or be null.
If fewer than 5 total reviews: return empty clusters, insufficient_data for both
sentiment_shift and sentiment_alignment, and note sparsity.

OUTPUT FORMAT — JSON only, no markdown fences:
{"theme_clusters": [...], "sentiment_shift": "...", "sentiment_alignment": "...",
 "key_concerns": [...], "auditor_summary": "..."}"""


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
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.0, max_tokens=1024, api_key=api_key)


def check_eligibility(state: SentimentState) -> dict:
    total = len(state.get("recent_reviews", [])) + len(state.get("older_reviews", []))
    if total == 0:
        return {
            "theme_clusters": [], "sentiment_shift": "insufficient_data",
            "sentiment_alignment": "insufficient_data",
            "key_concerns": [], "auditor_summary": "No reviews available.", "error": None,
        }
    return {}


def should_skip(state: SentimentState) -> str:
    return "end" if state.get("auditor_summary") is not None else "analyse"


def analyse_sentiment(state: SentimentState) -> dict:
    llm    = _get_llm()
    prompt = _build_prompt(state)
    msgs   = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]

    trace = state.get("trace")
    try:
        from utils.langfuse_client import generation_span
        ctx = generation_span(trace, name="auditor_llm", model="llama-3.1-8b-instant", input_data=prompt)
        if ctx is None:
            ctx = nullcontext(None)
    except Exception:
        ctx = nullcontext(None)

    try:
        with ctx as span:
            response: AIMessage = llm.invoke(msgs)
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.content.strip(), flags=re.MULTILINE).strip()
            parsed = json.loads(raw)

            if span and hasattr(span, "set_output"):
                span.set_output(raw)
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    span.set_usage(input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"))

            alignment = parsed.get("sentiment_alignment", "insufficient_data")
            if alignment not in ("aligned", "conflicted", "insufficient_data"):
                alignment = "insufficient_data"

            return {
                "theme_clusters": parsed.get("theme_clusters", []),
                "sentiment_shift": parsed.get("sentiment_shift", "insufficient_data"),
                "sentiment_alignment": alignment,
                "key_concerns": parsed.get("key_concerns", []),
                "auditor_summary": parsed.get("auditor_summary", ""),
                "error": None, "messages": [response],
            }

    except json.JSONDecodeError as e:
        return {"theme_clusters": None, "sentiment_shift": None, "sentiment_alignment": None,
                "key_concerns": None, "auditor_summary": None, "error": f"JSON parse error: {e}"}
    except Exception as e:
        return {"theme_clusters": None, "sentiment_shift": None, "sentiment_alignment": None,
                "key_concerns": None, "auditor_summary": None,
                "error": f"LLM call failed: {type(e).__name__}: {e}"}


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
        "trace": trace, "theme_clusters": None, "sentiment_shift": None,
        "sentiment_alignment": None,
        "key_concerns": None, "auditor_summary": None, "error": None,
    }
    final = get_graph().invoke(initial)
    return SentimentResult(
        appid=appid, snapshot_date=snapshot_date,
        theme_clusters=final.get("theme_clusters"),
        sentiment_shift=final.get("sentiment_shift"),
        sentiment_alignment=final.get("sentiment_alignment"),
        key_concerns=final.get("key_concerns"),
        auditor_summary=final.get("auditor_summary"),
        error=final.get("error"),
    )