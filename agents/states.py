"""
agents/states.py
EARLY — Agent state typed dicts and Pydantic output models.
"""

from typing import Annotated, TypedDict
from pydantic import BaseModel
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# Forensic Agent
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
    error_msg: str | None


class ForensicOutputModel(BaseModel):
    update_substance_score: float
    fake_heartbeat_flag: int
    momentum: str
    event_state_mismatch: int
    reasoning: str


# ---------------------------------------------------------------------------
# Sentiment Auditor
# ---------------------------------------------------------------------------

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
    theme_clusters: list[dict] | None
    sentiment_shift: str | None
    sentiment_alignment: str | None   # NEW
    key_concerns: list[str] | None
    auditor_summary: str | None
    error_msg: str | None


class ThemeClusterModel(BaseModel):
    theme: str
    valence: str
    frequency: str
    representative_quote: str | None
    quote_translation: str | None


class SentimentOutputModel(BaseModel):
    theme_clusters: list[ThemeClusterModel]
    sentiment_shift: str
    sentiment_alignment: str
    key_concerns: list[str]
    auditor_summary: str


# ---------------------------------------------------------------------------
# Critic Agent
# ---------------------------------------------------------------------------

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
    # Forensic
    forensic_ran: bool
    update_substance_score: float | None
    fake_heartbeat_flag: int | None
    momentum: str | None
    event_state_mismatch: int | None
    forensic_reasoning: str | None
    # Auditor
    auditor_ran: bool
    theme_clusters: list[dict] | None
    sentiment_shift: str | None
    sentiment_alignment: str | None
    key_concerns: list[str] | None
    auditor_summary: str | None
    # Triangulation output
    signal_alignment: str | None
    # Verdicts
    consumer_verdict: str | None
    developer_brief: str | None
    confidence_note: str | None
    error_msg: str | None