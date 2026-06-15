"""
api/models.py
-------------
Pydantic response models for all API endpoints.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, field_validator


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class DimensionScores(BaseModel):
    update_health:    float | None
    player_retention: float | None
    dev_engagement:   float | None
    sentiment:        float | None
    price_market:     float | None


# ---------------------------------------------------------------------------
# /games  (list)
# ---------------------------------------------------------------------------

class GameSummary(BaseModel):
    appid:                          int
    name:                           str | None
    ea_start_date:                  str | None
    ea_age_days:                    int | None
    l1_state:                       str | None
    p_distressed:                   float | None
    is_distressed:                  int | None
    ml_eligible:                    int | None
    review_count_at_T:              int | None
    snap_date:                      str | None
    outcome:                        str | None
    days_since_last_build_update:   int | None = None


class GameListResponse(BaseModel):
    total:  int
    offset: int
    limit:  int
    items:  list[GameSummary]

# ---------------------------------------------------------------------------
# /games/{appid}/score
# ---------------------------------------------------------------------------

class GameScore(BaseModel):
    appid:             int
    name:              str | None
    ea_start_date:     str | None
    ea_age_days:       int | None
    primary_genre:     str | None
    l1_state:          str | None
    p_distressed:      float | None
    is_distressed:     int | None
    ml_eligible:       int | None
    model_version:     str | None
    snap_date:         str | None
    review_count_at_T: int | None
    null_features:     list[str]
    data_quality:      str          # "high" | "medium" | "low" — based on null feature count
    dimensions:        DimensionScores | None
    outcome:           str | None
    currently_in_ea:   int | None
    days_since_last_build_update: int | None = None

    @field_validator("null_features", mode="before")
    @classmethod
    def parse_null_features(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (ValueError, TypeError):
                return []
        return v

    @field_validator("data_quality", mode="before")
    @classmethod
    def compute_data_quality(cls, v: Any, info: Any) -> str:
        # If explicitly provided, use it; otherwise derive from null_features
        if isinstance(v, str) and v in ("high", "medium", "low"):
            return v
        # Derive from null_features if available in model data
        # Fallback: accept int null count directly during construction
        if isinstance(v, int):
            return "high" if v <= 5 else "medium" if v <= 15 else "low"
        return "medium"  # safe default


# ---------------------------------------------------------------------------
# /games/{appid}/history
# ---------------------------------------------------------------------------

class ScoreSnapshot(BaseModel):
    snap_date:         str | None
    l1_state:          str | None
    p_distressed:      float | None
    is_distressed:     int | None
    ea_age_days:       int | None
    review_count_at_T: int | None
    null_features:     list[str]
    dimensions:        DimensionScores | None
    days_since_last_build_update: int | None = None

    @field_validator("null_features", mode="before")
    @classmethod
    def parse_null_features(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (ValueError, TypeError):
                return []
        return v


class ScoreHistoryResponse(BaseModel):
    appid:     int
    name:      str | None
    snapshots: list[ScoreSnapshot]


# ---------------------------------------------------------------------------
# /games/{appid}/features  (live_snapshots table)
# ---------------------------------------------------------------------------

class GameFeatures(BaseModel):
    appid:             int
    name:              str | None
    snap_date:         str | None
    ea_age_days:       int | None
    primary_genre:     str | None
    review_count_at_T: int | None
    features:          dict[str, float | int | str | None]
    shap_values:       dict[str, float] | None   # top-N SHAP values, None if not computed

    @field_validator("features", mode="before")
    @classmethod
    def parse_features(cls, v: Any) -> dict:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (ValueError, TypeError):
                return {}
        return v or {}

    @field_validator("shap_values", mode="before")
    @classmethod
    def parse_shap(cls, v: Any) -> dict | None:
        if v is None:
            return None
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (ValueError, TypeError):
                return None
        return v


# ---------------------------------------------------------------------------
# /games/{appid}/analysis  (agent_analysis table)
# ---------------------------------------------------------------------------

class ForensicOutput(BaseModel):
    ran:                    bool
    update_substance_score: float | None
    fake_heartbeat_flag:    int | None
    momentum:               str | None   # NEW
    event_state_mismatch:   int | None   # NEW
    reasoning:              str | None


class AuditorOutput(BaseModel):
    ran:                  bool
    sentiment_shift:      str | None
    sentiment_alignment:  str | None   # NEW
    key_concerns:         list[str] | None
    theme_clusters:       list[dict] | None
    summary:              str | None

    @field_validator("key_concerns", mode="before")
    @classmethod
    def parse_key_concerns(cls, v: Any) -> list[str] | None:
        if v is None:
            return None
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (ValueError, TypeError):
                return None
        return v

    @field_validator("theme_clusters", mode="before")
    @classmethod
    def parse_theme_clusters(cls, v: Any) -> list[dict] | None:
        if v is None:
            return None
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (ValueError, TypeError):
                return None
        return v


class CriticOutput(BaseModel):
    ran:              bool
    consumer_verdict: str | None
    developer_brief:  str | None
    confidence_note:  str | None


class AgentAnalysisResponse(BaseModel):
    appid:           int
    name:            str | None
    snapshot_date:   str | None
    analysed_at:     int | None         # Unix ts, None if never analysed
    trigger_reason:  str | None
    status:          str                # "ready" | "pending" | "not_eligible" | "never_run" | "error"
    signal_alignment: str | None   # NEW — top-level triangulation verdict
    forensic:        ForensicOutput | None
    auditor:         AuditorOutput | None
    critic:          CriticOutput | None
    error:           str | None


class AnalysisTriggerResponse(BaseModel):
    appid:   int
    status:  str       # "queued" | "not_eligible" | "already_running"
    message: str


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class PipelineHealth(BaseModel):
    status:                 str           # "ok" | "stale" | "empty"
    last_scored_at:         int | None
    snapshot_date:          str | None = None
    games_scored_this_week: int
    games_total:            int
    at_risk_count:          int
    watch_count:            int
    healthy_count:          int
    null_rate_warning:      list[str]


# ---------------------------------------------------------------------------
# POST /search/similar
# ---------------------------------------------------------------------------

class SimilarGame(BaseModel):
    appid:               int
    name:                str | None
    snapshot_date:       str
    ea_age_days:         int
    primary_genre:       str
    l1_state:            str
    outcome:             str            # "SUCCESS" | "ABANDONED"
    p_distressed:        float
    distance:            float          # Cosine similarity from query vector (higher is closer)
    match_quality:       str            # "high" | "medium" | "low"
    null_feature_count:  int            # how many SHAP features were imputed


class SimilaritySearchResponse(BaseModel):
    query_appid:    int
    query_snap_date: str | None
    results:        list[SimilarGame]
    message:        str | None          # populated when results are empty