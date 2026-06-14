"""
agents/orchestrator.py
EARLY — Phase 2 Agent Orchestrator

Coordinates Forensic Agent, Sentiment Auditor, Critic Agent.
Creates a top-level Langfuse trace per run_analysis() call and passes
it down to each agent as a nested span.

Trigger condition: l1_state in ("Watch", "At Risk")

Forensic input: last 3 announcements (any event type) within 60 days —
see agents/forensic_agent.py module docstring for rationale.
Auditor receives l1_state for sentiment_alignment triangulation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from agents.forensic_agent import (
    AnnouncementInput,
    ForensicResult,
    LOOKBACK_DAYS,
    MAX_EVENTS_CONSIDERED,
    run_forensic_agent,
)
from agents.sentiment_auditor import SentimentResult, run_sentiment_auditor
from agents.critic_agent import CriticResult, run_critic_agent

logger = logging.getLogger(__name__)


@dataclass
class ScorecardResult:
    l1_state: str
    composite_score: float
    update_health: float | None = None
    player_retention: float | None = None
    dev_engagement: float | None = None
    sentiment: float | None = None
    price_market: float | None = None


@dataclass
class XGBoostResult:
    ml_eligible: bool
    p_distressed: float | None = None
    is_distressed: int | None = None


@dataclass
class AnnouncementEvent:
    """Raw announcement from event_history — any event type."""
    event_type: int
    title: str
    body_stripped: str
    word_count: int
    posted_at: date


@dataclass
class GameContext:
    appid: int
    game_name: str
    snapshot_date: date
    ea_age_days: int
    scorecard: ScorecardResult
    xgboost: XGBoostResult
    recent_announcements: list[AnnouncementEvent] = field(default_factory=list)
    days_since_last_build_update: int = 9999
    recent_reviews: list[dict] = field(default_factory=list)
    older_reviews: list[dict] = field(default_factory=list)
    review_score_at_T: float = 0.0
    review_score_last_90d: float | None = None
    review_count_at_T: int = 0
    session_id: str | None = None   # passed from API for Langfuse correlation


def should_run_phase2(scorecard: ScorecardResult) -> bool:
    return scorecard.l1_state in ("Watch", "At Risk")


def _select_announcements(
    events: list[AnnouncementEvent],
    snapshot_date: date,
) -> list[AnnouncementInput]:
    """
    Select up to MAX_EVENTS_CONSIDERED announcements (any event type),
    within LOOKBACK_DAYS of snapshot_date, most recent first.
    """
    cutoff = snapshot_date - timedelta(days=LOOKBACK_DAYS)
    in_window = [e for e in events if e.posted_at >= cutoff]
    in_window.sort(key=lambda e: e.posted_at, reverse=True)

    selected = []
    for e in in_window[:MAX_EVENTS_CONSIDERED]:
        selected.append(AnnouncementInput(
            event_type=e.event_type,
            title=e.title,
            body_stripped=e.body_stripped,
            word_count=e.word_count,
            days_ago=(snapshot_date - e.posted_at).days,
        ))
    return selected


@dataclass
class AnalysisResult:
    appid: int
    snapshot_date: str
    phase2_triggered: bool
    forensic_ran: bool
    auditor_ran: bool
    critic_ran: bool
    forensic: ForensicResult | None = None
    auditor: SentimentResult | None = None
    critic: CriticResult | None = None

    @property
    def update_substance_score(self) -> float | None:
        return self.forensic.update_substance_score if self.forensic else None
    @property
    def fake_heartbeat_flag(self) -> int | None:
        return self.forensic.fake_heartbeat_flag if self.forensic else None
    @property
    def event_state_mismatch(self) -> int | None:
        return self.forensic.event_state_mismatch if self.forensic else None
    @property
    def signal_alignment(self) -> str | None:
        return self.critic.signal_alignment if self.critic else None
    @property
    def consumer_verdict(self) -> str | None:
        return self.critic.consumer_verdict if self.critic else None
    @property
    def developer_brief(self) -> str | None:
        return self.critic.developer_brief if self.critic else None
    @property
    def confidence_note(self) -> str | None:
        return self.critic.confidence_note if self.critic else None
    @property
    def sentiment_shift(self) -> str | None:
        return self.auditor.sentiment_shift if self.auditor else None
    @property
    def sentiment_alignment(self) -> str | None:
        return self.auditor.sentiment_alignment if self.auditor else None
    @property
    def key_concerns(self) -> list[str] | None:
        return self.auditor.key_concerns if self.auditor else None
    @property
    def success(self) -> bool:
        if self.forensic and self.forensic.error: return False
        if self.auditor  and self.auditor.error:  return False
        if self.critic   and self.critic.error:   return False
        return True


def run_analysis(ctx: GameContext) -> AnalysisResult:
    snap_date_str = ctx.snapshot_date.isoformat()

    result = AnalysisResult(
        appid=ctx.appid, snapshot_date=snap_date_str,
        phase2_triggered=False, forensic_ran=False, auditor_ran=False, critic_ran=False,
    )

    if not should_run_phase2(ctx.scorecard):
        logger.info("Phase 2 not triggered for appid=%d (state=%s)", ctx.appid, ctx.scorecard.l1_state)
        return result

    result.phase2_triggered = True

    # --- Top-level Langfuse trace ---
    trace = None
    try:
        from utils.langfuse_client import get_callback_handler
        trace = get_callback_handler()
    except Exception as e:
        logger.debug("Langfuse trace init failed: %s", e)

    # --- Forensic Agent ---
    announcements = _select_announcements(ctx.recent_announcements, ctx.snapshot_date)
    if announcements:
        logger.info(
            "Forensic Agent: running for appid=%d (%d announcements in last %dd)",
            ctx.appid, len(announcements), LOOKBACK_DAYS,
        )
        try:
            forensic = run_forensic_agent(
                appid=ctx.appid, game_name=ctx.game_name, snapshot_date=snap_date_str,
                ea_age_days=ctx.ea_age_days,
                days_since_last_build_update=ctx.days_since_last_build_update,
                announcements=announcements,
                trace=trace,
            )
            result.forensic = forensic
            result.forensic_ran = True
            if forensic.error:
                logger.warning("Forensic error appid=%d: %s", ctx.appid, forensic.error)
        except Exception as e:
            logger.error("Forensic exception appid=%d: %s", ctx.appid, e)
    else:
        logger.info("Forensic skipped appid=%d (no announcements in last %dd)", ctx.appid, LOOKBACK_DAYS)

    # --- Sentiment Auditor ---
    if ctx.xgboost.ml_eligible:
        logger.info("Sentiment Auditor: running for appid=%d", ctx.appid)
        try:
            auditor = run_sentiment_auditor(
                appid=ctx.appid, game_name=ctx.game_name, snapshot_date=snap_date_str,
                review_score_at_T=ctx.review_score_at_T,
                review_score_last_90d=ctx.review_score_last_90d,
                review_count_at_T=ctx.review_count_at_T,
                recent_reviews=ctx.recent_reviews, older_reviews=ctx.older_reviews,
                l1_state=ctx.scorecard.l1_state,
                trace=trace,
            )
            result.auditor = auditor
            result.auditor_ran = True
            if auditor.error:
                logger.warning("Auditor error appid=%d: %s", ctx.appid, auditor.error)
        except Exception as e:
            logger.error("Auditor exception appid=%d: %s", ctx.appid, e)
    else:
        logger.info("Auditor skipped appid=%d (ml_eligible=False)", ctx.appid)

    # --- Critic Agent ---
    logger.info("Critic Agent: running for appid=%d", ctx.appid)
    try:
        forensic_res = result.forensic
        auditor_res  = result.auditor
        critic = run_critic_agent(
            appid=ctx.appid, game_name=ctx.game_name, snapshot_date=snap_date_str,
            ea_age_days=ctx.ea_age_days, l1_state=ctx.scorecard.l1_state,
            l1_composite_score=ctx.scorecard.composite_score,
            update_health=ctx.scorecard.update_health, player_retention=ctx.scorecard.player_retention,
            dev_engagement=ctx.scorecard.dev_engagement, sentiment=ctx.scorecard.sentiment,
            price_market=ctx.scorecard.price_market, p_distressed=ctx.xgboost.p_distressed,
            is_distressed=ctx.xgboost.is_distressed, ml_eligible=ctx.xgboost.ml_eligible,
            forensic_ran=result.forensic_ran and not (forensic_res and forensic_res.error),
            update_substance_score=forensic_res.update_substance_score if forensic_res else None,
            fake_heartbeat_flag=forensic_res.fake_heartbeat_flag if forensic_res else None,
            momentum=forensic_res.momentum if forensic_res else None,
            event_state_mismatch=forensic_res.event_state_mismatch if forensic_res else None,
            forensic_reasoning=forensic_res.reasoning if forensic_res else None,
            auditor_ran=result.auditor_ran and not (auditor_res and auditor_res.error),
            theme_clusters=auditor_res.theme_clusters if auditor_res else None,
            sentiment_shift=auditor_res.sentiment_shift if auditor_res else None,
            sentiment_alignment=auditor_res.sentiment_alignment if auditor_res else None,
            key_concerns=auditor_res.key_concerns if auditor_res else None,
            auditor_summary=auditor_res.auditor_summary if auditor_res else None,
            trace=trace,
        )
        result.critic = critic
        result.critic_ran = True
        if critic.error:
            logger.warning("Critic error appid=%d: %s", ctx.appid, critic.error)
    except Exception as e:
        logger.error("Critic exception appid=%d: %s", ctx.appid, e)

    return result