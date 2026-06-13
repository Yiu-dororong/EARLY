"""
agents/orchestrator.py
EARLY — Phase 2 Agent Orchestrator

Coordinates Forensic Agent, Sentiment Auditor, Critic Agent.
Creates a top-level Langfuse trace per run_analysis() call and passes
it down to each agent as a nested span.

Trigger condition: l1_state in ("Watch", "At Risk")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from agents.forensic_agent import ForensicResult, run_forensic_agent
from agents.sentiment_auditor import SentimentResult, run_sentiment_auditor
from agents.critic_agent import CriticResult, run_critic_agent

logger = logging.getLogger(__name__)

FORENSIC_LOOKBACK_DAYS = 30


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
class BuildEvent:
    event_type: int
    announcement_title: str
    announcement_body_stripped: str
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
    recent_build_events: list[BuildEvent] = field(default_factory=list)
    recent_reviews: list[dict] = field(default_factory=list)
    older_reviews: list[dict] = field(default_factory=list)
    review_score_at_T: float = 0.0
    review_score_last_90d: float | None = None
    review_count_at_T: int = 0
    session_id: str | None = None   # passed from API for Langfuse correlation


def should_run_phase2(scorecard: ScorecardResult) -> bool:
    return scorecard.l1_state in ("Watch", "At Risk")


def _find_recent_build_event(events: list[BuildEvent], snapshot_date: date) -> BuildEvent | None:
    cutoff = snapshot_date - timedelta(days=FORENSIC_LOOKBACK_DAYS)
    for event in events:
        if event.event_type in (12, 13) and event.posted_at >= cutoff:
            return event
    return None


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
    def key_concerns(self) -> list[str] | None:
        return self.auditor.key_concerns if self.auditor else None
    @property
    def success(self) -> bool:
        if self.forensic and self.forensic.error: return False
        if self.auditor  and self.auditor.error:  return False
        if self.critic   and self.critic.error:   return False
        return True


async def run_analysis(ctx: GameContext) -> AnalysisResult:
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
        from utils.langfuse_client import create_trace, flush
        trace = create_trace(
            name="phase2_analysis",
            appid=ctx.appid,
            session_id=ctx.session_id,
            metadata={
                "l1_state": ctx.scorecard.l1_state,
                "composite_score": ctx.scorecard.composite_score,
                "ml_eligible": ctx.xgboost.ml_eligible,
                "ea_age_days": ctx.ea_age_days,
            },
        )
    except Exception as e:
        logger.debug("Langfuse trace init failed: %s", e)

    # --- Forensic Agent ---
    recent_build = _find_recent_build_event(ctx.recent_build_events, ctx.snapshot_date)
    if recent_build:
        logger.info("Forensic Agent: running for appid=%d", ctx.appid)
        try:
            forensic = await run_forensic_agent(
                appid=ctx.appid, game_name=ctx.game_name, snapshot_date=snap_date_str,
                event_type=recent_build.event_type,
                announcement_title=recent_build.announcement_title,
                announcement_body_stripped=recent_build.announcement_body_stripped,
                word_count=recent_build.word_count,
                ea_age_days=ctx.ea_age_days,
                days_since_last_build_update=(ctx.snapshot_date - recent_build.posted_at).days,
                trace=trace,
            )
            result.forensic = forensic
            result.forensic_ran = True
            if forensic.error:
                logger.warning("Forensic error appid=%d: %s", ctx.appid, forensic.error)
        except Exception as e:
            logger.error("Forensic exception appid=%d: %s", ctx.appid, e)
    else:
        logger.info("Forensic skipped appid=%d (no build update in last %dd)", ctx.appid, FORENSIC_LOOKBACK_DAYS)

    # --- Sentiment Auditor ---
    if ctx.xgboost.ml_eligible:
        logger.info("Sentiment Auditor: running for appid=%d", ctx.appid)
        try:
            auditor = await run_sentiment_auditor(
                appid=ctx.appid, game_name=ctx.game_name, snapshot_date=snap_date_str,
                review_score_at_T=ctx.review_score_at_T,
                review_score_last_90d=ctx.review_score_last_90d,
                review_count_at_T=ctx.review_count_at_T,
                recent_reviews=ctx.recent_reviews, older_reviews=ctx.older_reviews,
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
        critic = await run_critic_agent(
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
            forensic_reasoning=forensic_res.reasoning if forensic_res else None,
            auditor_ran=result.auditor_ran and not (auditor_res and auditor_res.error),
            theme_clusters=auditor_res.theme_clusters if auditor_res else None,
            sentiment_shift=auditor_res.sentiment_shift if auditor_res else None,
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

    # Flush Langfuse events
    try:
        if trace:
            from utils.langfuse_client import flush
            flush()
    except Exception:
        pass

    return result