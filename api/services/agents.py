"""
api/services/agents.py
----------------------
Thin adapter between the FastAPI layer and the LangGraph orchestrator.

Responsibilities:
  - Load game context from DB (live_scores, live_snapshots, review_history)
  - Check staleness / eligibility before running
  - Call agents.orchestrator.run_analysis()
  - Persist results back to agent_analysis table

Staleness policy:
  Re-run if:
    a) No prior analysis exists
    b) l1_state changed since last analysis
    c) analysed_at is older than STALE_DAYS
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone

import libsql

from agents.orchestrator import (
    AnalysisResult,
    AnnouncementEvent,
    GameContext,
    ScorecardResult,
    XGBoostResult,
    run_analysis,
)

from api.services.reviews import fetch_reviews_for_auditor

logger = logging.getLogger(__name__)

STALE_DAYS = 14          # re-run analysis if older than this
MAX_REVIEWS_PER_WINDOW = 20  # cap reviews fetched per window


# ---------------------------------------------------------------------------
# Text processing helpers
# ---------------------------------------------------------------------------

_BBCODE_TAG_RE = re.compile(
    r"\[/?\w[^\]]*\]|\{STEAM_CLAN_IMAGE\}[^\s]*",
    re.IGNORECASE
)

def strip_bbcode(text: str) -> str:
    if not text:
        return ""
    cleaned = _BBCODE_TAG_RE.sub(" ", text)
    return " ".join(cleaned.split())


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------

def is_analysis_eligible(l1_state: str | None) -> bool:
    """Only Watch and At Risk games get agent analysis."""
    return l1_state in ("Watch", "At Risk")


def needs_rerun(
    existing_row: dict | None,
    current_l1_state: str,
    stale_days: int = STALE_DAYS,
) -> tuple[bool, str]:
    """
    Returns (should_rerun, reason).
    reason is one of: "first_run" | "state_change" | "stale" | "fresh"
    """
    if existing_row is None:
        return True, "first_run"

    if existing_row.get("l1_state_at_analysis") != current_l1_state:
        return True, "state_change"

    analysed_at = existing_row.get("analysed_at") or 0
    age_days = (time.time() - analysed_at) / 86400
    if age_days > stale_days:
        return True, "stale"

    return False, "fresh"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _fetch_live_score(db: libsql.Connection, appid: int) -> dict | None:
    row = db.execute("""
        SELECT
            ls.appid, ls.scored_at, ls.ea_age_days, ls.l1_state,
            ls.p_distressed, ls.is_distressed, ls.ml_eligible,
            ls.update_health, ls.player_retention, ls.dev_engagement,
            ls.sentiment, ls.price_market, ls.review_count_at_T, ls.snapshot_date
        FROM live_scores ls
        WHERE ls.appid = ?
        ORDER BY ls.scored_at DESC
        LIMIT 1
    """, (appid,)).fetchone()

    if not row:
        return None

    return {
        "appid": row[0], "scored_at": row[1], "ea_age_days": row[2],
        "l1_state": row[3], "p_distressed": row[4], "is_distressed": row[5],
        "ml_eligible": row[6], "update_health": row[7], "player_retention": row[8],
        "dev_engagement": row[9], "sentiment": row[10], "price_market": row[11],
        "review_count_at_T": row[12],
        "snapshot_date": row[13],
    }


def _fetch_game_meta(db: libsql.Connection, appid: int) -> dict:
    row = db.execute(
        "SELECT name, ea_start_date FROM games_v2 WHERE appid = ?", (appid,)
    ).fetchone()
    return {"name": row[0] if row else None, "ea_start_date": row[1] if row else None}


def _fetch_build_events(db: libsql.Connection, appid: int) -> list[AnnouncementEvent]:
    """Fetch build update events (type 12/13/14) sorted most recent first."""
    rows = db.execute("""
        SELECT event_type, event_name, announcement_body, word_count, event_ts
        FROM event_history
        WHERE appid = ?
          AND event_type IN (12, 13)
        ORDER BY event_ts DESC
        LIMIT 10
    """, (appid,)).fetchall()

    events = []
    for r in rows:
        try:
            posted = datetime.fromtimestamp(r[4], tz=timezone.utc).date()
            events.append(AnnouncementEvent(
                event_type=r[0],
                title=r[1] or "",
                body_stripped=strip_bbcode(r[2] or ""),
                word_count=r[3] or 0,
                posted_at=posted,
            ))
        except (ValueError, TypeError, OSError):
            continue
    return events


def _fetch_existing_analysis(db: libsql.Connection, appid: int) -> dict | None:
    row = db.execute(
        "SELECT * FROM agent_analysis WHERE appid = ?", (appid,)
    ).fetchone()
    if not row:
        return None
    cols = [d[0] for d in db.execute("SELECT * FROM agent_analysis LIMIT 0").description]
    # Re-fetch with description
    cursor = db.execute("SELECT * FROM agent_analysis WHERE appid = ?", (appid,))
    cols = [d[0] for d in cursor.description]
    row = cursor.fetchone()
    return dict(zip(cols, row)) if row else None


def _persist_result(
    db: libsql.Connection,
    appid: int,
    result: AnalysisResult,
    trigger_reason: str,
    l1_state: str,
    snapshot_date: str,
) -> None:
    forensic = result.forensic
    auditor  = result.auditor
    critic   = result.critic

    db.execute("""
        INSERT OR REPLACE INTO agent_analysis (
            appid, snapshot_date, analysed_at, trigger_reason, l1_state_at_analysis,
            forensic_ran, update_substance_score, fake_heartbeat_flag, momentum, event_state_mismatch, forensic_reasoning,
            auditor_ran, sentiment_shift, sentiment_alignment, key_concerns, theme_clusters, auditor_summary,
            signal_alignment, critic_ran, consumer_verdict, developer_brief, confidence_note,
            error
        ) VALUES (
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?
        )
    """, (
        appid,
        snapshot_date,
        int(time.time()),
        trigger_reason,
        l1_state,
        # Forensic
        int(result.forensic_ran),
        forensic.update_substance_score if forensic else None,
        forensic.fake_heartbeat_flag if forensic else None,
        forensic.momentum if forensic else None,
        forensic.event_state_mismatch if forensic else None,
        forensic.reasoning if forensic else None,
        # Auditor
        int(result.auditor_ran),
        auditor.sentiment_shift if auditor else None,
        auditor.sentiment_alignment if auditor else None,
        json.dumps(auditor.key_concerns) if auditor and auditor.key_concerns else None,
        json.dumps(auditor.theme_clusters) if auditor and auditor.theme_clusters else None,
        auditor.auditor_summary if auditor else None,
        # Critic
        result.signal_alignment,
        int(result.critic_ran),
        critic.consumer_verdict if critic else None,
        critic.developer_brief if critic else None,
        critic.confidence_note if critic else None,
        # Error
        ((forensic.error if forensic else None) or (auditor.error if auditor else None) or (critic.error if critic else None)),
    ))
    db.commit()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def trigger_analysis(db: libsql.Connection, appid: int, force: bool = False) -> dict:
    """
    Check eligibility, load context, run agents, persist result.
    Returns a status dict consumed by the router.

    This is called inside a BackgroundTask — errors are logged, not raised.
    """
    try:
        score = _fetch_live_score(db, appid)
        if not score:
            logger.warning("trigger_analysis: no live score for appid=%d", appid)
            return {"status": "error", "message": "No live score found"}

        l1_state = score["l1_state"]
        if not is_analysis_eligible(l1_state):
            return {"status": "not_eligible", "message": f"l1_state={l1_state} does not require analysis"}

        existing = _fetch_existing_analysis(db, appid)
        should_run, reason = needs_rerun(existing, l1_state)

        if not should_run and not force:
            return {"status": "fresh", "message": "Analysis is up to date"}

        trigger_reason = "user_request" if force else reason

        meta        = _fetch_game_meta(db, appid)
        build_events = _fetch_build_events(db, appid)
        snap_ts     = score["scored_at"]
        snap_date   = score.get("snapshot_date") or datetime.fromtimestamp(snap_ts, tz=timezone.utc).strftime("%Y-%m-%d")

        recent_reviews, older_reviews = fetch_reviews_for_auditor(
        appid,
        n_recent=MAX_REVIEWS_PER_WINDOW,   # existing constant, default 50 — see note below
        n_older=MAX_REVIEWS_PER_WINDOW,
        )

        ctx = GameContext(
            appid=appid,
            game_name=meta["name"] or str(appid),
            snapshot_date=datetime.strptime(snap_date, "%Y-%m-%d").date(),
            ea_age_days=score["ea_age_days"] or 0,
            scorecard=ScorecardResult(
                l1_state=l1_state,
                composite_score=score.get("p_distressed") or 0.0,
                update_health=score.get("update_health"),
                player_retention=score.get("player_retention"),
                dev_engagement=score.get("dev_engagement"),
                sentiment=score.get("sentiment"),
                price_market=score.get("price_market"),
            ),
            xgboost=XGBoostResult(
                ml_eligible=bool(score.get("ml_eligible")),
                p_distressed=score.get("p_distressed"),
                is_distressed=score.get("is_distressed"),
            ),
            recent_build_events=build_events,
            recent_reviews=recent_reviews,
            older_reviews=older_reviews,
            review_score_at_T=0.0,   # not stored in live_scores; auditor handles gracefully
            review_score_last_90d=None,
            review_count_at_T=score.get("review_count_at_T") or 0,
        )

        result = asyncio.run(run_analysis(ctx))
        _persist_result(db, appid, result, trigger_reason, l1_state, snap_date)

        logger.info(
            "trigger_analysis: appid=%d reason=%s success=%s",
            appid, trigger_reason, result.success,
        )
        return {"status": "done", "message": "Analysis complete"}

    except Exception as e:
        logger.error("trigger_analysis: appid=%d error=%s", appid, e, exc_info=True)
        return {"status": "error", "message": str(e)}
