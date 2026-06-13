"""
api/routers/health.py
---------------------
GET /health — pipeline heartbeat.
"""

import time

from fastapi import APIRouter, HTTPException

from api.db import get_db
from api.schemas import PipelineHealth

router = APIRouter(tags=["health"])

# Score is considered stale if last run was more than 7 days ago
_STALE_THRESHOLD_SECS = 7 * 24 * 3600

# Null rate warning threshold (fraction)
_NULL_RATE_WARN = 0.15

# Key features to spot-check null rates on
_MONITORED_FEATURES = [
    "p_distressed",
    "update_health",
    "player_retention",
    "dev_engagement",
    "sentiment",
    "price_market",
]


@router.get("/health", response_model=PipelineHealth)
def get_health():
    db = get_db()

    # Last scored_at and total game count
    row = db.execute("""
        SELECT MAX(scored_at), COUNT(DISTINCT appid)
        FROM live_scores
    """).fetchone()

    if not row or row[1] == 0:
        return PipelineHealth(
            status="empty",
            last_scored_at=None,
            games_scored_this_week=0,
            games_total=0,
            at_risk_count=0,
            null_rate_warning=[],
        )

    last_scored_at, games_total = row

    # Games scored in the last week
    cutoff_7d = int(time.time()) - _STALE_THRESHOLD_SECS
    (games_scored_this_week,) = db.execute("""
        SELECT COUNT(DISTINCT appid)
        FROM live_scores
        WHERE scored_at >= ?
    """, (cutoff_7d,)).fetchone()

    # At Risk count (latest score per game)
    (at_risk_count,) = db.execute("""
        SELECT COUNT(*)
        FROM (
            SELECT appid, MAX(scored_at) AS latest
            FROM live_scores
            GROUP BY appid
        ) latest_scores
        JOIN live_scores ls
          ON ls.appid = latest_scores.appid
         AND ls.scored_at = latest_scores.latest
        WHERE ls.l1_state = 'At Risk'
    """).fetchone()

    # Null rate check on monitored features
    null_warnings: list[str] = []
    for col in _MONITORED_FEATURES:
        try:
            total_row = db.execute("""
                SELECT COUNT(*), SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END)
                FROM (
                    SELECT appid, MAX(scored_at) AS latest
                    FROM live_scores GROUP BY appid
                ) ls
                JOIN live_scores s ON s.appid = ls.appid AND s.scored_at = ls.latest
            """.replace("{col}", col)).fetchone()
            if total_row and total_row[0] > 0:
                null_rate = (total_row[1] or 0) / total_row[0]
                if null_rate > _NULL_RATE_WARN:
                    null_warnings.append(f"{col} ({null_rate:.1%} null)")
        except Exception:
            pass  # column may not exist in older schema versions

    # Staleness
    age_secs = int(time.time()) - last_scored_at
    status = "stale" if age_secs > _STALE_THRESHOLD_SECS else "ok"

    return PipelineHealth(
        status=status,
        last_scored_at=last_scored_at,
        games_scored_this_week=games_scored_this_week,
        games_total=games_total,
        at_risk_count=at_risk_count,
        null_rate_warning=null_warnings,
    )
