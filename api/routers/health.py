"""
api/routers/health.py
---------------------
GET /health — pipeline heartbeat.
"""

import json
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

    cutoff_7d = int(time.time()) - _STALE_THRESHOLD_SECS

    # Single combined query restricted to the latest snapshot batch
    query = """
        WITH latest_batch AS (
            SELECT * FROM live_scores
            WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM live_scores)
        ),
        deduped AS (
            SELECT * FROM latest_batch
            WHERE (appid, scored_at) IN (
                SELECT appid, MAX(scored_at) FROM latest_batch GROUP BY appid
            )
        ),
        state_agg AS (
            SELECT IFNULL(l1_state, 'Unknown') AS state, COUNT(*) as cnt
            FROM deduped
            GROUP BY l1_state
        )
        SELECT
            MAX(scored_at) AS last_scored_at,
            SUM(CASE WHEN scored_at >= ? THEN 1 ELSE 0 END) AS games_scored_this_week,
            (SELECT json_group_object(state, cnt) FROM state_agg) AS state_counts,
            SUM(CASE WHEN p_distressed IS NULL THEN 1 ELSE 0 END) AS null_p_distressed,
            SUM(CASE WHEN update_health IS NULL THEN 1 ELSE 0 END) AS null_update_health,
            SUM(CASE WHEN player_retention IS NULL THEN 1 ELSE 0 END) AS null_player_retention,
            SUM(CASE WHEN dev_engagement IS NULL THEN 1 ELSE 0 END) AS null_dev_engagement,
            SUM(CASE WHEN sentiment IS NULL THEN 1 ELSE 0 END) AS null_sentiment,
            SUM(CASE WHEN price_market IS NULL THEN 1 ELSE 0 END) AS null_price_market
        FROM deduped
    """

    row = db.execute(query, (cutoff_7d,)).fetchone()

    if not row or row[0] is None:
        return PipelineHealth(
            status="empty",
            last_scored_at=None,
            games_scored_this_week=0,
            games_total=0,
            at_risk_count=0,
            watch_count=0,
            healthy_count=0,
            null_rate_warning=[],
        )

    last_scored_at = row[0]
    games_scored_this_week = row[1] or 0

    state_counts = json.loads(row[2] or "{}")
    games_total = sum(state_counts.values())

    # Null rate check on monitored features
    null_warnings = []
    for i, col in enumerate(_MONITORED_FEATURES):
        null_count = row[3 + i] or 0
        if games_total > 0:
            null_rate = null_count / games_total
            if null_rate > _NULL_RATE_WARN:
                null_warnings.append(f"{col} ({null_rate:.1%} null)")

    # Staleness
    age_secs = int(time.time()) - last_scored_at
    status = "stale" if age_secs > _STALE_THRESHOLD_SECS else "ok"

    return PipelineHealth(
        status=status,
        last_scored_at=last_scored_at,
        games_scored_this_week=games_scored_this_week,
        games_total=games_total,
        at_risk_count=state_counts.get("At Risk", 0),
        watch_count=state_counts.get("Watch", 0),
        healthy_count=state_counts.get("Healthy", 0),
        null_rate_warning=null_warnings,
    )
