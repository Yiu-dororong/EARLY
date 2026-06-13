"""
api/routers/games.py
--------------------
GET  /games                      — paginated game list with filters
GET  /games/{appid}/score        — latest full score + dimension breakdown
GET  /games/{appid}/history      — full score time series
GET  /games/{appid}/features     — latest feature vector from live_snapshots
POST /games/{appid}/analyse      — trigger background agent analysis
GET  /games/{appid}/analysis     — retrieve cached agent analysis
"""

import json
import time

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from api.db import get_db
from api.schemas import (
    AgentAnalysisResponse,
    AnalysisTriggerResponse,
    AuditorOutput,
    CriticOutput,
    DimensionScores,
    ForensicOutput,
    GameFeatures,
    GameListResponse,
    GameScore,
    GameSummary,
    ScoreHistoryResponse,
    ScoreSnapshot,
)
from api.services.agents import is_analysis_eligible, trigger_analysis

router = APIRouter(tags=["games"])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIMENSION_COLS = (
    "update_health",
    "player_retention",
    "dev_engagement",
    "sentiment",
    "price_market",
)


def _build_dimensions(row: dict) -> DimensionScores | None:
    vals = {col: row.get(col) for col in _DIMENSION_COLS}
    if all(v is None for v in vals.values()):
        return None
    return DimensionScores(**vals)


# ---------------------------------------------------------------------------
# GET /games
# ---------------------------------------------------------------------------

@router.get("", response_model=GameListResponse)
def list_games(
    l1_state:        str | None = Query(None, description="Healthy | Watch | At Risk"),
    ml_eligible:     int | None = Query(None, description="1 = ML eligible only"),
    currently_in_ea: int | None = Query(None, description="1 = active EA only"),
    outcome:         str | None = Query(None, description="EXIT_SUCCESS | EXIT_ABANDONED | EXIT_SILENT | STAYS_ACTIVE"),
    min_reviews:     int | None = Query(None, description="Minimum review_count_at_T"),
    offset:          int = Query(0, ge=0),
    limit:           int = Query(50, ge=1, le=200),
):
    db = get_db()

    filters = []
    params: list = []

    if l1_state is not None:
        filters.append("ls.l1_state = ?")
        params.append(l1_state)
    if ml_eligible is not None:
        filters.append("ls.ml_eligible = ?")
        params.append(ml_eligible)
    if currently_in_ea is not None:
        filters.append("g.currently_in_ea = ?")
        params.append(currently_in_ea)
    if outcome is not None:
        filters.append("g.outcome = ?")
        params.append(outcome)
    if min_reviews is not None:
        filters.append("ls.review_count_at_T >= ?")
        params.append(min_reviews)

    where = ("WHERE " + " AND ".join(filters)) if filters else ""

    base_query = f"""
        FROM (
            SELECT appid, MAX(scored_at) AS latest
            FROM live_scores
            GROUP BY appid
        ) latest
        JOIN live_scores ls ON ls.appid = latest.appid AND ls.scored_at = latest.latest
        LEFT JOIN games_v2 g ON g.appid = ls.appid
        {where}
    """

    (total,) = db.execute(f"SELECT COUNT(*) {base_query}", params).fetchone()

    rows = db.execute(f"""
        SELECT
            ls.appid,
            g.name,
            g.ea_start_date,
            ls.ea_age_days,
            ls.l1_state,
            ls.p_distressed,
            ls.is_distressed,
            ls.ml_eligible,
            ls.review_count_at_T,
            ls.snapshot_date,
            g.outcome
        {base_query}
        ORDER BY ls.p_distressed DESC NULLS LAST
        LIMIT ? OFFSET ?
    """, params + [limit, offset]).fetchall()

    items = [
        GameSummary(
            appid=r[0], name=r[1], ea_start_date=r[2], ea_age_days=r[3],
            l1_state=r[4], p_distressed=r[5], is_distressed=r[6],
            ml_eligible=r[7], review_count_at_T=r[8], snap_date=r[9], outcome=r[10],
        )
        for r in rows
    ]

    return GameListResponse(total=total, offset=offset, limit=limit, items=items)


# ---------------------------------------------------------------------------
# GET /games/{appid}/score
# ---------------------------------------------------------------------------

@router.get("/{appid}/score", response_model=GameScore)
def get_game_score(appid: int):
    db = get_db()

    row = db.execute("""
        SELECT
            ls.appid, g.name, g.ea_start_date, ls.ea_age_days,
            ls.primary_genre, ls.l1_state, ls.p_distressed, ls.is_distressed,
            ls.ml_eligible, ls.model_version, ls.snapshot_date, ls.review_count_at_T,
            ls.null_features, ls.update_health, ls.player_retention,
            ls.dev_engagement, ls.sentiment, ls.price_market,
            g.outcome, g.currently_in_ea
        FROM live_scores ls
        LEFT JOIN games_v2    g  ON g.appid  = ls.appid
        WHERE ls.appid = ?
        ORDER BY ls.scored_at DESC
        LIMIT 1
    """, (appid,)).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"Game {appid} not found.")

    row_dict = {
        "appid": row[0], "name": row[1], "ea_start_date": row[2],
        "ea_age_days": row[3], "primary_genre": row[4], "l1_state": row[5],
        "p_distressed": row[6], "is_distressed": row[7], "ml_eligible": row[8],
        "model_version": row[9], "snap_date": row[10], "review_count_at_T": row[11],
        "null_features": row[12], "update_health": row[13], "player_retention": row[14],
        "dev_engagement": row[15], "sentiment": row[16], "price_market": row[17],
        "outcome": row[18], "currently_in_ea": row[19],
    }

    null_list = json.loads(row_dict["null_features"]) if isinstance(row_dict["null_features"], str) else (row_dict["null_features"] or [])
    data_quality = "high" if len(null_list) <= 5 else "medium" if len(null_list) <= 15 else "low"

    return GameScore(
        **{k: v for k, v in row_dict.items() if k not in _DIMENSION_COLS and k != "null_features"},
        null_features=row_dict["null_features"],
        data_quality=data_quality,
        dimensions=_build_dimensions(row_dict),
    )


# ---------------------------------------------------------------------------
# GET /games/{appid}/history
# ---------------------------------------------------------------------------

@router.get("/{appid}/history", response_model=ScoreHistoryResponse)
def get_game_history(appid: int):
    db = get_db()

    rows = db.execute("""
        SELECT
            snapshot_date, l1_state, p_distressed, is_distressed,
            ea_age_days, review_count_at_T, null_features,
            update_health, player_retention, dev_engagement, sentiment, price_market
        FROM live_scores
        WHERE appid = ?
        ORDER BY scored_at ASC
    """, (appid,)).fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail=f"Game {appid} not found.")

    name_row = db.execute("SELECT name FROM games_v2 WHERE appid = ?", (appid,)).fetchone()

    snapshots = []
    for r in rows:
        row_dict = {
            "snap_date": r[0], "l1_state": r[1], "p_distressed": r[2],
            "is_distressed": r[3], "ea_age_days": r[4], "review_count_at_T": r[5],
            "null_features": r[6], "update_health": r[7], "player_retention": r[8],
            "dev_engagement": r[9], "sentiment": r[10], "price_market": r[11],
        }
        snapshots.append(ScoreSnapshot(
            **{k: v for k, v in row_dict.items() if k not in _DIMENSION_COLS and k != "null_features"},
            null_features=row_dict["null_features"],
            dimensions=_build_dimensions(row_dict),
        ))

    return ScoreHistoryResponse(
        appid=appid,
        name=name_row[0] if name_row else None,
        snapshots=snapshots,
    )


# ---------------------------------------------------------------------------
# GET /games/{appid}/features
# ---------------------------------------------------------------------------

@router.get("/{appid}/features", response_model=GameFeatures)
def get_game_features(appid: int):
    db = get_db()

    row = db.execute("""
        SELECT snapshot_date, ea_age_days, primary_genre, review_count_at_T,
               shap_json
        FROM live_scores
        WHERE appid = ?
        ORDER BY scored_at DESC LIMIT 1
    """, (appid,)).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"No live snapshot found for game {appid}.")

    name_row = db.execute("SELECT name FROM games_v2 WHERE appid = ?", (appid,)).fetchone()

    return GameFeatures(
        appid=appid,
        name=name_row[0] if name_row else None,
        snap_date=row[0],
        ea_age_days=row[1],
        primary_genre=row[2],
        review_count_at_T=row[3],
        features="{}",          # live_snapshots no longer has a single features_json blob
        shap_values=row[4],     # parsed by field_validator
    )


# ---------------------------------------------------------------------------
# POST /games/{appid}/analyse
# ---------------------------------------------------------------------------

@router.post("/{appid}/analyse", response_model=AnalysisTriggerResponse)
def trigger_game_analysis(
    appid: int,
    background_tasks: BackgroundTasks,
    force: bool = Query(False, description="Re-run even if analysis is fresh"),
):
    db = get_db()

    # Quick eligibility check before queuing
    score_row = db.execute("""
        SELECT l1_state FROM live_scores
        WHERE appid = ?
        ORDER BY scored_at DESC LIMIT 1
    """, (appid,)).fetchone()

    if not score_row:
        raise HTTPException(status_code=404, detail=f"Game {appid} not found.")

    l1_state = score_row[0]
    if not is_analysis_eligible(l1_state):
        return AnalysisTriggerResponse(
            appid=appid,
            status="not_eligible",
            message=f"Game is '{l1_state}' — agent analysis only runs for Watch and At Risk games.",
        )

    background_tasks.add_task(trigger_analysis, db, appid, force)

    return AnalysisTriggerResponse(
        appid=appid,
        status="queued",
        message="Analysis queued. Poll GET /games/{appid}/analysis for results.",
    )


# ---------------------------------------------------------------------------
# GET /games/{appid}/analysis
# ---------------------------------------------------------------------------

@router.get("/{appid}/analysis", response_model=AgentAnalysisResponse)
def get_game_analysis(appid: int):
    db = get_db()

    # Check game exists
    score_row = db.execute("""
        SELECT l1_state FROM live_scores
        WHERE appid = ?
        ORDER BY scored_at DESC LIMIT 1
    """, (appid,)).fetchone()

    if not score_row:
        raise HTTPException(status_code=404, detail=f"Game {appid} not found.")

    l1_state = score_row[0]
    name_row = db.execute("SELECT name FROM games_v2 WHERE appid = ?", (appid,)).fetchone()

    # Not eligible — return immediately with status
    if not is_analysis_eligible(l1_state):
        return AgentAnalysisResponse(
            appid=appid,
            name=name_row[0] if name_row else None,
            snapshot_date=None,
            analysed_at=None,
            trigger_reason=None,
            status="not_eligible",
            forensic=None,
            auditor=None,
            critic=None,
            error=None,
        )

    cursor = db.execute("SELECT * FROM agent_analysis WHERE appid = ?", (appid,))
    cols = [d[0] for d in cursor.description]
    row  = cursor.fetchone()

    if not row:
        return AgentAnalysisResponse(
            appid=appid,
            name=name_row[0] if name_row else None,
            snapshot_date=None,
            analysed_at=None,
            trigger_reason=None,
            status="never_run",
            forensic=None,
            auditor=None,
            critic=None,
            error=None,
        )

    r = dict(zip(cols, row))

    status = "error" if r.get("error") else "ready"

    return AgentAnalysisResponse(
        appid=appid,
        name=name_row[0] if name_row else None,
        snapshot_date=r.get("snapshot_date"),
        analysed_at=r.get("analysed_at"),
        trigger_reason=r.get("trigger_reason"),
        status=status,
        forensic=ForensicOutput(
            ran=bool(r.get("forensic_ran")),
            update_substance_score=r.get("update_substance_score"),
            fake_heartbeat_flag=r.get("fake_heartbeat_flag"),
            reasoning=r.get("forensic_reasoning"),
        ) if r.get("forensic_ran") else None,
        auditor=AuditorOutput(
            ran=bool(r.get("auditor_ran")),
            sentiment_shift=r.get("sentiment_shift"),
            key_concerns=r.get("key_concerns"),
            theme_clusters=r.get("theme_clusters"),
            summary=r.get("auditor_summary"),
        ) if r.get("auditor_ran") else None,
        critic=CriticOutput(
            ran=bool(r.get("critic_ran")),
            consumer_verdict=r.get("consumer_verdict"),
            developer_brief=r.get("developer_brief"),
            confidence_note=r.get("confidence_note"),
        ) if r.get("critic_ran") else None,
        error=r.get("error"),
    )