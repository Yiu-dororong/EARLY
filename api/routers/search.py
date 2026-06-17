"""
api/routers/search.py
---------------------
POST /search/similar — find historically similar games via Zilliz ANN search.

Accepts a SHAP vector payload (from live_snapshots) and returns up to 5
deduplicated historical games with known outcomes.

Optional — only available when Zilliz is configured.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from api.db import get_db
from api.schemas import SimilarGame, SimilaritySearchResponse
from api.services.zilliz import VECTOR_DIM, search_similar
from api.rate_limit import limiter, search_rate_limit, search_ip_rate_limit, get_real_ip

router = APIRouter(tags=["search"])


@router.post("/similar", response_model=SimilaritySearchResponse)
@limiter.limit(search_rate_limit)
@limiter.limit(search_ip_rate_limit, key_func=get_real_ip)
def find_similar_games(request: Request, appid: int, n_results: int = 5):
    """
    Find historically similar games for a given appid using Zilliz ANN search.

    Loads the SHAP vector from live_snapshots for the given appid,
    queries Zilliz with metadata filters (ea_age ±90d, same genre),
    returns up to n_results deduplicated games with known outcomes.

    Returns 503 if Zilliz is not configured.
    Returns 404 if no live snapshot exists for the appid.
    """
    from api.services.zilliz import get_client
    if get_client() is None:
        raise HTTPException(
            status_code=503,
            detail="Similarity search is not available — Zilliz not configured.",
        )

    db = get_db()

    # Load live snapshot for query game
    row = db.execute("""
        SELECT snapshot_date, ea_age_days, primary_genre, shap_json, model_version
        FROM live_scores
        WHERE appid = ?
        ORDER BY scored_at DESC LIMIT 1
    """, (appid,)).fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No live snapshot found for appid {appid}. Run scoring first.",
        )

    snapshot_date, ea_age_days, primary_genre, shap_json_str, model_version = row

    if not shap_json_str:
        raise HTTPException(
            status_code=404,
            detail=f"SHAP values not yet computed for appid {appid}.",
        )

    try:
        shap_dict: dict[str, float | None] = json.loads(shap_json_str)
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(status_code=500, detail="Malformed SHAP JSON in live_snapshots.")

    # Load canonical feature order from the model artifact to guarantee exact vector dimension alignment
    model_dir = Path(__file__).resolve().parent.parent.parent / "models"
    top25_path = model_dir / f"shap_top25_{model_version}.json"

    feature_order = []
    if top25_path.exists():
        try:
            with open(top25_path) as f:
                feature_order = json.load(f).get("features", [])
        except Exception:
            pass

    # Fallback to dictionary keys if the artifact is missing
    if not feature_order:
        feature_order = list(shap_dict.keys())

    if len(feature_order) != VECTOR_DIM:
        raise HTTPException(
            status_code=500,
            detail=f"SHAP vector has {len(feature_order)} features, expected {VECTOR_DIM}.",
        )

    # Run ANN search
    hits = search_similar(
        query_shap_dict=shap_dict,
        feature_order=feature_order,
        query_appid=appid,
        query_ea_age_days=ea_age_days or 0,
        query_primary_genre=primary_genre or "",
        n_results=min(n_results, 5),
    )

    if not hits:
        return SimilaritySearchResponse(
            query_appid=appid,
            query_snap_date=snapshot_date,
            results=[],
            message="No similar games found. Zilliz collection may be empty or filters too strict.",
        )

    # Enrich with game names from DB
    appids = [h.appid for h in hits]
    placeholders = ",".join("?" * len(appids))
    name_rows = db.execute(
        f"SELECT appid, name FROM games_v2 WHERE appid IN ({placeholders})", appids
    ).fetchall()
    name_map = {r[0]: r[1] for r in name_rows}

    results = [
        SimilarGame(
            appid=h.appid,
            name=name_map.get(h.appid),
            snapshot_date=h.snapshot_date,
            ea_age_days=h.ea_age_days,
            primary_genre=h.primary_genre,
            l1_state=h.l1_state,
            outcome=h.outcome,
            p_distressed=h.p_distressed,
            distance=h.distance,
            match_quality=h.match_quality,
            null_feature_count=h.null_feature_count,
        )
        for h in hits
    ]

    return SimilaritySearchResponse(
        query_appid=appid,
        query_snap_date=snapshot_date,
        results=results,
        message=None,
    )
