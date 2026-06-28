"""
seed_vector_db.py
------------------
One-time script to populate the Zilliz early_historical_anchors collection
with historical labeled snapshots from the training set.

Run once after initial model training, and again after retraining if the
top-25 SHAP feature list changes (requires collection rebuild).

Usage:
    python -m training.seed_vector_db
    python -m training.seed_vector_db --model-version xgb_v1.3
    python -m training.seed_vector_db --dry-run     # validate without writing
    python -m training.seed_vector_db --rebuild     # drop + recreate collection first

Requirements:
    - TURSO_URL, TURSO_AUTH_TOKEN       (source DB)
    - ZILLIZ_URI, ZILLIZ_TOKEN                   (target)
    - models/shap_top25_{MODEL_VERSION}.json     (feature order)
    - models/{MODEL_VERSION}.json                (XGBoost model)

Source:
    Queries the snapshots table filtered to labeled outcomes only
    (outcome IN ('EXIT_SUCCESS', 'EXIT_ABANDONED')).
    Each game can have multiple snapshots — all are included as anchors.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from dotenv import load_dotenv


load_dotenv()


log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_DIR    = PROJECT_ROOT / "models"
DEFAULT_MODEL_VERSION = "v1.5"
BATCH_SIZE   = 200      # upsert batch size
CONFIG_VERSION = "v1.1" #scorecard version

# ---------------------------------------------------------------------------
# DB connection
# ---------------------------------------------------------------------------

def get_db():
    import libsql
    return libsql.connect(
        database=os.environ["TURSO_URL"],
        auth_token=os.environ["TURSO_AUTH_TOKEN"],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model_and_features(model_version: str) -> tuple[xgb.Booster,
                                                         list[str],
                                                         list[str]]:
    """Load XGBoost booster, top-25 feature list, and genre columns."""
    model_path = MODEL_DIR / f"xgb_{model_version}.json"
    top25_path = MODEL_DIR / f"shap_top25_{model_version}.json"
    feature_path = MODEL_DIR / f"xgb_{model_version}_features.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not top25_path.exists():
        raise FileNotFoundError(
            f"Top-25 feature list not found: {top25_path}\n"
            f"Run train_xgboost.py to generate it."
        )
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature metadata not found: {feature_path}")

    booster = xgb.Booster()
    booster.load_model(str(model_path))

    with open(top25_path) as f:
        meta_top25 = json.load(f)

    with open(feature_path) as f:
        meta_features = json.load(f)

    log.info(
        "Model: %s | Top-25 features cover %.1f%% variance",
        model_version, meta_top25.get("cumulative_variance_pct", 0.0),
    )
    return booster, meta_top25["features"], meta_features.get("genre_columns", [])


def fetch_labeled_snapshots(db) -> list[dict]:
    """
    Fetch all snapshots with known outcomes from the training DB.
    Returns list of row dicts with features + metadata.
    """
    # Fetch all columns
    cols_query = db.execute("PRAGMA table_info(snapshots)").fetchall()
    col_names = [c[1] for c in cols_query]
    select_cols = [f"s.{c}" for c in col_names]

    config_version = CONFIG_VERSION

    rows = db.execute(f"""
        SELECT
            {','.join(select_cols)},
            g.outcome AS g_outcome,
            sc.l1_state AS sc_l1_state,
            sc.l1_composite_score AS sc_l1_composite_score,
            sc.l1_update_health_score AS sc_l1_update_health_score,
            sc.l1_player_retention_score AS sc_l1_player_retention_score,
            sc.l1_dev_engagement_score AS sc_l1_dev_engagement_score,
            sc.l1_sentiment_score AS sc_l1_sentiment_score,
            sc.l1_price_market_score AS sc_l1_price_market_score,
            gg.genre_scope AS gg_genre_scope
        FROM snapshots s
        JOIN games_v2 g ON s.appid = g.appid
        JOIN scorecard sc ON s.appid = sc.appid AND s.snapshot_date = sc.snapshot_date
        LEFT JOIN game_genres gg ON s.appid = gg.appid
        WHERE g.outcome IN ('EXIT_SUCCESS', 'EXIT_ABANDONED', 'EXIT_SILENT')
          AND sc.l1_state IS NOT NULL
          AND s.ml_eligible = 1
          AND sc.config_version = '{config_version}'
        ORDER BY s.appid, s.snapshot_date
    """).fetchall()

    extended_cols = col_names + [
        "sc_l1_state", "sc_l1_composite_score", "sc_l1_update_health_score",
        "sc_l1_player_retention_score", "sc_l1_dev_engagement_score",
        "sc_l1_sentiment_score", "sc_l1_price_market_score",
        "gg_genre_scope","g_outcome"
    ]

    L1_STATE_MAP = {"Healthy": 0, "Watch": 1, "At Risk": 2}
    snapshots = []
    for row in rows:
        d = dict(zip(extended_cols, row))
        # Normalise outcome label
        d["outcome"] = (
            "SUCCESS"   if d["outcome"] == "EXIT_SUCCESS"   else
            "ABANDONED" if d["outcome"] in ("EXIT_ABANDONED", "EXIT_SILENT") else
            d["outcome"]
        )

        # Merge derived L1 and genre features
        d["l1_state"] = d.pop("sc_l1_state")
        d["l1_composite_score"] = d.pop("sc_l1_composite_score")
        d["l1_update_health_score"] = d.pop("sc_l1_update_health_score")
        d["l1_player_retention_score"] = d.pop("sc_l1_player_retention_score")
        d["l1_dev_engagement_score"] = d.pop("sc_l1_dev_engagement_score")
        d["l1_sentiment_score"] = d.pop("sc_l1_sentiment_score")
        d["l1_price_market_score"] = d.pop("sc_l1_price_market_score")
        d["genre_scope"] = d.pop("gg_genre_scope")
        d["outcome"] = d.pop("g_outcome")

        l1_val = L1_STATE_MAP.get(d["l1_state"])
        d["l1_state_encoded"] = l1_val if l1_val is not None else 2

        snapshots.append(d)

    log.info("Fetched %d labeled snapshots (%d unique games)",
             len(snapshots),
             len({r["appid"] for r in snapshots}))
    return snapshots


PRICE_TREND_ENCODING = {"decreased": -1, "stable": 0, "increased": 1}

def encode_price_trend(features: dict) -> dict:
    encoded = dict(features)
    raw = encoded.get("price_trend")
    if raw is not None:
        encoded["price_trend_encoded"] = PRICE_TREND_ENCODING.get(raw)
    else:
        encoded["price_trend_encoded"] = None
    return encoded


def encode_genre(primary_genre: str | None, genre_cols: list[str]) -> dict[str, int]:
    encoded = {col: 0 for col in genre_cols}
    if not primary_genre:
        return encoded
    expected_col = f"genre_{primary_genre}"
    if expected_col in encoded:
        encoded[expected_col] = 1
    return encoded


def compute_shap_vectors(
    snapshots: list[dict],
    booster: xgb.Booster,
    all_feature_cols: list[str],
    top25_features: list[str],
    genre_cols: list[str],
) -> list[dict]:
    """
    Compute SHAP vector for each snapshot.
    Returns snapshots enriched with 'shap_vector' and 'null_feature_count'.
    """
    log.info("Computing SHAP vectors for %d snapshots...", len(snapshots))

    X_data = []
    enriched = []

    for s in snapshots:
        # Encode price trend
        features = encode_price_trend(s)

        # Inject derived feature: review_update_divergence
        rev_score = features.get("review_score_at_T")
        upd_trend = features.get("update_frequency_trend")
        if rev_score is not None and upd_trend is not None:
            try:
                clipped_trend = max(-1.0, min(1.0, float(upd_trend)))
                features["review_update_divergence"] = (
                    float(rev_score) * (1.0 - clipped_trend)
                )
            except (ValueError, TypeError):
                features["review_update_divergence"] = None
        else:
            features["review_update_divergence"] = None

        # Encode genres
        genre_encoded = encode_genre(features.get("primary_genre"), genre_cols)
        features.update(genre_encoded)

        # Assemble ordered feature row
        row = []
        null_count = 0
        for col in all_feature_cols:
            val = features.get(col)
            if val is None:
                null_count += 1
                row.append(np.nan)
            else:
                try:
                    row.append(float(val))
                except (TypeError, ValueError):
                    null_count += 1
                    row.append(np.nan)

        X_data.append(row)

        # Keep track of enriched dict and null_count
        enriched_snap = dict(s)
        enriched_snap["null_feature_count"] = null_count
        enriched.append(enriched_snap)

    X = np.array(X_data, dtype=np.float32)

    dmat = xgb.DMatrix(data=X, feature_names=all_feature_cols)

    # pred_contribs: shape (n_samples, n_features + 1), last col = bias
    contributions = booster.predict(dmat, pred_contribs=True)
    shap_matrix   = contributions[:, :-1]   # (n_samples, n_features)

    # Map feature name → column index for fast lookup
    feat_idx = {f: i for i, f in enumerate(all_feature_cols)}

    for i, snap in enumerate(enriched):
        row_shap = shap_matrix[i]

        # Extract top-25 values in canonical order
        vector = [float(row_shap[feat_idx[f]]) for f in top25_features if f in feat_idx]

        if len(vector) != 25:
            log.warning(
                "appid=%d snap=%s: vector length %d != 25, skipping",
                snap["appid"], snap.get("snapshot_date"), len(vector),
            )
            continue

        snap["shap_vector"] = vector

    log.info("SHAP computed for %d snapshots", len(enriched))
    return enriched


def upsert_to_zilliz(snapshots: list[dict], dry_run: bool = False) -> tuple[int, int]:
    """
    Upsert snapshots into Zilliz in batches.
    Returns (success_count, error_count).
    """
    from api.services.zilliz import COLLECTION_NAME, ensure_collection, get_client

    client = get_client()
    if client is None:
        raise RuntimeError("Zilliz client not available — "
                           "check ZILLIZ_URI and ZILLIZ_TOKEN")

    if not ensure_collection():
        raise RuntimeError("Failed to ensure Zilliz collection exists")

    success = error = 0
    total   = len(snapshots)

    for batch_start in range(0, total, BATCH_SIZE):
        batch = snapshots[batch_start: batch_start + BATCH_SIZE]

        rows = []
        for s in batch:
            # Skip items where shap_vector generation failed
            if "shap_vector" not in s:
                continue
            rows.append({
                "id":                 f"{s['appid']}_{s['snapshot_date']}",
                "shap_vector":        s["shap_vector"],
                "appid":              int(s["appid"]),
                "snapshot_date":      str(s["snapshot_date"]),
                "ea_age_days":        int(s.get("ea_age_days") or 0),
                "primary_genre":      str(s.get("primary_genre") or "unknown"),
                "l1_state":           str(s.get("l1_state") or "unknown"),
                "outcome":            str(s["outcome"]),
                "null_feature_count": int(s["null_feature_count"]),
                "p_distressed":       float(s.get("p_distressed") or 0.0),
            })

        if dry_run:
            log.info("[DRY RUN] Would upsert batch %d–%d (%d rows)",
                     batch_start, batch_start + len(batch), len(rows))
            success += len(rows)
            continue

        try:
            client.upsert(collection_name=COLLECTION_NAME, data=rows)
            success += len(rows)
            log.info(
                "[%d/%d] Upserted batch (%d rows)  appids %d–%d",
                batch_start + len(batch), total, len(rows),
                rows[0]["appid"], rows[-1]["appid"],
            )
        except Exception as e:
            error += len(rows)
            log.error("Batch %d–%d failed: %s",
                      batch_start, batch_start + len(batch), e)

    return success, error


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Populate Zilliz with historical SHAP anchors"
    )
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--dry-run",  action="store_true",
                        help="Validate without writing to Zilliz")
    parser.add_argument("--rebuild",  action="store_true",
                        help="Drop and recreate collection first")
    args = parser.parse_args()

    log.info("=== seed_vector_db.py ===")
    log.info("Model version : %s", args.model_version)
    log.info("Dry run       : %s", args.dry_run)
    log.info("Rebuild       : %s", args.rebuild)

    # Load model + top-25 feature list
    booster, top25_features, genre_cols = load_model_and_features(args.model_version)

    # Get all feature columns from model
    all_feature_cols = booster.feature_names
    log.info("Model has %d features total", len(all_feature_cols))

    # Optional: drop + recreate collection
    if args.rebuild and not args.dry_run:
        from api.services.zilliz import COLLECTION_NAME, get_client
        client = get_client()
        if client and client.has_collection(COLLECTION_NAME):
            client.drop_collection(COLLECTION_NAME)
            log.info("Dropped existing collection '%s'", COLLECTION_NAME)

    # Fetch labeled snapshots from DB
    db = get_db()
    snapshots = fetch_labeled_snapshots(db)

    if not snapshots:
        log.warning("No labeled snapshots found — check outcome labels in games_v2")
        return

    # Compute SHAP vectors
    snapshots = compute_shap_vectors(snapshots, booster,
                                     all_feature_cols, top25_features, genre_cols)

    # Upsert to Zilliz
    t0 = time.time()
    success, errors = upsert_to_zilliz(snapshots, dry_run=args.dry_run)
    elapsed = time.time() - t0

    log.info("=" * 60)
    log.info("seed_vector_db.py complete  [%.1fs]", elapsed)
    log.info("  Upserted : %d", success)
    log.info("  Errors   : %d", errors)
    log.info("  Dry run  : %s", args.dry_run)
    if errors:
        log.warning("Some batches failed — re-run to retry")


if __name__ == "__main__":
    main()
