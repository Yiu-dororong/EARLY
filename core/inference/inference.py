"""
inference.py — EARLY pipeline: live scorer for STAYS_ACTIVE games
==================================================================

Scores all games in the STAYS_ACTIVE pool using the trained XGBoost model.
For each game, produces:
  - P(IS_DISTRESSED)        — raw probability from the model
  - is_distressed           — binary classification at OOF threshold
  - l1_state                — Layer 1 scorecard state (rule-based, always run)
  - null_features           — list of features that were None at inference time
  - scored_at               — UTC timestamp of this run

Feature construction is delegated entirely to feature_builder.py. The same
build_features() call used in build_snapshots.py is used here — this is the
point of the refactor. Any feature computed differently here vs training is
a silent model bug.

─────────────────────────────────────────────────────────────────────────────
MODEL ARTIFACT REQUIREMENTS
─────────────────────────────────────────────────────────────────────────────
Inference requires two files produced by train_xgboost.py:

  models/xgb_v1.0.json          — trained XGBoost booster
  models/xgb_v1.0_features.json — exact ordered feature list + OOF threshold
                                   {
                                     "features": ["col_a", "col_b", ...],
                                     "threshold": 0.5364,
                                     "genre_columns": ["genre__action", ...]
                                   }

The feature list governs column ordering passed to xgb.DMatrix. Any column
in the list that is absent from the computed feature dict is filled with None
(maps to NaN in DMatrix — XGBoost null routing applies). This is correct
and expected for stub features not yet populated.

─────────────────────────────────────────────────────────────────────────────
GENRE ENCODING
─────────────────────────────────────────────────────────────────────────────
Genre columns are stored in xgb_v1.0_features.json under "genre_columns".
At inference:
  1. primary_genre is looked up from games_v2
  2. One-hot encoded using known genre_columns (training set)
  3. Unknown genres → all-zero row (logged as warning)
  4. Missing genre → all-zero row (logged as warning)

─────────────────────────────────────────────────────────────────────────────
SNAPSHOT DATE = TODAY
─────────────────────────────────────────────────────────────────────────────
Inference uses today as the snapshot date (design decision: session 9).
ea_age_days = today - ea_start_date.
All lookback windows (30d, 90d, 180d) are relative to today.
Caching (nightly job → live_scores table) is the correct staleness strategy,
not inferring from a stale snapshot_date.

─────────────────────────────────────────────────────────────────────────────
ML ELIGIBILITY
─────────────────────────────────────────────────────────────────────────────
ml_eligible is evaluated here at score time using current review count.
  - ml_eligible = 0 → Layer 1 only; XGBoost score = None
  - ml_eligible = 1 → both layers; full output
This matches training (design decision 8): ml_eligible is snapshot-time,
not discovery-time.

─────────────────────────────────────────────────────────────────────────────
OUTPUT TABLE: live_scores
─────────────────────────────────────────────────────────────────────────────
Scores are written to the live_scores table (upsert on appid).
Each run overwrites the previous score for each game.
Columns:
  appid, scored_at, ea_age_days,
  p_distressed, is_distressed, l1_state,
  ml_eligible, model_version,
  null_features (JSON array of null feature names),
  review_count_at_T (for UI eligibility display),
  update_health, player_retention, dev_engagement, sentiment, price_market

─────────────────────────────────────────────────────────────────────────────
USAGE
─────────────────────────────────────────────────────────────────────────────
  python inference.py                      # score all STAYS_ACTIVE games
  python inference.py --appid 1145360      # single game debug
  python inference.py --model v1.1         # use a specific model version
  python inference.py --dry-run            # compute but don't write
  python inference.py --verbose            # debug logging
  python inference.py --no-itad            # skip ITAD price features
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import time
from datetime import date, datetime, timezone
from typing import Any

import libsql
import numpy as np
import xgboost as xgb
from dotenv import load_dotenv

load_dotenv()

DB_URL  = os.getenv("TURSO_URL", "")
DB_AUTH = os.getenv("TURSO_AUTH_TOKEN", "")

DEFAULT_MODEL_VERSION = "v1.3"

# Dynamically locate the project root to ensure models directory is found regardless of cwd
PROJECT_ROOT          = Path(__file__).resolve().parent.parent.parent
MODEL_DIR             = PROJECT_ROOT / "models"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_conn() -> libsql.Connection:
    if DB_URL and DB_AUTH:
        return libsql.connect(DB_URL, auth_token=DB_AUTH)
    return libsql.connect("early.db")


def ensure_live_scores_table(conn: libsql.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS live_scores (
            appid               INTEGER PRIMARY KEY,
            scored_at           INTEGER NOT NULL,
            ea_age_days         INTEGER,
            p_distressed        REAL,
            is_distressed       INTEGER,
            l1_state            TEXT,
            ml_eligible         INTEGER,
            model_version       TEXT,
            null_features       TEXT,    -- JSON array of null feature names
            review_count_at_T   INTEGER,
            update_health       REAL,
            player_retention    REAL,
            dev_engagement      REAL,
            sentiment           REAL,
            price_market        REAL
        )
    """)
    conn.commit()
    log.info("live_scores table ready")


def load_latest_live_snapshots(
    conn: libsql.Connection,
    appid_filter: int | None,
) -> list[dict]:
    cols = conn.execute("PRAGMA table_info(live_snapshots)").fetchall()
    col_names = [c[1] for c in cols]

    where  = "WHERE outcome = 'STAYS_ACTIVE'"
    params: list = []

    if appid_filter is not None:
        where += " AND appid = ?"
        params.append(appid_filter)

    rows = conn.execute(f"""
        SELECT {','.join(col_names)}
        FROM live_snapshots
        {where}
        AND snapshot_date = (
            SELECT MAX(snapshot_date) FROM live_snapshots ls2 WHERE ls2.appid = live_snapshots.appid
        )
        ORDER BY appid
    """, params).fetchall()

    return [
        dict(zip(col_names, row)) for row in rows
    ]


def upsert_score(conn: libsql.Connection, score: dict) -> None:
    conn.execute("""
        INSERT OR REPLACE INTO live_scores (
            appid, scored_at, ea_age_days,
            p_distressed, is_distressed, l1_state,
            ml_eligible, model_version,
            null_features, review_count_at_T,
            update_health, player_retention, dev_engagement, sentiment, price_market
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        score["appid"],
        score["scored_at"],
        score["ea_age_days"],
        score["p_distressed"],
        score["is_distressed"],
        score["l1_state"],
        score["ml_eligible"],
        score["model_version"],
        json.dumps(score["null_features"]),
        score["review_count_at_T"],
        score["update_health"],
        score["player_retention"],
        score["dev_engagement"],
        score["sentiment"],
        score["price_market"],
    ))


# ---------------------------------------------------------------------------
# Model artifact loading
# ---------------------------------------------------------------------------

def load_model_artifacts(version: str) -> tuple[xgb.Booster, list[str], list[str], float]:
    """
    Load model booster + feature metadata from models/{version}.json and
    models/{version}_features.json.

    Returns
    -------
    booster       : trained XGBoost Booster
    feature_cols  : ordered list of all feature columns (non-genre)
    genre_cols    : ordered list of one-hot genre columns
    threshold     : OOF F1-maximising classification threshold
    """
    model_path   = MODEL_DIR / f"xgb_{version}.json"
    feature_path = MODEL_DIR / f"xgb_{version}_features.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run train_xgboost.py first to produce model artifacts."
        )
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature metadata not found: {feature_path}\n"
            f"Ensure train_xgboost.py writes features + threshold to this file."
        )

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    log.info("Loaded model: %s", model_path)

    with open(feature_path) as f:
        meta = json.load(f)

    feature_cols = meta["features"]
    genre_cols   = meta.get("genre_columns", [])
    threshold    = float(meta["threshold"])

    log.info(
        "Feature metadata: %d features, %d genre columns, threshold=%.4f",
        len(feature_cols), len(genre_cols), threshold,
    )
    return booster, feature_cols, genre_cols, threshold


# ---------------------------------------------------------------------------
# Genre one-hot encoding
# ---------------------------------------------------------------------------

def encode_genre(primary_genre: str | None, genre_cols: list[str]) -> dict[str, int]:
    """
    Produce a {col: 0|1} dict for all genre_cols.
    Unknown or missing genres → all-zero row.

    genre_cols entries are formatted as "genre_<name>" (e.g., "genre_Action").
    """
    encoded = {col: 0 for col in genre_cols}

    if not primary_genre:
        log.warning("primary_genre is None — all genre columns set to 0")
        return encoded

    expected_col = f"genre_{primary_genre}"
    if expected_col in encoded:
        encoded[expected_col] = 1
    else:
        log.warning(
            "Unknown genre %r (col %r not in training set) — all genre columns set to 0. "
            "Consider patching xgb_%s_features.json if this is a new Steam tag.",
            primary_genre, expected_col, DEFAULT_MODEL_VERSION,
        )
    return encoded


# ---------------------------------------------------------------------------
# Layer 1 scorecard — thin wrapper
# ---------------------------------------------------------------------------

def run_scorecard(features: dict) -> dict | None:
    """
    Run the Layer 1 scorecard on a feature dict.
    Returns the result dict from compute_scorecard, or None if unavailable.
    """
    try:
        from training.scorecard import compute_scorecard
        result = compute_scorecard(features)
        return result
    except ImportError:
        log.debug("scorecard.py not available — l1_state will be None")
        return None
    except Exception as e:
        log.warning("Scorecard error for appid %s: %s", features.get("appid"), e)
        return None


# ---------------------------------------------------------------------------
# Feature vector assembly
# ---------------------------------------------------------------------------

def assemble_feature_row(
    features: dict,
    genre_encoded: dict[str, int],
    all_feature_cols: list[str],
) -> tuple[list[float | None], list[str]]:
    """
    Merge raw features + genre one-hot into a single ordered row matching
    all_feature_cols. Tracks which features were None (null_features list).

    Returns
    -------
    row           : list of float | None in column order (None → NaN in DMatrix)
    null_features : list of column names that were None
    """
    merged       = {**features, **genre_encoded}
    row          = []
    null_features = []

    for col in all_feature_cols:
        val = merged.get(col)
        if val is None:
            null_features.append(col)
            row.append(float("nan"))
        else:
            try:
                row.append(float(val))
            except (TypeError, ValueError):
                # Non-numeric feature (e.g. price_trend text before encoding)
                # These should have been encoded upstream; log and null-route.
                log.warning("Non-numeric value for feature %r: %r — treating as null", col, val)
                null_features.append(col)
                row.append(float("nan"))

    return row, null_features


# ---------------------------------------------------------------------------
# price_trend encoding
# ---------------------------------------------------------------------------
# price_trend is stored as text in features dict ('decreased'/'stable'/'increased').
# XGBoost expects numeric. Encode before assembling the feature row.
# Decision 37: decrease=-1, stable=0, increase=1 (semantic ordinal).

PRICE_TREND_ENCODING = {"decreased": -1, "stable": 0, "increased": 1}


def encode_price_trend(features: dict) -> dict:
    """Return a copy of features with price_trend replaced by its numeric encoding."""
    encoded = dict(features)
    raw = encoded.get("price_trend")
    if raw is not None:
        encoded["price_trend"] = PRICE_TREND_ENCODING.get(raw)
        if encoded["price_trend"] is None:
            log.warning("Unexpected price_trend value %r — treating as null", raw)
    return encoded


# ---------------------------------------------------------------------------
# Single-game scorer
# ---------------------------------------------------------------------------

def score_game(
    features: dict,
    booster: xgb.Booster,
    all_feature_cols: list[str],
    genre_cols: list[str],
    threshold: float,
    model_version: str,
) -> dict:
    """
    Take a pre-computed live_snapshots feature row and produce a score record.

    Returns a score dict ready for upsert_score(), or raises on unrecoverable error.
    """
    appid     = features["appid"]
    scored_at = int(datetime.now(timezone.utc).timestamp())
    ea_age_days = features.get("ea_age_days", 0)

    # Encode price_trend to numeric (decision 37)
    features = encode_price_trend(features)

    # Inject context columns required by evaluation.scorecard.compute_scorecard

    # Layer 1 — always run, regardless of ml_eligible
    l1_result = run_scorecard(features)
    if l1_result:
        l1_state = l1_result.get("l1_state")
        update_health = l1_result.get("l1_update_health_score")
        player_retention = l1_result.get("l1_player_retention_score")
        dev_engagement = l1_result.get("l1_dev_engagement_score")
        sentiment = l1_result.get("l1_sentiment_score")
        price_market = l1_result.get("l1_price_market_score")

        # Inject scores back into features dict using column names from build_snapshots.py
        features["l1_composite_score"] = l1_result.get("l1_composite_score")
        features["l1_update_health_score"] = update_health
        features["l1_player_retention_score"] = player_retention
        features["l1_dev_engagement_score"] = dev_engagement
        features["l1_community_sentiment_score"] = sentiment
        features["l1_price_signals_score"] = price_market

        state_map = {"Healthy": 2, "Watch": 1, "At risk": 0}
        features["l1_state_encoded"] = state_map.get(l1_state)
    else:
        l1_state = None
        update_health = None
        player_retention = None
        dev_engagement = None
        sentiment = None
        price_market = None

    review_count = features.get("review_count_at_T") or 0
    ml_eligible  = features.get("ml_eligible", 0)

    # Layer 2 — XGBoost, only if ml_eligible
    p_distressed: float | None  = None
    is_distressed: int | None   = None
    null_features: list[str]    = []

    if ml_eligible:
        # Genre one-hot encoding (decision 24)
        genre_encoded = encode_genre(features.get("primary_genre"), genre_cols)

        # Assemble ordered feature row
        row, null_features = assemble_feature_row(features, genre_encoded, all_feature_cols)

        # Predict
        dmat = xgb.DMatrix(
            data=np.array([row], dtype=np.float32),
            feature_names=all_feature_cols,
        )
        prob = float(booster.predict(dmat)[0])

        p_distressed  = round(prob, 6)
        is_distressed = 1 if prob >= threshold else 0

        log.debug(
            "appid %-10d  p=%.4f  distressed=%d  nulls=%d  l1=%s",
            appid, p_distressed, is_distressed, len(null_features), l1_state,
        )
    else:
        log.debug(
            "appid %-10d  ml_ineligible (reviews=%d)  l1=%s",
            appid, review_count, l1_state,
        )

    return {
        "appid":            appid,
        "scored_at":        scored_at,
        "ea_age_days":      ea_age_days,
        "p_distressed":     p_distressed,
        "is_distressed":    is_distressed,
        "l1_state":         l1_state,
        "ml_eligible":      ml_eligible,
        "model_version":    model_version,
        "null_features":    null_features,
        "review_count_at_T": review_count,
        "update_health":    update_health,
        "player_retention": player_retention,
        "dev_engagement":   dev_engagement,
        "sentiment":        sentiment,
        "price_market":     price_market,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score STAYS_ACTIVE games with EARLY model")
    p.add_argument("--appid",   type=int,  help="Single appid (debug)")
    p.add_argument("--model",   type=str,  default=DEFAULT_MODEL_VERSION,
                   help="Model version string, e.g. v1.0 (default) or v1.1")
    p.add_argument("--dry-run", action="store_true", help="Compute but don't write scores")
    p.add_argument("--no-itad", action="store_true", help="Skip ITAD price features")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    today = datetime.now(timezone.utc).date()
    conn  = get_conn()
    ensure_live_scores_table(conn)

    # Load model artifacts
    try:
        booster, feature_cols, genre_cols, threshold = load_model_artifacts(args.model)
    except FileNotFoundError as e:
        log.error("%s", e)
        return

    all_feature_cols = feature_cols + genre_cols

    # Load pre-computed snapshot candidates
    candidates = load_latest_live_snapshots(conn, args.appid)
    log.info(
        "Scoring %d STAYS_ACTIVE game%s with model xgb_%s (threshold=%.4f)",
        len(candidates), "s" if len(candidates) != 1 else "",
        args.model, threshold,
    )

    if not candidates:
        log.info("No STAYS_ACTIVE games found.")
        return

    n_ok             = 0
    n_error          = 0
    n_ineligible     = 0
    total_null_sum   = 0

    for i, features in enumerate(candidates, 1):
        appid = features["appid"]
        try:
            score = score_game(
                features=features,
                booster=booster,
                all_feature_cols=all_feature_cols,
                genre_cols=genre_cols,
                threshold=threshold,
                model_version=args.model,
            )

            if not args.dry_run:
                upsert_score(conn, score)

            n_ok += 1
            if not score["ml_eligible"]:
                n_ineligible += 1
            if score["null_features"]:
                total_null_sum += len(score["null_features"])

            # Progress log every 50 games or for single-appid runs
            if args.appid or i % 50 == 0 or i == len(candidates):
                log.info(
                    "[%d/%d] appid %-10d  ea_age=%dd  p=%-8s  l1=%-18s  nulls=%d%s",
                    i, len(candidates),
                    score["appid"],
                    score["ea_age_days"],
                    f"{score['p_distressed']:.4f}" if score["p_distressed"] is not None else "N/A",
                    score["l1_state"] or "N/A",
                    len(score["null_features"]),
                    "  [DRY RUN]" if args.dry_run else "",
                )

        except Exception as e:
            log.error("[%d/%d] appid %d: failed — %s", i, len(candidates), appid, e)
            n_error += 1

    if not args.dry_run:
        conn.commit()

    conn.close()

    log.info("=" * 64)
    log.info("inference.py complete  [%s]", "DRY RUN" if args.dry_run else "LIVE")
    log.info("  Model         : xgb_%s  threshold=%.4f", args.model, threshold)
    log.info("  Scored (ok)   : %d", n_ok)
    log.info("  ML ineligible : %d  (Layer 1 only — reviews < 50)", n_ineligible)
    log.info("  Errors        : %d", n_error)
    if n_ok > 0:
        avg_nulls = total_null_sum / n_ok
        log.info("  Avg null features per game: %.1f", avg_nulls)
        if avg_nulls > 5:
            log.warning(
                "  High null rate (%.1f avg). Check stub features "
                "(dev_*, changelog_*, l1_*) and whether ITAD is active.",
                avg_nulls,
            )
    log.info("=" * 64)


if __name__ == "__main__":
    main()
