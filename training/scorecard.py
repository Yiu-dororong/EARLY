"""
scorecard.py
------------
Computes L1 scorecard scores for all snapshots and writes results to
the scorecard table.

Architecture:
    1. Load snapshot rows joined with game_genres
    2. For each snapshot:
       a. Check hard abandon override
       b. Normalise features using FEATURE_SCALES
       c. Compute backbone base_score per dimension (weighted avg of 90d features)
       d. Compute momentum_delta per dimension (weighted avg of centered 30d features)
       e. dimension_score applies momentum proportionally to remaining headroom (0 to 1)
       f. composite_score = weighted avg of dimension scores
       g. Classify state from STATE_THRESHOLDS
    3. Write to scorecard table

Output table: scorecard
    appid, snapshot_date, config_version,
    l1_update_health_score, l1_player_retention_score,
    l1_dev_engagement_score, l1_sentiment_score, l1_price_market_score,
    l1_composite_score, l1_state, ml_eligible,
    null_feature_count, computed_at

Usage:
    python scorecard.py [--dry-run] [--appid APPID] [--limit N]
"""

import argparse
import logging
import math
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

import libsql
import pandas as pd

from evaluation.scorecard_config import (
    CONFIG_VERSION,
    DIMENSION_FEATURES,
    DIMENSION_WEIGHTS,
    FEATURE_SCALES,
    HARD_ABANDON_BUILD_GAP_DAYS,
    HARD_ABANDON_MIN_EA_AGE,
    ML_ELIGIBLE_MIN_REVIEWS,
    MOMENTUM_LR,
    STATE_THRESHOLDS,
)

# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

DB_URL = os.getenv("TURSO_URL")
DB_AUTH = os.getenv("TURSO_AUTH_TOKEN")


def get_conn() -> libsql.Connection:
    if DB_URL and DB_AUTH:
        return libsql.connect(DB_URL, auth_token=DB_AUTH)
    return libsql.connect("early.db")


CREATE_SCORECARD_TABLE = """
CREATE TABLE IF NOT EXISTS scorecard (
    appid                       INTEGER NOT NULL,
    snapshot_date               TEXT    NOT NULL,
    config_version              TEXT,
    l1_update_health_score      REAL,
    l1_player_retention_score   REAL,
    l1_dev_engagement_score     REAL,
    l1_sentiment_score          REAL,
    l1_price_market_score       REAL,
    l1_composite_score          REAL,
    l1_state                    TEXT,
    ml_eligible                 INTEGER,
    null_feature_count          INTEGER,
    computed_at                 TEXT,
    PRIMARY KEY (appid, snapshot_date)
)
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature normalisation
# ---------------------------------------------------------------------------

def normalise(value: float | None, scale: dict) -> float | None:
    """
    Normalises a raw feature value to [0.0, 1.0] using the given scale config.
    Returns None if value is None (caller handles NULL redistribution).
    1.0 = best health signal, 0.0 = worst.
    """
    if value is None:
        return None

    t = scale["type"]

    if t == "inverted_cap":
        cap = scale["cap"]
        return max(0.0, 1.0 - (value / cap))

    elif t == "log_cap":
        cap = scale["cap"]
        if value <= 0:
            return 0.0
        return min(1.0, math.log1p(value) / math.log1p(cap))

    elif t == "linear":
        lo, hi = scale["min"], scale["max"]
        if hi == lo:
            return 0.5
        if scale.get("inverse", False):
            if value == 0:
                value = hi
            else:
                value = 1 / value
        return max(0.0, min(1.0, (value - lo) / (hi - lo)))

    elif t == "clamp":
        # Used for MOMENTUM: Simply caps extreme values but preserves the zero-center.
        # e.g., "min": -1.0, "max": 1.0
        lo, hi = scale["min"], scale["max"]
        
        # No division! Just a strict ceiling and floor.
        return max(lo, min(hi, value))

    elif t == "centered_ratio":
        # Used for MOMENTUM: Translates a raw ratio (e.g., 2.0x, 0.5x) 
        # into a bounded [-1.0, 1.0] momentum score.
        
        # Extract the half_range from the config, defaulting to 1.0 for safety
        half_range = scale.get("half_range", 1.0)
        
        if half_range <= 0:
            raise ValueError("half_range for centered_ratio must be strictly positive.")
            
        # Step 1: Shift the ratio so a steady state (1.0) becomes exactly 0.0
        shifted = value - 1.0
        
        # Step 2: Apply the range divisor
        raw_momentum = shifted / half_range
        
        # Step 3: Hard clamp to [-1.0, 1.0] to protect the scorecard
        return max(-1.0, min(1.0, raw_momentum))
    
    elif t == "symlog_norm":
        cap = scale.get("cap", 1000)
        if value is None:
            return 0.5  # Neutral fallback for 0 slope / no data
        
        # math.log1p is a fast, safe built-in for log(|x| + 1)
        symlog_val = math.copysign(math.log1p(abs(value)), value)
        symlog_cap = math.log1p(cap)
        
        # Normalize to [-1.0, 1.0], then shift to [0.0, 1.0]
        v_norm = max(-1.0, min(1.0, symlog_val / symlog_cap))
        score = (v_norm + 1.0) / 2.0

        return score

    elif t == "binary":
        return 1.0 if value else 0.0

    elif t == "binary_inverted":
        return 0.0 if value else 1.0

    elif t == "inverted_distance":
        anchor = scale["anchor"]
        if value <= 0:
            return 0.0
        if value < anchor:
            # Below anchor — linear penalty, 0 at 0, 1 at anchor
            return value / anchor
        else:
            # Above anchor — mild penalty for extreme overpricing
            # Soft decay: 1.0 at anchor, ~0.7 at 2x anchor
            return max(0.0, 1.0 - ((value - anchor) / (2.0 * anchor)) * 0.3)

    else:
        log.warning("Unknown scale type '%s' — returning None", t)
        return None


# ---------------------------------------------------------------------------
# Weighted average with NULL redistribution
# ---------------------------------------------------------------------------

def weighted_avg_with_redistribution(
    values: dict[str, float | None],
    weights: dict[str, float],
) -> tuple[float | None, int]:
    """
    Computes weighted average of normalised values, redistributing weight
    from NULL features proportionally to non-NULL features.

    Returns (score, null_count).
    Returns (None, total) if all features are NULL.
    """
    null_count = sum(1 for v in values.values() if v is None)
    valid = {k: v for k, v in values.items() if v is not None}

    if not valid:
        return None, null_count

    total_valid_weight = sum(weights[k] for k in valid)
    if total_valid_weight == 0:
        return None, null_count

    score = sum(
        (v * weights[k] / total_valid_weight)
        for k, v in valid.items()
    )
    return score, null_count


# ---------------------------------------------------------------------------
# Dimension scoring
# ---------------------------------------------------------------------------

def score_dimension(
    dim: str,
    row: dict,
    lr: float,
) -> tuple[float | None, int]:
    """
    Computes a single dimension score for one snapshot row.

    Returns (dimension_score, null_count_for_this_dimension).

    Steps:
        1. Normalise backbone features → base_score
        2. Normalise momentum features, center at 0.5 → momentum_delta
        3. dimension_score applies momentum proportionally to remaining room
           so it never falls outside [0, 1].
    """
    cfg      = DIMENSION_FEATURES[dim]
    backbone = cfg["backbone"]
    momentum = cfg["momentum"]
    total_nulls = 0

    # ---- Base score (90d backbone) ---------------------------------------
    backbone_normalised = {}
    for feat, _ in backbone.items():
        raw   = row.get(feat)
        scale = FEATURE_SCALES.get(feat)
        backbone_normalised[feat] = normalise(raw, scale) if scale else None

    base_score, n_nulls = weighted_avg_with_redistribution(
        backbone_normalised, backbone
    )
    total_nulls += n_nulls

    if base_score is None:
        return None, total_nulls

    # ---- Momentum delta (30d, zero-centered) -----------------------------
    if not momentum:
        return round(base_score, 6), total_nulls

    momentum_normalised = {}
    for feat, _ in momentum.items():
        raw   = row.get(feat)
        scale = FEATURE_SCALES.get(feat)
        norm  = normalise(raw, scale) if scale else None
        # Center: shift [0,1] → [-0.5, +0.5]
        momentum_normalised[feat] = (norm - 0.5) if norm is not None else None

    momentum_delta, n_nulls = weighted_avg_with_redistribution(
        momentum_normalised, momentum
    )
    total_nulls += n_nulls

    # If momentum is all null, fall back to base score only
    if momentum_delta is None:
        return round(base_score, 6), total_nulls

    if momentum_delta > 0:
        dimension_score = base_score + (lr * momentum_delta * (1.0 - base_score))
    elif momentum_delta < 0:
        dimension_score = base_score + (lr * momentum_delta * base_score)
    else:
        dimension_score = base_score

    return round(dimension_score, 6), total_nulls


# ---------------------------------------------------------------------------
# Hard override
# ---------------------------------------------------------------------------

def check_hard_abandon(row: dict) -> bool:
    """
    Returns True if game meets hard abandon criteria:
        days_since_last_build_update >= HARD_ABANDON_BUILD_GAP_DAYS
        AND ea_age_days >= HARD_ABANDON_MIN_EA_AGE
    """
    build_gap = row.get("days_since_last_build_update")
    ea_age    = row.get("ea_age_days")

    if build_gap is None or ea_age is None:
        return False

    return (
        build_gap >= HARD_ABANDON_BUILD_GAP_DAYS
        and ea_age >= HARD_ABANDON_MIN_EA_AGE
    )


# ---------------------------------------------------------------------------
# State classification
# ---------------------------------------------------------------------------

def classify_state(composite: float | None) -> str:
    if composite is None:
        return "Unknown"
    for threshold, label in STATE_THRESHOLDS:
        if composite >= threshold:
            return label
    return "At risk"


# ---------------------------------------------------------------------------
# Full scorecard for one row
# ---------------------------------------------------------------------------

def compute_scorecard(row: dict) -> dict:
    """
    Computes all scorecard fields for a single snapshot row dict.
    Returns a result dict ready for DB insertion.
    """
    result = {
        "appid":                       row["appid"],
        "snapshot_date":               row["snapshot_date"],
        "config_version":              CONFIG_VERSION,
        "l1_update_health_score":      None,
        "l1_player_retention_score":   None,
        "l1_dev_engagement_score":     None,
        "l1_sentiment_score":          None,
        "l1_price_market_score":       None,
        "l1_composite_score":          None,
        "l1_state":                    None,
        "ml_eligible":                 None,
        "null_feature_count":          0,
        "computed_at":                 datetime.now(timezone.utc).isoformat(),
    }

    # ---- ml_eligible -------------------------------------------------------
    review_count = row.get("review_count_at_T")
    result["ml_eligible"] = int(
        review_count is not None and review_count >= ML_ELIGIBLE_MIN_REVIEWS
    )

    # ---- Hard override -----------------------------------------------------
    if check_hard_abandon(row):
        result["l1_state"] = "At risk"
        result["l1_composite_score"] = 0.0
        return result

    # ---- Dimension scores --------------------------------------------------
    dim_map = {
        "update_health":    "l1_update_health_score",
        "player_retention": "l1_player_retention_score",
        "dev_engagement":   "l1_dev_engagement_score",
        "sentiment":        "l1_sentiment_score",
        "price_market":     "l1_price_market_score",
    }

    total_nulls  = 0
    dim_scores   = {}

    for dim, col in dim_map.items():
        lr    = MOMENTUM_LR[dim]
        score, n_nulls = score_dimension(dim, row, lr)
        result[col]    = score
        total_nulls   += n_nulls
        if score is not None:
            dim_scores[dim] = score

    result["null_feature_count"] = total_nulls

    # ---- Composite score ---------------------------------------------------
    if dim_scores:
        total_weight = sum(DIMENSION_WEIGHTS[d] for d in dim_scores)
        composite    = sum(
            dim_scores[d] * DIMENSION_WEIGHTS[d] / total_weight
            for d in dim_scores
        )
        result["l1_composite_score"] = round(composite, 6)
    else:
        result["l1_composite_score"] = None

    # ---- State classification ----------------------------------------------
    result["l1_state"] = classify_state(result["l1_composite_score"])

    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# All features referenced in DIMENSION_FEATURES + context columns
SNAPSHOT_COLUMNS = [
    # Context
    "appid", "snapshot_date", "ea_age_days", "review_count_at_T",
    # Update Health
    "days_since_last_build_update", "build_update_count_last_90d",
    "build_update_count_last_30d",  "update_frequency_trend",
    "max_hiatus_ever_days",         "hiatus_recovery_count",
    # Player Retention
    "ccu_vs_peak_ratio",        "ccu_trend_slope_90d",  "ccu_trend_slope_30d",
    "ccu_floor_established",    "ccu_vs_launch_ratio",  "days_since_ccu_above_100",
    # Dev Engagement
    "build_to_post_ratio", "days_since_dev_post",
    "dev_posts_last_90d",  "dev_posts_last_30d",
    "dev_previous_ea_count", "dev_has_prior_success", "dev_total_games_shipped",
    # Sentiment
    "review_score_at_T",      "review_score_delta_30d",
    "review_velocity_90d",    "review_velocity_30d",
    "review_score_last_90d",  "review_score_last_30d",
    # Price & Market
    "price_vs_genre_median", "early_deep_discount_flag", "discount_frequency",
]


def load_snapshots(
    conn: libsql.Connection,
    appid: int | None = None,
    limit: int | None = None,
) -> list[dict]:
    cols      = ", ".join(f"s.{c}" for c in SNAPSHOT_COLUMNS)
    where     = f"WHERE s.appid = {appid}" if appid else ""
    limit_sql = f"LIMIT {limit}" if limit else ""

    query = f"""
        SELECT {cols}
        FROM snapshots s
        {where}
        ORDER BY s.appid, s.snapshot_date
        {limit_sql}
    """
    rows = conn.execute(query).fetchall()
    return [dict(zip(SNAPSHOT_COLUMNS, r)) for r in rows]


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def upsert_batch(conn: libsql.Connection, results: list[dict]) -> None:
    batch_size = 500
    total = len(results)
    for i in range(0, total, batch_size):
        chunk = results[i:i + batch_size]
        conn.executemany(
            """
            INSERT INTO scorecard (
                appid, snapshot_date, config_version,
                l1_update_health_score, l1_player_retention_score,
                l1_dev_engagement_score, l1_sentiment_score,
                l1_price_market_score, l1_composite_score,
                l1_state, ml_eligible, null_feature_count, computed_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(appid, snapshot_date) DO UPDATE SET
                config_version              = excluded.config_version,
                l1_update_health_score      = excluded.l1_update_health_score,
                l1_player_retention_score   = excluded.l1_player_retention_score,
                l1_dev_engagement_score     = excluded.l1_dev_engagement_score,
                l1_sentiment_score          = excluded.l1_sentiment_score,
                l1_price_market_score       = excluded.l1_price_market_score,
                l1_composite_score          = excluded.l1_composite_score,
                l1_state                    = excluded.l1_state,
                ml_eligible                 = excluded.ml_eligible,
                null_feature_count          = excluded.null_feature_count,
                computed_at                 = excluded.computed_at
            """,
            [
                (
                    r["appid"], r["snapshot_date"], r["config_version"],
                    r["l1_update_health_score"], r["l1_player_retention_score"],
                    r["l1_dev_engagement_score"], r["l1_sentiment_score"],
                    r["l1_price_market_score"], r["l1_composite_score"],
                    r["l1_state"], r["ml_eligible"],
                    r["null_feature_count"], r["computed_at"],
                )
                for r in chunk
            ],
        )
        conn.commit()

        processed = i + len(chunk)
        log.info("  Upserted %d/%d rows (%.1f%%)", processed, total, processed / total * 100)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(dry_run: bool, appid: int | None, limit: int | None) -> None:
    conn = get_conn()
    conn.execute(CREATE_SCORECARD_TABLE)
    conn.commit()

    existing_versions = {r[0] for r in conn.execute("SELECT DISTINCT config_version FROM scorecard").fetchall() if r[0]}
    if CONFIG_VERSION in existing_versions:
        log.warning(
            "CONFIG_VERSION '%s' already exists in the database. "
            "Records will be overwritten. Did you forget to bump the version?", CONFIG_VERSION
        )

    log.info("Loading snapshots (config=%s)...", CONFIG_VERSION)
    rows = load_snapshots(conn, appid=appid, limit=limit)
    log.info("Loaded %d snapshot rows.", len(rows))

    results    = []
    state_dist = {}
    hard_overrides = 0
    hard_override_appids = set()

    for i, row in enumerate(rows, 1):
        res = compute_scorecard(row)
        results.append(res)

        state = res["l1_state"]
        state_dist[state] = state_dist.get(state, 0) + 1

        if state == "At risk" and res["l1_composite_score"] == 0.0:
            hard_overrides += 1
            hard_override_appids.add(res["appid"])

        if i % 1000 == 0:
            log.info("[%d/%d] state_dist=%s", i, len(rows), state_dist)

    log.info("=" * 60)
    log.info("Scorecard complete. config=%s", CONFIG_VERSION)
    log.info("State distribution:")
    total = len(results)
    for state, count in sorted(state_dist.items(), key=lambda x: -x[1]):
        log.info("  %-20s %4d  (%.1f%%)", state, count, 100 * count / total)
    log.info("Hard abandon overrides: %d snapshots (%d games)", hard_overrides, len(hard_override_appids))
    log.info("=" * 60)

    if dry_run:
        log.info("Dry run — sample results (first 5):")
        for r in results[:5]:
            log.info(
                "  appid=%-8d  date=%s  composite=%.3f  state=%s  nulls=%d",
                r["appid"], r["snapshot_date"],
                r["l1_composite_score"] or 0,
                r["l1_state"],
                r["null_feature_count"],
            )
        log.info("Dry run complete — no writes.")
    else:
        log.info("Writing %d rows to scorecard table...", len(results))
        upsert_batch(conn, results)
        log.info("Done.")

    conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute L1 scorecard for all snapshots."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute but do not write to DB")
    parser.add_argument("--appid", type=int, default=None,
                        help="Process a single appid only (for debugging)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N snapshots (for testing)")
    args = parser.parse_args()

    run(args.dry_run, args.appid, args.limit)
