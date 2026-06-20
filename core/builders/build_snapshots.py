"""
build_snapshots.py — EARLY pipeline: training snapshot assembler
================================================================

Builds the `snapshots` table — one row per game per snapshot point.
Each row is a fully materialised feature vector used for XGBoost training
(resolved-outcome games) or live inference (STAYS_ACTIVE games).

Feature construction is delegated entirely to feature_builder.py.
This module owns: snapshot date planning, DB I/O, and orchestration.

─────────────────────────────────────────────────────────────────────────────
SNAPSHOT STRATEGY
─────────────────────────────────────────────────────────────────────────────
Training snapshots (resolved outcomes: EXIT_SUCCESS / EXIT_ABANDONED /
EXIT_SILENT) are taken at elastic percentile intervals of completed EA duration:

  Base percentiles:   configurable via --lower, --upper, and --n-base
                      default: 4 snapshots evenly spaced between 0.25 and 0.70
  Elastic extra:      +1 snapshot per ELASTIC_INTERVAL_DAYS beyond
                      ELASTIC_THRESHOLD_DAYS of EA duration
                      default: +1 per 180d beyond 360d, capped at absolute max
  Minimum EA age:     MIN_EA_AGE_DAYS (default 90) — snapshots below this
                      are skipped, not floored
  graduation_window:  if ALL percentile snapshots fall below MIN_EA_AGE_DAYS
                      (fast-graduating games), one snapshot is taken at
                      ea_end_date - GRADUATION_WINDOW_OFFSET_DAYS

STAYS_ACTIVE games contribute to validation/inference only — their snapshots
are taken at the same percentile schedule relative to ea_age_days as of today.

─────────────────────────────────────────────────────────────────────────────
LOOK-AHEAD DISCIPLINE (design decision 16)
─────────────────────────────────────────────────────────────────────────────
ALL features are computed strictly from data available BEFORE snapshot_date.
  - CCU: only ccu_history rows with month <= snapshot_date
  - Reviews: only review_history buckets with bucket_start < snapshot_date
    (partial bucket straddling snapshot_date is linearly interpolated)
  - Events: only event_history rows with event_date < snapshot_date
  - ITAD: price history filtered to records at or before snapshot_date
  - Labels: outcome determined as of label_date = snapshot_date + 365 days
    (label_date must be <= today for resolved outcomes; STAYS_ACTIVE exempt)

─────────────────────────────────────────────────────────────────────────────
USAGE
─────────────────────────────────────────────────────────────────────────────
  python build_snapshots.py                    # full run
  python build_snapshots.py --appid 1145360    # single game debug
  python build_snapshots.py --dry-run          # plan only, no writes
  python build_snapshots.py --force            # rebuild all (drop existing)
  python build_snapshots.py --stays-active     # include STAYS_ACTIVE games
  python build_snapshots.py --verbose          # debug logging

  # Override elastic snapshot config
  python build_snapshots.py \\
      --lower 0.25 --upper 0.75 --n-base 3 \\
      --elastic-interval 120 \\
      --elastic-threshold 360 \\
      --max-snapshots 10
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone

import libsql
from dotenv import load_dotenv

from core.builders.feature_builder import (
    SnapshotPlan,
    build_features,
    compute_snapshot_percentiles,
)


load_dotenv()

# ---------------------------------------------------------------------------
# Config — all overridable via CLI
# ---------------------------------------------------------------------------

DB_URL  = os.getenv("TURSO_URL", "")
DB_AUTH = os.getenv("TURSO_AUTH_TOKEN", "")

# Pipeline constants (NOT feature-semantic — safe to tune without retraining)
MIN_EA_AGE_DAYS                = 90
GRADUATION_WINDOW_OFFSET_DAYS  = 7
DELTA_GRADUATION_DAYS          = 90

# Snapshot schedule defaults
DEFAULT_LOWER                  = 0.25
DEFAULT_UPPER                  = 0.70
DEFAULT_N_BASE                 = 4
DEFAULT_ELASTIC_THRESHOLD_DAYS = 360
DEFAULT_ELASTIC_INTERVAL_DAYS  = 180
DEFAULT_MAX_SNAPSHOTS          = 8
DB_MAX_RETRIES                 = 3
DB_RETRY_DELAY                 = 5.0

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


def ensure_snapshots_table(conn: libsql.Connection) -> None:
    """
    Create the snapshots table if it doesn't exist.
    All stub columns are present from day one — NULL until populated.
    snapshot_type is metadata only; excluded from model feature matrix.
    Primary key: (appid, snapshot_date) — one snapshot per game per date.
    """
    schema_cols = """
        -- Identity
        appid                           INTEGER NOT NULL,
        snapshot_date                   TEXT    NOT NULL,
        snapshot_type                   TEXT    NOT NULL,
        percentile_label                TEXT,
        ea_age_days                     INTEGER,
        ea_age_lt_180d                  INTEGER,

        -- ── Dimension 1: Update Health ───────────────────────────────
        days_since_last_build_update    INTEGER,
        build_update_count_last_30d     INTEGER,
        build_update_count_last_90d     INTEGER,
        build_update_count_last_180d    INTEGER,
        mean_days_between_build_updates REAL,
        std_days_between_build_updates  REAL,
        max_hiatus_ever_days            INTEGER,
        hiatus_recovery_count           INTEGER,
        update_frequency_trend          REAL,
        allowable_build_gap_days        INTEGER,
        build_gap_vs_allowable_ratio    REAL,
        avg_changelog_word_count        REAL,
        changelog_word_count_trend      REAL,
        substance_score_latest          REAL,
        fake_heartbeat_flag             INTEGER,

        -- ── Dimension 2: Player Retention ────────────────────────────
        ccu_avg_last_30d                REAL,
        ccu_avg_last_90d                REAL,
        ccu_avg_last_180d               REAL,
        ccu_median_all                  REAL,
        ccu_at_launch_30d               REAL,
        peak_ccu_alltime                REAL,
        ccu_vs_peak_ratio               REAL,
        ccu_vs_launch_ratio             REAL,
        -- ccu_trend_slope_30d: despite the name, stores month-over-month ratio
        -- (current_month_avg / prior_month_avg). Slope was 99.8% null due to
        -- monthly CCU granularity. Name preserved for schema compatibility.
        ccu_trend_slope_30d             REAL,
        ccu_trend_slope_90d             REAL,
        ccu_trend_slope_180d            REAL,
        ccu_floor_established           INTEGER,
        days_since_ccu_above_100        INTEGER,
        ccu_low_regime                  INTEGER,
        ccu_unavailable                 INTEGER,
        ccu_recovery_per_update_avg     REAL,
        ccu_recovery_trend              REAL,

        -- ── Dimension 3: Developer Engagement ────────────────────────
        dev_posts_last_30d              INTEGER,
        dev_posts_last_90d              INTEGER,
        dev_posts_last_180d             INTEGER,
        dev_engagement_trend            REAL,
        days_since_dev_post             INTEGER,
        build_to_post_ratio             REAL,
        dev_previous_ea_count           INTEGER,
        dev_has_prior_success           INTEGER,
        dev_total_games_shipped         INTEGER,

        -- ── Dimension 4: Community Sentiment ─────────────────────────
        review_count_at_T               INTEGER,
        review_positive_at_T            INTEGER,
        review_negative_at_T            INTEGER,
        review_score_at_T               REAL,
        review_score_last_30d           REAL,
        review_score_last_90d           REAL,
        review_score_last_180d          REAL,
        review_score_delta_30d          REAL,
        review_velocity_30d             REAL,
        review_velocity_90d             REAL,
        review_velocity_180d            REAL,
        review_velocity_trend           REAL,
        review_sentiment_shock          REAL,
        review_low_regime               INTEGER,

        -- ── Dimension 5: Price & Market Signals ──────────────────────
        initial_price_usd               REAL,
        current_price_at_T              REAL,
        discount_count_to_date          INTEGER,
        max_discount_ever_pct           REAL,
        early_deep_discount_flag        INTEGER,
        discount_frequency              REAL,
        price_trend                     TEXT,

        -- ── Cross-dimension / XGBoost-only ───────────────────────────
        owner_estimate_at_T             INTEGER,
        primary_genre                   TEXT,
        price_vs_genre_median           REAL,
        ccu_vs_genre_weighted_median    REAL,
        update_freq_vs_genre_median     REAL,
        review_score_vs_genre_median    REAL,
        l1_composite_score              REAL,
        l1_update_health_score          REAL,
        l1_player_retention_score       REAL,
        l1_dev_engagement_score         REAL,
        l1_community_sentiment_score    REAL,
        l1_price_signals_score          REAL,
        l1_state_encoded                INTEGER,

        -- ── Label ─────────────────────────────────────────────────────
        outcome                         TEXT,
        label_date                      TEXT,
        label_is_resolved               INTEGER,
        ml_eligible                     INTEGER,

        -- ── Metadata ──────────────────────────────────────────────────
        collected_at                    INTEGER,

        PRIMARY KEY (appid, snapshot_date)
    """
    
    conn.execute(f"CREATE TABLE IF NOT EXISTS snapshots ({schema_cols})")
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_snapshots_appid ON snapshots (appid)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_snapshots_outcome ON snapshots (outcome)
    """)

    conn.execute(f"CREATE TABLE IF NOT EXISTS live_snapshots ({schema_cols})")
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_live_snapshots_appid ON live_snapshots (appid)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_live_snapshots_outcome 
                 ON live_snapshots (outcome)
    """)
    conn.commit()
    log.info("snapshots and live_snapshots tables ready")


# ---------------------------------------------------------------------------
# Snapshot date planning
# ---------------------------------------------------------------------------

def plan_snapshots(
    appid: int,
    ea_start: date,
    ea_end: date | None,
    abandoned_date: date | None,
    outcome: str,
    today: date,
    lower: float,
    upper: float,
    n_base: int,
    elastic_threshold_days: int,
    elastic_interval_days: int,
    max_snapshots: int,
) -> list[SnapshotPlan]:
    """
    Compute all snapshot dates for one game.

    Schedule:
      n = n_base + elastic extras, capped at max_snapshots.
      Elastic extras increase density (not range) within [lower, upper].

    Rules:
      1. EA duration = ea_end - ea_start (graduated) or today - ea_start (active)
      2. n = min(n_base + elastic_extras, max_snapshots)
      3. Percentiles evenly spaced across [lower, upper] for computed n
      4. Drop any snapshot where ea_age < MIN_EA_AGE_DAYS (skip, not floor)
      5. graduation_window: if ALL snapshots dropped (fast EXIT_SUCCESS),
         add one snapshot at ea_end - GRADUATION_WINDOW_OFFSET_DAYS
      6. STAYS_ACTIVE games use today as ea_end — same rules apply

    Returns list of SnapshotPlan sorted by snapshot_date ascending.
    """
    if outcome == "EXIT_SUCCESS" and ea_end:
        effective_end = ea_end
    elif outcome in ("EXIT_ABANDONED", "EXIT_SILENT") and abandoned_date:
        effective_end = abandoned_date
    else:
        effective_end = today

    ea_duration = (effective_end - ea_start).days
    if ea_duration <= 0:
        log.warning("appid %d: ea_duration=%d — skipping", appid, ea_duration)
        return []

    # Compute n: base + elastic extras
    n_extra = 0
    if ea_duration > elastic_threshold_days:
        excess  = ea_duration - elastic_threshold_days
        n_extra = min(
            int(excess // elastic_interval_days),
            max_snapshots - n_base,
        )
    n = n_base + n_extra

    all_percentiles = compute_snapshot_percentiles(lower, upper, n)

    log.debug(
        "appid %d: ea_duration=%dd n=%d (%d base + %d elastic) pcts=%s",
        appid, ea_duration, n, n_base, n_extra,
        [f"{p:.2f}" for p in all_percentiles],
    )

    plans: list[SnapshotPlan] = []

    for pct in all_percentiles:
        snap_offset = int(ea_duration * pct)
        snap_date   = ea_start + timedelta(days=snap_offset)
        snap_date   = min(snap_date, effective_end)
        ea_age      = (snap_date - ea_start).days

        if ea_age < MIN_EA_AGE_DAYS:
            log.debug(
                "appid %d pct=%.2f: ea_age=%dd < %dd floor — dropped",
                appid, pct, ea_age, MIN_EA_AGE_DAYS,
            )
            continue

        plans.append(SnapshotPlan(
            appid=appid,
            snapshot_date=snap_date,
            snapshot_type="percentile",
            percentile_label=f"pct_{int(round(pct * 100))}",
            ea_age_days=ea_age,
        ))

    # graduation_window fallback for fast-graduating EXIT_SUCCESS games
    if not plans and outcome == "EXIT_SUCCESS" and ea_end is not None:
        gw_date = ea_end - timedelta(days=GRADUATION_WINDOW_OFFSET_DAYS)
        gw_age  = (gw_date - ea_start).days

        if gw_age >= MIN_EA_AGE_DAYS:
            plans.append(SnapshotPlan(
                appid=appid,
                snapshot_date=gw_date,
                snapshot_type="graduation_window",
                percentile_label=None,
                ea_age_days=gw_age,
            ))
            log.debug(
                "appid %d: all percentiles below floor — "
                "graduation_window at %s (ea_age=%dd)",
                appid, gw_date, gw_age,
            )
        else:
            log.warning(
                "appid %d: EXIT_SUCCESS ea_duration=%dd — "
                "graduation_window also below %dd floor; no snapshots generated",
                appid, ea_duration, MIN_EA_AGE_DAYS,
            )

    return sorted(plans, key=lambda p: p.snapshot_date)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_ccu_history(conn: libsql.Connection, appid: int) -> tuple[list[dict], 
                                                            libsql.Connection]:
    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            rows = conn.execute("""
                SELECT month_date, avg_players, peak_players
                FROM ccu_history
                WHERE appid = ?
                ORDER BY month_date ASC
            """, (appid,)).fetchall()
            return [{"month": r[0], "avg": r[1], "peak": r[2]} for r in rows], conn
        except Exception as e:
            if attempt == DB_MAX_RETRIES: 
                log.warning("DB read error in load_ccu_history: "
                                  "%s - reconnecting", e)
                raise e
            time.sleep(DB_RETRY_DELAY)
            try: 
                conn.close()
            except Exception as e: 
                pass
            conn = get_conn()


def load_review_history(conn: libsql.Connection, appid: int) -> tuple[list[dict], 
                                                                libsql.Connection]:
    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            rows = conn.execute("""
                SELECT bucket_start, bucket_end, positive, negative
                FROM review_history
                WHERE appid = ?
                ORDER BY bucket_start ASC
            """, (appid,)).fetchall()
            return [
                {"start": r[0], "end": r[1], "positive": r[2], "negative": r[3]}
                for r in rows
            ], conn
        except Exception as e:
            if attempt == DB_MAX_RETRIES: 
                log.warning("DB read error in load_review_history: "
                                  "%s - reconnecting", e)
                raise e
            time.sleep(DB_RETRY_DELAY)
            try: 
                conn.close()
            except Exception as e: 
                pass
            conn = get_conn()


def load_event_history(conn: libsql.Connection, appid: int) -> tuple[list[dict], 
                                                                libsql.Connection]:
    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            rows = conn.execute("""
                SELECT datetime(event_ts, 'unixepoch') AS event_date,
                       event_type,
                       word_count
                FROM event_history
                WHERE appid = ?
                  AND event_gid != 'NONE'
                  AND is_automated = 0
                ORDER BY event_ts ASC
            """, (appid,)).fetchall()
            return [{"date": r[0], 
                     "type": r[1], 
                     "word_count": r[2]} for r in rows], conn
        except Exception as e:
            if attempt == DB_MAX_RETRIES: 
                log.warning("DB read error in load_event_history: "
                                  "%s - reconnecting", e)
                raise e
            time.sleep(DB_RETRY_DELAY)
            try: 
                conn.close()
            except Exception as e: 
                pass
            conn = get_conn()


def load_ccu_availability(conn: libsql.Connection, appid: int) -> tuple[str, 
                                                                        libsql.Connection]:
    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            row = conn.execute("""
                SELECT ccu_available FROM ccu_availability WHERE appid = ?
            """, (appid,)).fetchone()
            return (row[0] if row else "UNKNOWN"), conn
        except Exception as e:
            if attempt == DB_MAX_RETRIES: 
                log.warning("DB read error in load_ccu_availability: "
                                  "%s - reconnecting", e)
                raise e
            time.sleep(DB_RETRY_DELAY)
            try: 
                conn.close()
            except Exception as e: 
                pass
            conn = get_conn()


def load_initial_price(conn: libsql.Connection, appid: int) -> tuple[float | None, 
                                                                libsql.Connection]:
    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            row = conn.execute("""
                SELECT initial_price_usd FROM games_v2 WHERE appid = ?
            """, (appid,)).fetchone()
            return (row[0] if row else None), conn
        except Exception as e:
            if attempt == DB_MAX_RETRIES: 
                log.warning("DB read error in load_initial_price: "
                                  "%s - reconnecting", e)
                raise e
            time.sleep(DB_RETRY_DELAY)
            try: 
                conn.close()
            except Exception as e: 
                pass
            conn = get_conn()


# ---------------------------------------------------------------------------
# DB write
# ---------------------------------------------------------------------------

SNAPSHOT_COLUMNS = [
    "appid", "snapshot_date", "snapshot_type", "percentile_label",
    "ea_age_days", "ea_age_lt_180d",
    # D1
    "days_since_last_build_update",
    "build_update_count_last_30d", "build_update_count_last_90d",
    "build_update_count_last_180d",
    "mean_days_between_build_updates", "std_days_between_build_updates",
    "max_hiatus_ever_days", "hiatus_recovery_count",
    "update_frequency_trend", "allowable_build_gap_days",
    "build_gap_vs_allowable_ratio",
    "avg_changelog_word_count", "changelog_word_count_trend",
    "substance_score_latest", "fake_heartbeat_flag",
    # D2
    "ccu_avg_last_30d", "ccu_avg_last_90d", "ccu_avg_last_180d",
    "ccu_median_all", "ccu_at_launch_30d", "peak_ccu_alltime",
    "ccu_vs_peak_ratio", "ccu_vs_launch_ratio",
    "ccu_trend_slope_30d",
    "ccu_trend_slope_90d", "ccu_trend_slope_180d",
    "ccu_floor_established", "days_since_ccu_above_100",
    "ccu_low_regime", "ccu_unavailable",
    "ccu_recovery_per_update_avg", "ccu_recovery_trend",
    # D3
    "dev_posts_last_30d", "dev_posts_last_90d", "dev_posts_last_180d",
    "dev_engagement_trend", "days_since_dev_post", "build_to_post_ratio",
    "dev_previous_ea_count", "dev_has_prior_success", "dev_total_games_shipped",
    # D4
    "review_count_at_T", "review_positive_at_T", "review_negative_at_T",
    "review_score_at_T",
    "review_score_last_90d", "review_score_last_180d", "review_score_delta_30d",
    "review_velocity_30d", "review_velocity_90d", "review_velocity_180d",
    "review_velocity_trend",
    "review_score_last_30d", "review_sentiment_shock",
    "review_low_regime",
    # D5
    "initial_price_usd", "current_price_at_T",
    "discount_count_to_date", "max_discount_ever_pct",
    "early_deep_discount_flag", "discount_frequency", "price_trend",
    # Cross
    "owner_estimate_at_T",
    "primary_genre",
    "price_vs_genre_median",
    "ccu_vs_genre_weighted_median", "update_freq_vs_genre_median",
    "review_score_vs_genre_median",
    "l1_composite_score", "l1_update_health_score", "l1_player_retention_score",
    "l1_dev_engagement_score", "l1_community_sentiment_score",
    "l1_price_signals_score", "l1_state_encoded",
    # Label + metadata
    "outcome", "label_date", "label_is_resolved", "ml_eligible",
    "collected_at",
]


def insert_snapshot(conn: libsql.Connection, features: dict) -> libsql.Connection:
    cols         = SNAPSHOT_COLUMNS
    placeholders = ",".join("?" * len(cols))
    values       = tuple(features.get(c) for c in cols)
    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            conn.execute(
                f"INSERT OR REPLACE INTO snapshots ({','.join(cols)}) "
                f"VALUES ({placeholders})",
                values,
            )
            return conn
        except Exception as e:
            if attempt == DB_MAX_RETRIES: 
                log.warning("DB write error in insert_snapshot: "
                                  "%s - reconnecting", e)
                raise e
            time.sleep(DB_RETRY_DELAY)
            try: 
                conn.close()
            except Exception as e: 
                pass
            conn = get_conn()


def insert_live_snapshot(conn: libsql.Connection, features: dict) -> libsql.Connection:
    cols         = SNAPSHOT_COLUMNS
    placeholders = ",".join("?" * len(cols))
    values       = tuple(features.get(c) for c in cols)
    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            conn.execute(
                f"INSERT OR IGNORE INTO live_snapshots ({','.join(cols)}) "
                f"VALUES ({placeholders})",
                values,
            )
            return conn
        except Exception as e:
            if attempt == DB_MAX_RETRIES: 
                log.warning("DB write error in insert_live_snapshot: "
                                  "%s - reconnecting", e)
                raise e
            time.sleep(DB_RETRY_DELAY)
            try: 
                conn.close()
            except Exception as e: 
                pass
            conn = get_conn()


# ---------------------------------------------------------------------------
# Candidate loader
# ---------------------------------------------------------------------------

def load_candidates(
    conn: libsql.Connection,
    appid_filter: int | None,
    include_stays_active: bool,
    delta: bool = False,
) -> tuple[list[dict], libsql.Connection]:
    if delta:
        outcome_filter = "('STAYS_ACTIVE')"
    else:
        outcome_filter = (
            "('EXIT_SUCCESS', 'EXIT_ABANDONED', 'EXIT_SILENT', 'STAYS_ACTIVE')"
            if include_stays_active
            else "('EXIT_SUCCESS', 'EXIT_ABANDONED', 'EXIT_SILENT')"
        )
    where  = (
        f"WHERE outcome IN {outcome_filter}"
        f"  AND ea_start_date IS NOT NULL"
        f"  AND c.ccu_available IN ('AVAILABLE', 'UNAVAILABLE')"
    )
    params: list = []

    if delta:
        where += f"""
          AND games_v2.appid IN (
              SELECT appid FROM games_v2 
              WHERE currently_in_ea = 1 
                 OR (currently_in_ea = 0 AND graduation_date IS NOT NULL 
                 AND graduation_date >= date('now', '-{DELTA_GRADUATION_DAYS} days'))
          )
        """

    if appid_filter is not None:
        where += " AND games_v2.appid = ?"
        params.append(appid_filter)

    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            rows = conn.execute(f"""
                SELECT games_v2.appid, ea_start_date, 
                                graduation_date, outcome, abandoned_date
                FROM games_v2
                JOIN ccu_availability c ON c.appid = games_v2.appid
                {where}
                ORDER BY games_v2.appid
            """, params).fetchall()

            res = [
                {
                    "appid":           r[0],
                    "ea_start_date":   r[1],
                    "graduation_date": r[2],
                    "outcome":         r[3],
                    "abandoned_date":  r[4],
                }
                for r in rows
            ]
            return res, conn
        except Exception as e:
            if attempt == DB_MAX_RETRIES: 
                log.warning("DB read error in load_candidates: "
                                  "%s - reconnecting", e)
                raise e
            time.sleep(DB_RETRY_DELAY)
            try: 
                conn.close()
            except Exception as e: 
                pass
            conn = get_conn()


def load_existing_snapshot_dates(conn: libsql.Connection, appid: int) -> tuple[set[str],
                                                                               libsql.Connection]:
    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            rows = conn.execute(
                "SELECT snapshot_date FROM snapshots WHERE appid = ?", (appid,)
            ).fetchall()
            return {r[0] for r in rows}, conn
        except Exception as e:
            if attempt == DB_MAX_RETRIES: 
                log.warning("DB read error in load_existing_snapshot_dates: "
                        "%s - reconnecting", e)
                raise e
            time.sleep(DB_RETRY_DELAY)
            try: 
                conn.close()
            except Exception as e: 
                pass
            conn = get_conn()


# ---------------------------------------------------------------------------
# Delta helpers
# ---------------------------------------------------------------------------

def load_all_dev_features_current(conn: libsql.Connection) -> tuple[dict[int, dict], 
                                                                    libsql.Connection]:
    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            rows = conn.execute("""
                SELECT appid, dev_previous_ea_count, 
                                dev_has_prior_success, dev_total_games_shipped
                FROM game_dev_features_current
            """).fetchall()
            res = {
                r[0]: {
                    "dev_previous_ea_count": r[1],
                    "dev_has_prior_success": r[2],
                    "dev_total_games_shipped": r[3],
                }
                for r in rows
            }
            return res, conn
        except Exception as e:
            if attempt == DB_MAX_RETRIES:
                log.warning("Could not load game_dev_features_current: %s", e)
                return {}, conn
            log.warning("DB read error in load_all_dev_features_current: "
                        "%s - reconnecting", e)
            time.sleep(DB_RETRY_DELAY)
            try: 
                conn.close()
            except Exception as e: 
                pass
            conn = get_conn()


def load_all_genre_and_price_medians(
        conn: libsql.Connection, 
        delta: bool = False
        ) -> tuple[dict[int, dict], libsql.Connection]:
    table = "live_genre_price_medians" if delta else "genre_price_medians"
    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            rows = conn.execute(f"""
                SELECT gg.appid, gg.primary_genre, gpm.median_price_usd
                FROM game_genres gg
                LEFT JOIN {table} gpm ON gg.primary_genre = gpm.primary_genre
            """).fetchall()
            res = {r[0]: {"primary_genre": r[1], 
                          "median_price_usd": r[2]} for r in rows}
            return res, conn
        except Exception as e:
            if attempt == DB_MAX_RETRIES:
                log.warning("Could not load genre_price_medians: %s", e)
                return {}, conn
            log.warning("DB read error in load_all_genre_and_price_medians: "
                        "%s - reconnecting", e)
            time.sleep(DB_RETRY_DELAY)
            try: 
                conn.close()
            except Exception as e: 
                pass
            conn = get_conn()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build EARLY training snapshots")
    p.add_argument("--appid",    type=int,  
                            help="Single appid (debug)")
    p.add_argument("--force",    action="store_true", 
                            help="Rebuild all (INSERT OR REPLACE)")
    p.add_argument("--delta",    action="store_true", 
                            help="Delta run: create latest snapshots in live_snapshots")
    p.add_argument("--dry-run",  action="store_true", help="Plan only, no writes")
    p.add_argument("--stays-active", action="store_true", 
                            help="Include STAYS_ACTIVE games")
    p.add_argument("--verbose",  action="store_true")
    p.add_argument("--no-itad",  action="store_true", 
                            help="Skip ITAD price features")
    p.add_argument("--lower",    type=float, default=DEFAULT_LOWER)
    p.add_argument("--upper",    type=float, default=DEFAULT_UPPER)
    p.add_argument("--n-base",   type=int,   default=DEFAULT_N_BASE)
    p.add_argument("--elastic-threshold", type=int, 
                            default=DEFAULT_ELASTIC_THRESHOLD_DAYS)
    p.add_argument("--elastic-interval",  type=int, 
                            default=DEFAULT_ELASTIC_INTERVAL_DAYS)
    p.add_argument("--max-snapshots",     type=int, 
                            default=DEFAULT_MAX_SNAPSHOTS)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    today = datetime.now(timezone.utc).date()
    conn  = get_conn()
    ensure_snapshots_table(conn)

    itad_client = None
    if not args.no_itad:
        try:
            from utils.itad_client import ITADClient
            itad_client = ITADClient()
            log.info("ITAD client initialised")
        except Exception as e:
            log.warning("ITAD client unavailable (%s) — price features will be NULL", e)

    candidates, conn = load_candidates(conn, args.appid, args.stays_active, args.delta)
    log.info("Candidates: %d games (stays_active=%s, delta=%s)", 
             len(candidates), args.stays_active, args.delta)

    if not candidates:
        log.info("Nothing to process.")
        return

    dev_features_map = {}
    genre_price_map  = {}
    if args.delta:
        dev_features_map, conn = load_all_dev_features_current(conn)
        genre_price_map, conn  = load_all_genre_and_price_medians(conn, delta=True)

    n_games_ok      = 0
    n_games_skipped = 0
    n_games_error   = 0
    total_snaps     = 0
    total_skipped   = 0

    for i, game in enumerate(candidates, 1):
        appid   = game["appid"]
        outcome = game["outcome"]

        try:
            ea_start = date.fromisoformat(game["ea_start_date"])
        except (ValueError, TypeError):
            log.warning("[%d/%d] appid %d: bad ea_start_date %r — skip",
                        i, len(candidates), appid, game["ea_start_date"])
            n_games_skipped += 1
            continue

        ea_end: date | None = None
        if game["graduation_date"]:
            try:
                ea_end = date.fromisoformat(game["graduation_date"])
            except ValueError:
                pass

        abandoned_dt: date | None = None
        if game.get("abandoned_date"):
            try:
                abandoned_dt = date.fromisoformat(game["abandoned_date"])
            except ValueError:
                pass

        if args.delta:
            # Time lag fix: shift live snapshot date by 1 day (End of Day boundary).
            # Because feature_builder enforces strict look-ahead (< snapshot_date),
            # using 'today' (00:00:00) excludes the telemetry we just collected.
            # Using 'tomorrow' ensures all fresh pipeline data is safely included.
            live_snap_date = today + timedelta(days=1)
            ea_age = (live_snap_date - ea_start).days
            plans = [SnapshotPlan(
                appid=appid,
                snapshot_date=live_snap_date,
                snapshot_type="live",
                percentile_label=None,
                ea_age_days=max(0, ea_age),
            )]
        else:
            plans = plan_snapshots(
                appid=appid,
                ea_start=ea_start,
                ea_end=ea_end,
                abandoned_date=abandoned_dt,
                outcome=outcome,
                today=today,
                lower=args.lower,
                upper=args.upper,
                n_base=args.n_base,
                elastic_threshold_days=args.elastic_threshold,
                elastic_interval_days=args.elastic_interval,
                max_snapshots=args.max_snapshots,
            )

            if not plans:
                log.debug("[%d/%d] appid %d: no valid snapshots", i, len(candidates), 
                          appid)
                n_games_skipped += 1
                continue

            if not args.force:
                existing, conn = load_existing_snapshot_dates(conn, appid)
                plans    = [p for p in plans 
                            if p.snapshot_date.isoformat() not in existing]

        if not plans:
            log.debug("[%d/%d] appid %d: all snapshots already built", 
                      i, len(candidates), appid)
            n_games_skipped += 1
            continue

        if args.dry_run:
            for p in plans:
                log.info(
                    "[DRY RUN] appid=%-10d  %s  %-20s  ea_age=%dd",
                    appid, p.snapshot_date, p.percentile_label or p.snapshot_type,
                    p.ea_age_days,
                )
            total_snaps += len(plans)
            continue

        # Load raw data once per game
        try:
            ccu_all, conn           = load_ccu_history(conn, appid)
            review_bkts, conn       = load_review_history(conn, appid)
            events_all, conn        = load_event_history(conn, appid)
            ccu_available, conn     = load_ccu_availability(conn, appid)
            initial_price_usd, conn = load_initial_price(conn, appid)
        except Exception as e:
            log.error("[%d/%d] appid %d: data load error: %s", 
                      i, len(candidates), appid, e)
            n_games_error += 1
            continue

        game_snaps = 0
        game_skip  = 0

        for plan in plans:
            try:
                features = build_features(
                    plan=plan,
                    ccu_all=ccu_all,
                    review_buckets=review_bkts,
                    events_all=events_all,
                    ccu_available=ccu_available,
                    initial_price_usd=initial_price_usd,
                    ea_start=ea_start,
                    ea_end=ea_end,
                    outcome=outcome,
                    today=today,
                    itad_client=itad_client,
                )

                if args.delta:
                    # Inject latest cross-game dev features
                    if appid in dev_features_map:
                        features.update(dev_features_map[appid])

                    # Inject primary genre and price median ratios
                    if appid in genre_price_map:
                        gdata = genre_price_map[appid]
                        features["primary_genre"] = gdata["primary_genre"]
                        median_price = gdata["median_price_usd"]
                        curr_price = features.get("current_price_at_T")
                        if median_price and curr_price is not None:
                            features["price_vs_genre_median"] = round(
                                curr_price / median_price, 4
                                )
                        else:
                            features["price_vs_genre_median"] = None
                    else:
                        features["price_vs_genre_median"] = None

                    conn = insert_live_snapshot(conn, features)
                else:
                    conn = insert_snapshot(conn, features)
                game_snaps += 1

            except Exception as e:
                log.warning(
                    "appid %d snapshot %s failed: %s",
                    appid, plan.snapshot_date, e,
                )
                game_skip += 1

        conn.commit()
        total_snaps   += game_snaps
        total_skipped += game_skip
        n_games_ok    += 1

        log.info(
            "[%d/%d] appid %-10d  %-18s  %d snapshots built  (%d skipped)",
            i, len(candidates), appid, outcome, game_snaps, game_skip,
        )

    log.info("=" * 64)
    log.info("build_snapshots complete")
    log.info("  Games processed   : %d", n_games_ok)
    log.info("  Games skipped     : %d  (no valid plans or already built)", 
                                                                n_games_skipped)
    log.info("  Games errored     : %d  (data load failures)", 
                                                                n_games_error)
    log.info("  Snapshots written : %d", total_snaps)
    log.info("  Snapshots failed  : %d  (per-snapshot errors)", total_skipped)
    log.info("=" * 64)

    conn.close()


if __name__ == "__main__":
    main()
