"""
api/db.py
---------
Shared libsql connection for the FastAPI process.
One connection is created at startup and reused across requests.
libsql's Python client is thread-safe for reads.

Single source of truth for all CREATE TABLE statements.
Called by init_db() at startup — all statements are idempotent (IF NOT EXISTS).

Tables:
    live_scores       — pre-computed weekly scores (serving layer)
    live_snapshots    — latest feature vector per game (overwritten weekly)
    agent_analysis    — cached LangGraph agent output (on-demand, user-triggered)
    review_history    — historical review buckets
                        (owned by collect pipeline, read-only here)
"""

import logging
import os
import threading

import libsql
from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger(__name__)
_conn: libsql.Connection | None = None
_reconnect_lock = threading.Lock()

# ---------------------------------------------------------------------------
# live_scores
# Populated by score.yml → inference.py weekly.
# One row per appid per scoring run; API reads the latest by scored_at.
# ---------------------------------------------------------------------------

LIVE_SCORES = """
CREATE TABLE IF NOT EXISTS live_scores (
    appid               INTEGER NOT NULL,
    scored_at           INTEGER NOT NULL,
    ea_age_days         INTEGER,
    p_distressed        REAL,
    is_distressed       INTEGER,
    l1_state            TEXT,
    ml_eligible         INTEGER,
    model_version       TEXT,
    null_features       TEXT,       -- JSON array of null feature names
    review_count_at_T   INTEGER,
    update_health       REAL,
    player_retention    REAL,
    dev_engagement      REAL,
    sentiment           REAL,
    price_market        REAL,
    days_since_last_build_update INTEGER,
    l1_composite_score  REAL,
    PRIMARY KEY (appid, scored_at)
)
"""

# ---------------------------------------------------------------------------
# live_snapshots
# Current feature vector per game — one row per appid, overwritten weekly.
# Used for: feature explainability, Zilliz ANN query vector, downstream tasks.
# All 76 model features stored as JSON blob + key scalar fields as columns
# for cheap filtering without JSON parsing.
# ---------------------------------------------------------------------------

LIVE_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS live_snapshots (
    appid               INTEGER PRIMARY KEY,
    snap_date           TEXT    NOT NULL,  -- YYYY-MM-DD of last score run
    ea_age_days         INTEGER,
    genre_bucket        TEXT,              -- coarse genre for Zilliz filter
    review_count_at_T   INTEGER,
    features_json       TEXT    NOT NULL,  -- full feature dict as JSON
    shap_json           TEXT,              -- top-N SHAP values as JSON {feature: value}
    updated_at          INTEGER NOT NULL   -- Unix ts
)
"""

# ---------------------------------------------------------------------------
# agent_analysis
# Cached LangGraph agent output — written on demand when user triggers analysis.
# Re-run when: l1_state changes OR analysis is stale (> staleness threshold).
# ---------------------------------------------------------------------------

AGENT_ANALYSIS = """
CREATE TABLE IF NOT EXISTS agent_analysis (
    appid                   INTEGER PRIMARY KEY,

    -- YYYY-MM-DD of the live_scores row used
    snapshot_date           TEXT    NOT NULL,

    -- Unix ts of agent run
    analysed_at             INTEGER NOT NULL,

    -- "user_request" | "state_change" | "stale"
    trigger_reason          TEXT    NOT NULL,

    -- l1_state when analysis ran (staleness check)
    l1_state_at_analysis    TEXT,

    -- Forensic Agent
    forensic_ran            INTEGER NOT NULL DEFAULT 0,
    update_substance_score  REAL,
    fake_heartbeat_flag     INTEGER,
    momentum                TEXT,
    event_state_mismatch    INTEGER,
    forensic_reasoning      TEXT,

    -- Sentiment Auditor
    auditor_ran             INTEGER NOT NULL DEFAULT 0,
    sentiment_shift         TEXT,
    sentiment_alignment     TEXT,
    key_concerns            TEXT,               -- JSON array of strings
    theme_clusters          TEXT,               -- JSON array of cluster dicts
    auditor_summary         TEXT,

    -- Critic Agent
    signal_alignment        TEXT,
    critic_ran              INTEGER NOT NULL DEFAULT 0,
    consumer_verdict        TEXT,
    developer_brief         TEXT,
    confidence_note         TEXT,

    -- Error tracking
    error                   TEXT                -- last error if any agent failed
)
"""

# ---------------------------------------------------------------------------
# review_history  (reference — owned by collect pipeline, not created here)
# ---------------------------------------------------------------------------

REVIEW_HISTORY = """
CREATE TABLE IF NOT EXISTS review_history (
    appid           INTEGER NOT NULL,
    bucket_start    TEXT    NOT NULL,
    bucket_end      TEXT    NOT NULL,
    positive        INTEGER NOT NULL,
    negative        INTEGER NOT NULL,
    collected_at    INTEGER NOT NULL,
    PRIMARY KEY (appid, bucket_start)
)
"""

# ---------------------------------------------------------------------------
# All tables in init order (dependencies first)
# ---------------------------------------------------------------------------

ALL_TABLES = [
    LIVE_SCORES,
    LIVE_SNAPSHOTS,
    AGENT_ANALYSIS,
    REVIEW_HISTORY,
]

async def init_db() -> None:
    global _conn

    use_local = os.getenv("USE_LOCAL_DB", "false").lower() == "true"
    if use_local:
        db_path = "./demo_data/early_sample.db"
        _conn = libsql.connect(database=db_path)
    else:
        url   = os.environ["TURSO_URL"]
        token = os.environ["TURSO_AUTH_TOKEN"]
        _conn = libsql.connect(database=url, auth_token=token)

    # Ensure all tables exist (idempotent)
    for ddl in ALL_TABLES:
        _conn.execute(ddl)
    _conn.commit()

async def close_db() -> None:
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None


class ResilientDB:
    def execute(self, sql: str, parameters: tuple | list | dict = ()):
        global _conn
        for attempt in range(1, 4):
            try:
                if _conn is None:
                    self._reconnect()
                return _conn.execute(sql, parameters)
            except Exception as e:
                if attempt == 3:
                    raise
                logger.warning("DB execute error (attempt %d): %s — reconnecting",
                               attempt, e)
                self._reconnect()

    def commit(self) -> None:
        global _conn
        if _conn:
            _conn.commit()

    def _reconnect(self) -> None:
        global _conn
        with _reconnect_lock:
            try:
                if _conn is not None:
                    _conn.close()
            except Exception:
                pass
            use_local = os.getenv("USE_LOCAL_DB", "false").lower() == "true"
            if use_local:
                db_path = "./demo_data/early_sample.db"
                _conn = libsql.connect(database=db_path)
            else:
                url = os.environ["TURSO_URL"]
                token = os.environ["TURSO_AUTH_TOKEN"]
                _conn = libsql.connect(database=url, auth_token=token)


def get_db() -> ResilientDB:
    if _conn is None:
        raise RuntimeError("Database not initialised — call init_db() first.")
    return ResilientDB()
