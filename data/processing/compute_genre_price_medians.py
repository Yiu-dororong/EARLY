"""
compute_genre_price_medians.py
------------------------------
1. Computes median initial_price_usd per primary_genre from eligible games.
2. Handles thin genre buckets via fallback hierarchy:
       genre median (n >= 20)  →  scope-tier median  →  corpus-wide median
3. Stores results in genre_price_medians.
4. Populates price_vs_genre_median in snapshots as:
       current_price_at_T / reference_median

Eligibility filter for median computation:
    - initial_price_usd > 0
    - ea_start_date >= 2022-01-01   (post-inflation baseline)
    - appid present in game_genres  (has a valid primary_genre)
    - review_count_at_check >= 50   (from ccu_availability)

genre_median_source column records which fallback was used:
    "genre"          — direct genre median (n >= 20)
    "scope_fallback" — genre bucket too thin, used scope-tier median
    "corpus_fallback"— scope bucket also thin, used corpus-wide median

Usage:
    python compute_genre_price_medians.py [--dry-run] [--thin-threshold N]
    python compute_genre_price_medians.py --delta
"""

import argparse
import logging
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

import libsql
import pandas as pd

# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

DB_URL = os.getenv("TURSO_URL")
DB_AUTH = os.getenv("TURSO_AUTH_TOKEN")


def get_conn() -> libsql.Connection:
    if DB_URL and DB_AUTH:
        return libsql.connect(DB_URL, auth_token=DB_AUTH)
    return libsql.connect("early.db")


CREATE_MEDIANS_TABLE = """
CREATE TABLE IF NOT EXISTS genre_price_medians (
    primary_genre        TEXT    PRIMARY KEY,
    genre_scope          INTEGER,
    median_price_usd     REAL,
    game_count           INTEGER,
    genre_median_source  TEXT,
    computed_at          TEXT
)
"""

ALTER_SNAPSHOTS_SQL = """
ALTER TABLE snapshots ADD COLUMN price_vs_genre_median REAL
"""

THIN_BUCKET_DEFAULT = 20
DB_BATCH_SIZE = 500

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — Load data
# ---------------------------------------------------------------------------

def load_eligible_games(conn: libsql.Connection) -> pd.DataFrame:
    rows = conn.execute("""
        SELECT
            g.appid,
            g.initial_price_usd,
            g.ea_start_date,
            gg.primary_genre,
            gg.genre_scope
        FROM games_v2 g
        INNER JOIN game_genres gg
            ON g.appid = gg.appid
        INNER JOIN ccu_availability ca
            ON g.appid = ca.appid
        WHERE
            g.initial_price_usd          >  0
            AND g.ea_start_date          >= '2022-01-01'
            AND ca.review_count_at_check >= 50
            AND gg.primary_genre         IS NOT NULL
    """).fetchall()
    df = pd.DataFrame(rows, columns=[
        "appid", "initial_price_usd", "ea_start_date",
        "primary_genre", "genre_scope",
    ])
    log.info("Eligible games for median computation: %d", len(df))
    return df


def load_snapshots(conn: libsql.Connection) -> pd.DataFrame:
    rows = conn.execute("""
        SELECT
            s.appid,
            s.snapshot_date,
            s.current_price_at_T,
            gg.primary_genre,
            gg.genre_scope
        FROM snapshots s
        INNER JOIN game_genres gg
            ON s.appid = gg.appid
        """).fetchall()
    df = pd.DataFrame(rows, columns=[
        "appid", "snapshot_date", "current_price_at_T", "primary_genre", "genre_scope",
    ])

    # Null out zero prices — these are invalid (discounted to free, data error)
    zero_count = (df["current_price_at_T"] == 0).sum()
    if zero_count:
        log.info("Nulling %d zero current_price_at_T values", zero_count)
        df["current_price_at_T"] = df["current_price_at_T"].replace(0, pd.NA)

    log.info("Snapshots loaded: %d", len(df))
    return df


# ---------------------------------------------------------------------------
# Step 2 — Compute medians with fallback hierarchy
# ---------------------------------------------------------------------------

def compute_medians(
    eligible: pd.DataFrame,
    thin_threshold: int,
) -> pd.DataFrame:
    """
    Returns one row per primary_genre with resolved median and source label.
    Fallback hierarchy: genre → scope-tier → corpus-wide
    """
    # Genre-level medians
    genre_medians = (
        eligible
        .groupby(["primary_genre", "genre_scope"])["initial_price_usd"]
        .agg(median_price_usd="median", game_count="count")
        .reset_index()
    )

    # Scope-tier medians (fallback level 1)
    scope_medians = (
        eligible
        .groupby("genre_scope")["initial_price_usd"]
        .median()
        .rename("scope_median")
        .reset_index()
    )

    # Corpus-wide median (fallback level 2)
    corpus_median = eligible["initial_price_usd"].median()
    log.info("Corpus-wide median: $%.2f", corpus_median)

    result = genre_medians.merge(scope_medians, on="genre_scope", how="left")

    def resolve_median(row) -> tuple[float, str]:
        if row["game_count"] >= thin_threshold:
            return row["median_price_usd"], "genre"

        scope_med = row.get("scope_median")
        if pd.notna(scope_med):
            log.warning(
                "Thin bucket: genre='%s' n=%d < %d → scope_%d fallback ($%.2f)",
                row["primary_genre"], row["game_count"],
                thin_threshold, row["genre_scope"], scope_med,
            )
            return scope_med, "scope_fallback"

        log.warning(
            "Thin bucket: genre='%s' n=%d → corpus fallback ($%.2f)",
            row["primary_genre"], row["game_count"], corpus_median,
        )
        return corpus_median, "corpus_fallback"

    resolved = result.apply(
        lambda row: pd.Series(resolve_median(row), index=["median_price_usd", "genre_median_source"]),
        axis=1,
    )
    result["median_price_usd"]    = resolved["median_price_usd"]
    result["genre_median_source"] = resolved["genre_median_source"]

    log.info("Genre price medians (final):")
    for _, row in result.sort_values("genre_scope").iterrows():
        log.info(
            "  %-25s  scope=%d  median=$%5.2f  n=%-4d  source=%s",
            row["primary_genre"], row["genre_scope"],
            row["median_price_usd"], row["game_count"],
            row["genre_median_source"],
        )

    return result[[
        "primary_genre", "genre_scope", "median_price_usd",
        "game_count", "genre_median_source",
    ]]


# ---------------------------------------------------------------------------
# Step 3 — Compute price_vs_genre_median per snapshot
# ---------------------------------------------------------------------------

def compute_price_ratios(
    snapshots: pd.DataFrame,
    medians: pd.DataFrame,
) -> pd.DataFrame:
    df = snapshots.merge(
        medians[["primary_genre", "median_price_usd"]],
        on="primary_genre",
        how="left",
    )

    df["price_vs_genre_median"] = (
        df["current_price_at_T"] / df["median_price_usd"]
    )

    null_count = df["price_vs_genre_median"].isna().sum()
    log.info(
        "price_vs_genre_median: %d computed, %d NULL (%.1f%%)",
        len(df) - null_count, null_count,
        100 * null_count / len(df),
    )

    return df[["appid", "snapshot_date", "price_vs_genre_median", "primary_genre"]]


# ---------------------------------------------------------------------------
# Step 4 — Write
# ---------------------------------------------------------------------------

def ensure_column_exists(conn: libsql.Connection) -> None:
    try:
        conn.execute(ALTER_SNAPSHOTS_SQL)
        log.info("Added price_vs_genre_median column to snapshots.")
    except Exception:
        log.info("price_vs_genre_median column already exists.")


def write_medians(
    conn: libsql.Connection,
    medians: pd.DataFrame,
    computed_at: str,
) -> None:
    for _, row in medians.iterrows():
        conn.execute(
            """
            INSERT INTO genre_price_medians
                (primary_genre, genre_scope, median_price_usd,
                 game_count, genre_median_source, computed_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(primary_genre) DO UPDATE SET
                genre_scope         = excluded.genre_scope,
                median_price_usd    = excluded.median_price_usd,
                game_count          = excluded.game_count,
                genre_median_source = excluded.genre_median_source,
                computed_at         = excluded.computed_at
            """,
            [
                row["primary_genre"],
                int(row["genre_scope"]),
                float(row["median_price_usd"]),
                int(row["game_count"]),
                row["genre_median_source"],
                computed_at,
            ],
        )
        conn.commit()

    log.info("Wrote %d rows to genre_price_medians.", len(medians))


def write_ratios(
    conn: libsql.Connection,
    ratios: pd.DataFrame,
) -> None:
    """
    Batch update snapshots with both price_vs_genre_median and primary_genre.
    """
    ratios_clean = ratios.where(pd.notnull(ratios), None)
    updates = [
        (
            float(row["price_vs_genre_median"]) if row["price_vs_genre_median"] is not None else None,
            row["primary_genre"],
            row["appid"],
            row["snapshot_date"]
        )
        for _, row in ratios_clean.iterrows()
    ]

    total = len(updates)
    for i in range(0, total, DB_BATCH_SIZE):
        chunk = updates[i:i + DB_BATCH_SIZE]
        conn.executemany(
            """
            UPDATE snapshots
            SET price_vs_genre_median = ?,
                primary_genre = ?
            WHERE appid = ? AND snapshot_date = ?
            """,
            chunk,
        )
        processed = i + len(chunk)
        log.info("  Updated %d/%d snapshots (%.1f%%)", processed, total, processed / total * 100)

    conn.commit()
    null_ratios = ratios["price_vs_genre_median"].isna().sum()
    log.info(
        "Updated %d snapshots with primary_genre (%d have NULL price_vs_genre_median).",
        total, null_ratios,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(dry_run: bool, thin_threshold: int, delta: bool) -> None:
    conn = get_conn()
    conn.execute(CREATE_MEDIANS_TABLE)

    # Load
    eligible = load_eligible_games(conn)

    # Compute
    medians = compute_medians(eligible, thin_threshold)

    if dry_run:
        log.info("Dry run — sample medians:")
        print(medians.head(10).to_string(index=False))
        log.info("Dry run complete — no writes.")
        conn.close()
        return

    # Write
    computed_at = datetime.now(timezone.utc).isoformat()
    write_medians(conn, medians, computed_at)

    if delta:
        log.info("Delta run: skipping snapshot price_vs_genre_median updates.")
    else:
        snapshots = load_snapshots(conn)
        ratios = compute_price_ratios(snapshots, medians)
        ensure_column_exists(conn)
        write_ratios(conn, ratios)

    log.info("Done.")
    conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute genre price medians and populate snapshots."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute but do not write to DB")
    parser.add_argument(
        "--thin-threshold", type=int, default=THIN_BUCKET_DEFAULT,
        help=f"Min games for a genre median to be used directly (default: {THIN_BUCKET_DEFAULT})",
    )
    parser.add_argument("--delta", action="store_true",
                        help="Delta run: only recompute and write to the genre_price_medians table")
    args = parser.parse_args()

    run(args.dry_run, args.thin_threshold, args.delta)