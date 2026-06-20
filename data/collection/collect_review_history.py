"""
collect_review_history.py — EARLY pipeline: review histogram collector

Fetches monthly review buckets (positive + negative) per game from the Steam
review histogram API and stores them in the review_history table.

These buckets are the training-time review data source. build_snapshots.py
will later reconstruct cumulative review counts and ratios at any snapshot
date T by summing buckets up to T, with linear interpolation for the partial
bucket that straddles T.

  NOTE: Do NOT use bucket sums as a direct review count total.
  Histogram buckets are for chart rendering, not precise accounting.
  Use only for: (1) ea_start_date (already stored in games_v2),
  (2) historical positive/negative ratio reconstruction at snapshot T.

Candidate selection:
  Reads from ccu_availability where ccu_available IN ('AVAILABLE', 'UNAVAILABLE').
  Rationale: these are the games that passed the CCU collection pipeline's
  review gate (>= 50 reviews). Games that were SKIP_LOW_REVIEWS or ERROR
  in CCU collection are not yet ml_eligible and have no training value here.

Append-only by default:
  For each appid already in review_history, finds the most recent bucket
  end_date stored and only fetches/inserts buckets newer than that date.
  This makes the script safe to run as a cron job — only new months are added.
  Use --force to re-fetch and replace all buckets for an appid.

Rate limiting:
  Steam histogram API: ~200 req / 5 min → 1.5 s minimum between requests.
  Default REQUEST_DELAY is 1.5 s. Adjust via HISTOGRAM_REQUEST_DELAY env var
  or --delay flag.

Storage:
  review_history table: one row per appid per bucket (monthly period).
  Each row stores the bucket's start_date, end_date, positive count,
  negative count, and a collected_at timestamp.

Usage:
  python collect_review_history.py                  # full eligible universe
  python collect_review_history.py --appid 1145360  # single game (debug)
  python collect_review_history.py --force          # re-fetch all buckets
  python collect_review_history.py --limit 200      # cap run (testing)
  python collect_review_history.py --dry-run        # plan only, no writes
  python collect_review_history.py --verbose        # debug logging
"""

import argparse
import logging
import os
import time
from datetime import datetime, timezone

import libsql
import requests
from dotenv import load_dotenv


load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_URL = os.getenv("TURSO_URL")
DB_AUTH = os.getenv("TURSO_AUTH_TOKEN")

# Steam histogram API: ~200 req / 5 min → 1.5 s floor
REQUEST_DELAY = float(os.getenv("HISTOGRAM_REQUEST_DELAY", "1.5"))
DELTA_GRADUATION_DAYS = 90
DB_MAX_RETRIES = 3
DB_RETRY_DELAY = 5.0

HISTOGRAM_URL = "https://store.steampowered.com/appreviewhistogram/{appid}"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
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


def ensure_tables(conn: libsql.Connection) -> None:
    """
    review_history: one row per appid per monthly bucket.

    Columns:
      appid        — Steam appid
      bucket_start — bucket start date as ISO text (YYYY-MM-DD), usually the 1st
      bucket_end   — bucket end date as ISO text (YYYY-MM-DD), exclusive upper bound
      positive     — positive review count in this bucket
      negative     — negative review count in this bucket
      collected_at — unix timestamp when this row was inserted

    Primary key is (appid, bucket_start) — bucket_start uniquely identifies a
    monthly period per game. On re-collection (--force), INSERT OR REPLACE
    overwrites existing rows cleanly.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS review_history (
            appid           INTEGER NOT NULL,
            bucket_start    TEXT    NOT NULL,   -- YYYY-MM-DD (bucket open, usually 1st)
            bucket_end      TEXT    NOT NULL,   -- YYYY-MM-DD (bucket close, exclusive)
            positive        INTEGER NOT NULL,
            negative        INTEGER NOT NULL,
            collected_at    INTEGER NOT NULL,
            PRIMARY KEY (appid, bucket_start)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_review_history_appid
        ON review_history (appid)
    """)
    conn.commit()
    log.info("review_history table ready")


def get_candidates(
        conn: libsql.Connection,
        delta: bool = False
        ) -> tuple[list[int], libsql.Connection]:
    """
    Select appids from ccu_availability where collection succeeded (AVAILABLE)
    or where the game bled out but still has review signal (UNAVAILABLE).
    Both passed the >= 50 review gate in collect_ccu_history.py.
    SKIP_LOW_REVIEWS and ERROR are excluded — not yet ml_eligible.
    """
    delta_filter = ""
    if delta:
        delta_filter = f"""
        AND appid IN (
            SELECT appid FROM games_v2
            WHERE currently_in_ea = 1
               OR (currently_in_ea = 0
               AND graduation_date IS NOT NULL
               AND graduation_date >= date('now', '-{DELTA_GRADUATION_DAYS} days'))
        )
        """

    query = f"""
        SELECT appid
        FROM ccu_availability
        WHERE ccu_available IN ('AVAILABLE', 'UNAVAILABLE')
        {delta_filter}
        ORDER BY appid
    """
    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            rows = conn.execute(query).fetchall()
            return [r[0] for r in rows], conn
        except Exception as e:
            if attempt == DB_MAX_RETRIES:
                log.warning("DB read error: %s", e)
                time.sleep(DB_RETRY_DELAY)
                raise e
            try:
                conn.close()
            except Exception as e:
                log.warning("Error closing database connection: %s", e)
            conn = get_conn()


def get_latest_bucket_dates(
        conn: libsql.Connection,
        appids: list[int]
        ) -> tuple[dict[int, str], libsql.Connection]:
    """
    For each appid already in review_history, return the most recent
    bucket_start stored. Used by append logic to skip already-collected buckets.
    Returns dict of {appid: latest_bucket_start_str}.
    """
    if not appids:
        return {}, conn
    placeholders = ",".join("?" * len(appids))
    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            rows = conn.execute(f"""
                SELECT appid, MAX(bucket_start)
                FROM review_history
                WHERE appid IN ({placeholders})
                GROUP BY appid
            """, appids).fetchall()
            return {r[0]: r[1] for r in rows}, conn
        except Exception as e:
            if attempt == DB_MAX_RETRIES:
                log.warning("DB read error: %s", e)
                time.sleep(DB_RETRY_DELAY)
                raise e
            try:
                conn.close()
            except Exception as e:
                log.warning("Error closing database connection: %s", e)
            conn = get_conn()


def insert_buckets(
        conn: libsql.Connection,
        rows: list[dict]
        ) -> tuple[int, libsql.Connection]:
    """
    Insert review_history rows. Uses INSERT OR REPLACE so --force re-runs
    cleanly overwrite stale data without constraint errors.
    Returns number of rows inserted/replaced.
    """
    if not rows:
        return 0, conn

    now_ts = int(datetime.now(timezone.utc).timestamp())
    tuples = [
        (row["appid"], row["bucket_start"], row["bucket_end"],
         row["positive"], row["negative"], now_ts)
        for row in rows
    ]

    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            conn.executemany("""
                INSERT OR REPLACE INTO review_history
                    (appid, bucket_start, bucket_end, positive, negative, collected_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, tuples)
            conn.commit()
            return len(tuples), conn
        except Exception as e:
            if attempt == DB_MAX_RETRIES:
                log.warning("review_history insert failed: %s", e)
                return 0, conn
            log.warning("DB insert error attempt %d: %s - reconnecting", attempt, e)
            time.sleep(DB_RETRY_DELAY)
            try:
                conn.close()
            except Exception as e:
                log.warning("Error closing database connection: %s", e)
            conn = get_conn()


# ---------------------------------------------------------------------------
# Steam histogram API
# ---------------------------------------------------------------------------

def fetch_histogram(appid: int, session: requests.Session) -> dict | None:
    """
    Fetch the Steam review histogram JSON for one appid.

    Endpoint: /appreviewhistogram/<appid>?json=1
    Returns the parsed JSON dict on success, None on any failure.

    Response shape (relevant fields):
      {
        "success": 1,
        "results": {
          "start_date": <unix ts>,   -- EA entry date (authoritative)
          "weeks": [                 -- individual weekly buckets (not used here)
            { "date": <ts>, "recommendations_up": N, "recommendations_down": N },
            ...
          ],
          "rollups": [               -- monthly aggregated buckets (what we store)
            {
              "date": <unix ts>,               -- bucket start (unix)
              "recommendations_up": N,
              "recommendations_down": N,
            },
            ...
          ]
        }
      }

    NOTE: The API returns rollups sorted oldest-first. Each rollup's "date"
    is the bucket start. The bucket end is the start of the next rollup (or
    now for the most recent bucket). We compute bucket_end from this ordering.
    """
    url = HISTOGRAM_URL.format(appid=appid)
    try:
        resp = session.get(
            url,
            params={"json": "1"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("success") != 1:
            log.warning("appid %d: histogram API returned success=%s",
                        appid, data.get("success"))
            return None
        return data
    except requests.HTTPError as e:
        status = e.response.status_code if e.response else 0
        log.warning("appid %d histogram HTTP %d", appid, status)
        return None
    except Exception as e:
        log.warning("appid %d histogram fetch failed: %s", appid, e)
        return None


def parse_histogram(appid: int, data: dict) -> list[dict]:
    """
    Parse histogram API response into a list of bucket dicts ready for DB insert.

    Each bucket:
      appid        — as passed in
      bucket_start — ISO date string (YYYY-MM-DD), derived from rollup["date"] unix ts
      bucket_end   — ISO date string (YYYY-MM-DD), derived from next rollup's start
                     (for the final bucket: today's date, since the bucket is open)
      positive     — recommendations_up for this bucket
      negative     — recommendations_down for this bucket

    Degenerate response guard:
      If rollups is empty or absent, returns []. The caller logs and skips.
      (Degenerate responses — weeks:[] with single rollup — are expected for
      very-low-traffic games, but those should already be filtered by the
      ccu_availability gate. Log a warning if encountered here.)
    """
    results = data.get("results", {})
    rollups = results.get("rollups", [])

    if not rollups:
        return []

    today_str = datetime.now(timezone.utc).date().isoformat()
    buckets = []

    for i, rollup in enumerate(rollups):
        # bucket_start: unix ts → ISO date
        try:
            bucket_start_dt = datetime.fromtimestamp(rollup["date"],
                                                     tz=timezone.utc).date()
            bucket_start = bucket_start_dt.isoformat()
        except (KeyError, ValueError, OSError) as e:
            log.warning("appid %d rollup[%d] bad date: %s", appid, i, e)
            continue

        # bucket_end: start of next rollup, or today for the last bucket
        if i + 1 < len(rollups):
            try:
                next_dt = datetime.fromtimestamp(rollups[i + 1]["date"],
                                                 tz=timezone.utc).date()
                bucket_end = next_dt.isoformat()
            except (KeyError, ValueError, OSError):
                bucket_end = today_str
        else:
            bucket_end = today_str

        positive = int(rollup.get("recommendations_up", 0) or 0)
        negative = int(rollup.get("recommendations_down", 0) or 0)

        buckets.append({
            "appid": appid,
            "bucket_start": bucket_start,
            "bucket_end": bucket_end,
            "positive": positive,
            "negative": negative,
        })

    return buckets


def filter_new_buckets(buckets: list[dict], latest_stored: str | None) -> list[dict]:
    """
    For append mode: drop any buckets whose bucket_start <= latest_stored.
    This avoids re-inserting months we already have.

    The most recent stored bucket IS intentionally replaced even in append mode,
    because the current month's bucket_end is "today" and its counts are still
    accumulating. We use bucket_start > latest_stored (strict), not >=, to
    leave all completed past months untouched.

    Exception: the last bucket in the new response (open/current month) is
    always included regardless, as its counts will have grown since last fetch.
    """
    if latest_stored is None:
        return buckets  # nothing stored yet — take everything

    # Keep buckets strictly newer than latest_stored, plus always re-insert
    # the final bucket (open month) in case it has grown since last run.
    new_buckets = [b for b in buckets if b["bucket_start"] > latest_stored]

    # If the last bucket in the API response is the same as our latest stored
    # (i.e. we're in the same calendar month as the last collection), re-include
    # it so its accumulating counts are updated.
    if buckets and buckets[-1]["bucket_start"] == latest_stored:
        new_buckets = [buckets[-1]] + new_buckets

    return new_buckets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect review histogram buckets for EARLY pipeline"
    )
    p.add_argument("--appid", type=int,
                   help="Single appid (debug)")
    p.add_argument("--force", action="store_true",
                   help="Re-fetch and replace all buckets (overrides append logic)")
    p.add_argument("--delta", action="store_true",
                   help="Delta run: only fetch for active and recently graduated games")
    p.add_argument("--limit", type=int,
                   help="Cap number of appids processed (testing)")
    p.add_argument("--dry-run", action="store_true",
                   help="Plan only, no API calls or writes")
    p.add_argument("--delay", type=float, default=REQUEST_DELAY,
                   help=f"Seconds between histogram requests (default {REQUEST_DELAY})")
    p.add_argument("--verbose", action="store_true",
                   help="Debug logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    delay = args.delay

    conn = get_conn()
    ensure_tables(conn)

    # --- Candidate selection ---
    if args.appid:
        candidates = [args.appid]
        log.info("Single-appid mode: %d", args.appid)
    else:
        candidates, conn = get_candidates(conn, delta=args.delta)
        log.info(
            "Candidates from ccu_availability (AVAILABLE + UNAVAILABLE): %d",
            len(candidates),
        )

    if args.limit:
        candidates = candidates[: args.limit]
        log.info("Capped to %d appids via --limit", len(candidates))

    if not candidates:
        log.info("Nothing to collect.")
        return

    # --- Append mode: find latest stored bucket per appid ---
    latest_stored: dict[int, str] = {}
    if not args.force:
        latest_stored, conn = get_latest_bucket_dates(conn, candidates)
        already_have = len(latest_stored)
        log.info(
            "Append mode: %d appids already have buckets — will extend from latest",
            already_have,
        )
    else:
        log.info("--force: replacing all buckets for all appids")

    if args.dry_run:
        log.info("[DRY RUN] Would process %d appids. First 10: %s",
                 len(candidates), candidates[:10])
        return

    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (compatible; EARLY-pipeline/1.0)"

    # Counters
    n_ok = 0
    n_degenerate = 0
    n_error = 0
    n_no_new = 0
    total_buckets = 0

    for i, appid in enumerate(candidates, 1):

        data = fetch_histogram(appid, session)

        if data is None:
            n_error += 1
            log.warning("[%d/%d] appid %d: fetch failed — skipping",
                        i, len(candidates), appid)
            if i < len(candidates):
                time.sleep(delay)
            continue

        buckets = parse_histogram(appid, data)

        if not buckets:
            n_degenerate += 1
            log.warning(
                "[%d/%d] appid %d: degenerate histogram response (no rollups) — "
                "unexpected given ccu_availability gate; skipping",
                i, len(candidates), appid,
            )
            if i < len(candidates):
                time.sleep(delay)
            continue

        # Append mode: drop buckets we already have (except open current month)
        if not args.force:
            prior = latest_stored.get(appid)
            to_insert = filter_new_buckets(buckets, prior)
        else:
            to_insert = buckets

        if not to_insert:
            n_no_new += 1
            log.debug(
                "[%d/%d] appid %d: no new buckets since %s",
                i, len(candidates), appid, latest_stored.get(appid),
            )
            if i < len(candidates):
                time.sleep(delay)
            continue

        inserted, conn = insert_buckets(conn, to_insert)
        total_buckets += inserted
        n_ok += 1
        log.info(
            "[%d/%d] appid %d: %d buckets inserted (total in response: %d, "
            "latest stored was: %s)",
            i, len(candidates), appid, inserted, len(buckets),
            latest_stored.get(appid, "none"),
        )

        if i < len(candidates):
            time.sleep(delay)

    # --- Summary ---
    log.info("=" * 60)
    log.info("Review history collection complete")
    log.info("  Candidates processed  : %d", len(candidates))
    log.info("  OK (buckets inserted) : %d  (%d total buckets)", n_ok, total_buckets)
    log.info("  No new buckets        : %d", n_no_new)
    log.info("  Degenerate response   : %d  "
             "(investigate — should be filtered upstream)",
             n_degenerate)
    log.info("  Errors (retry needed) : %d", n_error)
    log.info("=" * 60)

    if n_degenerate > 0:
        log.warning(
            "%d games returned a degenerate histogram (no rollups). These passed the "
            "ccu_availability gate so this is unexpected. "
            "Check whether their review counts "
            "have dropped below the histogram threshold since CCU collection ran.",
            n_degenerate,
        )

    conn.close()


if __name__ == "__main__":
    main()
