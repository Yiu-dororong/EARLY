"""
collect_pre2022_ea_games.py
----------------------------
Collects historical EA game data for developers present in the snapshot corpus,
targeting games that launched in Early Access before 2022-01-01.

This populates the pre2022_ea_games table used by compute_dev_features.py to
fill dev_previous_ea_count and dev_has_prior_success for developers whose
history predates our 2022 collection start.

─────────────────────────────────────────────────────────────────────────────
CANDIDATE DISCOVERY (games_v2 only — no SteamSpy)
─────────────────────────────────────────────────────────────────────────────
Candidates are appids already in games_v2 that:
  1. Share a normalised dev string with a game in the active snapshot corpus
  2. Have ea_start_date < 2022-01-01
  3. Are not themselves in the active snapshot corpus (already handled)

games_v2 has 160k games so most relevant siblings are already present.
The residual gap (siblings never ingested at all) is accepted as a known
limitation — adding SteamSpy for marginal coverage is not worth the
rate-limit dependency.

─────────────────────────────────────────────────────────────────────────────
DATA SOURCES
─────────────────────────────────────────────────────────────────────────────
1. games_v2 — candidate discovery + ea_start_date, is_free
2. Steam appdetails API — outcome inference, graduation_date, app_type check
     success: false (delisted) → SKIP entirely (can't infer outcome reliably;
     contributes nothing to the conservative gate in compute_dev_features.py)
3. Steam review histogram API — full lifetime history → crossed_50_reviews

─────────────────────────────────────────────────────────────────────────────
RATE LIMITING
─────────────────────────────────────────────────────────────────────────────
Steam allows ~200 requests per 5 minutes. Default sleep: 1.5s between calls.
Each game makes 2 calls (appdetails + histogram). Configurable via --steam-sleep.

─────────────────────────────────────────────────────────────────────────────
USAGE
─────────────────────────────────────────────────────────────────────────────
  python collect_pre2022_ea_games.py               # full run
  python collect_pre2022_ea_games.py --dry-run     # print candidates, no API calls
  python collect_pre2022_ea_games.py --resume      # skip already-collected appids
  python collect_pre2022_ea_games.py --steam-sleep 2.0
  python collect_pre2022_ea_games.py --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from datetime import date, datetime, timezone

import requests
import libsql
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_URL  = os.getenv("TURSO_URL", "")
DB_AUTH = os.getenv("TURSO_AUTH_TOKEN", "")

PRE2022_CUTOFF      = date(2022, 1, 1)
EA_REVIEW_THRESHOLD = 50
EA_AGE_MIN_DAYS     = 90
DELTA_GRADUATION_DAYS = 90

DEFAULT_STEAM_SLEEP = 1.5
FLUSH_BATCH_SIZE    = 2
DB_MAX_RETRIES      = 3
DB_RETRY_DELAY      = 5.0

STEAM_APPDETAILS_URL = "https://store.steampowered.com/api/appdetails"
STEAM_HISTOGRAM_URL  = "https://store.steampowered.com/appreviewhistogram/{appid}?json=1"
EARLY_ACCESS_GENRE_ID = "70"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

def get_conn() -> libsql.Connection:
    if DB_URL and DB_AUTH:
        return libsql.connect(DB_URL, auth_token=DB_AUTH)
    return libsql.connect("early.db")


def ensure_table(conn: libsql.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pre2022_ea_games (
            appid               INTEGER PRIMARY KEY,
            developer_norm      TEXT    NOT NULL,
            ea_start_date       TEXT,
            graduation_date     TEXT,
            first_review_date   TEXT,
            outcome             TEXT,        -- EXIT_SUCCESS / STAYS_ACTIVE / UNKNOWN
            crossed_50_reviews  INTEGER,     -- 1 / 0 / NULL (histogram unavailable)
            ea_age_days         INTEGER,
            is_free             INTEGER DEFAULT 0,
            confidence          TEXT,        -- 'full' / 'no_histogram'
            collected_at        INTEGER
        )
    """)
    conn.commit()
    log.info("pre2022_ea_games table ready")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pre2022_review_history (
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
        CREATE INDEX IF NOT EXISTS idx_pre2022_review_history_appid
        ON pre2022_review_history (appid)
    """)
    conn.commit()    
    log.info("pre2022_review_history table ready")


def load_already_collected(conn: libsql.Connection) -> tuple[set[int], libsql.Connection]:
    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            res = {r[0] for r in conn.execute("SELECT appid FROM pre2022_ea_games").fetchall()}
            return res, conn
        except Exception as e:
            if attempt == DB_MAX_RETRIES:
                raise
            log.warning("DB read error attempt %d: %s - reconnecting in %ds", attempt, e, DB_RETRY_DELAY)
            time.sleep(DB_RETRY_DELAY)
            try:
                conn.close()
            except Exception:
                pass
            conn = get_conn()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_date(s: str | None) -> date | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%b %d, %Y", "%d %b, %Y", "%d %b %Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            continue
    return None


def parse_developers(raw: str | None) -> list[str]:
    if not raw:
        return []
    raw = raw.strip()
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [d.lower().strip() for d in parsed if isinstance(d, str) and d.strip()]
        except json.JSONDecodeError:
            pass
    return [raw.lower().strip()] if raw else []


# ---------------------------------------------------------------------------
# Candidate discovery from games_v2
# ---------------------------------------------------------------------------

def discover_candidates(conn: libsql.Connection, delta: bool = False) -> list[tuple[int, str, date, bool]]:
    """
    Return (appid, developer_norm, ea_start_date, is_free) for all pre-2022
    EA games in games_v2 that share a dev with the active snapshot corpus,
    excluding appids already in the snapshot corpus themselves.

    Uses games_v2 exclusively — no external API calls.
    """
    # Active corpus: appids that have snapshots
    snapshotted = {
        r[0] for r in conn.execute("SELECT DISTINCT appid FROM snapshots").fetchall()
    }

    delta_filter = ""
    if delta:
        delta_filter = f"""
          AND g.appid IN (
              SELECT appid FROM games_v2 
              WHERE currently_in_ea = 1 
                 OR (currently_in_ea = 0 AND graduation_date IS NOT NULL AND graduation_date >= date('now', '-{DELTA_GRADUATION_DAYS} days'))
          )
        """

    # Active corpus dev strings
    active_devs: set[str] = set()
    query = f"""
        SELECT DISTINCT g.developers
        FROM games_v2 g
        WHERE g.appid IN (SELECT appid FROM ccu_availability WHERE ccu_available IN ('AVAILABLE', 'UNAVAILABLE'))
          AND g.developers IS NOT NULL
          {delta_filter}
    """
    for (raw,) in conn.execute(query).fetchall():
        for d in parse_developers(raw):
            active_devs.add(d)

    if delta:
        existing_devs = {r[0] for r in conn.execute("SELECT DISTINCT developer_norm FROM pre2022_ea_games").fetchall() if r[0]}
        before_count = len(active_devs)
        active_devs = active_devs - existing_devs
        log.info("Delta filter: ignored %d already-processed developers (%d remaining)", before_count - len(active_devs), len(active_devs))

    log.info("Active corpus: %d games, %d unique dev strings to process", len(snapshotted), len(active_devs))

    # All games in games_v2 with ea_start_date < 2022 that share a dev
    all_rows = conn.execute("""
        SELECT appid, developers, ea_start_date, is_free
        FROM games_v2
        WHERE ea_start_date IS NOT NULL
          AND ea_start_date < '2022-01-01'
    """).fetchall()

    candidates: list[tuple[int, str, date, bool]] = []
    seen: set[int] = set()

    for appid, developers, ea_start_date_str, is_free in all_rows:
        if appid in snapshotted:
            continue  # already handled by compute_dev_features.py

        ea_start = parse_date(ea_start_date_str)
        if ea_start is None or ea_start >= PRE2022_CUTOFF:
            continue

        devs = parse_developers(developers)
        matching_devs = [d for d in devs if d in active_devs]
        if not matching_devs:
            continue

        if appid in seen:
            continue
        seen.add(appid)

        # Use first matching dev as the canonical developer_norm for this record
        candidates.append((appid, matching_devs[0], ea_start, bool(is_free)))

    log.info("Candidates from games_v2: %d pre-2022 EA sibling games", len(candidates))
    return candidates


# ---------------------------------------------------------------------------
# Steam appdetails
# ---------------------------------------------------------------------------

def fetch_appdetails(appid: int, sleep_s: float) -> dict | None:
    """
    Returns the inner data dict or None.
    None means: delisted, unavailable, or not a game.
    Callers should SKIP None — we cannot infer outcome reliably from absence.
    """
    try:
        resp = requests.get(
            STEAM_APPDETAILS_URL,
            params={"appids": appid,"cc": "us", "l": "en"},
            timeout=10,
        )
        resp.raise_for_status()
        outer = resp.json()
        entry = outer.get(str(appid), {})

        if not entry.get("success"):
            # Delisted or region-locked — skip entirely (design decision: session 9)
            log.debug("appid %d: success=false — skip", appid)
            time.sleep(sleep_s)
            return None

        data = entry.get("data", {})

        # Skip non-games (DLC, soundtrack, etc.)
        if data.get("type", "") != "game":
            log.debug("appid %d: type=%r — skip", appid, data.get("type"))
            time.sleep(sleep_s)
            return None

        time.sleep(sleep_s)
        return data

    except Exception as e:
        log.debug("appdetails failed appid=%d: %s", appid, e)
        time.sleep(sleep_s)
        return None


def infer_outcome_and_graduation(data: dict) -> tuple[str, date | None]:
    """
    Infer outcome from current appdetails state.

    STAYS_ACTIVE : still tagged Early Access (unexpected for pre-2022 but handled)
    EXIT_SUCCESS : no EA tag, has release_date → use release_date as graduation proxy
    UNKNOWN      : no EA tag, no release_date → can't confirm graduation
    """
    genre_ids = {str(g.get("id", "")) for g in data.get("genres", [])}

    if EARLY_ACCESS_GENRE_ID in genre_ids:
        return "STAYS_ACTIVE", None

    # raw_release   = data.get("release_date", {}).get("date", "")
    # graduation_dt = parse_date(raw_release)

    # if graduation_dt is not None:
    #     return "EXIT_SUCCESS", graduation_dt

    return "UNKNOWN", None


# ---------------------------------------------------------------------------
# Steam review histogram
# ---------------------------------------------------------------------------

def fetch_review_histogram(appid: int, sleep_s: float) -> tuple[list[dict] | None, int | None]:
    """
    Full lifetime review histogram, sorted ASC with synthetic end dates added.
    Returns (buckets, start_date_ts) or (None, None) on failure.
    """
    try:
        resp = requests.get(
            STEAM_HISTOGRAM_URL.format(appid=appid),
            #params={"l": "english"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", {})
        start_date_ts = results.get("start_date")
        if start_date_ts is not None:
            try:
                start_date_ts = int(start_date_ts)
            except (ValueError, TypeError):
                start_date_ts = None
        rollups = results.get("rollups", [])

        if not rollups:
            time.sleep(sleep_s)
            return None, start_date_ts

        today_str = datetime.now(timezone.utc).date().isoformat()
        buckets = []
        collected_at = int(datetime.now(timezone.utc).timestamp())

        for i, rollup in enumerate(rollups):
            # bucket_start: unix ts → ISO date
            try:
                bucket_start_dt = datetime.fromtimestamp(int(rollup["date"]), tz=timezone.utc).date()
                bucket_start = bucket_start_dt.isoformat()
            except (KeyError, ValueError, OSError, TypeError) as e:
                log.warning("appid %d rollup[%d] bad date: %s", appid, i, e)
                continue

            # bucket_end: start of next rollup, or today for the last bucket
            if i + 1 < len(rollups):
                try:
                    next_dt = datetime.fromtimestamp(int(rollups[i + 1]["date"]), tz=timezone.utc).date()
                    bucket_end = next_dt.isoformat()
                except (KeyError, ValueError, OSError, TypeError):
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
                "collected_at": collected_at,
            })

        time.sleep(sleep_s)
        return buckets, start_date_ts

    except Exception as e:
        log.debug("histogram failed appid=%d: %s", appid, e)
        time.sleep(sleep_s)
        return None, None


def cumulative_reviews_at(buckets: list[dict], target: date) -> int:
    """
    Reconstruct cumulative reviews at target date via linear interpolation.
    Identical logic to feature_builder.reviews_at_date — must stay in sync.
    """
    total      = 0
    target_iso = target.isoformat()

    for b in buckets:
        if b["bucket_end"] <= target_iso:
            total += b["positive"] + b["negative"]
        elif b["bucket_start"] <= target_iso < b["bucket_end"]:
            try:
                start_dt   = date.fromisoformat(b["bucket_start"])
                end_dt     = date.fromisoformat(b["bucket_end"])
                total_days = (end_dt - start_dt).days
                if total_days > 0:
                    frac    = (target - start_dt).days / total_days
                    total  += round((b["positive"] + b["negative"]) * frac)
            except (ValueError, ZeroDivisionError):
                pass
            break

    return total


def compute_crossed_50(buckets: list[dict], ea_end: date | None) -> int:
    """
    1 if cumulative reviews at ea_end >= 50, else 0.
    Reviews accumulate monotonically so checking at ea_end is sufficient.
    Uses today as upper bound for STAYS_ACTIVE (ea_end=None).
    """
    check_date = ea_end if ea_end is not None else date.today()
    return 1 if cumulative_reviews_at(buckets, check_date) >= EA_REVIEW_THRESHOLD else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(dry_run: bool, resume: bool, steam_sleep: float, delta: bool) -> None:
    conn = get_conn()
    ensure_table(conn)

    if resume:
        already_collected, conn = load_already_collected(conn)
    else:
        already_collected = set()
    candidates        = discover_candidates(conn, delta=delta)

    if dry_run:
        log.info("Dry run — candidates found: %d. No API calls, no writes.", len(candidates))
        for appid, dev_norm, ea_start, is_free in candidates[:20]:
            log.info("  appid=%-10d  dev=%-40s  ea_start=%s  free=%d",
                     appid, dev_norm[:40], ea_start, is_free)
        if len(candidates) > 20:
            log.info("  ... and %d more", len(candidates) - 20)
        conn.close()
        return

    n_collected = 0
    n_skipped   = 0   # delisted / not a game / ea_age too short
    n_resumed   = 0
    n_error     = 0
    records_buf: list[dict] = []
    buckets_buf: list[dict] = []

    for i, (appid, dev_norm, ea_start, is_free) in enumerate(candidates, 1):
        if resume and appid in already_collected:
            n_resumed += 1
            continue

        try:
            buckets, hist_ea_start_ts = fetch_review_histogram(appid, steam_sleep)
            
            if ea_start >= PRE2022_CUTOFF:
                log.debug("appid %d: real ea_start=%s >= 2022 — skip", appid, ea_start)
                n_skipped += 1
                continue

            data = fetch_appdetails(appid, steam_sleep)
            if data is None:
                n_skipped += 1
                continue

            outcome, ea_end = infer_outcome_and_graduation(data)

            first_review_date = None

            if hist_ea_start_ts is not None:
                try:
                    first_review_date = datetime.fromtimestamp(int(hist_ea_start_ts), tz=timezone.utc).date()
                    ea_end = datetime.fromtimestamp(int(hist_ea_start_ts), tz=timezone.utc).date()
                except (ValueError, TypeError, OSError):
                    pass

            if ea_end:
                if ea_end < ea_start:
                    ea_start, ea_end = ea_end, ea_start
                else:
                    ea_end = None


            ea_age_days = (
                (ea_end - ea_start).days
                if ea_end is not None
                else (date.today() - ea_start).days
            )

            if ea_age_days < EA_AGE_MIN_DAYS or ea_end is None:
                log.debug("appid %d: ea_age=%dd < %dd — skip", appid, ea_age_days, EA_AGE_MIN_DAYS)
                n_skipped += 1
                n_collected -= 1
                #continue
            else:
                outcome = "EXIT_SUCCESS"

            # is_free: trust games_v2 as primary; cross-check appdetails
            is_free_final = is_free or bool(data.get("is_free", False))

            if buckets is None:
                crossed    = None
                confidence = "no_histogram"
            else:
                crossed    = compute_crossed_50(buckets, ea_end)
                confidence = "full"
                buckets_buf.extend(buckets)

            records_buf.append({
                "appid":              appid,
                "developer_norm":     dev_norm,
                "ea_start_date":      ea_start.isoformat(),
                "graduation_date":    ea_end.isoformat() if ea_end else None,
                "first_review_date":  first_review_date.isoformat() if first_review_date else None,
                "outcome":            outcome,
                "crossed_50_reviews": crossed,
                "ea_age_days":        ea_age_days,
                "is_free":            1 if is_free_final else 0,
                "confidence":         confidence,
                "collected_at":       int(datetime.now(timezone.utc).timestamp()),
            })
            n_collected += 1

            # Flush every FLUSH_BATCH_SIZE to preserve progress
            if len(records_buf) % FLUSH_BATCH_SIZE == 0:
                conn = _flush(conn, records_buf, buckets_buf)
                records_buf.clear()
                buckets_buf.clear()

        except Exception as e:
            log.warning("[%d/%d] appid %d unexpected error: %s",
                        i, len(candidates), appid, e)
            n_error += 1

        if i % 10 == 0 or i == len(candidates):
            log.info(
                "[%d/%d] collected=%d  skipped=%d  resumed=%d  errors=%d",
                i, len(candidates), n_collected, n_skipped, n_resumed, n_error,
            )

    # Flush remaining
    if records_buf or buckets_buf:
        conn = _flush(conn, records_buf, buckets_buf)

    conn.commit()

    # Summary
    full    = sum(1 for r in records_buf if r["confidence"] == "full")
    no_hist = sum(1 for r in records_buf if r["confidence"] == "no_histogram")
    c_yes   = sum(1 for r in records_buf if r["crossed_50_reviews"] == 1)
    c_no    = sum(1 for r in records_buf if r["crossed_50_reviews"] == 0)
    c_unk   = sum(1 for r in records_buf if r["crossed_50_reviews"] is None)

    log.info("=" * 60)
    log.info("collect_pre2022_ea_games complete")
    log.info("  Candidates       : %d", len(candidates))
    log.info("  Collected        : %d", n_collected)
    log.info("  Skipped          : %d  (delisted / not-game / ea_age < 90d)", n_skipped)
    log.info("  Resumed (skip)   : %d", n_resumed)
    log.info("  Errors           : %d", n_error)
    log.info("  Confidence       : full=%d  no_histogram=%d", full, no_hist)
    log.info("  crossed_50       : yes=%d  no=%d  unknown=%d", c_yes, c_no, c_unk)
    log.info("=" * 60)

    conn.close()


def _flush(conn: libsql.Connection, records: list[dict], buckets: list[dict] = None) -> libsql.Connection:
    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            conn.executemany(
                """
                INSERT OR REPLACE INTO pre2022_ea_games (
                    appid, developer_norm, ea_start_date, graduation_date,
                    first_review_date, outcome, crossed_50_reviews, ea_age_days, is_free,
                    confidence, collected_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r["appid"], r["developer_norm"], r["ea_start_date"],
                        r["graduation_date"], r["first_review_date"], r["outcome"], r["crossed_50_reviews"],
                        r["ea_age_days"], r["is_free"], r["confidence"], r["collected_at"],
                    )
                    for r in records
                ],
            )

            if buckets:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO pre2022_review_history (
                        appid, bucket_start, bucket_end, positive, negative, collected_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (b["appid"], b["bucket_start"], b["bucket_end"], b["positive"], b["negative"], b["collected_at"])
                        for b in buckets
                    ],
                )

            conn.commit()
            log.info("Successfully uploaded %d records and %d buckets to the database.", len(records), len(buckets) if buckets else 0)
            return conn
        except Exception as e:
            if attempt == DB_MAX_RETRIES:
                raise
            log.warning("DB flush error attempt %d: %s - reconnecting in %ds", attempt, e, DB_RETRY_DELAY)
            time.sleep(DB_RETRY_DELAY)
            try:
                conn.close()
            except Exception:
                pass
            conn = get_conn()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect pre-2022 EA game history for active corpus developers."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print candidates only — no API calls, no writes")
    parser.add_argument("--resume", action="store_true",
                        help="Skip appids already in pre2022_ea_games")
    parser.add_argument("--steam-sleep", type=float, default=DEFAULT_STEAM_SLEEP,
                        help="Seconds between Steam API calls (default: %(default)s)")
    parser.add_argument("--delta", action="store_true",
                        help="Delta run: only check devs for active and recently graduated games, skipping existing devs")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run(args.dry_run, args.resume, args.steam_sleep, args.delta)