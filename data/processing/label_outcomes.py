"""
label_outcomes.py
-----------------
Applies outcome labels to games_v2 (and optionally the legacy `games` table).

Logic (in priority order):
  1. EXIT_SUCCESS  — appdetails graduation_date is set (non-null, non-empty)
  2. EXIT_ABANDONED — no build (type 12/13/14) AND no dev post (type 28) for > 365 days
  3. EXIT_SILENT   — no build for > allowable_build_gap, but dev posted within 365 days
  4. STAYS_ACTIVE  — build activity within allowable gap (open label)

  allowable_build_gap = max(365, median_historical_build_gap * 1.5)
  Requires MIN_EVENTS_FOR_PERSONAL_THRESHOLD build events to use personal baseline;
  falls back to flat FLOOR_GAP_DAYS (365) for games with insufficient history.

  NOTE: EXIT_SILENT requires event_history with type 28 (dev posts).
  If event_history is absent, EXIT_SILENT collapses into STAYS_ACTIVE — logged as warning.

Writes to:
  games_v2.outcome         — 'EXIT_SUCCESS' | 'EXIT_ABANDONED' | 'EXIT_SILENT' | 'STAYS_ACTIVE'
  games_v2.outcome_date    — ISO date string (graduation_date or last_build_update_date)
  games_v2.outcome_source  — 'appdetails_graduation' | 'abandonment_rule' | 'open_label'
  games_v2.abandoned_date  — ISO date string of the day the abandonment clock expired

Run modes:
  python label_outcomes.py                        — labels games_v2 only
  python label_outcomes.py --dry-run              — prints decisions, writes nothing
  python label_outcomes.py --appid 123            — single game debug mode
  python label_outcomes.py --delta                — only process active and recently graduated games
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from itertools import groupby
from dotenv import load_dotenv

load_dotenv()

import libsql

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_URL = os.environ.get("TURSO_URL", "")
DB_AUTH_TOKEN = os.environ.get("TURSO_AUTH_TOKEN", "")
LOCAL_DB_PATH = os.environ.get("LOCAL_DB_PATH", "early.db")  # fallback for local dev

FLOOR_GAP_DAYS = 365                   # minimum allowable build gap (Steam's own bar)
TOLERANCE_MULTIPLIER = 1.5             # allow 50% longer than personal norm
MIN_EVENTS_FOR_PERSONAL_THRESHOLD = 5  # below this, use FLOOR_GAP_DAYS
POST_GAP_DAYS = 365                    # dev post clock — independent of build clock
MIN_EA_AGE_DAYS = 90                   # clock doesn't start until game has been in EA 90+ days
DELTA_GRADUATION_DAYS = 90             # delta run look-back window
DB_MAX_RETRIES = 3                     # max DB connection retries
DB_RETRY_DELAY = 5.0                   # delay between DB retries

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DB connection
# ---------------------------------------------------------------------------

def get_conn():
    if DB_URL:
        log.info("Connecting to Turso: %s", DB_URL)
        return libsql.connect(DB_URL, auth_token=DB_AUTH_TOKEN)
    log.info("No TURSO_DATABASE_URL set — using local file: %s", LOCAL_DB_PATH)
    return libsql.connect(LOCAL_DB_PATH)


# ---------------------------------------------------------------------------
# Core labelling logic
# ---------------------------------------------------------------------------

def compute_allowable_build_gap(historical_build_gaps: list[int]) -> int:
    if len(historical_build_gaps) < MIN_EVENTS_FOR_PERSONAL_THRESHOLD:
        return FLOOR_GAP_DAYS
    from statistics import median
    return max(FLOOR_GAP_DAYS, int(median(historical_build_gaps) * TOLERANCE_MULTIPLIER))


def compute_outcome(
    appid: int,
    ea_start_date: str | None,
    graduation_date: str | None,
    last_build_update_date: str | None,
    last_dev_post_date: str | None,
    historical_build_gaps: list[int],
    reference_date: datetime | None = None,
) -> dict:
    """
    Returns a dict with keys:
      outcome, outcome_date, outcome_source, reason, abandoned_date

    abandoned_date:
      For EXIT_ABANDONED and EXIT_SILENT: ISO date string of the day the
      abandonment clock expired = last_build_update_date + allowable_build_gap.
      Capped at reference_date (today) — does not project into the future.
      None for EXIT_SUCCESS and STAYS_ACTIVE.
    """
    now = reference_date or datetime.now(timezone.utc)

    # ── 1. EXIT_SUCCESS ───────────────────────────────────────────────────
    if graduation_date and graduation_date.strip():
        return {
            "outcome":        "EXIT_SUCCESS",
            "outcome_date":   graduation_date.strip(),
            "outcome_source": "appdetails_graduation",
            "abandoned_date": None,
            "reason":         f"graduation_date={graduation_date}",
        }

    # ── EA age ────────────────────────────────────────────────────────────
    ea_age_days: float | None = None
    if ea_start_date and ea_start_date.strip():
        try:
            ea_start    = datetime.fromisoformat(ea_start_date.strip()).replace(tzinfo=timezone.utc)
            ea_age_days = (now - ea_start).days
        except ValueError:
            log.warning("appid=%s — unparseable ea_start_date: %r", appid, ea_start_date)

    # ── Build gap ─────────────────────────────────────────────────────────
    allowable_build_gap = compute_allowable_build_gap(historical_build_gaps)

    days_since_build: int | None = None
    last_build_dt: datetime | None = None

    if last_build_update_date and last_build_update_date.strip():
        try:
            last_build_dt    = datetime.fromisoformat(
                last_build_update_date.strip()
            ).replace(tzinfo=timezone.utc)
            days_since_build = (now - last_build_dt).days
        except ValueError:
            log.warning(
                "appid=%s — unparseable last_build_update_date: %r",
                appid, last_build_update_date,
            )
    else:
        days_since_build = int(ea_age_days) if ea_age_days is not None else None

    # ── Post gap ──────────────────────────────────────────────────────────
    days_since_post: int | None = None
    if last_dev_post_date and last_dev_post_date.strip():
        try:
            last_post_dt     = datetime.fromisoformat(
                last_dev_post_date.strip()
            ).replace(tzinfo=timezone.utc)
            days_since_post  = (now - last_post_dt).days
        except ValueError:
            log.warning(
                "appid=%s — unparseable last_dev_post_date: %r",
                appid, last_dev_post_date,
            )

    # ── Helper: compute abandoned_date ────────────────────────────────────
    def _abandoned_date() -> str | None:
        """
        The day the abandonment clock expired:
          last_build_update_date + allowable_build_gap
        Capped at today — does not project into the future.
        Falls back to ea_start_date if no build ever recorded.
        """
        if last_build_dt is not None:
            expiry = last_build_dt + timedelta(days=allowable_build_gap)
        elif ea_start_date and ea_start_date.strip():
            # No build ever — clock started at EA entry
            try:
                expiry = datetime.fromisoformat(
                    ea_start_date.strip()
                ).replace(tzinfo=timezone.utc) + timedelta(days=allowable_build_gap)
            except ValueError:
                return None
        else:
            return None

        # Cap at today — the expiry may be in the future for very recent silences
        expiry = min(expiry, now)
        return expiry.date().isoformat()

    # ── 2 & 3. Distressed branch ──────────────────────────────────────────
    if (
        ea_age_days is not None
        and ea_age_days > MIN_EA_AGE_DAYS
        and days_since_build is not None
        and days_since_build > allowable_build_gap
    ):
        post_expired = (
            days_since_post is None
            or days_since_post > POST_GAP_DAYS
        )

        if post_expired:
            # 2. EXIT_ABANDONED
            return {
                "outcome":        "EXIT_ABANDONED",
                "outcome_date":   (
                    last_build_update_date.strip()
                    if last_build_update_date
                    else ea_start_date
                ),
                "outcome_source": "abandonment_rule",
                "abandoned_date": _abandoned_date(),
                "reason": (
                    f"days_since_build={days_since_build} (>allowable={allowable_build_gap}), "
                    f"days_since_post={days_since_post} (>{POST_GAP_DAYS} or never), "
                    f"ea_age={ea_age_days}d"
                ),
            }
        else:
            # 3. EXIT_SILENT
            return {
                "outcome":        "EXIT_SILENT",
                "outcome_date":   (
                    last_build_update_date.strip()
                    if last_build_update_date
                    else ea_start_date
                ),
                "outcome_source": "silent_rule",
                "abandoned_date": _abandoned_date(),
                "reason": (
                    f"days_since_build={days_since_build} (>allowable={allowable_build_gap}), "
                    f"days_since_post={days_since_post} (<={POST_GAP_DAYS}, still posting), "
                    f"ea_age={ea_age_days}d"
                ),
            }

    # ── 4. STAYS_ACTIVE ───────────────────────────────────────────────────
    return {
        "outcome":        "STAYS_ACTIVE",
        "outcome_date":   None,
        "outcome_source": "open_label",
        "abandoned_date": None,
        "reason": (
            f"ea_age={ea_age_days}d, "
            f"days_since_build={days_since_build}, "
            f"allowable_gap={allowable_build_gap}d"
        ),
    }


# ---------------------------------------------------------------------------
# Data fetching helpers
# ---------------------------------------------------------------------------

def fetch_last_build_update_dates(conn, appids: list[int]) -> dict[int, str | None]:
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='event_history'"
    )
    if not cursor.fetchone():
        log.warning(
            "event_history table not found — all last_build_update_dates will be NULL. "
            "Run collect_events.py first."
        )
        return {appid: None for appid in appids}

    result = {}
    chunk_size = 500
    for i in range(0, len(appids), chunk_size):
        chunk = appids[i : i + chunk_size]
        placeholders = ",".join("?" * len(chunk))
        
        for attempt in range(1, DB_MAX_RETRIES + 1):
            try:
                rows = conn.execute(
                    f"""
                    SELECT appid, datetime(MAX(event_ts), 'unixepoch') AS last_build_update_date
                    FROM event_history
                    WHERE appid IN ({placeholders})
                      AND event_type IN (12, 13, 14)
                    GROUP BY appid
                    """,
                    chunk,
                ).fetchall()
                for row in rows:
                    result[row[0]] = row[1]
                break
            except Exception as e:
                if attempt == DB_MAX_RETRIES:
                    raise
                log.warning("DB read error attempt %d: %s - retrying in %ds", attempt, e, DB_RETRY_DELAY)
                time.sleep(DB_RETRY_DELAY)
    for appid in appids:
        result.setdefault(appid, None)
    return result


def fetch_last_dev_post_dates(conn, appids: list[int]) -> dict[int, str | None]:
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='event_history'"
    )
    if not cursor.fetchone():
        log.warning(
            "event_history table not found — last_dev_post_dates will be NULL. "
            "EXIT_SILENT cannot be distinguished from STAYS_ACTIVE."
        )
        return {appid: None for appid in appids}

    result = {}
    chunk_size = 500
    for i in range(0, len(appids), chunk_size):
        chunk = appids[i : i + chunk_size]
        placeholders = ",".join("?" * len(chunk))
        
        for attempt in range(1, DB_MAX_RETRIES + 1):
            try:
                rows = conn.execute(
                    f"""
                    SELECT appid, datetime(MAX(event_ts), 'unixepoch') AS last_dev_post_date
                    FROM event_history
                    WHERE appid IN ({placeholders})
                      AND event_type = 28
                    GROUP BY appid
                    """,
                    chunk,
                ).fetchall()
                for row in rows:
                    result[row[0]] = row[1]
                break
            except Exception as e:
                if attempt == DB_MAX_RETRIES:
                    raise
                log.warning("DB read error attempt %d: %s - retrying in %ds", attempt, e, DB_RETRY_DELAY)
                time.sleep(DB_RETRY_DELAY)
    for appid in appids:
        result.setdefault(appid, None)
    return result


def fetch_historical_build_gaps(conn, appids: list[int]) -> dict[int, list[int]]:
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='event_history'"
    )
    if not cursor.fetchone():
        return {appid: [] for appid in appids}

    result: dict[int, list[int]] = {appid: [] for appid in appids}
    chunk_size = 500
    for i in range(0, len(appids), chunk_size):
        chunk = appids[i : i + chunk_size]
        placeholders = ",".join("?" * len(chunk))
        
        for attempt in range(1, DB_MAX_RETRIES + 1):
            try:
                rows = conn.execute(
                    f"""
                    SELECT appid, datetime(event_ts, 'unixepoch') AS event_date
                    FROM event_history
                    WHERE appid IN ({placeholders})
                      AND event_type IN (12, 13, 14)
                    ORDER BY appid, event_ts ASC
                    """,
                    chunk,
                ).fetchall()

                for appid, events in groupby(rows, key=lambda r: r[0]):
                    dates = []
                    for row in events:
                        try:
                            dates.append(
                                datetime.fromisoformat(row[1]).replace(tzinfo=timezone.utc)
                            )
                        except (ValueError, TypeError):
                            continue
                    gaps = [
                        (dates[j] - dates[j - 1]).days
                        for j in range(1, len(dates))
                        if (dates[j] - dates[j - 1]).days > 0
                    ]
                    result[appid] = gaps
                break
            except Exception as e:
                if attempt == DB_MAX_RETRIES:
                    raise
                log.warning("DB read error attempt %d: %s - retrying in %ds", attempt, e, DB_RETRY_DELAY)
                time.sleep(DB_RETRY_DELAY)

    return result


def fetch_games_v2(conn, appid_filter: int | None = None, delta: bool = False) -> list[dict]:
    delta_filter = ""
    if delta:
        delta_filter = f"AND (currently_in_ea = 1 OR (currently_in_ea = 0 AND graduation_date IS NOT NULL AND graduation_date >= date('now', '-{DELTA_GRADUATION_DAYS} days')))"

    where = f"""
        WHERE eligibility_status = 'ELIGIBLE'
        AND ea_start_ts IS NOT NULL
        AND outcome != 'EXIT_SUCCESS'
        AND (
            (currently_in_ea = 1)
            OR
            (currently_in_ea = 0
            AND graduation_date IS NOT NULL
            AND CAST(
                (julianday(graduation_date) - julianday(ea_start_date))
                AS INTEGER) >= 90)
            {delta_filter}
            )
        """
    params = []
    if appid_filter is not None:
        where += " AND appid = ?"
        params.append(appid_filter)

    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            rows = conn.execute(
                f"""
                SELECT appid, ea_start_date, graduation_date
                FROM games_v2
                {where}
                """,
                params,
            ).fetchall()
            break
        except Exception as e:
            if attempt == DB_MAX_RETRIES:
                raise
            log.warning("DB fetch_games_v2 error attempt %d: %s - retrying in %ds", attempt, e, DB_RETRY_DELAY)
            time.sleep(DB_RETRY_DELAY)

    return [
        {"appid": r[0], "ea_start_date": r[1], "graduation_date": r[2]}
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def write_outcomes_v2(conn, decisions: list[dict], dry_run: bool = False):
    existing_cols = set()
    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(games_v2)").fetchall()}
            break
        except Exception as e:
            if attempt == DB_MAX_RETRIES:
                raise
            log.warning("DB PRAGMA read error attempt %d: %s - retrying in %ds", attempt, e, DB_RETRY_DELAY)
            time.sleep(DB_RETRY_DELAY)
            
    if "abandoned_date" not in existing_cols:
        if not dry_run:
            for attempt in range(1, DB_MAX_RETRIES + 1):
                try:
                    conn.execute("ALTER TABLE games_v2 ADD COLUMN abandoned_date TEXT")
                    conn.commit()
                    break
                except Exception as e:
                    if attempt == DB_MAX_RETRIES:
                        raise
                    log.warning("DB ALTER TABLE error attempt %d: %s - retrying in %ds", attempt, e, DB_RETRY_DELAY)
                    time.sleep(DB_RETRY_DELAY)
            log.info("Added column games_v2.abandoned_date")
        else:
            log.info("[DRY RUN] Would add column games_v2.abandoned_date")

    updated = 0
    if not dry_run:
        chunk_size = 500
        for i in range(0, len(decisions), chunk_size):
            chunk = decisions[i : i + chunk_size]
            update_tuples = [
                (d["outcome"], d["outcome_date"], d["outcome_source"], d["abandoned_date"], d["appid"])
                for d in chunk
            ]
            for attempt in range(1, DB_MAX_RETRIES + 1):
                try:
                    conn.executemany(
                        """
                        UPDATE games_v2
                        SET outcome        = ?,
                            outcome_date   = ?,
                            outcome_source = ?,
                            abandoned_date = ?
                        WHERE appid = ?
                        """,
                        update_tuples,
                    )
                    conn.commit()
                    break
                except Exception as e:
                    if attempt == DB_MAX_RETRIES:
                        raise
                    log.warning("DB write batch error attempt %d: %s - retrying in %ds", attempt, e, DB_RETRY_DELAY)
                    time.sleep(DB_RETRY_DELAY)
            updated += len(chunk)
        log.info("games_v2: wrote %d outcome labels", updated)
    else:
        for d in decisions:
            log.info(
                "[DRY RUN] appid=%-10s  %-17s  source=%-26s  %s",
                d["appid"], d["outcome"], d["outcome_source"], d["reason"],
            )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(decisions: list[dict]):
    counts: dict[str, int] = {}
    for d in decisions:
        counts[d["outcome"]] = counts.get(d["outcome"], 0) + 1
    total = len(decisions)
    log.info("─" * 60)
    log.info("OUTCOME SUMMARY  (total eligible: %d)", total)
    for outcome in ("EXIT_SUCCESS", "EXIT_ABANDONED", "EXIT_SILENT", "STAYS_ACTIVE"):
        count = counts.get(outcome, 0)
        pct   = (count / total * 100) if total else 0
        log.info("  %-20s  %5d  (%5.1f%%)", outcome, count, pct)
    log.info("─" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Label game outcomes in EARLY DB")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print decisions without writing to DB")
    parser.add_argument("--appid",    type=int, default=None,
                        help="Debug a single appid")
    parser.add_argument("--delta",    action="store_true",
                        help="Delta run: only process active and recently graduated games")
    args = parser.parse_args()

    conn = get_conn()

    # ── Normal labelling pass ─────────────────────────────────────────────
    log.info("Fetching ELIGIBLE games from games_v2...")
    try:
        games = fetch_games_v2(conn, appid_filter=args.appid, delta=args.delta)
    except Exception as e:
        log.error("Could not fetch games_v2: %s", e)
        log.error("Has pipeline_discovery.py been run yet?")
        sys.exit(1)

    if not games:
        log.warning("No ELIGIBLE games found in games_v2. Nothing to label.")
    else:
        log.info("Found %d eligible games in games_v2", len(games))
        appids = [g["appid"] for g in games]

        log.info("Fetching event data from event_history...")
        last_updates = fetch_last_build_update_dates(conn, appids)
        last_posts   = fetch_last_dev_post_dates(conn, appids)
        build_gaps_  = fetch_historical_build_gaps(conn, appids)

        decisions = []
        for g in games:
            result         = compute_outcome(
                appid=g["appid"],
                ea_start_date=g["ea_start_date"],
                graduation_date=g["graduation_date"],
                last_build_update_date=last_updates.get(g["appid"]),
                last_dev_post_date=last_posts.get(g["appid"]),
                historical_build_gaps=build_gaps_.get(g["appid"], []),
            )
            result["appid"] = g["appid"]
            decisions.append(result)

        print_summary(decisions)
        write_outcomes_v2(conn, decisions, dry_run=args.dry_run)

    log.info("Done.")


if __name__ == "__main__":
    main()
