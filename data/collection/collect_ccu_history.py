"""
collect_ccu_history.py — EARLY pipeline: CCU history collector

Three-step pipeline (by design):

  Step 1 — EA age gate
    Select ELIGIBLE games from games_v2 where ea_age >= 90 days.
    Includes both currently_in_ea=1 (active) and currently_in_ea=0 (graduated).
    Games below 90 days are too young to have meaningful trajectory signals.

  Step 2 — Review count gate (Steam Reviews API)
    For each candidate, check current total review count via Steam Reviews API.
    Gate: >= 50 reviews required (ml_eligible boundary).
    Games below 50 are stored with ccu_available='SKIP_LOW_REVIEWS' and excluded.
    This is checked live rather than from games_v2 because review counts grow
    and a game may have crossed the threshold since discovery.

  Step 3 — Steam Charts scrape
    For games that pass Steps 1 + 2, attempt to fetch monthly CCU history
    from steamcharts.com.
    Outcomes:
      - Success: store monthly rows in ccu_history table
      - 500 / not found: store ccu_available='UNAVAILABLE' — game has bled
        out below Steam Charts' monitoring threshold. CCU features will be
        null in snapshots; XGBoost handles natively.
      - Other error: store ccu_available='ERROR' for retry

Storage design:
  ccu_history table: one row per appid per month.
  ccu_availability table: one row per appid recording collection outcome
    and the review count observed at collection time.

Idempotent: skips appids already in ccu_availability unless --force.

Usage:
  python collect_ccu_history.py                  # full eligible universe
  python collect_ccu_history.py --appid 1145360  # single game (debug)
  python collect_ccu_history.py --force          # re-fetch all
  python collect_ccu_history.py --delta          # update active & recently graduated
  python collect_ccu_history.py --limit 200      # cap run (testing)
  python collect_ccu_history.py --dry-run        # plan only, no writes
  python collect_ccu_history.py --skip-review-gate  # bypass Step 2 (bulk debug)
"""

import argparse
import logging
import os
import re
import time
from datetime import date, datetime, timezone

import libsql
import requests
from dotenv import load_dotenv


load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_URL = os.getenv("TURSO_URL")
DB_AUTH = os.getenv("TURSO_AUTH_TOKEN")
REQUEST_DELAY = float(os.getenv("CCU_REQUEST_DELAY", "2.0"))
REVIEW_API_DELAY = float(os.getenv("REVIEW_API_DELAY", "1.5"))

MIN_EA_AGE_DAYS = 90
DELTA_GRADUATION_DAYS = 90
MIN_REVIEW_COUNT = 50
DB_MAX_RETRIES = 3
DB_RETRY_DELAY = 5.0

STEAM_REVIEW_URL = "https://store.steampowered.com/appreviews/{appid}"
STEAM_CHARTS_URL = "https://steamcharts.com/app/{appid}"

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
    # Monthly CCU rows
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ccu_history (
            appid           INTEGER NOT NULL,
            month_date      TEXT    NOT NULL,  -- YYYY-MM-01 (first of month)
            avg_players     REAL,              -- monthly average CCU
            peak_players    INTEGER,           -- monthly peak CCU
            collected_at    INTEGER NOT NULL,
            PRIMARY KEY (appid, month_date)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ccu_history_appid
        ON ccu_history (appid)
    """)

    # One row per appid recording collection outcome
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ccu_availability (
            appid               INTEGER PRIMARY KEY,
            ccu_available       TEXT    NOT NULL,
            -- AVAILABLE / UNAVAILABLE / SKIP_LOW_REVIEWS / SKIP_EA_AGE / ERROR
            review_count_at_check INTEGER,     -- review count observed at Step 2
            review_checked_at   INTEGER,       -- unix ts when review count was fetched
            months_collected    INTEGER,        -- how many monthly rows stored
            collected_at        INTEGER NOT NULL
        )
    """)
    try:
        conn.execute(
            "ALTER TABLE ccu_availability ADD COLUMN review_checked_at INTEGER"
            )
    except Exception as e:
        log.warning("Error occurred while altering ccu_availability table: %s", e)

    conn.commit()
    log.info("ccu_history and ccu_availability tables ready")


def get_candidates(conn: libsql.Connection, delta: bool = False) -> tuple[list[dict],
                                                                          libsql.Connection]:
    """
    Step 1: ELIGIBLE games with ea_age >= 90 days.
    Includes active EA and graduated games.
    Returns list of dicts with appid and ea_start_date.
    """
    today_ts = int(datetime.now(timezone.utc).timestamp())
    min_age_ts = today_ts - (MIN_EA_AGE_DAYS * 86400)

    delta_filter = ""
    if delta:
        delta_filter = (f"AND (g.currently_in_ea = 1 OR (g.currently_in_ea = 0 "
                        f"AND g.graduation_date IS NOT NULL "
                        f"AND g.graduation_date >= "
                        f"date('now', '-{DELTA_GRADUATION_DAYS} days')))")

    query = f"""
        SELECT g.appid, g.ea_start_date, g.ea_start_ts,
                g.graduation_date, ca.review_count_at_check, ca.review_checked_at
        FROM games_v2 g
        LEFT JOIN ccu_availability ca ON g.appid = ca.appid
        WHERE g.eligibility_status = 'ELIGIBLE'
        AND g.ea_start_ts IS NOT NULL
        AND (
            -- Active EA: still in EA and has been for >= 90 days
            (g.currently_in_ea = 1
            AND g.ea_start_ts <= ?)
            OR
            -- Graduated: EA duration from start to graduation was >= 90 days
            (g.currently_in_ea = 0
            AND g.graduation_date IS NOT NULL
            AND CAST(
                (julianday(g.graduation_date) - julianday(g.ea_start_date))
                AS INTEGER) >= 90)
        )
        {delta_filter}
        ORDER BY g.appid
    """
    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            rows = conn.execute(query, (min_age_ts,)).fetchall()
            return [{"appid": r[0], "ea_start_date": r[1],
                     "ea_start_ts": r[2], "review_count": r[4],
                     "review_checked_at": r[5]} for r in rows], conn
        except Exception as e:
            if attempt == DB_MAX_RETRIES:
                raise
            log.warning("DB read error attempt %d in get_candidates: "
                        "%s - reconnecting", attempt, e)
            time.sleep(DB_RETRY_DELAY)
            try:
                conn.close()
            except Exception as e:
                log.warning("Error occurred while closing connection: %s", e)
            conn = get_conn()
    return [], conn


def get_already_collected(conn: libsql.Connection) -> tuple[set[int],
                                                            libsql.Connection]:
    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            rows = conn.execute(
                "SELECT appid FROM ccu_availability WHERE appid"
                ).fetchall()
            return {r[0] for r in rows}, conn
        except Exception as e:
            if attempt == DB_MAX_RETRIES:
                raise
            log.warning("DB read error attempt %d: %s - reconnecting", attempt, e)
            time.sleep(DB_RETRY_DELAY)
            try:
                conn.close()
            except Exception:
                pass
            conn = get_conn()


def write_availability(
    conn: libsql.Connection,
    appid: int,
    status: str,
    review_count: int | None,
    review_checked_at: int | None,
    months_collected: int,
) -> libsql.Connection:
    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            conn.execute("""
                INSERT OR REPLACE INTO ccu_availability
                    (appid, ccu_available, review_count_at_check,
                         review_checked_at, months_collected, collected_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                appid, status, review_count, review_checked_at, months_collected,
                int(datetime.now(timezone.utc).timestamp()),
            ))
            conn.commit()
            return conn
        except Exception as e:
            if attempt == DB_MAX_RETRIES:
                raise
            log.warning("DB write_availability error attempt %d for appid %d: %s - "
                        "reconnecting in %ds", attempt, appid, e, DB_RETRY_DELAY)
            time.sleep(DB_RETRY_DELAY)
            try:
                conn.close()
            except Exception as e:
                log.warning("Error occurred while closing database connection: %s", e)
            conn = get_conn()


def upsert_ccu_rows(conn: libsql.Connection,
                    rows: list[dict],
                    delta: bool = False) -> tuple[int, libsql.Connection]:
    if not rows:
        return 0, conn

    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            current_rows = rows
            if delta:
                appid = current_rows[0]["appid"]
                max_date_row = conn.execute(
                    "SELECT MAX(month_date) FROM ccu_history WHERE appid = ?", (appid,)
                ).fetchone()
                if max_date_row and max_date_row[0]:
                    max_date = max_date_row[0]
                    current_rows = [r for r in current_rows
                                    if r["month_date"] >= max_date]

            tuples = [
                (r["appid"], r["month_date"], r["avg_players"],
                 r["peak_players"], r["collected_at"])
                for r in current_rows
            ]

            conn.executemany("""
                INSERT OR REPLACE INTO ccu_history
                    (appid, month_date, avg_players, peak_players, collected_at)
                VALUES (?, ?, ?, ?, ?)
            """, tuples)
            conn.commit()
            return len(tuples), conn
        except Exception as e:
            if attempt == DB_MAX_RETRIES:
                log.warning("CCU row insert failed after %d attempts for appid=%d: %s",
                            DB_MAX_RETRIES, rows[0]["appid"], e)
                return 0, conn
            log.warning("DB upsert_ccu_rows error attempt %d for appid %d: %s - "
                        "reconnecting in %ds",
                        attempt, rows[0]["appid"], e, DB_RETRY_DELAY)
            time.sleep(DB_RETRY_DELAY)
            try:
                conn.close()
            except Exception as e:
                log.warning("Error occurred while closing database connection: %s", e)
            conn = get_conn()

    return 0, conn


# ---------------------------------------------------------------------------
# Step 2 — Steam Reviews API (review count gate)
# ---------------------------------------------------------------------------

def get_review_count(appid: int, session: requests.Session) -> int | None:
    """
    Fetch total review count (positive + negative) from Steam Reviews API.
    Returns None on failure.
    """
    try:
        resp = session.get(
            STEAM_REVIEW_URL.format(appid=appid),
            params={
                "json": "1",
                "language": "all",
                "purchase_type": "steam",
                "num_per_page": "0",   # metadata only, no review text needed
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("success") != 1:
            log.warning("appid %d review count API returned success != 1", appid)
            return None

        summary = data.get("query_summary") or {}
        total = summary.get("total_reviews")
        if total is not None:
            return int(total)
        # Fallback: sum positive + negative if total_reviews absent
        pos = summary.get("total_positive", 0) or 0
        neg = summary.get("total_negative", 0) or 0
        return pos + neg
    except Exception as e:
        log.warning("appid %d review count fetch failed: %s", appid, e)
        return None


# ---------------------------------------------------------------------------
# Step 3 — Steam Charts scrape
# ---------------------------------------------------------------------------

def parse_steamcharts_page(html: str, appid: int) -> list[dict]:
    """
    Parse the Steam Charts app page HTML to extract monthly CCU data.

    Steam Charts renders a JS table with rows like:
      <tr>
        <td>March 2024</td>
        <td>1,234.5</td>   <!-- avg players -->
        <td>...</td>        <!-- gain -->
        <td>...</td>        <!-- % gain -->
        <td>5,678</td>      <!-- peak players -->
      </tr>

    We extract month, avg_players, peak_players only.
    Returns list of dicts sorted oldest-first.
    """
    now_ts = int(datetime.now(timezone.utc).timestamp())
    rows = []

    # Match table rows containing month data
    # Steam Charts table has a specific structure — month is first td,
    # avg is second, peak is last (5th) td
    row_pattern = re.compile(
        r"<tr[^>]*>\s*"
        r"<td[^>]*>\s*([A-Za-z]+ \d{4}|Last 30 Days)\s*</td>\s*"   # month name
        r"<td[^>]*>\s*([\d,.\-]+)\s*</td>\s*"           # avg players
        r"<td[^>]*>.*?</td>\s*"                          # gain (skip)
        r"<td[^>]*>.*?</td>\s*"                          # % gain (skip)
        r"<td[^>]*>\s*([\d,]+)\s*</td>",                 # peak players
        re.IGNORECASE | re.DOTALL,
    )

    month_map = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }

    for m in row_pattern.finditer(html):
        month_str, avg_str, peak_str = m.group(1), m.group(2), m.group(3)

        if month_str.strip().lower() == "last 30 days":
            now = datetime.now(timezone.utc)
            month_date = date(now.year, now.month, 1).isoformat()
        else:
            # Parse month → YYYY-MM-01
            parts = month_str.strip().split()
            if len(parts) != 2:
                continue
            month_name, year_str = parts
            month_num = month_map.get(month_name.lower())
            if not month_num:
                continue
            try:
                year = int(year_str)
                month_date = date(year, month_num, 1).isoformat()
            except ValueError:
                continue

        # Parse avg (can be float, e.g. "1,234.5"; or "-" for current month)
        try:
            avg = float(avg_str.replace(",", "")) if avg_str.strip() != "-" else None
        except ValueError:
            avg = None

        # Parse peak (integer)
        try:
            peak = int(peak_str.replace(",", ""))
        except ValueError:
            peak = None

        rows.append({
            "appid": appid,
            "month_date": month_date,
            "avg_players": avg,
            "peak_players": peak,
            "collected_at": now_ts,
        })

    # Deduplicate by month_date keeping the first (most recent / Last 30 Days)
    seen = set()
    unique_rows = []
    for r in rows:
        if r["month_date"] not in seen:
            seen.add(r["month_date"])
            unique_rows.append(r)

    # Sort oldest first for clean storage
    unique_rows.sort(key=lambda r: r["month_date"])
    return unique_rows


def fetch_ccu_history(
    appid: int,
    session: requests.Session,
) -> tuple[str, list[dict]]:
    """
    Fetch and parse Steam Charts for one appid.

    Returns:
      ("AVAILABLE", rows)    — success, rows is non-empty list
      ("UNAVAILABLE", [])    — 500 / not found — game below monitoring threshold
      ("ERROR", [])          — unexpected failure, worth retrying
    """
    url = STEAM_CHARTS_URL.format(appid=appid)
    try:
        resp = session.get(url, timeout=15)

        if resp.status_code in (500, 404):
            log.debug("appid %d: Steam Charts %d — UNAVAILABLE",
                      appid, resp.status_code)
            return "UNAVAILABLE", []

        resp.raise_for_status()
        rows = parse_steamcharts_page(resp.text, appid)

        if not rows:
            # Page loaded but no data rows parsed — treat as unavailable
            log.debug("appid %d: Steam Charts returned page but no data rows", appid)
            return "UNAVAILABLE", []

        return "AVAILABLE", rows

    except requests.HTTPError as e:
        status = e.response.status_code if e.response else 0
        if status in (500, 404):
            return "UNAVAILABLE", []
        log.warning("appid %d Steam Charts HTTP error %d", appid, status)
        return "ERROR", []
    except requests.RequestException as e:
        log.warning("appid %d Steam Charts request failed: %s", appid, e)
        return "ERROR", []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect CCU history for EARLY pipeline (3-step pipeline)"
    )
    p.add_argument("--appid", type=int,
                   help="Single appid (debug)")
    p.add_argument("--force", action="store_true",
                   help="Re-fetch already-collected appids")
    p.add_argument("--delta", action="store_true",
                   help="Delta run: update CCU history "
                   "for active and recently graduated games")
    p.add_argument("--limit", type=int, help="Cap number of appids processed")
    p.add_argument("--dry-run", action="store_true",
                   help="Plan only, no API calls or writes")
    p.add_argument("--skip-review-gate", action="store_true",
                   help="Skip Step 2 review count check (debug/bulk mode)")
    p.add_argument("--delay", type=float, default=REQUEST_DELAY,
                   help=f"Seconds between Steam Charts requests "
                   f"(default {REQUEST_DELAY})")
    p.add_argument("--verbose", action="store_true", help="Debug logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    delay = args.delay

    conn = get_conn()
    ensure_tables(conn)

    # --- Step 1: EA age gate ---
    if args.appid:
        # Single-appid mode bypasses DB candidate query
        candidates = [{"appid": args.appid, "ea_start_date": None,
                       "ea_start_ts": None, "review_count": None,
                       "review_checked_at": None}]
        log.info("Single-appid mode: %d", args.appid)
    else:
        candidates, conn = get_candidates(conn, delta=args.delta)
        log.info(
            "Step 1 — EA age gate (>= %d days): %d candidates",
            MIN_EA_AGE_DAYS, len(candidates),
        )

        if not args.force and not args.delta:
            already, conn = get_already_collected(conn)
            before = len(candidates)
            candidates = [c for c in candidates if c["appid"] not in already]
            log.info(
                "Skipping %d already-collected (%d remaining)",
                before - len(candidates), len(candidates),
            )
        elif args.delta:
            log.info(
                "Delta run: updating history for %d active/recently graduated games",
                len(candidates)
            )

    if args.limit:
        candidates = candidates[: args.limit]
        log.info("Capped to %d appids via --limit", len(candidates))

    if not candidates:
        log.info("Nothing to collect.")
        return

    log.info("Processing %d candidates", len(candidates))

    if args.dry_run:
        log.info("[DRY RUN] First 10: %s", [c["appid"] for c in candidates[:10]])
        return

    session = requests.Session()
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (compatible; EARLY-pipeline/1.0)"
    )

    # Counters
    n_skip_reviews = 0
    n_unavailable = 0
    n_available = 0
    n_error = 0
    total_months = 0

    for i, candidate in enumerate(candidates, 1):
        appid = candidate["appid"]
        prev_review_count = candidate.get("review_count")
        review_checked_at = candidate.get("review_checked_at")

        # --- Step 2: Review count gate ---
        review_count = None

        # Fast path for delta runs: skip API call
        # if we know it's already safely > MIN_REVIEW_COUNT + 10
        bypass_api = args.skip_review_gate
        if (args.delta
            and prev_review_count is not None
            and prev_review_count > MIN_REVIEW_COUNT + 10):
            bypass_api = True
            review_count = prev_review_count
            log.debug("[%d/%d] appid %d: bypass review API (previous count %d > %d)",
                      i, len(candidates), appid,
                      prev_review_count, MIN_REVIEW_COUNT + 10)

        # If bypassing via CLI flag
        # ensure we carry over the previous count instead of leaving it None
        if bypass_api and review_count is None and prev_review_count is not None:
            review_count = prev_review_count

        if not bypass_api:
            review_count = get_review_count(appid, session)
            review_checked_at = int(datetime.now(timezone.utc).timestamp())
            time.sleep(REVIEW_API_DELAY)

            # Rescue mission: If the API failed today
            # keep the valid count from yesterday!
            if review_count is None and prev_review_count is not None:
                log.debug(
                    "[%d/%d] appid %d: API failed, rescuing previous review count (%d)",
                    i, len(candidates), appid, prev_review_count
                )
                review_count = prev_review_count
                review_checked_at = candidate.get("review_checked_at")

            if review_count is not None and review_count < MIN_REVIEW_COUNT:
                log.info(
                    "[%d/%d] appid %d: SKIP — %d reviews (need %d)",
                    i, len(candidates), appid, review_count, MIN_REVIEW_COUNT,
                )
                conn = write_availability(conn, appid, "SKIP_LOW_REVIEWS",
                                          review_count, review_checked_at, 0)
                n_skip_reviews += 1
                continue

            if review_count is None:
                log.warning(
                    "[%d/%d] appid %d: review count fetch failed — "
                    "proceeding cautiously",
                    i, len(candidates), appid,
                )
                # Proceed to Step 3 anyway — don't discard on API failure

        # --- Step 3: Steam Charts ---
        status, rows = fetch_ccu_history(appid, session)

        if status == "AVAILABLE":
            inserted, conn = upsert_ccu_rows(conn, rows, delta=args.delta)
            conn = write_availability(conn, appid, "AVAILABLE",
                                      review_count, review_checked_at, inserted)
            n_available += 1
            total_months += inserted
            log.info(
                "[%d/%d] appid %d: AVAILABLE — %d months stored (reviews: %s)",
                i, len(candidates), appid, inserted,
                str(review_count) if review_count is not None else "unchecked",
            )

        elif status == "UNAVAILABLE":
            conn = write_availability(conn, appid, "UNAVAILABLE",
                                      review_count, review_checked_at, 0)
            n_unavailable += 1
            log.info(
                "[%d/%d] appid %d: UNAVAILABLE — below Steam Charts threshold "
                "(reviews: %s) — CCU features will be null in snapshots",
                i, len(candidates), appid,
                str(review_count) if review_count is not None else "unchecked",
            )

        else:  # ERROR
            conn = write_availability(conn, appid, "ERROR",
                                      review_count, review_checked_at, 0)
            n_error += 1
            log.warning(
                "[%d/%d] appid %d: ERROR — will need retry",
                i, len(candidates), appid,
            )

        if i < len(candidates):
            time.sleep(delay)

    # --- Summary ---
    log.info("=" * 60)
    log.info("CCU collection complete")
    log.info("  Candidates processed : %d", len(candidates))
    log.info("  Available (stored)   : %d  (%d monthly rows)",
                                            n_available, total_months)
    log.info("  Unavailable (500/dead): %d", n_unavailable)
    log.info("  Skipped (< %d reviews): %d", MIN_REVIEW_COUNT, n_skip_reviews)
    log.info("  Errors (retry needed) : %d", n_error)
    log.info("=" * 60)

    if n_unavailable > 0:
        log.info(
            "NOTE: %d games returned 500 from Steam Charts. This is expected for "
            "games that have bled out to near-zero CCU. These games still contribute "
            "training signal via event_history and review_history. "
            "CCU features will be null for these rows in build_snapshots.py.",
            n_unavailable,
        )

    conn.close()


if __name__ == "__main__":
    main()
