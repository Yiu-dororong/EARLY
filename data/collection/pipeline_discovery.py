"""
EARLY — Game Discovery Pipeline
=================================
Discovers all Steam EA games (post-2022) using official Steam APIs only.

Step 1: IStoreService/GetAppList — paginated, games only, delta on repeat runs
Step 2: appdetails — eligibility + has_ea_genre + release_date
         Optimisation: if has_ea_genre=True AND release_date < 2022 → skip
         histogram call entirely (release_date IS the ea_start_date, pre-cutoff)
Step 3: histogram API — ea_start_date (results.start_date)
         Also determines graduation: has_ea_genre=False + valid start_date
         → graduated game, store appdetails release_date as graduation_date

Tables:  games_v2, pipeline_log_v2, run_meta
Storage: Turso (libSQL)
run meta: 1779297626
Run modes:
  python pipeline_discovery.py                      # delta run (uses last run timestamp)
  python pipeline_discovery.py --bootstrap          # force full re-fetch
  python pipeline_discovery.py --dry-run            # fetch app list only, no API calls
  python pipeline_discovery.py --check-graduated    # check graduated games
  python pipeline_discovery.py --retry-errors       # rerun games with ERROR status
"""

import time
import logging
import argparse
from datetime import datetime, timezone
import json

import requests
import libsql
from dotenv import load_dotenv
import os


load_dotenv()


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline_discovery.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
TURSO_URL      = os.environ.get("TURSO_URL")      # e.g. libsql://xxx.turso.io
TURSO_TOKEN    = os.environ.get("TURSO_AUTH_TOKEN")
STEAM_API_KEY = os.environ.get("STEAM_API_KEY")  # steamcommunity.com/dev/apikey

ISTORESERVICE_URL = "https://api.steampowered.com/IStoreService/GetAppList/v1/"
APPDETAILS_URL    = "https://store.steampowered.com/api/appdetails"
HISTOGRAM_URL     = "https://store.steampowered.com/appreviewhistogram/{appid}?json=1"

PAGE_SIZE     = 45_000
REQUEST_DELAY = 1.5   # seconds between appdetails / histogram calls
MAX_RETRIES   = 3
RETRY_DELAY   = 15    # seconds on 429 or transient error

EA_CUTOFF_TS  = int(datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp())
EA_GENRE_ID   = "70"

# ── Schema ────────────────────────────────────────────────────────────────────
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS games_v2 (
    appid               INTEGER PRIMARY KEY,
    name                TEXT,
    -- EA dates
    ea_start_date       TEXT,       -- ISO date from histogram start_date
    ea_start_ts         INTEGER,    -- raw unix timestamp
    -- Graduation (populated at discovery time if already graduated)
    currently_in_ea     INTEGER,    -- 1 = active EA, 0 = graduated
    graduation_date     TEXT,       -- ISO date from appdetails release_date
                                    -- only set when currently_in_ea = 0
    -- Eligibility metadata
    initial_price_usd   REAL,
    is_free             INTEGER,
    type                TEXT,
    developers          TEXT,       -- JSON list
    publishers          TEXT,       -- JSON list
    categories          TEXT,       -- JSON list
    -- Outcome — populated later by label_outcomes.py
    outcome             TEXT,       -- EXIT_SUCCESS / EXIT_ABANDONED / STAYS_ACTIVE
    outcome_date        TEXT,
    outcome_source      TEXT,
    -- Pipeline metadata
    eligibility_status  TEXT,       -- ELIGIBLE / SKIP_NOT_GAME / SKIP_FREE /
                                    -- SKIP_PRE_2022 / SKIP_NO_HISTOGRAM /
                                    -- SKIP_NOT_EA / ERROR
    fetched_at          TEXT,
    last_modified_steam INTEGER,    -- last_modified from IStoreService
    notes               TEXT
);

CREATE TABLE IF NOT EXISTS pipeline_log_v2 (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    appid     INTEGER,
    run_at    TEXT,
    step      TEXT,
    status    TEXT,
    message   TEXT
);

CREATE TABLE IF NOT EXISTS run_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

# ── DB ────────────────────────────────────────────────────────────────────────
def get_conn():
    return libsql.connect(database=TURSO_URL, auth_token=TURSO_TOKEN)


def init_schema(conn):
    for stmt in SCHEMA_SQL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
            
    conn.commit()
    log.info("Schema initialised.")


def load_existing_appids(conn) -> set:
    rows = conn.execute("SELECT appid FROM games_v2").fetchall()
    return {r[0] for r in rows}


def get_last_run_ts(conn) -> int:
    row = conn.execute(
        "SELECT value FROM run_meta WHERE key = 'last_run_ts'"
    ).fetchone()
    return int(row[0]) if row else 0


def save_run_ts(conn, ts: int):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            conn.execute(
                "INSERT INTO run_meta (key, value) VALUES ('last_run_ts', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(ts),),
            )
            conn.commit()
            return
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            log.warning(f"DB save_run_ts error attempt {attempt}: {e} — retrying in {RETRY_DELAY}s")
            time.sleep(RETRY_DELAY)


def log_step(conn, appid, step, status, message=""):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            conn.execute(
                "INSERT INTO pipeline_log_v2 (appid, run_at, step, status, message) "
                "VALUES (?,?,?,?,?)",
                (appid, datetime.now(timezone.utc).isoformat(), step, status, message),
            )
            conn.commit()
            return
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            log.warning(f"DB log_step error attempt {attempt} for appid {appid}: {e} — retrying in {RETRY_DELAY}s")
            time.sleep(RETRY_DELAY)


def upsert_game(conn, row: dict):
    cols = ", ".join(row.keys())
    placeholders = ", ".join("?" * len(row))
    conflict_update = ", ".join(
        f"{k}=excluded.{k}" for k in row if k != "appid"
    )
    sql = (
        f"INSERT INTO games_v2 ({cols}) VALUES ({placeholders}) "
        f"ON CONFLICT(appid) DO UPDATE SET {conflict_update}"
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            conn.execute(sql, list(row.values()))
            conn.commit()
            return
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            log.warning(f"DB upsert error attempt {attempt} for appid {row.get('appid')}: {e} — retrying in {RETRY_DELAY}s")
            time.sleep(RETRY_DELAY)


# ── HTTP ──────────────────────────────────────────────────────────────────────
def safe_get(url: str, params: dict = None) -> dict | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 429:
                log.warning(
                    f"Rate limited — waiting {RETRY_DELAY}s (attempt {attempt})"
                )
                time.sleep(RETRY_DELAY)
                continue
            if r.status_code != 200:
                log.warning(f"HTTP {r.status_code} for {url}")
                return None
            return r.json()
        except Exception as e:
            log.warning(f"Request error attempt {attempt}: {e}")
            time.sleep(RETRY_DELAY)
    return None


def parse_steam_date(date_str: str) -> int | None:
    """
    Parse Steam's release_date string to unix timestamp.
    Steam returns formats like '25 May, 2022' or 'May 25, 2022'.
    Returns None if unparseable (e.g. 'Coming soon').
    """
    if not date_str:
        return None
    for fmt in ("%d %b, %Y", "%b %d, %Y", "%d %B, %Y", "%B %d, %Y"):
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return int(dt.replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue
    log.debug(f"Could not parse Steam date: {date_str!r}")
    return None


# ── Step 1: IStoreService/GetAppList ─────────────────────────────────────────
def fetch_app_list(if_modified_since: int = 0) -> list[dict]:
    """
    Paginates IStoreService until have_more_results=False.
    Returns list of {appid, name, last_modified}.
    if_modified_since=0  → bootstrap (full fetch ~49k games).
    if_modified_since=T  → delta (only games modified since T).
    """
    all_apps = []
    last_appid = 0
    page = 1

    log.info(
        "Fetching app list — "
        + (
            "bootstrap"
            if if_modified_since == 0
            else f"delta since "
                 f"{datetime.fromtimestamp(if_modified_since, tz=timezone.utc).isoformat()}"
        )
    )

    while True:
        params = {
            "key":              STEAM_API_KEY,
            "include_games":    1,
            "include_dlc":      0,
            "include_software": 0,
            "include_videos":   0,
            "include_hardware": 0,
            "max_results":      PAGE_SIZE,
            "last_appid":       last_appid,
        }
        if if_modified_since > 0:
            params["if_modified_since"] = if_modified_since

        data = safe_get(ISTORESERVICE_URL, params=params)
        if not data:
            log.error(f"IStoreService fetch failed on page {page} — stopping.")
            break

        response = data.get("response", {})
        apps = response.get("apps", [])
        if not apps:
            break

        all_apps.extend(apps)
        log.info(f"  Page {page}: {len(apps)} apps (running total: {len(all_apps)})")

        if not response.get("have_more_results", False):
            break

        last_appid = apps[-1]["appid"]
        page += 1
        time.sleep(0.5)

    log.info(f"App list complete: {len(all_apps)} apps.")
    return all_apps


# ── Step 2: appdetails ────────────────────────────────────────────────────────
def check_eligibility(appid: int) -> tuple[str, dict]:
    """
    Returns (status, meta).

    Statuses:
      ELIGIBLE_ACTIVE     — in EA, release_date >= 2022, proceed to histogram
      ELIGIBLE_ACTIVE_UNK — in EA, release_date unparseable, proceed to histogram
      ELIGIBLE_GRADUATED  — no EA genre, proceed to histogram to confirm EA history
      SKIP_PRE_2022       — in EA, release_date < 2022, skip histogram entirely
      SKIP_NOT_GAME       — type != game
      SKIP_FREE           — is_free or price = 0
      ERROR               — API failure or delisted
    """
    data = safe_get(APPDETAILS_URL, params={"appids": appid, "cc": "us", "l": "en"})

    if not data or str(appid) not in data:
        return "ERROR", {}

    app_data = data[str(appid)]
    if not app_data.get("success"):
        return "ERROR", {"notes": "appdetails success=false — possibly delisted"}

    info = app_data.get("data", {})

    app_type = info.get("type", "").lower()
    if app_type != "game":
        return "SKIP_NOT_GAME", {
            "type": app_type,
            "name": info.get("name", ""),
        }

    is_free = info.get("is_free", False)
    price_data = info.get("price_overview", {})
    initial_price = price_data.get("initial", 0) / 100.0

    genres = info.get("genres", [])
    has_ea_genre = any(g.get("id") == EA_GENRE_ID for g in genres)

    release_date_str = info.get("release_date", {}).get("date", "")
    release_ts = parse_steam_date(release_date_str)

    developers = info.get("developers", [])
    publishers = info.get("publishers", [])
    categories_raw = info.get("categories", [])
    categories = [c.get("description", "") for c in categories_raw if isinstance(c, dict)]

    meta = {
        "name":              info.get("name", ""),
        "type":              app_type,
        "is_free":           int(is_free),
        "initial_price_usd": initial_price,
        "has_ea_genre":      has_ea_genre,
        "release_date_str":  release_date_str,
        "release_ts":        release_ts,
        "developers":        json.dumps(developers),
        "publishers":        json.dumps(publishers),
        "categories":        json.dumps(categories),
    }

    if is_free or initial_price == 0:
        return "SKIP_FREE", meta

    # Optimisation: parseable date clearly before cutoff → skip histogram
    # For active EA, this means it started before 2022.
    # For graduated games, this means it graduated before 2022.
    if release_ts is not None and release_ts < EA_CUTOFF_TS:
        return "SKIP_PRE_2022", meta

    if has_ea_genre:
        # Post-2022 or unparseable ("Coming soon") → proceed to histogram
        return "ELIGIBLE_ACTIVE", meta

    # No EA genre → may be graduated, histogram will confirm
    return "ELIGIBLE_GRADUATED", meta


# ── Step 3: histogram ─────────────────────────────────────────────────────────
def fetch_ea_start(appid: int) -> int | None:
    """
    Returns ea_start_ts (unix) from histogram results.start_date, or None.
    None means either no reviews yet or game was never in EA.
    """
    data = safe_get(HISTOGRAM_URL.format(appid=appid))
    if not data or data.get("success") != 1:
        return None
    return data.get("results", {}).get("start_date")


# ── Step 4: Check Graduated Games ─────────────────────────────────────────────
def run_check_graduated(dry_run: bool = False):
    conn = get_conn()
    init_schema(conn)

    # Select all current EA appids from the database
    rows = conn.execute("SELECT appid, name, ea_start_ts FROM games_v2 WHERE currently_in_ea = 1").fetchall()

    log.info(f"Found {len(rows)} active EA games to check for graduation.")

    if dry_run:
        log.info("DRY RUN — stopping before API calls.")
        return

    graduated_count = 0
    error_count = 0

    for i, (appid, name, ea_start_ts) in enumerate(rows, 1):
        log.info(f"[{i}/{len(rows)}] Checking graduation status for appid={appid} name={name!r}")
        time.sleep(REQUEST_DELAY)

        data = safe_get(APPDETAILS_URL, params={"appids": appid, "cc": "us", "l": "en"})

        if not data or str(appid) not in data:
            log.warning(f"  ERROR — Failed to fetch appdetails for {appid}")
            error_count += 1
            continue

        app_data = data[str(appid)]
        if not app_data.get("success"):
            log.warning(f"  WARNING — appdetails success=false for {appid} (possibly delisted)")
            continue

        info = app_data.get("data", {})

        # Check if the EA genre ("70") is still present
        genres = info.get("genres", [])
        has_ea_genre = any(g.get("id") == EA_GENRE_ID for g in genres)

        if not has_ea_genre:
            release_date_str = info.get("release_date", {}).get("date", "")
            release_ts = parse_steam_date(release_date_str)

            graduation_date = None
            if release_ts:
                graduation_date = datetime.fromtimestamp(release_ts, tz=timezone.utc).date().isoformat()

            log.info(f"  ✓ GRADUATED — {name} | graduation_date={graduation_date}")

            now_iso = datetime.now(timezone.utc).isoformat()
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    conn.execute(
                        "UPDATE games_v2 SET currently_in_ea = 0, graduation_date = ?, fetched_at = ? WHERE appid = ?",
                        (graduation_date, now_iso, appid)
                    )
                    conn.commit()
                    break
                except Exception as e:
                    if attempt == MAX_RETRIES:
                        raise
                    log.warning(f"DB update error attempt {attempt} for appid {appid}: {e} — retrying in {RETRY_DELAY}s")
                    time.sleep(RETRY_DELAY)

            log_step(conn, appid, "check_graduated", "GRADUATED", f"graduation_date={graduation_date}")
            graduated_count += 1

    log.info(
        f"\n── Graduation Check Complete ──\n"
        f"  Total checked: {len(rows)}\n"
        f"  Graduated:     {graduated_count}\n"
        f"  Errors:        {error_count}\n"
    )

# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(force_bootstrap: bool = False, dry_run: bool = False, retry_errors: bool = False):

    run_start_ts = int(datetime.now(timezone.utc).timestamp())

    conn = get_conn()
    init_schema(conn)

    if retry_errors:
        rows = conn.execute(
            "SELECT appid, name, last_modified_steam FROM games_v2 WHERE eligibility_status = 'ERROR'"
        ).fetchall()
        to_process = [{"appid": r[0], "name": r[1], "last_modified": r[2]} for r in rows]
        log.info(f"Retry errors run — Found {len(to_process)} games with ERROR status to retry.")
    else:
        existing    = load_existing_appids(conn)
        last_run_ts = 0 if force_bootstrap else get_last_run_ts(conn)

        if last_run_ts == 0:
            log.info("Bootstrap run — fetching full app list.")
        else:
            log.info(
                f"Delta run — modified since "
                f"{datetime.fromtimestamp(last_run_ts, tz=timezone.utc).isoformat()}"
            )

        apps = fetch_app_list(if_modified_since=last_run_ts)
        to_process = [a for a in apps if a["appid"] not in existing]

        log.info(
            f"From Steam: {len(apps)} | "
            f"Already in DB: {len(apps) - len(to_process)} | "
            f"To process: {len(to_process)}"
        )

    if dry_run:
        log.info("DRY RUN — stopping before API calls.")
        return

    stats = {
        "eligible_active":    0,
        "eligible_graduated": 0,
        "skip_not_game":      0,
        "skip_free":          0,
        "skip_pre_2022":      0,
        "skip_no_histogram":  0,
        "skip_not_ea":        0,
        "errors":             0,
    }

    for i, app in enumerate(to_process, 1):
        appid         = app["appid"]
        app_name      = app.get("name", "")
        last_modified = app.get("last_modified", 0)
        now_iso       = datetime.now(timezone.utc).isoformat()

        log.info(f"[{i}/{len(to_process)}] appid={appid} name={app_name!r}")

        # ── Step 2 ──
        time.sleep(REQUEST_DELAY)
        status, meta = check_eligibility(appid)

        base_row = {
            "appid":               appid,
            "name":                meta.get("name", app_name),
            "type":                meta.get("type", ""),
            "is_free":             int(meta.get("is_free", False)),
            "initial_price_usd":   meta.get("initial_price_usd", 0.0),
            "currently_in_ea":     1 if meta.get("has_ea_genre") else 0,
            "developers":          meta.get("developers", "[]"),
            "publishers":          meta.get("publishers", "[]"),
            "categories":          meta.get("categories", "[]"),
            "last_modified_steam": last_modified,
            "fetched_at":          now_iso,
        }

        # ── Hard skips — no histogram needed ──
        if status == "ERROR":
            stats["errors"] += 1
            log.warning(f"  ERROR — {meta.get('notes', 'appdetails failed')}")
            log_step(conn, appid, "eligibility", "ERROR", meta.get("notes", ""))
            upsert_game(conn, {
                **base_row,
                "eligibility_status": "ERROR",
                "notes": meta.get("notes", ""),
            })
            continue

        if status == "SKIP_NOT_GAME":
            stats["skip_not_game"] += 1
            log.info(f"  SKIP_NOT_GAME ({meta.get('type')}) — {meta.get('name', app_name)}")
            log_step(conn, appid, "eligibility", status, meta.get("type", ""))
            upsert_game(conn, {**base_row, "eligibility_status": status})
            continue

        if status == "SKIP_FREE":
            stats["skip_free"] += 1
            log.info(f"  SKIP_FREE — {meta.get('name', app_name)}")
            log_step(conn, appid, "eligibility", status, "free or zero price")
            upsert_game(conn, {**base_row, "eligibility_status": status})
            continue

        if status == "SKIP_PRE_2022":
            # release_date confirmed < 2022 — no histogram needed
            stats["skip_pre_2022"] += 1
            ea_date_str = datetime.fromtimestamp(
                meta["release_ts"], tz=timezone.utc
            ).date().isoformat()
            log.info(
                f"  SKIP_PRE_2022 (release={meta['release_date_str']}) "
                f"— {meta['name']}"
            )
            log_step(conn, appid, "eligibility", "SKIP_PRE_2022",
                     meta["release_date_str"])
            upsert_game(conn, {
                **base_row,
                "ea_start_date":      ea_date_str,
                "ea_start_ts":        meta["release_ts"],
                "eligibility_status": "SKIP_PRE_2022",
            })
            continue

        # ── Step 3: histogram ──
        # Reached for ELIGIBLE_ACTIVE and ELIGIBLE_GRADUATED
        time.sleep(REQUEST_DELAY)
        ea_start_ts = fetch_ea_start(appid)
        ea_start_ts = int(ea_start_ts) if ea_start_ts is not None else None

        if ea_start_ts is None:
            if status == "ELIGIBLE_GRADUATED":
                # No histogram + no EA genre = was never in EA
                stats["skip_not_ea"] += 1
                log.info(f"  SKIP_NOT_EA (graduated, no histogram) — {meta['name']}")
                log_step(conn, appid, "histogram", "SKIP_NOT_EA", "")
                upsert_game(conn, {
                    **base_row,
                    "eligibility_status": "SKIP_NOT_EA",
                })
            else:
                # Active EA game but no histogram yet (very new, zero reviews)
                # Keep as ELIGIBLE — will be picked up on next delta run once
                # it accumulates reviews and histogram becomes available
                stats["skip_no_histogram"] += 1
                log.info(f"  SKIP_NO_HISTOGRAM (active EA, no reviews yet) — {meta['name']}")
                log_step(conn, appid, "histogram", "SKIP_NO_HISTOGRAM",
                         "active EA but no histogram yet")
                upsert_game(conn, {
                    **base_row,
                    "eligibility_status": "SKIP_NO_HISTOGRAM",
                    "notes": "no histogram yet — retry on next run",
                })
            continue

        # Temporal filter — applies to both ELIGIBLE_GRADUATED and ELIGIBLE_ACTIVE
        # (catches ELIGIBLE_ACTIVE with unparseable release_date that turns out pre-2022)
        if ea_start_ts < EA_CUTOFF_TS:
            stats["skip_pre_2022"] += 1
            ea_date_str = datetime.fromtimestamp(
                ea_start_ts, tz=timezone.utc
            ).date().isoformat()
            log.info(
                f"  SKIP_PRE_2022 via histogram (ea_start={ea_date_str}) "
                f"— {meta['name']}"
            )
            log_step(conn, appid, "temporal_filter", "SKIP_PRE_2022", ea_date_str)
            upsert_game(conn, {
                **base_row,
                "ea_start_date":      ea_date_str,
                "ea_start_ts":        ea_start_ts,
                "eligibility_status": "SKIP_PRE_2022",
            })
            continue

        # ── All checks passed ──
        ea_date_str     = datetime.fromtimestamp(
            ea_start_ts, tz=timezone.utc
        ).date().isoformat()
        currently_in_ea = base_row["currently_in_ea"]

        # For graduated games, appdetails release_date = graduation date
        graduation_date = None
        if currently_in_ea == 0 and meta.get("release_ts"):
            graduation_date = datetime.fromtimestamp(
                meta["release_ts"], tz=timezone.utc
            ).date().isoformat()

        stats["eligible_active" if currently_in_ea else "eligible_graduated"] += 1

        log.info(
            f"  ✓ {'ACTIVE EA' if currently_in_ea else 'GRADUATED'} — "
            f"{meta['name']} | ea_start={ea_date_str}"
            + (f" | graduated={graduation_date}" if graduation_date else "")
        )

        upsert_game(conn, {
            **base_row,
            "ea_start_date":      ea_date_str,
            "ea_start_ts":        ea_start_ts,
            "graduation_date":    graduation_date,
            "eligibility_status": "ELIGIBLE",
        })
        log_step(
            conn, appid, "complete", "ELIGIBLE",
            f"ea_start={ea_date_str} graduated={graduation_date}",
        )

    if not retry_errors:
        save_run_ts(conn, run_start_ts)

    summary = (
        f"\n── Pipeline complete ──\n"
        f"  Eligible (active EA):   {stats['eligible_active']}\n"
        f"  Eligible (graduated):   {stats['eligible_graduated']}\n"
        f"  Skip (not game):        {stats['skip_not_game']}\n"
        f"  Skip (free):            {stats['skip_free']}\n"
        f"  Skip (pre-2022):        {stats['skip_pre_2022']}\n"
        f"  Skip (no histogram):    {stats['skip_no_histogram']}\n"
        f"  Skip (not EA):          {stats['skip_not_ea']}\n"
        f"  Errors:                 {stats['errors']}"
    )
    if not retry_errors:
        summary += f"\n  Run timestamp saved:    {run_start_ts}"

    log.info(summary)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EARLY game discovery pipeline")
    parser.add_argument(
        "--bootstrap", action="store_true",
        help="Force full fetch ignoring previous run timestamp",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch app list only, no appdetails or histogram calls",
    )
    parser.add_argument(
        "--check-graduated", action="store_true",
        help="Check current EA games in DB to see if they have graduated to 1.0",
    )
    parser.add_argument(
        "--retry-errors", action="store_true",
        help="Rerun pipeline for games where eligibility_status = 'ERROR'",
    )
    args = parser.parse_args()

    if args.check_graduated:
        run_check_graduated(dry_run=args.dry_run)
    else:
        run_pipeline(force_bootstrap=args.bootstrap, dry_run=args.dry_run, retry_errors=args.retry_errors)
