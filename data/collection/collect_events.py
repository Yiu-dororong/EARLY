"""
collect_events.py — EARLY pipeline: Steam event history collector

Fetches event_type 12, 13 (build updates) and 14 (dev posts) for every
ELIGIBLE game in games_v2, stores them in the event_history table.

Event type reference (confirmed against live Steam API):
  12  k_EClanSmallUpdateEvent   — small update / hotfix / patch notes
  13  k_EClanPreAnnounceMajorUpdateEvent — in practice: regular patch update
  14  k_EClanMajorUpdateEvent   — major content update / season launch

Build clock:  types 12 + 13 +14 only
Post clock:   type 28 only (developer-authored posts, filtered)
Branch note:  build_branch field is empty for most games in the events API.
              Branch filtering is NOT implemented — live API data confirms the
              field is unpopulated. All 12/13/14 events are treated as main-branch
              unless a future Steam API change exposes branch data.

Key design decisions (from project context):
  - Type 14 filtered: excludes automated store content (no announcement_body,
    or known automated tag signatures).
  - No look-ahead enforcement here — build_snapshots.py applies snapshot_date
    cutoffs when assembling training features.
  - Idempotent: re-running skips appids already collected unless --force passed.
  - Rate limiting: 1 req/s default, configurable.
  - Pagination: fetches in batches of 100 until exhausted.

Usage:
  python collect_events.py                  # all ELIGIBLE games
  python collect_events.py --appid 1145360  # single game (debug)
  python collect_events.py --force          # re-fetch already-collected games
  python collect_events.py --delta          # update active & recently graduated
  python collect_events.py --limit 500      # cap run size (testing)
  python collect_events.py --dry-run        # print what would be fetched

DB table created if not exists:
  event_history (appid, event_gid, event_type, event_name,
                 event_ts, announcement_body, word_count,
                 is_automated, build_id, build_branch,
                 collected_at)
"""

import argparse
import logging
import os
import json
import re
import time
from typing import Union
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
REQUEST_DELAY = float("1.5")  # seconds between API calls
BATCH_SIZE = 100          # Steam API max per page
BUILD_EVENT_TYPES = {12, 13, 14}
POST_EVENT_TYPES = {28}
DELTA_GRADUATION_DAYS = 90
ALL_WANTED_TYPES = BUILD_EVENT_TYPES | POST_EVENT_TYPES
DB_MAX_RETRIES = 3
DB_RETRY_DELAY = 5.0

# Steam event API endpoint (undocumented but stable)
EVENTS_URL = "https://store.steampowered.com/events/ajaxgetpartnereventspageable/"

# Tags that indicate automated/store-injected content in type 14 events.
# Events with these tags are NOT developer-authored posts.
AUTOMATED_TAGS = frozenset({
    "sale",
    "weekend_deal",
    "daily_deal",
    "spotlight",
    "curator_connect",
    "steam_store",
    "cross_promo",       # Devs advertising their OTHER games
    "franchise_sale",    # Publisher-wide automated sales
    "steam_award"        # "Vote for us!" spam (no dev value)
})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------
# PRE-COMPILED REGEX (Hoisted outside functions for speed)
# ---------------------------------------------------------
_BBCODE_TAG_RE = re.compile(
    r"\[/?\w[^\]]*\]"               # Catches [h1], [/b], [url=...]
    r"|\{STEAM_CLAN_IMAGE\}[^\s]*", # Catches Steam CDN image tokens
    re.IGNORECASE
)

# CJK Unicode Blocks: Hanzi/Kanji, Hiragana, Katakana, Hangul
_CJK_RE = re.compile(
    r'[\u4e00-\u9fff'      # CJK Unified Ideographs (Chinese/Japanese)
    r'\u3040-\u309f'       # Hiragana
    r'\u30a0-\u30ff'       # Katakana
    r'\uac00-\ud7af'       # Hangul (Korean)
    r'\uff00-\uffef]'      # Full-width characters
)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_conn() -> libsql.Connection:
    if DB_URL and DB_AUTH:
        return libsql.connect(DB_URL, auth_token=DB_AUTH)
    # local fallback
    return libsql.connect("early.db")


def ensure_table(conn: libsql.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS event_history (
            appid           INTEGER NOT NULL,
            event_gid       TEXT    NOT NULL,
            event_type      INTEGER NOT NULL,
            event_name      TEXT,
            event_ts        INTEGER NOT NULL,  -- rtime32_start_time (Unix)
            announcement_body TEXT,            -- raw text content of the post
            word_count      INTEGER,           -- word count of announcement_body
            is_automated    INTEGER NOT NULL DEFAULT 0,  -- 1 = filtered out
            build_id        INTEGER,
            build_branch    TEXT,
            collected_at    INTEGER NOT NULL,  -- Unix ts of collection
            PRIMARY KEY (appid, event_gid)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_event_history_appid_type
        ON event_history (appid, event_type)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_event_history_appid_ts
        ON event_history (appid, event_ts)
    """)
    conn.commit()
    log.info("event_history table ready")


def get_eligible_appids(conn: libsql.Connection, delta: bool = False) -> tuple[list[int], libsql.Connection]:
    delta_filter = ""
    if delta:
        delta_filter = f"""
        AND appid IN (
            SELECT appid FROM games_v2 
            WHERE currently_in_ea = 1 
               OR (currently_in_ea = 0 AND graduation_date IS NOT NULL AND graduation_date >= date('now', '-{DELTA_GRADUATION_DAYS} days'))
        )
        """
    query = f"""
        SELECT appid FROM ccu_availability 
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
                raise
            log.warning("DB read error attempt %d: %s - reconnecting", attempt, e)
            time.sleep(DB_RETRY_DELAY)
            try: conn.close()
            except: pass
            conn = get_conn()


def get_already_collected(conn: libsql.Connection) -> tuple[set[int], libsql.Connection]:
    """Return appids that have at least one row in event_history."""
    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            rows = conn.execute("SELECT DISTINCT appid FROM event_history").fetchall()
            return {r[0] for r in rows}, conn
        except Exception as e:
            if attempt == DB_MAX_RETRIES:
                raise
            log.warning("DB read error attempt %d: %s - reconnecting", attempt, e)
            time.sleep(DB_RETRY_DELAY)
            try: conn.close()
            except: pass
            conn = get_conn()


def upsert_events(conn: libsql.Connection, events: list[dict]) -> tuple[int, libsql.Connection]:
    """Insert events, ignoring duplicates. Returns count inserted."""
    if not events:
        return 0, conn

    tuples = [
        (ev["appid"], ev["event_gid"], ev["event_type"], ev["event_name"],
         ev["event_ts"], ev["announcement_body"], ev["word_count"],
         ev["is_automated"], ev["build_id"], ev["build_branch"], ev["collected_at"])
        for ev in events
    ]

    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            conn.executemany("""
                INSERT OR IGNORE INTO event_history
                    (appid, event_gid, event_type, event_name,
                     event_ts, announcement_body, word_count,
                     is_automated, build_id, build_branch, collected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, tuples)
            conn.commit()
            return len(tuples), conn
        except Exception as e:
            if attempt == DB_MAX_RETRIES:
                log.warning("upsert failed: %s", e)
                return 0, conn
            log.warning("DB upsert_events error attempt %d: %s - reconnecting", attempt, e)
            time.sleep(DB_RETRY_DELAY)
            try: conn.close()
            except: pass
            conn = get_conn()


def mark_no_events(conn: libsql.Connection, appid: int) -> libsql.Connection:
    """
    Insert a sentinel row so we know this appid was checked but had no events.
    Uses event_gid='NONE' and event_ts=0 as sentinel values.
    This prevents re-fetching on every run for quiet games.
    """
    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            conn.execute("""
                INSERT OR IGNORE INTO event_history
                    (appid, event_gid, event_type, event_name,
                     event_ts, announcement_body, word_count,
                     is_automated, build_id, build_branch, collected_at)
                VALUES (?, 'NONE', 0, 'NO_EVENTS_SENTINEL', 0, NULL, 0, 0, NULL, NULL, ?)
            """, (appid, int(datetime.now(timezone.utc).timestamp())))
            conn.commit()
            return conn
        except Exception as e:
            if attempt == DB_MAX_RETRIES: raise
            time.sleep(DB_RETRY_DELAY)
            try: conn.close()
            except: pass
            conn = get_conn()


# ---------------------------------------------------------------------------
# Event parsing and filtering
# ---------------------------------------------------------------------------

def strip_bbcode(text: str) -> str:
    """
    Strip Steam BBCode markup and return plain text.
    Optimized to use native string methods for whitespace collapsing.
    """
    if not text:
        return ""
    
    # Replace tags with a space to prevent word-mashing (e.g., "Fixed[b]bug[/b]")
    cleaned = _BBCODE_TAG_RE.sub(" ", text)
    
    # .split() + .join() is vastly faster than re.sub(r"\s+", " ", text)
    return " ".join(cleaned.split())


def count_words(text: Union[str, None]) -> int:
    """
    Count developer effort intelligently for mixed-language patch notes.
    Normalizes CJK characters to Western word-equivalents for ML fairness.
    """
    if not text or not isinstance(text, str):
        return 0

    # 1. Clean the text of Steam markup FIRST
    clean_text = strip_bbcode(text)
    if not clean_text:
        return 0

    # 2. Extract and count CJK characters
    cjk_chars = _CJK_RE.findall(clean_text)
    
    # 3. Remove CJK characters to isolate English/Western words
    text_without_cjk = _CJK_RE.sub(' ', clean_text)
    
    # \w+ perfectly extracts English words, numbers, and strips punctuation
    english_words = re.findall(r'\w+', text_without_cjk)

    # 4. Machine Learning Normalization
    # Translate CJK characters into English "word effort" equivalents (~2.5 chars per word)
    normalized_cjk_effort = int(len(cjk_chars) / 2.5)

    # 5. Return combined effort
    return len(english_words) + normalized_cjk_effort

def is_automated_post(event: dict) -> bool:
    """
    Detect automated / store-injected type 14 events that are not
    developer-authored posts.

    Heuristics:
    1. Tags contain known automated markers.
    2. No announcement_body (content is empty or missing).
    3. event_name matches known store-injection patterns.
    """
    # Check announcement_body existence
    body = event.get("announcement_body")
    if not body:
        return True
    body_text = body.get("body", "") or ""
    if not body_text.strip():
        return True

    # Check tags
    tags = body.get("tags", []) or []
    tag_set = {t.lower() for t in tags}
    if tag_set & AUTOMATED_TAGS:
        return True

    # Steam-injected sale events often have no posterid or posterid = "0"
    # and their event_name contains store keywords
    name = (event.get("event_name") or "").lower()
    automated_name_signals = ["weekend deal", "daily deal", "sale ends", "free weekend"]
    if any(sig in name for sig in automated_name_signals):
        return True

    return False


def parse_event(raw: dict, appid: int) -> dict:
    """
    Parse a raw Steam event dict into our storage format.
    Works for types 12, 13, and 14.
    """
    event_type = raw.get("event_type", 0)
    body_obj = raw.get("announcement_body") or {}
    body_text = body_obj.get("body", "") or ""

    automated = False
    if event_type in POST_EVENT_TYPES:
        automated = is_automated_post(raw)

    return {
        "appid": appid,
        "event_gid": str(raw.get("gid", "")),
        "event_type": event_type,
        "event_name": raw.get("event_name", "")[:500] if raw.get("event_name") else None,
        "event_ts": raw.get("rtime32_start_time", 0),
        "announcement_body": body_text if body_text else None,
        "word_count": count_words(body_text),
        "is_automated": 1 if automated else 0,
        "build_id": raw.get("build_id") or None,
        "build_branch": raw.get("build_branch") or None,
        "collected_at": int(datetime.now(timezone.utc).timestamp()),
    }


# ---------------------------------------------------------------------------
# API fetching
# ---------------------------------------------------------------------------

def fetch_events_for_appid(
    appid: int,
    session: requests.Session,
    dry_run: bool = False,
    delta: bool = False,
    existing_gids: set[str] = None,
) -> list[dict]:
    """
    Paginate through all events for a single appid, returning only
    types 12, 13, 14.

    Steam API paginates via offset. We keep fetching until the returned
    batch is smaller than BATCH_SIZE (last page) or empty.
    """
    if existing_gids is None:
        existing_gids = set()

    if dry_run:
        log.info("[DRY RUN] would fetch events for appid %d", appid)
        return []

    all_events = []
    offset = 0

    while True:
        params = {
            "appid": appid,
            "offset": offset,
            "count": BATCH_SIZE,
            "l": "english",
        }
        try:
            resp = session.get(EVENTS_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            log.warning("appid %d offset %d request failed: %s", appid, offset, e)
            break
        except ValueError as e:
            log.warning("appid %d offset %d JSON parse failed: %s", appid, offset, e)
            break

        raw_events = data.get("events") or []
        if not raw_events:
            break

        # Filter to wanted types before parsing
        wanted = [e for e in raw_events if e.get("event_type") in ALL_WANTED_TYPES]
        
        stop_pagination = False
        new_in_batch = 0
        for raw in wanted:
            gid = str(raw.get("gid", ""))
            if delta and gid in existing_gids:
                stop_pagination = True
                break
            all_events.append(parse_event(raw, appid))
            new_in_batch += 1

        log.debug(
            "appid %d: offset %d → %d total events, %d wanted, %d new",
            appid, offset, len(raw_events), len(wanted), new_in_batch
        )

        if stop_pagination:
            break

        # If we got a full page there may be more; if short, we're done
        if len(raw_events) < BATCH_SIZE:
            break

        offset += BATCH_SIZE
        time.sleep(REQUEST_DELAY)

    return all_events


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect Steam event history for EARLY pipeline")
    p.add_argument("--appid", type=int, help="Fetch a single appid (debug)")
    p.add_argument("--force", action="store_true", help="Re-fetch already-collected appids")
    p.add_argument("--delta", action="store_true", help="Delta run: only fetch for active and recently graduated games")
    p.add_argument("--limit", type=int, help="Max number of appids to process in this run")
    p.add_argument("--dry-run", action="store_true", help="Print plan without fetching or writing")
    p.add_argument("--delay", type=float, default=REQUEST_DELAY, help=f"Seconds between requests (default {REQUEST_DELAY})")
    p.add_argument("--verbose", action="store_true", help="Debug logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    delay = args.delay

    conn = get_conn()
    ensure_table(conn)

    # --- Determine target appids ---
    if args.appid:
        appids = [args.appid]
        log.info("Single-appid mode: %d", args.appid)
    else:
        appids, conn = get_eligible_appids(conn, delta=args.delta)
        log.info("Found %d ELIGIBLE games in games_v2", len(appids))

        if not args.force and not args.delta:
            already, conn = get_already_collected(conn)
            before = len(appids)
            appids = [a for a in appids if a not in already]
            log.info(
                "Skipping %d already-collected appids (%d remaining)",
                before - len(appids), len(appids),
            )
        elif args.delta:
            log.info(
                "Delta run: updating events for %d active/recently graduated games",
                len(appids)
            )

    if args.limit:
        appids = appids[: args.limit]
        log.info("Capped to %d appids via --limit", len(appids))

    if not appids:
        log.info("Nothing to collect. Use --force to re-fetch existing.")
        return

    log.info("Will collect events for %d appids (delay=%.1fs)", len(appids), delay)

    if args.dry_run:
        log.info("[DRY RUN] First 10 appids: %s", appids[:10])
        return

    # --- Collection loop ---
    session = requests.Session()
    session.headers["User-Agent"] = "EARLY-pipeline/1.0"

    total_events = 0
    total_build = 0
    total_post = 0
    total_automated_filtered = 0
    errors = 0
    no_events = 0

    for i, appid in enumerate(appids, 1):
        try:
            existing_gids = set()
            if args.delta:
                rows = conn.execute("SELECT event_gid FROM event_history WHERE appid = ?", (appid,)).fetchall()
                existing_gids = {r[0] for r in rows}

            events = fetch_events_for_appid(
                appid, session, dry_run=args.dry_run,
                delta=args.delta, existing_gids=existing_gids
            )

            if not events:
                if not existing_gids or existing_gids == {'NONE'}:
                    conn = mark_no_events(conn, appid)
                no_events += 1
                log.debug("appid %d: no new events found", appid)
            else:
                inserted, conn = upsert_events(conn, events)
                n_build = sum(1 for e in events if e["event_type"] in BUILD_EVENT_TYPES)
                n_post = sum(1 for e in events if e["event_type"] in POST_EVENT_TYPES and not e["is_automated"])
                n_auto = sum(1 for e in events if e["is_automated"])

                total_events += inserted
                total_build += n_build
                total_post += n_post
                total_automated_filtered += n_auto

                log.info(
                    "[%d/%d] appid %d: %d build events, %d dev posts, %d automated filtered",
                    i, len(appids), appid, n_build, n_post, n_auto,
                )

        except Exception as e:
            log.error("appid %d unexpected error: %s", appid, e)
            errors += 1

        # Rate limiting between appids (not just between pages)
        if i < len(appids):
            time.sleep(delay)

    # --- Summary ---
    log.info("=" * 60)
    log.info("Collection complete")
    log.info("  Appids processed : %d", len(appids))
    log.info("  Total rows stored: %d", total_events)
    log.info("  Build events (12+13): %d", total_build)
    log.info("  Dev posts (14, real): %d", total_post)
    log.info("  Automated filtered  : %d", total_automated_filtered)
    log.info("  No events found     : %d", no_events)
    log.info("  Errors              : %d", errors)
    log.info("=" * 60)

    conn.close()


if __name__ == "__main__":
    main()
