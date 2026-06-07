"""
collect_genres.py
-----------------
Fetches genre data for all ELIGIBLE appids in games_v2 and populates
game_genres using a three-level cascading fallback:

    Level 1 — Official (Steam appdetails genres[])
        GET store.steampowered.com/api/appdetails?appids=<id>&filters=genres
        Apply GENRE_PRIORITY intersection → primary_genre

    Level 2 — Community (Steam gettoptagsforapp)
        Only fetched when Level 1 yields no primary_genre.
        GET store.steampowered.com/api/gettoptagsforapp?appid=<id>
        Apply same GENRE_PRIORITY intersection against tag names.

    Level 3 — Hard default
        primary_genre = None, genre_scope = 2, genre_source = 'default'
        Only reached when both official genres and community tags are empty
        or contain no GENRE_PRIORITY match.

Schema written:
    game_genres (
        appid           INTEGER PRIMARY KEY,
        primary_genre   TEXT,
        genre_scope     INTEGER,
        genre_source    TEXT,       -- 'official' / 'community' / 'default'
        raw_genres      TEXT,       -- JSON list from appdetails
        community_tags  TEXT,       -- JSON list of {name, count} from gettoptagsforapp
                                    --   NULL if Level 1 succeeded (not fetched)
        fetched_at      TEXT
    )

Rate limits:
    Both endpoints are on store.steampowered.com — ~200 req/min safe.
    Script uses 0.35s delay. Level 2 fetch adds one extra call only for
    games that failed Level 1, so total calls = n_games + n_level2_fallbacks.

Usage:
    python collect_genres.py [--delay SECS] [--refetch] [--dry-run]
"""

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv()

import libsql
import requests

# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

DB_URL = os.getenv("TURSO_URL")
DB_AUTH = os.getenv("TURSO_AUTH_TOKEN")
REQUEST_DELAY = 1.5
DELTA_GRADUATION_DAYS = 90



def get_conn() -> libsql.Connection:
    if DB_URL and DB_AUTH:
        return libsql.connect(DB_URL, auth_token=DB_AUTH)
    return libsql.connect("early.db")


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS game_genres (
    appid           INTEGER PRIMARY KEY,
    primary_genre   TEXT,
    genre_scope     INTEGER,
    genre_source    TEXT,
    raw_genres      TEXT,
    community_tags  TEXT,
    fetched_at      TEXT
)
"""

# ---------------------------------------------------------------------------
# Genre taxonomy
# ---------------------------------------------------------------------------

# Priority-ordered list of (genre_label, scope).
# First match wins — more specific genres ranked above broad ones.
# "Indie" intentionally excluded — business model label, not a genre.
# "Early Access" and "Free To Play" excluded — lifecycle/pricing labels.
GENRE_PRIORITY: list[tuple[str, int]] = [
    ("Massively Multiplayer", 3),
    ("RPG",                   2),
    ("Strategy",              2),
    ("Simulation",            2),
    ("Survival",              2),   
    ("Sports",                2),
    ("Racing",                2),
    ("Adventure",             1),
    ("Action",                1),
    ("Casual",                1),
    ("Visual Novel",          1),   
    ("Rhythm",                1),  
    ("Puzzle",                1),
]

TAG_SYNONYM_MAP = {
    # Community tag string → canonical GENRE_PRIORITY label
    "open world survival craft": "Survival",
    "farming sim":               "Simulation",
    "base building":             "Strategy",
    "colony sim":                "Strategy",
    "city builder":              "Strategy",
    "tower defense":             "Strategy",
    "dating sim":                "Visual Novel",
    "interactive fiction":       "Visual Novel",
    "turn-based strategy":       "Strategy",
    "turn-based tactics":        "Strategy",
    "action roguelike":          "Action",
    "rhythm":                    "Casual",
    "runner":                    "Casual",
    "typing":                    "Casual",
    "4x":                        "Strategy",
    "grand strategy":            "Strategy",
    "management":                "Simulation",
    "resource management":       "Simulation",
}

# Normalised lookup for fast intersection
_GENRE_PRIORITY_NORM: dict[str, tuple[str, int]] = {
    label.lower(): (label, scope)
    for label, scope in GENRE_PRIORITY
}

DEFAULT_SCOPE = 2


def derive_from_labels(labels: list[str]) -> tuple[str | None, int]:
    """
    Given a list of genre/tag label strings, returns (primary_genre, scope)
    using GENRE_PRIORITY order. Returns (None, DEFAULT_SCOPE) if no match.
    """
    normalised = [l.lower().strip() for l in labels]
    # Check synonym map first (more specific → less specific)
    for tag in normalised:
        if tag in TAG_SYNONYM_MAP:
            canonical = TAG_SYNONYM_MAP[tag]
            scope = dict(GENRE_PRIORITY)[canonical]
            return canonical, scope
        
    for label, scope in GENRE_PRIORITY:
        if label.lower() in normalised:
            return label, scope
    return None, DEFAULT_SCOPE


# ---------------------------------------------------------------------------
# Steam API helpers
# ---------------------------------------------------------------------------

APPDETAILS_URL     = "https://store.steampowered.com/api/appdetails"
STORE_APP_URL      = os.getenv("STORE_APP_URL", "https://store.steampowered.com/app/")


def fetch_official_genres(appid: int, session: requests.Session) -> list[str] | None:
    """
    Returns list of genre description strings from Steam appdetails.
    Returns [] if game has no genres. Returns None on network/parse failure.
    """
    try:
        resp = session.get(
            APPDETAILS_URL,
            params={"appids": appid, "filters": "genres"},
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()

        app_data = payload.get(str(appid), {})
        if not isinstance(app_data, dict) or not app_data.get("success"):
            return []

        data_obj = app_data.get("data", {})
        if not isinstance(data_obj, dict):
            return []

        genres = data_obj.get("genres", [])
        if not isinstance(genres, list):
            return []

        return [g["description"] for g in genres if isinstance(g, dict) and "description" in g]

    except requests.exceptions.Timeout:
        log.warning("appid=%d — appdetails timed out", appid)
        return None
    except requests.exceptions.HTTPError as e:
        log.warning("appid=%d — appdetails HTTP %s", appid, e)
        return None
    except (json.JSONDecodeError, KeyError) as e:
        log.warning("appid=%d — appdetails parse error: %s", appid, e)
        return None


def fetch_community_tags(appid: int, session: requests.Session) -> list[dict] | None:
    """
    Scrapes Steam community tags and their vote counts directly from the official store page HTML.
    Returns list of {name, count} dicts.
    Returns [] if no tags. Returns None on failure.
    """
    url = f"{STORE_APP_URL}{appid}/"
    cookies = {
        'birthtime': '283993201', 
        'lastagecheckage': '1-January-1979',
        'wants_mature_content': '1' # This is the missing key for M-rated games
    }

    try:
        resp = session.get(url, cookies=cookies, timeout=15)
        resp.raise_for_status()

        # The exact tag counts are embedded in a JavaScript call on the page
        match = re.search(r'InitAppTagModal\s*\(\s*\d+\s*,\s*(\[\{.*?\}\])', resp.text, re.DOTALL)
        
        if match:
            try:
                tag_data = json.loads(match.group(1))
                return [{"name": t["name"], "count": int(t.get("count", 0))} for t in tag_data if "name" in t]
            except json.JSONDecodeError:
                pass

        # Fallback to BeautifulSoup scraping if the JS array isn't found (count defaults to 0)
        soup = BeautifulSoup(resp.text, 'html.parser')
        tags = [tag.text.strip() for tag in soup.find_all('a', class_='app_tag') if tag.text.strip() != '+']
        
        return [{"name": tag, "count": 0} for tag in tags]

    except requests.exceptions.Timeout:
        log.warning("appid=%d — store page scraping timed out", appid)
        return None
    except requests.exceptions.RequestException as e:
        log.warning("appid=%d — store page scraping HTTP error: %s", appid, e)
        return None
    except Exception as e:
        log.warning("appid=%d — store page scraping error: %s", appid, e)
        return None


# ---------------------------------------------------------------------------
# Three-level resolution
# ---------------------------------------------------------------------------

def resolve_genre(
    appid: int,
    session: requests.Session,
    delay: float,
) -> dict:
    """
    Executes the three-level fallback and returns a result dict with all
    fields needed for upsert.

    Returns:
        {
            primary_genre  : str | None,
            genre_scope    : int,
            genre_source   : str,          # 'official' / 'community' / 'default'
            raw_genres     : list[str],
            community_tags : list[dict] | None,
            failed         : bool,         # True if any network call failed
        }
    """
    result = {
        "primary_genre":  None,
        "genre_scope":    DEFAULT_SCOPE,
        "genre_source":   "default",
        "raw_genres":     [],
        "community_tags": None,
        "failed":         False,
    }

    # ---- Level 1: Official genres ----------------------------------------
    raw_genres = fetch_official_genres(appid, session)

    if raw_genres is None:
        result["failed"] = True
        return result

    result["raw_genres"] = raw_genres

    # Filter out non-genre labels before priority matching
    EXCLUDE = {"early access", "free to play", "indie"}
    filtered = [g for g in raw_genres if g.lower().strip() not in EXCLUDE]

    primary, scope = derive_from_labels(filtered)
    if primary is not None:
        result["primary_genre"] = primary
        result["genre_scope"]   = scope
        result["genre_source"]  = "official"
        return result

    # ---- Level 2: Community tags -----------------------------------------
    time.sleep(delay)  # extra delay for the second call
    community_tags = fetch_community_tags(appid, session)

    if community_tags is None:
        result["failed"] = True
        return result

    result["community_tags"] = community_tags

    tag_names = [t["name"] for t in community_tags]
    primary, scope = derive_from_labels(tag_names)

    if primary is not None:
        result["primary_genre"] = primary
        result["genre_scope"]   = scope
        result["genre_source"]  = "community"
        return result

    # ---- Level 3: Hard default -------------------------------------------
    # primary_genre stays None, genre_scope = DEFAULT_SCOPE, source = 'default'
    return result


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_target_appids(conn: libsql.Connection, refetch: bool, delta: bool = False) -> list[int]:
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
    all_rows = conn.execute(query).fetchall()
    all_appids = [r[0] for r in all_rows]

    if refetch:
        return sorted(all_appids)

    existing = {r[0] for r in conn.execute("SELECT appid FROM game_genres").fetchall()}
    targets = [a for a in all_appids if a not in existing]
    log.info(
        "Eligible: %d | Already fetched: %d | Remaining: %d",
        len(all_appids), len(existing), len(targets),
    )
    return sorted(targets)


def upsert(conn: libsql.Connection, appid: int, res: dict) -> None:
    conn.execute(
        """
        INSERT INTO game_genres
            (appid, primary_genre, genre_scope, genre_source,
             raw_genres, community_tags, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(appid) DO UPDATE SET
            primary_genre  = excluded.primary_genre,
            genre_scope    = excluded.genre_scope,
            genre_source   = excluded.genre_source,
            raw_genres     = excluded.raw_genres,
            community_tags = excluded.community_tags,
            fetched_at     = excluded.fetched_at
        """,
        [
            appid,
            res["primary_genre"],
            res["genre_scope"],
            res["genre_source"],
            json.dumps(res["raw_genres"]),
            json.dumps(res["community_tags"]) if res["community_tags"] is not None else None,
            datetime.now(timezone.utc).isoformat(),
        ],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run(delay: float, refetch: bool, dry_run: bool, delta: bool) -> None:
    conn = get_conn()
    conn.execute(CREATE_TABLE_SQL)

    targets = get_target_appids(conn, refetch, delta=delta)
    if not targets:
        log.info("Nothing to fetch.")
        conn.close()
        return

    total = len(targets)
    log.info(
        "Fetching %d appids (delay=%.2fs, ETA ~%.0f min)...",
        total, delay, total * delay / 60,
    )

    session = requests.Session()
    session.headers.update({"User-Agent": "EARLY-research/1.0"})

    # Tracking counters
    source_counts = {"official": 0, "community": 0, "default": 0}
    failed: list[int] = []

    for i, appid in enumerate(targets, 1):
        res = resolve_genre(appid, session, delay)

        if res["failed"]:
            failed.append(appid)
        else:
            source_counts[res["genre_source"]] += 1

            if dry_run:
                if i <= 10:
                    log.info(
                        "  appid=%-8d  genre=%-25s  scope=%s  source=%s",
                        appid,
                        res["primary_genre"] or "None",
                        res["genre_scope"],
                        res["genre_source"],
                    )
            else:
                upsert(conn, appid, res)
                conn.commit()

        if i % 100 == 0 or i == total:
            log.info(
                "[%d/%d] official=%d community=%d default=%d failed=%d",
                i, total,
                source_counts["official"],
                source_counts["community"],
                source_counts["default"],
                len(failed),
            )

        if i < total:
            time.sleep(delay)

    log.info("=" * 60)
    log.info("Done. source breakdown: %s", source_counts)
    if source_counts["default"] > 0:
        log.warning("Found %d games with unclassified/unknown primary genre (defaulted).", source_counts["default"])

    if failed:
        log.warning("Failed appids (re-run to retry): %s", failed[:20])

    conn.close()


# ---------------------------------------------------------------------------
# Training-time helpers — import in build_features.py
# ---------------------------------------------------------------------------

def build_genre_onehot(conn: libsql.Connection) -> "pd.DataFrame":
    """
    Returns DataFrame with [appid, primary_genre] for one-hot encoding
    at training time.

    Usage in build_features.py:
        from collect_genres import build_genre_onehot
        import pandas as pd

        genre_df = build_genre_onehot(conn)
        # Fit on train only:
        dummies = pd.get_dummies(train_df.merge(genre_df, on='appid')['primary_genre'], prefix='genre')
        # Apply same columns to test:
        test_dummies = pd.get_dummies(test_df.merge(genre_df, on='appid')['primary_genre'], prefix='genre')
        test_dummies = test_dummies.reindex(columns=dummies.columns, fill_value=0)
    """
    import pandas as pd
    rows = conn.execute("SELECT appid, primary_genre FROM game_genres").fetchall()
    return pd.DataFrame(rows, columns=["appid", "primary_genre"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Populate game_genres with three-level genre fallback."
    )
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY,
                        help="Seconds between requests (default: 1.5)")
    parser.add_argument("--refetch", action="store_true",
                        help="Re-fetch already-stored appids")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch and compute but do not write to DB")
    parser.add_argument("--delta", action="store_true",
                        help="Delta run: only fetch for active and recently graduated games")
    args = parser.parse_args()

    run(args.delay, args.refetch, args.dry_run, args.delta)