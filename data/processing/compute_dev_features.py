"""
compute_dev_features.py
------------------------
Populates developer cross-game features in the snapshots table:
  - dev_previous_ea_count   : Prior EA games crossing 50 peak EA reviews
                               AND ea_age > 90 days AND ea_start_date < T
                               All outcomes included (abandoned/active/success).
                               MAX across dev entities.
  - dev_has_prior_success   : 1 if any prior EA game crossed 50 peak EA reviews
                               AND graduated before T (ANY across dev entities)
  - dev_total_games_shipped : Paid games shipped (EA + non-EA) with
                               release_date < T. EA games only count if graduated.
                               Free games excluded (is_free = 1).
                               MAX across dev entities.

Identity resolution:
  - "Active corpus" devs: normalised dev strings from games_v2 WHERE appid IN
    snapshots. These are the developers whose siblings we care about.
  - Sibling lookup: all games in games_v2 sharing a dev string with an active
    corpus game (including pre-2022 games never snapshotted themselves).
  - Pre-2022 sibling games with no snapshot data: review count is unknown.
    They are excluded from the 50-review gate conservatively (skipped).
    Hook: if pre2022_ea_games table exists, it is merged in to fill these gaps.

Publisher detection: dev strings appearing on > PUBLISHER_FREQ_CAP games
in the full corpus are treated as publisher credits and excluded.

Changes vs prior version:
  - dev_previous_ea_count: added ea_age > 90d gate
  - dev_previous_ea_count: all outcomes now count (was implicitly success-biased)
  - dev_total_games_shipped: free games excluded via is_free flag
  - Sibling lookup now scoped to devs present in snapshot corpus (not all 160k)
    with full sibling expansion across games_v2 (catches pre-2022 games)
  - Pre-2022 games with no snapshot data: skipped conservatively with
    clear TODO hook for pre2022_ea_games table (Option A)

Usage:
    python compute_dev_features.py [--dry-run] [--publisher-cap N] [--verbose]

Assumptions:
    - games_v2: appid, developers (JSON), ea_start_date, graduation_date,
                outcome, abandoned_date, is_free
    - snapshots: appid, snapshot_date, dev_previous_ea_count,
                 dev_has_prior_success, dev_total_games_shipped, review_count_at_T
    - pre2022_ea_games (optional): appid, developer_norm, ea_start_date,
                 graduation_date, outcome, crossed_50_reviews, ea_age_days, is_free
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import date, datetime, timezone
import os

import libsql
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_URL  = os.getenv("TURSO_URL", "")
DB_AUTH = os.getenv("TURSO_AUTH_TOKEN", "")

PUBLISHER_FREQ_CAP   = 10
EA_REVIEW_THRESHOLD  = 50
EA_AGE_MIN_DAYS      = 90   # ea_age gate for dev_previous_ea_count
PRE2022_CUTOFF       = date(2022, 1, 1)
DELTA_GRADUATION_DAYS = 90
DB_BATCH_SIZE        = 500

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_conn() -> libsql.Connection:
    if DB_URL and DB_AUTH:
        return libsql.connect(DB_URL, auth_token=DB_AUTH)
    return libsql.connect("early.db")


def ensure_tables(conn: libsql.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS game_dev_features_current (
            appid INTEGER PRIMARY KEY,
            dev_previous_ea_count INTEGER,
            dev_has_prior_success INTEGER,
            dev_total_games_shipped INTEGER,
            updated_at TEXT
        )
    """)
    conn.commit()


def parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def normalise_dev(name: str) -> str:
    return name.lower().strip()


def parse_developers(raw: str | None) -> list[str]:
    if not raw:
        return []
    raw = raw.strip()
    if raw.startswith("["):
        try:
            devs = json.loads(raw)
            if isinstance(devs, list):
                return [normalise_dev(d) for d in devs if isinstance(d, str) and d.strip()]
        except json.JSONDecodeError:
            pass
    return [normalise_dev(raw)] if raw else []


def table_exists(conn: libsql.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Build in-memory game index
# ---------------------------------------------------------------------------

def build_game_index(conn: libsql.Connection, publisher_cap: int) -> dict[int, dict]:
    """
    Build a metadata dict for every game in games_v2.

    Scope for sibling resolution:
      - Active corpus devs: dev strings that appear in games_v2 WHERE appid
        is in snapshots. These are the developers whose history matters.
      - All games in games_v2 sharing any active-corpus dev string are
        included as potential siblings — this catches pre-2022 games that
        were never snapshotted themselves.

    ea_peak_review_count:
      - Resolved games: MAX(review_count_at_T) across snapshots <= ea_end
      - STAYS_ACTIVE: most recent snapshot review count
      - No snapshots (pre-2022 or unsnapshotted): None → excluded from
        50-review gate conservatively. Filled by pre2022_ea_games if present.

    is_free: games with is_free=1 are excluded from dev_total_games_shipped.
    """
    # Step 1: get all games from games_v2
    all_rows = conn.execute("""
        SELECT appid, developers, ea_start_date, graduation_date,
               outcome, abandoned_date, is_free
        FROM games_v2
    """).fetchall()

    # Step 2: which appids have snapshots (active corpus)
    snapshotted = {
        r[0] for r in conn.execute("SELECT DISTINCT appid FROM snapshots").fetchall()
    }
    log.info("Active corpus: %d snapshotted games out of %d total", len(snapshotted), len(all_rows))

    # Step 3: dev frequency count (across full corpus for publisher detection)
    dev_freq: dict[str, int] = defaultdict(int)
    raw_games = []
    for appid, developers, ea_start_date, graduation_date, outcome, abandoned_date, is_free in all_rows:
        devs = parse_developers(developers)
        raw_games.append((appid, devs, ea_start_date, graduation_date, outcome, abandoned_date, is_free))
        for d in devs:
            dev_freq[d] += 1

    publishers = {d for d, cnt in dev_freq.items() if cnt > publisher_cap}
    if publishers:
        log.warning(
            "Excluding %d high-frequency dev strings (publisher_cap=%d): %s",
            len(publishers), publisher_cap, sorted(publishers)[:10],
        )

    # Step 4: active corpus dev strings (devs present in snapshotted games)
    active_devs: set[str] = set()
    for appid, devs, *_ in raw_games:
        if appid in snapshotted:
            for d in devs:
                if d not in publishers:
                    active_devs.add(d)
    log.info("Active corpus dev strings: %d unique (post publisher filter)", len(active_devs))

    # Step 5: snapshot data for review counts — ASC so last element = most recent
    snap_rows = conn.execute("""
        SELECT appid, snapshot_date, review_count_at_T
        FROM snapshots
        WHERE review_count_at_T IS NOT NULL
        ORDER BY appid, snapshot_date ASC
    """).fetchall()

    snap_by_appid: dict[int, list[tuple[date, int]]] = defaultdict(list)
    for appid, snapshot_date, review_count in snap_rows:
        d = parse_date(snapshot_date)
        if d is not None:
            snap_by_appid[appid].append((d, review_count))

    # Step 6: build index — include ALL games that share a dev with active corpus
    #         (needed so pre-2022 games appear as valid siblings)
    game_index: dict[int, dict] = {}

    for appid, devs, ea_start_date, graduation_date, outcome, abandoned_date, is_free in raw_games:
        clean_devs = [d for d in devs if d not in publishers]

        # Only index games that are either in the active corpus OR share a dev
        # with the active corpus. Skip pure strangers (not worth the memory).
        is_active         = appid in snapshotted
        shares_active_dev = any(d in active_devs for d in clean_devs)
        if not is_active and not shares_active_dev:
            continue

        ea_start      = parse_date(ea_start_date)
        graduation_dt = parse_date(graduation_date)
        abandoned_dt  = parse_date(abandoned_date)
        is_free_flag  = bool(is_free) if is_free is not None else False

        if outcome == "EXIT_SUCCESS":
            ea_end = graduation_dt
        elif outcome in ("EXIT_ABANDONED", "EXIT_SILENT"):
            ea_end = abandoned_dt
        else:
            ea_end = None  # STAYS_ACTIVE

        is_ea = ea_start is not None

        # ea_age_days: resolved = ea_end - ea_start, active = unknown (None)
        ea_age_days: int | None = None
        if is_ea and ea_start is not None and ea_end is not None:
            ea_age_days = (ea_end - ea_start).days

        # ea_peak_review_count from snapshot data
        # Pre-2022 games with no snapshots → None (conservative skip)
        ea_peak_review_count: int | None = None
        if is_ea:
            snaps = snap_by_appid.get(appid, [])
            if snaps:
                if ea_end is not None:
                    ea_snaps = [rc for d, rc in snaps if d <= ea_end]
                    ea_peak_review_count = max(ea_snaps) if ea_snaps else None
                else:
                    # STAYS_ACTIVE: most recent snapshot (last in ASC list)
                    ea_peak_review_count = snaps[-1][1]
            # else: no snapshot data → remains None

        # release_date for dev_total_games_shipped:
        # EA games: only graduation_date (abandoned ≠ shipped)
        # Non-EA: graduation_date is the release date
        release_date = graduation_dt  # covers both cases cleanly

        game_index[appid] = {
            "devs":                  clean_devs,
            "ea_start":              ea_start,
            "graduation_date":       graduation_dt,
            "outcome":               outcome,
            "ea_end":                ea_end,
            "ea_age_days":           ea_age_days,
            "is_ea":                 is_ea,
            "is_free":               is_free_flag,
            "release_date":          release_date,
            "ea_peak_review_count":  ea_peak_review_count,  # None = unknown
        }

    log.info("Game index built: %d games (active corpus + siblings)", len(game_index))
    return game_index


def merge_pre2022_table(conn: libsql.Connection, game_index: dict[int, dict]) -> int:
    """
    TODO (Option A): merge pre2022_ea_games table into game_index.

    When pre2022_ea_games is populated, this fills ea_peak_review_count and
    ea_age_days for pre-2022 sibling games that have no snapshot data.
    Games already in game_index with snapshot-derived data are NOT overwritten.

    Expected schema:
        pre2022_ea_games (
            appid             INTEGER PRIMARY KEY,
            developer_norm    TEXT,
            ea_start_date     TEXT,
            graduation_date   TEXT,
            outcome           TEXT,
            crossed_50_reviews INTEGER,  -- 1 / 0 / NULL (unknown)
            ea_age_days       INTEGER,
            is_free           INTEGER
        )

    Returns number of games patched.
    """
    if not table_exists(conn, "pre2022_ea_games"):
        return 0

    rows = conn.execute("""
        SELECT appid, developer_norm, ea_start_date, graduation_date,
               outcome, crossed_50_reviews, ea_age_days, is_free
        FROM pre2022_ea_games
        WHERE (outcome = 'EXIT_SUCCESS') OR (outcome = 'STAYS_ACTIVE')
    """).fetchall()

    patched = 0
    for appid, dev_norm, ea_start_date, graduation_date, outcome, crossed_50, ea_age_days, is_free in rows:
        if appid not in game_index:
            # Add new entry for pre-2022 game not in games_v2 corpus at all
            ea_start      = parse_date(ea_start_date)
            graduation_dt = parse_date(graduation_date) if outcome != "STAYS_ACTIVE" else None
            game_index[appid] = {
                "devs":                  [dev_norm] if dev_norm else [],
                "ea_start":              ea_start,
                "graduation_date":       graduation_dt,
                "outcome":               outcome or "UNKNOWN",
                "ea_end":                graduation_dt,
                "ea_age_days":           ea_age_days if outcome != "STAYS_ACTIVE" else EA_AGE_MIN_DAYS, # Checked: latest pre-2022 ea release date - oldest snapshot date > 90
                "is_ea":                 True,
                "is_free":               bool(is_free),
                "release_date":          graduation_dt if graduation_dt else ea_start, # release here means 1.0 release, avoid active EA return None by replacing EA start date
                "ea_peak_review_count":  EA_REVIEW_THRESHOLD if crossed_50 else 0, # all pre-2022 EA games have graduated before 2022, so there is no need for real time calculation
            }
            patched += 1
        else:
            # Patch existing entry only if snapshot data is missing
            existing = game_index[appid]
            if existing["ea_peak_review_count"] is None and crossed_50 is not None:
                existing["ea_peak_review_count"] = EA_REVIEW_THRESHOLD if crossed_50 else 0
                if existing["ea_age_days"] is None and ea_age_days is not None:
                    existing["ea_age_days"] = ea_age_days
                patched += 1

    log.info("pre2022_ea_games: patched %d games into game_index", patched)
    return patched


def build_dev_to_games(game_index: dict[int, dict]) -> dict[str, list[int]]:
    dev_to_games: dict[str, list[int]] = defaultdict(list)
    for appid, meta in game_index.items():
        for dev in meta["devs"]:
            dev_to_games[dev].append(appid)
    return dev_to_games


# ---------------------------------------------------------------------------
# Per-snapshot feature computation
# ---------------------------------------------------------------------------

def compute_dev_features(
    target_appid: int,
    snapshot_t: date,
    game_index: dict[int, dict],
    dev_to_games: dict[str, list[int]],
) -> tuple[int, int, int]:
    """
    Returns (dev_previous_ea_count, dev_has_prior_success, dev_total_games_shipped).

    dev_previous_ea_count:
        Prior EA games where ALL of:
          - ea_start < T
          - ea_age_days > 90  (game was meaningfully in EA, not a quick pass-through)
          - ea_peak_review_count >= 50  (had real traction during EA)
          - All outcomes count (abandoned/active/success)
          - ea_peak_review_count is None → skip conservatively (pre-2022 unknown)
        MAX across dev entities.

    dev_has_prior_success:
        1 if ANY sibling game where:
          - ea_peak_review_count >= 50  (same gate)
          - outcome == EXIT_SUCCESS AND graduation_date < T
          - ea_peak_review_count is None → skip conservatively
        ANY across dev entities.

    dev_total_games_shipped:
        Paid games (is_free=False) with release_date < T.
        EA games: release_date = graduation_date (abandoned = not shipped).
        MAX across dev entities.
    """
    target_meta = game_index.get(target_appid)
    if not target_meta or not target_meta["devs"]:
        return 0, 0, 0

    per_dev_ea_count = []
    per_dev_shipped  = []
    has_prior_success = 0

    for dev in target_meta["devs"]:
        siblings = [a for a in dev_to_games.get(dev, []) if a != target_appid]

        ea_count = 0
        shipped  = 0

        for sib_appid in siblings:
            sib = game_index[sib_appid]

            # ── dev_total_games_shipped ───────────────────────────────────
            if (
                not sib["is_free"]
                and sib["release_date"] is not None
                and sib["release_date"] < snapshot_t
            ):
                shipped += 1

            # ── dev_previous_ea_count ─────────────────────────────────────
            # Skip if review count unknown (conservative: pre-2022 no data)
            if sib["ea_peak_review_count"] is None:
                continue

            ea_age_ok = (
                sib["ea_age_days"] is not None
                and sib["ea_age_days"] > EA_AGE_MIN_DAYS
            )

            if (
                sib["is_ea"]
                and sib["ea_start"] is not None
                and sib["ea_start"] < snapshot_t
                and ea_age_ok
                and sib["ea_peak_review_count"] >= EA_REVIEW_THRESHOLD
            ):
                ea_count += 1

                # ── dev_has_prior_success ─────────────────────────────────
                if (
                    sib["outcome"] == "EXIT_SUCCESS"
                    and sib["graduation_date"] is not None
                    and sib["graduation_date"] < snapshot_t
                ):
                    has_prior_success = 1

        per_dev_ea_count.append(ea_count)
        per_dev_shipped.append(shipped)

    return (
        max(per_dev_ea_count, default=0),
        has_prior_success,
        max(per_dev_shipped, default=0),
    )


# ---------------------------------------------------------------------------
# Publisher candidate inspector
# ---------------------------------------------------------------------------

def get_publisher_candidates(conn: libsql.Connection, top_n: int = 20) -> None:
    freq: dict[str, int] = defaultdict(int)
    for (raw,) in conn.execute(
        "SELECT developers FROM games_v2 WHERE developers IS NOT NULL"
    ).fetchall():
        for d in parse_developers(raw):
            freq[d] += 1
    top = sorted(freq.items(), key=lambda x: -x[1])[:top_n]
    print("\nTop developer strings by corpus frequency:")
    for name, cnt in top:
        print(f"  {cnt:4d}  {name}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(dry_run: bool, publisher_cap: int, delta: bool) -> None:
    conn = get_conn()
    ensure_tables(conn)

    get_publisher_candidates(conn)

    log.info("Building game index (publisher_cap=%d)...", publisher_cap)
    game_index = build_game_index(conn, publisher_cap)

    # Merge pre2022_ea_games if available (Option A hook)
    n_patched = merge_pre2022_table(conn, game_index)
    if n_patched == 0 and not table_exists(conn, "pre2022_ea_games"):
        log.info(
            "pre2022_ea_games table not found — pre-2022 siblings with no snapshot "
            "data will be skipped conservatively. See Option A in design notes."
        )

    dev_to_games = build_dev_to_games(game_index)

    updates: list[tuple[int, int, int, int, str]] = []
    skipped = 0
    unique_appids = set()

    if delta:
        log.info("Delta run: skipping historical snapshots update.")
        query = f"""
            SELECT appid FROM ccu_availability 
            WHERE ccu_available IN ('AVAILABLE', 'UNAVAILABLE')
              AND appid IN (
                  SELECT appid FROM games_v2 
                  WHERE currently_in_ea = 1 
                     OR (currently_in_ea = 0 AND graduation_date IS NOT NULL AND graduation_date >= date('now', '-{DELTA_GRADUATION_DAYS} days'))
              )
        """
        rows = conn.execute(query).fetchall()
        unique_appids = {r[0] for r in rows}
    else:
        log.info("Loading snapshots to process...")
        query = "SELECT appid, snapshot_date FROM snapshots ORDER BY appid, snapshot_date"
        snapshots = conn.execute(query).fetchall()
        log.info("Processing %d snapshots...", len(snapshots))

        for appid, snapshot_date_str in snapshots:
            unique_appids.add(appid)
            snapshot_t = parse_date(snapshot_date_str)
            if snapshot_t is None:
                log.warning("Unparseable snapshot_date appid=%d: %s", appid, snapshot_date_str)
                skipped += 1
                continue

            prev_ea, prior_success, total_shipped = compute_dev_features(
                appid, snapshot_t, game_index, dev_to_games,
            )
            updates.append((prev_ea, prior_success, total_shipped, appid, snapshot_date_str))

        log.info("Computed %d snapshots (%d skipped).", len(updates), skipped)

    today = date.today()
    current_updates = []
    for appid in unique_appids:
        prev_ea, prior_success, total_shipped = compute_dev_features(
            appid, today, game_index, dev_to_games
        )
        current_updates.append((
            appid, prev_ea, prior_success, total_shipped, datetime.now(timezone.utc).isoformat()
        ))
    log.info("Computed current state for %d games.", len(current_updates))

    if dry_run:
        if not delta:
            log.info("Dry run — sample (first 10 historical updates):")
            for prev_ea, prior_success, total_shipped, appid, snap_date in updates[:10]:
                log.info(
                    "  appid=%-8d  T=%s  prev_ea=%-3d  prior_success=%d  shipped=%d",
                    appid, snap_date, prev_ea, prior_success, total_shipped,
                )
        log.info("Dry run — sample (first 10 current states):")
        for appid, prev_ea, prior_success, total_shipped, updated_at in current_updates[:10]:
            log.info(
                "  appid=%-8d  prev_ea=%-3d  prior_success=%d  shipped=%d",
                appid, prev_ea, prior_success, total_shipped,
            )
        log.info("Dry run complete — no writes.")
        conn.close()
        return

    if updates:
        log.info("Writing to snapshots table...")
        total_updates = len(updates)
        for i in range(0, total_updates, DB_BATCH_SIZE):
            chunk = updates[i:i + DB_BATCH_SIZE]
            conn.executemany(
                """
                UPDATE snapshots
                SET
                    dev_previous_ea_count   = ?,
                    dev_has_prior_success   = ?,
                    dev_total_games_shipped = ?
                WHERE appid = ? AND snapshot_date = ?
                """,
                chunk,
            )
            processed = i + len(chunk)
            log.info("  Updated %d/%d snapshots (%.1f%%)", processed, total_updates, processed / total_updates * 100)
        log.info("Done. %d snapshots updated.", total_updates)

    if current_updates:
        log.info("Writing to game_dev_features_current table...")
        total_current = len(current_updates)
        for i in range(0, total_current, DB_BATCH_SIZE):
            chunk = current_updates[i:i + DB_BATCH_SIZE]
            conn.executemany(
                """
                INSERT OR REPLACE INTO game_dev_features_current (
                    appid, dev_previous_ea_count, dev_has_prior_success,
                    dev_total_games_shipped, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                chunk,
            )
            processed = i + len(chunk)
            log.info("  Updated %d/%d current states (%.1f%%)", processed, total_current, processed / total_current * 100)
        log.info("Done. %d current states updated.", total_current)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Populate developer cross-game features in snapshots."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute but do not write to DB")
    parser.add_argument("--publisher-cap", type=int, default=PUBLISHER_FREQ_CAP,
                        help="Exclude dev strings on > N games (default: %(default)s)")
    parser.add_argument("--delta", action="store_true",
                        help="Delta run: only process active and recently graduated games")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run(args.dry_run, args.publisher_cap, args.delta)