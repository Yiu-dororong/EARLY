"""
itad_client.py
--------------
IsThereAnyDeal (ITAD) API client for EARLY project. Targets API v2.9.0.

Fetches Dimension 5 — Price & Market Signals:
  - current_price_usd          price at snapshot_date
  - lowest_price_usd           all-time low to snapshot_date
  - discount_count_to_date     number of distinct discount events to snapshot_date
  - max_discount_ever_pct      highest cut% seen to snapshot_date
  - discount_frequency         discounts per month of EA lifetime
  - early_deep_discount_flag   >50% off within first 6 months (distress signal)

API v2.9 changes vs what was in original client:
  - /v02/game/lookup/  retired → 404.  Use GET /games/lookup/v1?appid=
  - /v02/game/prices/  retired → 404.  Use POST /games/prices/v3 for current
    prices + historyLow, and GET /games/history/v2 for full price log.
  - Auth header: ITAD-API-Key (Bearer still works but discouraged).
  - `nondeals` param gone from prices/v3 — all prices returned by default.
    Pass deals=true only if you want sale-only results (we don't).
  - historyLow now lives at game level in prices/v3 response (not per-deal),
    with sub-fields: .all / .y1 / .m3
  - /games/history/v2 returns a direct list of price-change records —
    no wrapping dict, no "history" key.

Games not indexed by ITAD:
  Some EA games exist on Steam but have no ITAD entry (never tracked by any
  price tracker). _resolve_game_id logs ITAD_NOT_INDEXED and returns None.
  get_price_signals returns a PriceSignals with fetch_errors=['itad_not_indexed'].
  build_snapshots.py treats this as a clean null — price features stay NULL,
  XGBoost handles natively.

Usage:
  from itad_client import ITADClient

  client = ITADClient()                      # reads ITAD_API_KEY from .env
  signals = client.get_price_signals(
      appid=1145360,
      ea_start_date="2023-01-15",
      snapshot_date="2024-06-01",            # None = today
  )

Smoke test CLI:
  python itad_client.py --appid 1145360
  python itad_client.py --appid 1145360 --ea-start 2022-03-01 --snapshot 2023-09-01
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


load_dotenv()
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ITAD_API_KEY  = os.environ.get("ITAD_API_KEY", "")
ITAD_BASE_URL = "https://api.isthereanydeal.com"

DEFAULT_COUNTRY = "US"

# Distress signal: >50% discount within first 6 months of EA
EARLY_DEEP_DISCOUNT_THRESHOLD    = 50
EARLY_DEEP_DISCOUNT_WINDOW_DAYS  = 180

# Steam shop ID on ITAD (confirmed from docs sample data)
STEAM_SHOP_ID = 61

RATE_LIMIT_DELAY = 0.5   # seconds between requests


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class PriceSignals:
    appid: int
    snapshot_date: str                        # ISO YYYY-MM-DD

    # Raw price data
    current_price_usd: float | None  = None  # price at snapshot_date
    lowest_price_usd: float | None   = None  # all-time low to snapshot_date

    # Discount history
    discount_count_to_date: int          = 0
    max_discount_ever_pct: float | None  = None
    discount_frequency: float | None     = None  # discounts per month of EA

    # Derived flags
    early_deep_discount_flag: bool   = False

    # Source tracking
    current_price_source: str | None = None  # 'history' | 'prices_v3_live'

    # Audit
    data_source: str       = "itad"
    itad_game_id: str | None = None           # ITAD UUID, stored for debug
    fetch_errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class ITADClient:

    def __init__(self, api_key: str = "", country: str = DEFAULT_COUNTRY):
        self.api_key = api_key or ITAD_API_KEY
        if not self.api_key:
            raise ValueError(
                "ITAD API key not set. "
                "Add ITAD_API_KEY=<key> to your .env or pass api_key= to ITADClient()."
            )
        self.country = country
        self.session = self._build_session()

    # -----------------------------------------------------------------------
    # Session
    # -----------------------------------------------------------------------

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            # Do NOT retry on 404 — ITAD_NOT_INDEXED is a clean signal, not an error
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        # v2.9 preferred auth header (Bearer still works but is discouraged)
        session.headers.update({
            "ITAD-API-Key": self.api_key,
            "Content-Type": "application/json",
        })
        return session

    # -----------------------------------------------------------------------
    # HTTP helpers
    # -----------------------------------------------------------------------

    def _get(self, endpoint: str, params: dict | None = None) -> Any | None:
        url = f"{ITAD_BASE_URL}{endpoint}"
        try:
            resp = self.session.get(url, params=params or {}, timeout=10)
            if resp.status_code == 404:
                log.debug("ITAD 404 for %s — not indexed", url)
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            log.warning("ITAD GET HTTP %s — %s: %s", resp.status_code, url, e)
            return None
        except requests.RequestException as e:
            log.warning("ITAD GET request error — %s: %s", url, e)
            return None

    def _post(self, 
              endpoint: str, 
              payload: Any, 
              params: dict | None = None) -> Any | None:
        url = f"{ITAD_BASE_URL}{endpoint}"
        try:
            resp = self.session.post(url, json=payload, params=params or {}, timeout=10)
            if resp.status_code == 404:
                log.debug("ITAD 404 for %s — not indexed", url)
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            log.warning("ITAD POST HTTP %s — %s: %s", resp.status_code, url, e)
            return None
        except requests.RequestException as e:
            log.warning("ITAD POST request error — %s: %s", url, e)
            return None

    # -----------------------------------------------------------------------
    # Step 1 — Resolve Steam appid → ITAD game UUID
    # -----------------------------------------------------------------------

    def _resolve_game_id(self, appid: int) -> str | None:
        """
        GET /games/lookup/v1?appid=<appid>
        Returns ITAD UUID string, or None if not indexed.

        Response shape:
          {"found": true,  "game": {"id": "<uuid>", "slug": "...", ...}}
          {"found": false, "game": null}
        """
        data = self._get("/games/lookup/v1", params={"appid": appid})
        if not data:
            log.info("appid %d: ITAD_NOT_INDEXED (no lookup response)", appid)
            return None

        if not data.get("found"):
            log.info("appid %d: ITAD_NOT_INDEXED (found=false)", appid)
            return None

        game = data.get("game") or {}
        game_id = game.get("id")
        if not game_id:
            log.warning("appid %d: ITAD lookup returned found=true but no id", appid)
            return None

        return str(game_id)

    # -----------------------------------------------------------------------
    # Step 2 — Fetch current prices + historyLow via POST /games/prices/v3
    # -----------------------------------------------------------------------

    def _fetch_prices_v3(self, game_id: str) -> dict | None:
        """
        POST /games/prices/v3
        Body: [<uuid>]
        Query: country=US

        Returns the first (and only) element of the response list for this game,
        or None on failure.

        Response shape per game:
          {
            "id": "<uuid>",
            "historyLow": {
              "all": {"amount": 4.99, "amountInt": 499, "currency": "USD"},
              "y1":  {"amount": 7.49, ...},
              "m3":  {"amount": 9.99, ...}
            },
            "deals": [
              {
                "shop": {"id": 61, "name": "Steam"},
                "price":   {"amount": 14.99, "amountInt": 1499, "currency": "USD"},
                "regular": {"amount": 14.99, ...},
                "cut": 0,
                "storeLow": {"amount": 4.99, ...},
                "timestamp": "2024-02-11T01:47:46+01:00",
                ...
              }
            ]
          }

        NOTE: nondeals parameter removed in v3 — all prices returned by default.
        """
        data = self._post(
            "/games/prices/v3",
            payload=[game_id],
            params={"country": self.country},
        )
        if not data or not isinstance(data, list):
            return None

        # Response is a list; find entry matching our game_id
        for entry in data:
            if entry.get("id") == game_id:
                return entry

        # Fallback: return first entry if id matching fails (shouldn't happen)
        return data[0] if data else None

    # -----------------------------------------------------------------------
    # Step 3 — Fetch full price change log via GET /games/history/v2
    # -----------------------------------------------------------------------

    def _fetch_price_history(self, 
                             game_id: str, 
                             since_date: str | None = None) -> list[dict]:
        """
        GET /games/history/v2?id=<uuid>&country=US&since=<iso>

        Returns a direct list of price-change records across ALL shops
        (no shops filter — see note below).

        NOTE: shops= filter is intentionally omitted. Passing an integer shop ID
        can be silently rejected by the endpoint, returning empty results. More
        importantly, discount events on Fanatical, GOG, etc. contribute to
        discount_count_to_date and max_discount_ever_pct — restricting to Steam
        only would undercount. Steam-specific current price comes from prices/v3.

        Fallback: if since_date produces an empty result (ITAD data gap — game
        tracked later than ea_start), retry without since to get whatever history
        ITAD does have.
        """
        def _call(since: str | None) -> list[dict]:
            params: dict = {
                "id":      game_id,
                "country": self.country,
            }
            if since:
                params["since"] = f"{since}T00:00:00Z"

            data = self._get("/games/history/v2", params=params)
            if not data or not isinstance(data, list):
                return []

            records = []
            for item in data:
                try:
                    ts_raw   = item.get("timestamp", "")
                    date_str = str(ts_raw)[:10]          # YYYY-MM-DD

                    deal      = item.get("deal", {})
                    price_obj = deal.get("price", {})
                    price_val = price_obj.get("amount")
                    cut_val   = deal.get("cut", 0) or 0
                    shop_name = (item.get("shop") or {}).get("name", "unknown")

                    if date_str and price_val is not None:
                        records.append({
                            "date":  date_str,
                            "price": price_val,
                            "cut":   int(cut_val),
                            "shop":  shop_name,
                        })
                except Exception:
                    continue

            records.sort(key=lambda x: x.get("date", ""))
            return records

        # First attempt: with since_date (full EA window)
        records = _call(since_date)

        # Fallback: ITAD may have started tracking after ea_start_date.
        # Retry without since to get whatever history exists.
        if not records and since_date:
            log.info(
                "game_id=%s: no history with since=%s — "
                "retrying without since (ITAD data gap)",
                game_id, since_date,
            )
            time.sleep(RATE_LIMIT_DELAY)
            records = _call(since=None)

        return records

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------

    def get_price_signals(
        self,
        appid: int,
        ea_start_date: str | None = None,
        snapshot_date: str | None = None,
    ) -> PriceSignals:
        """
        Main entry point for build_snapshots.py.

        Parameters
        ----------
        appid          : Steam appid
        ea_start_date  : ISO date — EA entry date (from histogram API).
                         Required for discount_frequency and early_deep_discount_flag.
        snapshot_date  : ISO date — look-ahead cutoff. Defaults to today.
                         All features computed strictly from data <= snapshot_date.
        """
        snap_dt = (
            datetime.fromisoformat(snapshot_date).replace(tzinfo=timezone.utc)
            if snapshot_date
            else datetime.now(timezone.utc)
        )
        snap_iso = snap_dt.date().isoformat()
        signals  = PriceSignals(appid=appid, snapshot_date=snap_iso)

        # ── Step 1: resolve ITAD UUID ─────────────────────────────────────
        game_id = self._resolve_game_id(appid)
        if not game_id:
            signals.fetch_errors.append("itad_not_indexed")
            return signals

        signals.itad_game_id = game_id
        time.sleep(RATE_LIMIT_DELAY)

        # ── Step 2: historyLow from prices/v3 ────────────────────────────
        # prices/v3 gives us the current Steam price and the all-time historyLow
        # in a single cheap call. We use historyLow.all as lowest_price_usd.
        prices_v3 = self._fetch_prices_v3(game_id)
        time.sleep(RATE_LIMIT_DELAY)

        if prices_v3:
            history_low_obj = (prices_v3.get("historyLow") or {}).get("all") or {}
            if history_low_obj.get("amount") is not None:
                signals.lowest_price_usd = _to_usd(history_low_obj["amount"])

            # Current Steam price from deals list
            for deal in prices_v3.get("deals", []):
                if (deal.get("shop") or {}).get("id") == STEAM_SHOP_ID:
                    price_amt = (deal.get("price") or {}).get("amount")
                    if price_amt is not None:
                        signals.current_price_usd = _to_usd(price_amt)
                        signals.current_price_source = "prices_v3_live"
                    break

        # ── Step 3: full price history log ───────────────────────────────
        # Use ea_start_date as `since` to capture the full EA window.
        # If no ea_start_date, default ITAD window is last 3 months — still useful.
        history = self._fetch_price_history(
            game_id,
            since_date=ea_start_date,
        )

        if not history:
            signals.fetch_errors.append("no_price_history")
            # current_price_usd and lowest_price_usd may still be set from prices/v3
            return signals

        # Filter strictly to records at or before snapshot_date (look-ahead discipline)
        history_to_snap = [r for r in history if r.get("date", "") <= snap_iso]

        if not history_to_snap:
            signals.fetch_errors.append("no_history_before_snapshot_date")
            return signals

        # ── initial_price ─────────────────────────────────────────────────
        # Earliest record in history. May slightly predate ea_start_date if
        # the game had a price before the EA flag was set — acceptable.
        signals.initial_price_usd = _to_usd(history[0].get("price"))

        # ── current_price (prefer history to maintain look-ahead discipline) ──
        history_current = None
        if history_to_snap:
            history_current = _to_usd(history_to_snap[-1].get("price"))
        if history_current is not None:
            signals.current_price_usd = history_current
            signals.current_price_source = "history"

        # ── lowest_price (override with history scan if prices/v3 missed) ─
        if signals.lowest_price_usd is None:
            valid_prices = [
                _to_usd(r["price"]) for r in history_to_snap
                if r.get("price") is not None
            ]
            valid_prices = [p for p in valid_prices if p is not None]
            if valid_prices:
                signals.lowest_price_usd = min(valid_prices)

        # ── price_trend ───────────────────────────────────────────────────
        if (signals.initial_price_usd is not None 
            and signals.current_price_usd is not None 
            and signals.current_price_source == "history"):
            delta = signals.current_price_usd - signals.initial_price_usd
            if delta > 0.5:
                signals.price_trend = "increased"
            elif delta < -0.5:
                signals.price_trend = "decreased"
            else:
                signals.price_trend = "stable"

        # ── discount analysis ─────────────────────────────────────────────
        discounts = [r for r in history_to_snap if r.get("cut", 0) > 0]
        signals.discount_count_to_date = len(discounts)

        if discounts:
            signals.max_discount_ever_pct = float(
                max(r.get("cut", 0) for r in discounts)
            )

        # ── discount_frequency + early_deep_discount_flag ─────────────────
        if ea_start_date:
            try:
                ea_start    = datetime.fromisoformat(
                                ea_start_date).replace(tzinfo=timezone.utc)
                ea_months   = max((snap_dt - ea_start).days / 30.0, 1.0)
                signals.discount_frequency = round(
                    signals.discount_count_to_date / ea_months, 3
                )

                # Early deep discount: >50% within first 180 days of EA
                ea_start_iso        = ea_start.date().isoformat()
                early_window_end    = (ea_start + timedelta(
                                                    days=EARLY_DEEP_DISCOUNT_WINDOW_DAYS))
                early_window_end_iso = early_window_end.date().isoformat()

                early_deep = [
                    r for r in discounts
                    if ea_start_iso <= r.get("date", "") <= early_window_end_iso
                    and r.get("cut", 0) >= EARLY_DEEP_DISCOUNT_THRESHOLD
                ]
                signals.early_deep_discount_flag = len(early_deep) > 0

            except ValueError:
                log.warning("appid=%d — unparseable ea_start_date: %r", 
                            appid, ea_start_date)

        return signals


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _to_usd(price_val: Any) -> float | None:
    """
    Coerce ITAD price value to float USD.
    ITAD v2.9 returns `amount` as dollars (float) and `amountInt` as cents (int).
    We always read `amount` directly, so no cents conversion needed.
    Guard is kept for safety in case a stale int sneaks through.
    """
    if price_val is None:
        return None
    try:
        val = float(price_val)
        # Safety: if value looks like cents (>500 for a game price) convert down.
        # Threshold of 500 avoids misidentifying a $300 game as cents.
        if val > 500:
            val = val / 100.0
        return round(val, 2)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Smoke test CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="ITAD price signals smoke test")
    parser.add_argument("--appid",    type=int, required=True, help="Steam appid")
    parser.add_argument("--ea-start", type=str, default=None,  help="EA start date ISO")
    parser.add_argument("--snapshot", type=str, default=None,  help="Snapshot date ISO")
    args = parser.parse_args()

    client  = ITADClient()
    signals = client.get_price_signals(
        appid=args.appid,
        ea_start_date=args.ea_start,
        snapshot_date=args.snapshot,
    )

    print("\n── ITAD Price Signals ──────────────────────────────")
    print(f"  appid                    : {signals.appid}")
    print(f"  itad_game_id             : {signals.itad_game_id}")
    print(f"  snapshot_date            : {signals.snapshot_date}")
    print(f"  current_price_usd        : {signals.current_price_usd}")
    print(f"  current_price_source     : {signals.current_price_source}")
    print(f"  lowest_price_usd         : {signals.lowest_price_usd}")
    print(f"  discount_count_to_date   : {signals.discount_count_to_date}")
    print(f"  max_discount_ever_pct    : {signals.max_discount_ever_pct}")
    print(f"  discount_frequency       : {signals.discount_frequency}")
    print(f"  early_deep_discount_flag : {signals.early_deep_discount_flag}")
    if signals.fetch_errors:
        print(f"  fetch_errors             : {signals.fetch_errors}")
    print("────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()