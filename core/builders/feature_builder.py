"""
feature_builder.py — EARLY pipeline: shared feature computation module
=======================================================================

Single source of truth for all feature construction logic used by both:
  - build_snapshots.py  (training snapshot assembly)
  - inference.py        (live scoring of STAYS_ACTIVE games)

CRITICAL: Any change to feature semantics here affects both training and
inference simultaneously. This is intentional — divergence between the two
is the primary source of silent model degradation.

─────────────────────────────────────────────────────────────────────────────
WHAT THIS MODULE OWNS
─────────────────────────────────────────────────────────────────────────────
  - All pure feature computation helpers (no DB, no I/O)
  - build_features() — the canonical feature vector constructor
  - Feature-semantic constants (window sizes, thresholds, multiplier tiers)
  - SnapshotPlan dataclass (shared between builder and inference)

WHAT THIS MODULE DOES NOT OWN
─────────────────────────────────────────────────────────────────────────────
  - DB connections or queries  → build_snapshots.py / inference.py
  - Snapshot date planning     → build_snapshots.py
  - External API calls         → caller passes pre-fetched data in
  - SteamSpy reference columns → caller's responsibility (non-training)
  - ITAD client lifecycle      → caller passes itad_client handle

─────────────────────────────────────────────────────────────────────────────
NULL DISCIPLINE (design decision 16)
─────────────────────────────────────────────────────────────────────────────
Null-producing conditions must be identical between training and inference.
XGBoost's learned null routing is trained on the null patterns produced here.
Do NOT add None-coalescing or fallback values in inference.py that aren't
present in this module — that silently changes the feature distribution.

─────────────────────────────────────────────────────────────────────────────
GENRE ENCODING (design decision 24)
─────────────────────────────────────────────────────────────────────────────
Genre one-hot encoding is NOT handled here. It requires a fitted encoder
(or a serialized column list) that must be applied identically at training
and inference time. The caller is responsible for:
  1. Fitting the encoder on training appids only
  2. Serializing the encoder / column list alongside the model
  3. At inference, applying reindex with fill_value=0 for unknown genres
     (new Steam tags → all-zero representation, logged for monitoring)

─────────────────────────────────────────────────────────────────────────────
USAGE
─────────────────────────────────────────────────────────────────────────────
  from feature_builder import build_features, SnapshotPlan

  plan = SnapshotPlan(appid=..., snapshot_date=..., ...)
  features = build_features(
      plan=plan,
      ccu_all=ccu_rows,          # list[dict] from load_ccu_history()
      review_buckets=rev_bkts,   # list[dict] from load_review_history()
      events_all=events,         # list[dict] from load_event_history()
                                 # each row must include 'announcement_body' (str|None)
                                 # for avg_changelog_word_count to be non-null
      ccu_available=avail_str,   # 'AVAILABLE' | 'UNAVAILABLE' | 'UNKNOWN'
      initial_price_usd=price,   # float | None
      ea_start=ea_start_date,    # date
      ea_end=ea_end_date,        # date | None
      outcome=outcome_str,       # 'EXIT_SUCCESS' | ... | 'STAYS_ACTIVE'
      today=date.today(),
      itad_client=itad,          # ITADClient | None
  )
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from statistics import median
from typing import Any


log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature-semantic constants
# These values are baked into the trained model — do not change without
# retraining. Bump the model version if you change any of these.
# ---------------------------------------------------------------------------

# Feature windows (days)
WINDOWS = [30, 90, 180]

# CCU regime thresholds
CCU_LOW_REGIME_THRESHOLD = 10   # ccu_low_regime=1 if avg_last_30d < this
CCU_FLOOR_THRESHOLD = 50        # ccu_floor_established: avg < 50 for 3+ months

# Label window
LABEL_WINDOW_DAYS = 365         # label_date = snapshot_date + 365d

# Owner estimate: GameDiscoverCo 2022 tiered multiplier + EA penalty
# Multiplier scales UP with game size — small niche games review at ~1/20,
# viral hits at ~1/60. EA penalty +25%: players defer reviews until 1.0.
EA_REVIEW_PENALTY = 1.25
OWNER_MULTIPLIER_TIERS: list[tuple[int | None, int]] = [
    (200,   20),   # review_count < 200  → 20x
    (1000,  30),   # review_count < 1000 → 30x
    (5000,  40),   # review_count < 5000 → 40x
    (None,  55),   # review_count >= 5000 → 55x
]

# ml_eligible gate: games below this review count skip XGBoost (Layer 1 only)
ML_ELIGIBLE_MIN_REVIEWS = 50


# ---------------------------------------------------------------------------
# SnapshotPlan — shared dataclass for builder and inference
# ---------------------------------------------------------------------------

@dataclass
class SnapshotPlan:
    appid: int
    snapshot_date: date
    snapshot_type: str          # 'percentile' | 'graduation_window' | 'live'
    percentile_label: str | None
    ea_age_days: int


# ---------------------------------------------------------------------------
# Pure computation helpers
# All functions below are stateless and free of I/O.
# ---------------------------------------------------------------------------

def reviews_at_date(
    buckets: list[dict],
    target: date,
) -> tuple[int, int]:
    """
    Reconstruct cumulative (positive, negative) review counts at target date
    using linear interpolation for the partial bucket straddling target.

    Buckets must be sorted by bucket_start ascending.
    Each bucket: {start: str, end: str, positive: int, negative: int}

    Returns (positive, negative).
    """
    pos = 0
    neg = 0
    target_iso = target.isoformat()

    for bucket in buckets:
        b_start = bucket["start"]
        b_end   = bucket["end"]

        if b_end <= target_iso:
            # Fully elapsed bucket — count everything
            pos += bucket["positive"]
            neg += bucket["negative"]
        elif b_start <= target_iso < b_end:
            # Partial bucket — linear interpolation by elapsed days
            try:
                start_dt  = date.fromisoformat(b_start)
                end_dt    = date.fromisoformat(b_end)
                total_days = (end_dt - start_dt).days
                if total_days > 0:
                    elapsed = (target - start_dt).days
                    frac    = elapsed / total_days
                    pos    += round(bucket["positive"] * frac)
                    neg    += round(bucket["negative"] * frac)
            except (ValueError, ZeroDivisionError):
                pass
            break  # nothing after this contributes

    return pos, neg


def reviews_in_window(
    buckets: list[dict],
    window_start: date,
    window_end: date,
) -> tuple[int, int]:
    """
    Reviews (positive, negative) between window_start and window_end (exclusive).
    Computed as: cumulative_at(window_end) - cumulative_at(window_start).
    """
    pos_end,   neg_end   = reviews_at_date(buckets, window_end)
    pos_start, neg_start = reviews_at_date(buckets, window_start)
    return max(0, pos_end - pos_start), max(0, neg_end - neg_start)


def ccu_rows_to_date(ccu: list[dict], snap_date: date) -> list[dict]:
    """
    Filter CCU rows to those with month <= snapshot_date.
    Each row: {month: str (YYYY-MM-DD), avg: float, peak: float}
    """
    snap_iso = snap_date.isoformat()
    return [r for r in ccu if r["month"] <= snap_iso]


def avg_ccu_in_window(
    ccu_rows: list[dict],
    window_start: date,
    window_end: date,
) -> float | None:
    """
    Average CCU over months whose start date falls within [window_start, window_end).
    Returns None if no rows in window.
    """
    ws = window_start.isoformat()
    we = window_end.isoformat()
    rows = [r for r in ccu_rows if ws <= r["month"] < we]
    if not rows:
        return None
    vals = [r["avg"] for r in rows if r["avg"] is not None]
    return sum(vals) / len(vals) if vals else None


def linear_slope(values: list[float]) -> float | None:
    """
    Least-squares slope of a 1D series (x = index, y = value).
    Returns None if fewer than 2 points.
    """
    n = len(values)
    if n < 2:
        return None
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den else None


# ccu_trend_slope_30d: despite the name, stores month-over-month ratio
# (current_month_avg / prior_month_avg). Slope was 99.8% null due to
# monthly CCU granularity. Name preserved for schema compatibility.
def ccu_mom_ratio(ccu_rows: list[dict], snap_date: date) -> float | None:
    """
    Month-over-month CCU ratio: current_month_avg / prior_month_avg.
    Returns None if either month has no data or prior_month_avg is zero.
    """
    rows_to_snap = ccu_rows_to_date(ccu_rows, snap_date)
    if len(rows_to_snap) < 2:
        return None

    current_avg = rows_to_snap[-1]["avg"]
    prior_avg   = rows_to_snap[-2]["avg"]

    if current_avg is None or prior_avg is None:
        return None
    if prior_avg == 0:
        return None

    return round(current_avg / prior_avg, 4)


def std_dev(values: list[float]) -> float | None:
    """Sample standard deviation. Returns None if fewer than 2 values."""
    if len(values) < 2:
        return None
    mean     = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def compute_build_gaps(build_dates: list[date]) -> list[int]:
    """Days between consecutive build events. Skips zero-day duplicates."""
    gaps = []
    for i in range(1, len(build_dates)):
        gap = (build_dates[i] - build_dates[i - 1]).days
        if gap > 0:
            gaps.append(gap)
    return gaps


def compute_allowable_build_gap(historical_gaps: list[int]) -> int:
    """
    Developer-relative allowable build gap (design decision 2).
    Floor of 365d; requires MIN_EVENTS gaps to be meaningful.
    """
    MIN_EVENTS = 5
    FLOOR      = 365
    TOLERANCE  = 1.5
    if len(historical_gaps) < MIN_EVENTS:
        return FLOOR
    return max(FLOOR, int(median(historical_gaps) * TOLERANCE))


def estimate_owners(review_count: int) -> int:
    """
    Tiered owner estimate from review count (GameDiscoverCo 2022).
    Multiplier scales UP with size. EA penalty: +25%.
    """
    multiplier = OWNER_MULTIPLIER_TIERS[-1][1]  # default top tier
    for threshold, mult in OWNER_MULTIPLIER_TIERS:
        if threshold is None:
            multiplier = mult
        elif review_count < threshold:
            multiplier = mult
            break
    return int(review_count * multiplier * EA_REVIEW_PENALTY)


def compute_changelog_features(
    build_events: list[dict],
    snap_date: date,
    w90_start: date,
) -> tuple[float | None, float | None]:
    """
    Compute avg_changelog_word_count and changelog_word_count_trend.

    Reads pre-computed word_count from event_history rows (via build_events).
    Only includes build events (type in {12,13,14}) with word_count > 0.
    Image-only / empty posts (word_count == 0 or None) are excluded from
    the mean so they don't drag down a developer's avg.

    avg_changelog_word_count
        Mean word_count across ALL qualifying build events up to snap_date.
        None if no events have word_count > 0.

    changelog_word_count_trend
        centered_ratio: mean_last_90d / mean_prior_90d.
        Follows the same null discipline as update_frequency_trend:
        - None  if prior window has no qualifying events (ratio undefined)
        - 0.0   is live: last window is empty, prior is not (going silent)
    """
    snap_iso        = snap_date.isoformat()
    prior_90_start  = w90_start - timedelta(days=90)

    all_wc, last_wc, prior_wc = [], [], []

    for e in build_events:
        if e.get("date", "") >= snap_iso:
            continue
        wc = e.get("word_count") or 0
        if wc <= 0:
            continue
        all_wc.append(wc)
        ev_date_str = e.get("date", "")[:10]
        try:
            ev_date = date.fromisoformat(ev_date_str)
        except ValueError:
            continue
        if ev_date >= w90_start:
            last_wc.append(wc)
        elif ev_date >= prior_90_start:
            prior_wc.append(wc)

    avg = round(sum(all_wc) / len(all_wc), 2) if all_wc else None

    if not prior_wc:
        trend = None
    else:
        mean_last  = sum(last_wc)  / len(last_wc)  if last_wc  else 0.0
        mean_prior = sum(prior_wc) / len(prior_wc)
        trend = round(mean_last / mean_prior, 4)

    return avg, trend


def compute_snapshot_percentiles(lower: float, upper: float, n: int) -> list[float]:
    """
    Divide [lower, upper] into n evenly-spaced percentile points.
    n >= 2 guaranteed by caller.
    """
    return [
        round(lower + (upper - lower) * i / (n - 1), 4)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Core feature builder
# ---------------------------------------------------------------------------

def build_features(
    plan: SnapshotPlan,
    ccu_all: list[dict],
    review_buckets: list[dict],
    events_all: list[dict],
    ccu_available: str,
    initial_price_usd: float | None,
    ea_start: date,
    ea_end: date | None,
    outcome: str,
    today: date,
    itad_client: Any | None,
) -> dict:
    """
    Compute all features for one snapshot. Returns a flat dict matching
    the snapshots table schema. Stub columns are included as None.

    Parameters
    ----------
    plan              : SnapshotPlan with appid, snapshot_date, ea_age_days
    ccu_all           : all CCU rows for appid (unfiltered — filtered internally)
    review_buckets    : all review histogram buckets for appid
    events_all        : all event_history rows for appid
    ccu_available     : 'AVAILABLE' | 'UNAVAILABLE' | 'UNKNOWN'
    initial_price_usd : launch price from games_v2 (None if unknown)
    ea_start          : EA launch date
    ea_end            : graduation date (None if still in EA)
    outcome           : game outcome string from games_v2
    today             : caller's reference date (date.today() or injected)
    itad_client       : ITADClient instance or None (price features null if None)

    Edge cases handled
    ------------------
    - ccu_unavailable   : all CCU features → None (XGBoost null routing)
    - ccu_low_regime    : flagged; velocity stubs remain None
    - review_low_regime : ml_eligible = 0 (Layer 1 only gate)
    - short EA games    : windows larger than ea_age_days degrade gracefully
                          to available data rather than returning None
    - graduation_window : same feature logic, no special casing

    NOT handled here
    ----------------
    - SteamSpy reference columns (non-training) → caller's responsibility
    - Genre one-hot encoding → caller applies fitted encoder post-return
    - Layer 1 scorecard passthrough → stub (scorecard.py not yet built)
    - Forensic Agent scores → stub (agents/ not yet built)
    """
    snap     = plan.snapshot_date
    snap_iso = snap.isoformat()

    f: dict[str, Any] = {}

    # ── Identity ──────────────────────────────────────────────────────────────
    f["appid"]           = plan.appid
    f["snapshot_date"]   = snap_iso
    f["snapshot_type"]   = plan.snapshot_type
    f["percentile_label"] = plan.percentile_label
    f["ea_age_days"]     = plan.ea_age_days
    f["ea_age_lt_180d"]  = 1 if plan.ea_age_days < 180 else 0

    # ── Label ─────────────────────────────────────────────────────────────────
    label_date = snap + timedelta(days=LABEL_WINDOW_DAYS)
    f["outcome"]           = outcome
    f["label_date"]        = label_date.isoformat()
    f["label_is_resolved"] = 1 if label_date <= today else 0
    f["collected_at"]      = int(datetime.now(timezone.utc).timestamp())

    # ── Events: filter strictly before snapshot_date ──────────────────────────
    build_types = {12, 13, 14}
    post_types  = {28}

    events_before = [e for e in events_all if e["date"] < snap_iso]
    build_events  = [e for e in events_before if e["type"] in build_types]
    post_events   = [e for e in events_before if e["type"] in post_types]

    build_dates: list[date] = []
    for e in build_events:
        try:
            build_dates.append(date.fromisoformat(e["date"][:10]))
        except ValueError:
            pass

    post_dates: list[date] = []
    for e in post_events:
        try:
            post_dates.append(date.fromisoformat(e["date"][:10]))
        except ValueError:
            pass

    # ── Window boundaries ─────────────────────────────────────────────────────
    w30_start  = snap - timedelta(days=30)
    w90_start  = snap - timedelta(days=90)
    w180_start = snap - timedelta(days=180)
    prior_90_start = snap - timedelta(days=180)   # window before w90

    # ── Dimension 1: Update Health ────────────────────────────────────────────

    # Days since last build update
    if build_dates:
        f["days_since_last_build_update"] = (snap - build_dates[-1]).days
    else:
        # No build ever — gap equals full ea_age
        f["days_since_last_build_update"] = plan.ea_age_days

    # Build counts per window
    f["build_update_count_last_30d"]  = sum(1 for d in build_dates if d >= w30_start)
    f["build_update_count_last_90d"]  = sum(1 for d in build_dates if d >= w90_start)
    f["build_update_count_last_180d"] = sum(1 for d in build_dates if d >= w180_start)

    # Build gap statistics (lifetime to T)
    build_gaps = compute_build_gaps(build_dates)
    f["mean_days_between_build_updates"] = (
        sum(build_gaps) / len(build_gaps) if build_gaps else None
    )
    f["std_days_between_build_updates"] = std_dev(build_gaps) if build_gaps else None
    f["max_hiatus_ever_days"]           = max(build_gaps) if build_gaps else plan.ea_age_days

    # Hiatus recovery count: transitions from gap > 90d back to activity
    hiatus_recoveries = 0
    in_hiatus         = False
    for gap in build_gaps:
        if gap > 90 and not in_hiatus:
            in_hiatus = True
        elif gap <= 90 and in_hiatus:
            hiatus_recoveries += 1
            in_hiatus = False
    f["hiatus_recovery_count"] = hiatus_recoveries

    # Update frequency trend: last_90d count vs prior_90d count
    count_last_90  = f["build_update_count_last_90d"]
    count_prior_90 = sum(1 for d in build_dates if prior_90_start <= d < w90_start)
    if count_prior_90 > 0:
        f["update_frequency_trend"] = round(count_last_90 / count_prior_90, 4)
    elif count_last_90 > 0:
        f["update_frequency_trend"] = None  # prior was zero — ratio undefined
    else:
        f["update_frequency_trend"] = None

    # Allowable build gap and ratio
    allowable = compute_allowable_build_gap(build_gaps)
    f["allowable_build_gap_days"]       = allowable
    f["build_gap_vs_allowable_ratio"]   = round(
        f["days_since_last_build_update"] / allowable, 4
    )

    # Changelog substance features (build events only: types 12/13/14)
    # strip_bbcode() handles Steam BBCode markup in announcement_body.
    # None when no build events have body text (mirrors update_frequency_trend
    # null discipline — XGBoost routes these semantically).
    _avg_wc, _wc_trend = compute_changelog_features(
        build_events=build_events,
        snap_date=snap,
        w90_start=w90_start,
    )
    f["avg_changelog_word_count"]   = _avg_wc
    f["changelog_word_count_trend"] = _wc_trend

    # Stubs (Forensic Agent — not yet built)
    f["substance_score_latest"] = None
    f["fake_heartbeat_flag"]    = None

    # ── Dimension 2: Player Retention ─────────────────────────────────────────
    f["ccu_unavailable"] = 1 if ccu_available == "UNAVAILABLE" else 0

    if f["ccu_unavailable"]:
        # All CCU features null — XGBoost handles natively (design decision 17)
        _null_ccu_cols = [
            "ccu_avg_last_30d", "ccu_avg_last_90d", "ccu_avg_last_180d",
            "ccu_median_all", "ccu_at_launch_30d", "peak_ccu_alltime",
            "ccu_vs_peak_ratio", "ccu_vs_launch_ratio",
            "ccu_trend_slope_30d",
            "ccu_trend_slope_90d", "ccu_trend_slope_180d",
            "ccu_floor_established", "days_since_ccu_above_100",
            "ccu_low_regime",
            "ccu_recovery_per_update_avg", "ccu_recovery_trend",
        ]
        for col in _null_ccu_cols:
            f[col] = None
    else:
        ccu_to_snap = ccu_rows_to_date(ccu_all, snap)

        avg_30  = avg_ccu_in_window(ccu_to_snap, w30_start,  snap)
        avg_90  = avg_ccu_in_window(ccu_to_snap, w90_start,  snap)
        avg_180 = avg_ccu_in_window(ccu_to_snap, w180_start, snap)

        f["ccu_avg_last_30d"]  = round(avg_30,  2) if avg_30  is not None else None
        f["ccu_avg_last_90d"]  = round(avg_90,  2) if avg_90  is not None else None
        f["ccu_avg_last_180d"] = round(avg_180, 2) if avg_180 is not None else None

        all_avgs = [r["avg"] for r in ccu_to_snap if r["avg"] is not None]
        f["ccu_median_all"] = round(median(all_avgs), 2) if all_avgs else None

        # CCU at launch: first 30d of EA
        launch_end = ea_start + timedelta(days=30)
        f["ccu_at_launch_30d"] = avg_ccu_in_window(ccu_to_snap, ea_start, launch_end)

        # Peak CCU alltime to snap
        peaks = [r["peak"] for r in ccu_to_snap if r["peak"] is not None]
        f["peak_ccu_alltime"] = max(peaks) if peaks else None

        # Derived ratios
        f["ccu_vs_peak_ratio"] = (
            round(f["ccu_avg_last_30d"] / f["peak_ccu_alltime"], 4)
            if f["peak_ccu_alltime"] and f["ccu_avg_last_30d"]
            else None
        )
        f["ccu_vs_launch_ratio"] = (
            round(f["ccu_avg_last_30d"] / f["ccu_at_launch_30d"], 4)
            if f["ccu_at_launch_30d"] and f["ccu_avg_last_30d"]
            else None
        )

        # ccu_trend_slope_30d: despite the name, stores month-over-month ratio
        # (current_month_avg / prior_month_avg). Slope was 99.8% null due to
        # monthly CCU granularity. Name preserved for schema compatibility.
        f["ccu_trend_slope_30d"] = ccu_mom_ratio(ccu_to_snap, snap)

        # Trend slopes (90d and 180d have enough monthly points for regression)
        avgs_90  = [
            r["avg"] for r in ccu_to_snap
            if r["avg"] is not None and r["month"] >= w90_start.isoformat()
        ]
        avgs_180 = [
            r["avg"] for r in ccu_to_snap
            if r["avg"] is not None and r["month"] >= w180_start.isoformat()
        ]
        f["ccu_trend_slope_90d"]  = linear_slope(avgs_90)
        f["ccu_trend_slope_180d"] = linear_slope(avgs_180)

        # CCU floor established: avg < CCU_FLOOR_THRESHOLD for 3+ consecutive months
        floor_count   = 0
        max_floor_run = 0
        for r in ccu_to_snap:
            if r["avg"] is not None and r["avg"] < CCU_FLOOR_THRESHOLD:
                floor_count += 1
                max_floor_run = max(max_floor_run, floor_count)
            else:
                floor_count = 0
        f["ccu_floor_established"] = 1 if max_floor_run >= 3 else 0

        # Days since CCU above 100
        last_above_100 = None
        for r in reversed(ccu_to_snap):
            if r["avg"] is not None and r["avg"] >= 100:
                try:
                    last_above_100 = date.fromisoformat(r["month"])
                except ValueError:
                    pass
                break
        f["days_since_ccu_above_100"] = (
            (snap - last_above_100).days if last_above_100 else plan.ea_age_days
        )

        # Low regime flag
        is_low_regime = (
            f["ccu_avg_last_30d"] is not None
            and f["ccu_avg_last_30d"] < CCU_LOW_REGIME_THRESHOLD
        )
        f["ccu_low_regime"] = 1 if is_low_regime else 0

        # Stubs (require additional collection)
        f["ccu_recovery_per_update_avg"] = None
        f["ccu_recovery_trend"]          = None

    # ── Dimension 3: Developer Engagement ─────────────────────────────────────
    f["dev_posts_last_30d"]  = sum(1 for d in post_dates if d >= w30_start)
    f["dev_posts_last_90d"]  = sum(1 for d in post_dates if d >= w90_start)
    f["dev_posts_last_180d"] = sum(1 for d in post_dates if d >= w180_start)

    # Dev engagement trend: last_90d vs prior_90d posts
    posts_prior_90 = sum(1 for d in post_dates if prior_90_start <= d < w90_start)
    if posts_prior_90 > 0:
        f["dev_engagement_trend"] = round(f["dev_posts_last_90d"] / posts_prior_90, 4)
    elif f["dev_posts_last_90d"] > 0:
        f["dev_engagement_trend"] = None  # prior was zero — ratio undefined
    else:
        f["dev_engagement_trend"] = None

    # Days since last dev post
    f["days_since_dev_post"] = (
        (snap - post_dates[-1]).days if post_dates else plan.ea_age_days
    )

    # Build-to-post ratio (key Model 2 discriminator)
    total_builds_to_snap = len(build_dates)
    total_posts_to_snap  = len(post_dates)
    if total_posts_to_snap > 0:
        f["build_to_post_ratio"] = round(total_builds_to_snap / total_posts_to_snap, 4)
    elif total_builds_to_snap > 0:
        f["build_to_post_ratio"] = None  # posts=0 — ratio undefined, not zero
    else:
        f["build_to_post_ratio"] = None

    # Stubs (require separate developer profile collection)
    for col in [
        "dev_previous_ea_count", "dev_has_prior_success",
        "dev_total_games_shipped",
    ]:
        f[col] = None

    # ── Dimension 4: Community Sentiment ──────────────────────────────────────
    pos_T, neg_T = reviews_at_date(review_buckets, snap)
    total_T      = pos_T + neg_T

    f["review_positive_at_T"] = pos_T
    f["review_negative_at_T"] = neg_T
    f["review_count_at_T"]    = total_T
    f["review_score_at_T"]    = round(pos_T / total_T, 4) if total_T > 0 else None
    f["review_low_regime"]    = 1 if total_T < ML_ELIGIBLE_MIN_REVIEWS else 0
    f["ml_eligible"]          = 0 if total_T < ML_ELIGIBLE_MIN_REVIEWS else 1

    # Review velocity per window
    for days, key in [
        (30,  "review_velocity_30d"),
        (90,  "review_velocity_90d"),
        (180, "review_velocity_180d"),
    ]:
        ws = snap - timedelta(days=days)
        p_w, n_w = reviews_in_window(review_buckets, ws, snap)
        f[key] = p_w + n_w

    # Review velocity trend: last_90d vs prior_90d
    p_prior, n_prior = reviews_in_window(
        review_buckets,
        snap - timedelta(days=180),
        snap - timedelta(days=90),
    )
    velocity_prior = p_prior + n_prior
    velocity_last  = f["review_velocity_90d"]
    if velocity_prior > 0:
        f["review_velocity_trend"] = round(velocity_last / velocity_prior, 4)
    elif velocity_last > 0:
        f["review_velocity_trend"] = None
    else:
        f["review_velocity_trend"] = None

    # Review score in 30d, 90d and 180d windows (ratio within window only)
    for days, key_score, key_delta in [
        (30,  "review_score_last_30d",  "review_score_delta_30d"),
        (90,  "review_score_last_90d",  None),
        (180, "review_score_last_180d", None),
    ]:
        ws = snap - timedelta(days=days)
        p_w, n_w  = reviews_in_window(review_buckets, ws, snap)
        total_w   = p_w + n_w
        score_w   = round(p_w / total_w, 4) if total_w > 0 else None
        f[key_score] = score_w
        if key_delta:
            if score_w is not None and f["review_score_at_T"] is not None:
                f[key_delta] = round(f["review_score_at_T"] - score_w, 4)
            else:
                f[key_delta] = None

    # Negative review rate last 30d
    if f["review_score_last_30d"] is not None:
        f["negative_review_rate_30d"] = round(1.0 - f["review_score_last_30d"], 4)
    else:
        f["negative_review_rate_30d"] = None

    # Sentiment shock: recent neg rate minus lifetime neg rate
    lifetime_neg_rate = round(neg_T / total_T, 4) if total_T > 0 else None
    if f["negative_review_rate_30d"] is not None and lifetime_neg_rate is not None:
        f["review_sentiment_shock"] = round(
            f["negative_review_rate_30d"] - lifetime_neg_rate, 4
        )
    else:
        f["review_sentiment_shock"] = None

    # ── Dimension 5: Price & Market Signals ───────────────────────────────────
    f["initial_price_usd"] = initial_price_usd

    if itad_client is not None:
        try:
            import time
            signals = itad_client.get_price_signals(
                appid=plan.appid,
                ea_start_date=ea_start.isoformat(),
                snapshot_date=snap_iso,
            )
            if signals.current_price_source == "history":
                f["current_price_at_T"]           = signals.current_price_usd
            else:
                f["current_price_at_T"]           = None
            f["discount_count_to_date"]   = signals.discount_count_to_date
            f["max_discount_ever_pct"]    = signals.max_discount_ever_pct
            f["early_deep_discount_flag"] = 1 if signals.early_deep_discount_flag else 0
            f["discount_frequency"]       = signals.discount_frequency
            time.sleep(0.5)  # ITAD rate limit
        except Exception as e:
            log.warning("appid %d ITAD error at %s: %s", plan.appid, snap_iso, e)
            for col in [
                "current_price_at_T",
                "discount_count_to_date", "max_discount_ever_pct",
                "early_deep_discount_flag", "discount_frequency",
            ]:
                f[col] = None
    else:
        for col in [
            "current_price_at_T",
            "discount_count_to_date", "max_discount_ever_pct",
            "early_deep_discount_flag", "discount_frequency",
        ]:
            f[col] = None

    # Price trend direction
    current = f.get("current_price_at_T")
    if initial_price_usd and initial_price_usd > 0 and current is not None:
        delta_pct = (current - initial_price_usd) / initial_price_usd * 100
        if delta_pct > 5:
            f["price_trend"] = "increased"
        elif delta_pct < -5:
            f["price_trend"] = "decreased"
        else:
            f["price_trend"] = "stable"
    else:
        f["price_trend"] = None

    # ── Cross-dimension ────────────────────────────────────────────────────────
    f["owner_estimate_at_T"] = estimate_owners(total_T)

    # Stubs — genre, Layer 1, Forensic Agent
    for col in [
        "primary_genre",
        "ccu_vs_genre_weighted_median", "update_freq_vs_genre_median",
        "review_score_vs_genre_median",
        "l1_composite_score", "l1_update_health_score", "l1_player_retention_score",
        "l1_dev_engagement_score", "l1_community_sentiment_score",
        "l1_price_signals_score", "l1_state_encoded",
    ]:
        f[col] = None

    return f