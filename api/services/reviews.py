"""
api/services/reviews.py
------------------------
Fetches and selects Steam reviews for the Sentiment Auditor.

Replaces the review_history aggregate-count proxy with real review text
from Steam's appreviews API, with selection logic tuned for:

  - Recency vs. helpfulness tradeoff (free-tier LLM context budget)
  - "Meme discount" — high funny+helpful negative reviews are often jokes,
    not substantive complaints; high helpful+zero funny = signal
  - CJK-aware length scoring (CJK chars carry more info per char than Latin)
  - "Great Wall of Text" guard — dedupe repeated lines/words before scoring
    length, so copy-pasted scripts/ASCII art don't inflate the length score
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

import requests


logger = logging.getLogger(__name__)

STEAM_REVIEWS_URL = "https://store.steampowered.com/appreviews/{appid}"
REQUEST_TIMEOUT   = 8

MAX_REVIEW_CHARS  = 300   # truncation cap sent to LLM
MIN_TEXT_LEN      = 20    # skip one-liners below this (raw char length)


# ---------------------------------------------------------------------------
# Steam API fetch
# ---------------------------------------------------------------------------

def _fetch_raw_reviews(
    appid: int,
    filter_type: str,       # "recent" | "all" (sorted by helpfulness)
    day_range: int | None,  # only used with filter="all" + review_type cuts
    num_per_page: int = 100,
) -> list[dict]:
    """Single page fetch from Steam's appreviews endpoint."""
    params = {
        "json": 1,
        "filter": filter_type,
        "language": "all",
        "num_per_page": num_per_page,
        "purchase_type": "steam",
    }
    if day_range is not None:
        params["day_range"] = day_range

    try:
        resp = requests.get(
            STEAM_REVIEWS_URL.format(appid=appid),
            params=params,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "EARLY-bot/1.0"},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("reviews", [])
    except Exception as e:
        logger.warning("Steam review fetch failed appid=%d filter=%s: %s", 
                       appid, filter_type, e)
        return []


# ---------------------------------------------------------------------------
# Text quality helpers
# ---------------------------------------------------------------------------

# CJK Unicode ranges (CJK Unified Ideographs, Hiragana, Katakana, Hangul)
_CJK_PATTERN = re.compile(
    r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]"
)


def _effective_length(text: str) -> int:
    """
    Length score that:
      1. Counts CJK characters at higher weight (they carry more info/char
         than Latin — a 50-char CJK review ≈ a 150-char English one)
      2. Deduplicates repeated lines/tokens before counting, so copy-pasted
         scripts or ASCII-art "Great Wall of Text" reviews don't inflate
         the score.
    """
    if not text:
        return 0

    # Dedup repeated lines (catches copy-pasted blocks repeated many times)
    lines = text.split("\n")
    unique_lines = list(dict.fromkeys(line.strip() for line in lines if line.strip()))
    deduped = " ".join(unique_lines)

    # Dedup repeated whitespace-separated tokens within the deduped text
    # (catches "ha ha ha ha ha..." / ASCII art repetition at word level)
    tokens = deduped.split()
    if tokens:
        # Count unique tokens but weight by sqrt(total) so legitimate
        # repetition ("very very good") isn't fully zeroed out
        unique_ratio = len(set(tokens)) / len(tokens)
        token_penalty = max(unique_ratio, 0.3)  # floor at 0.3 — don't over-punish
    else:
        token_penalty = 1.0

    cjk_chars   = len(_CJK_PATTERN.findall(deduped))
    other_chars = len(deduped) - cjk_chars

    # CJK chars weighted ~2.5x — rough heuristic for info density vs Latin
    weighted_len = (cjk_chars * 2.5) + other_chars

    return int(weighted_len * token_penalty)


def _smart_truncate(text: str, max_chars: int) -> str:
    """Truncate at the nearest sentence boundary under max_chars."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Try sentence-ending punctuation first
    for sep in (". ", "。", "! ", "? "):
        idx = truncated.rfind(sep)
        if idx > max_chars * 0.5:
            return truncated[: idx + 1]
    return truncated + "…"


# ---------------------------------------------------------------------------
# Selection scoring
# ---------------------------------------------------------------------------

@dataclass
class ScoredReview:
    text: str
    voted_up: bool
    score: float


def _score_review(raw: dict, now_ts: int) -> ScoredReview | None:
    text = (raw.get("review") or "").strip()

    eff_len = _effective_length(text)
    if eff_len < MIN_TEXT_LEN:
        return None  # skip near-empty / low-signal reviews

    voted_up   = bool(raw.get("voted_up", True))
    votes_up   = raw.get("votes_up", 0) or 0
    votes_fun  = raw.get("votes_funny", 0) or 0
    created_at = raw.get("timestamp_created", now_ts)

    days_old = max(0, (now_ts - created_at) / 86400)

    # ── Recency ───────────────────────────────────────────────────────────
    recency_score = max(0.0, 1 - days_old / 180)  # decays to 0 over 6 months

    # ── Helpfulness with "meme discount" ────────────────────────────────────
    # High helpful + high funny on a negative review → often a viral joke,
    # not a substantive complaint. Discount helpfulness proportionally to
    # how "funny-dominated" the votes are.
    if votes_up > 0:
        funny_ratio = votes_fun / (votes_up + votes_fun)
        # Discount only kicks in meaningfully when funny_ratio is high
        meme_discount = 1.0 - (funny_ratio ** 2)  # quadratic: mild until funny dominate
    else:
        meme_discount = 1.0

    helpfulness_score = min(votes_up / 10, 1.0) * meme_discount

    # ── Length (CJK-aware, dedup-guarded) ───────────────────────────────────
    length_score = min(eff_len / 500, 1.0)

    score = (recency_score * 0.5) + (helpfulness_score * 0.3) + (length_score * 0.2)

    return ScoredReview(
        text=_smart_truncate(text, MAX_REVIEW_CHARS),
        voted_up=voted_up,
        score=score,
    )


def _select_top(raws: list[dict], n: int, now_ts: int) -> list[dict]:
    scored = []
    seen_ids = set()

    for r in raws:
        rid = r.get("recommendationid")
        if rid in seen_ids:
            continue
        seen_ids.add(rid)

        sr = _score_review(r, now_ts)
        if sr:
            scored.append(sr)

    scored.sort(key=lambda x: -x.score)
    return [{"text": s.text, "voted_up": s.voted_up} for s in scored[:n]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_reviews_for_auditor(
    appid: int,
    n_recent: int = 25,
    n_older: int = 15,
) -> tuple[list[dict], list[dict]]:
    """
    Fetch and select reviews for the Sentiment Auditor.

    Returns:
        (recent_reviews, older_reviews)
        recent_reviews — last ~90 days, top n_recent by score
        older_reviews  — 90-180 days, top n_older by score

    Fetch strategy:
        - "recent" filter (last 90d) for recency-weighted pool
        - "all" filter (helpfulness-sorted, no day_range) for the
          older window, filtered client-side by timestamp
    """
    now_ts = int(time.time())

    # Recent pool: Steam's "recent" filter is last 30d by default for
    # day_range purposes, but returns reviews sorted by date regardless.
    # Fetch a larger pool and let scoring + day-window filtering handle it.
    recent_raw = _fetch_raw_reviews(appid, 
                                    filter_type="all", 
                                    day_range=90, 
                                    num_per_page=100)

    # Helpful pool: sorted by helpfulness, no day_range — gives us older
    # high-signal reviews that "recent" would miss.
    helpful_raw = _fetch_raw_reviews(appid, 
                                     filter_type="all", 
                                     day_range=None, 
                                     num_per_page=100)

    # Partition by age
    cutoff_90d  = now_ts - (90 * 86400)
    cutoff_180d = now_ts - (180 * 86400)

    combined = recent_raw + helpful_raw

    recent_pool = [r for r in combined if r.get("timestamp_created", 0) >= cutoff_90d]
    older_pool  = [
        r for r in combined
        if cutoff_180d <= r.get("timestamp_created", 0) < cutoff_90d
    ]

    recent_reviews = _select_top(recent_pool, n_recent, now_ts)
    older_reviews  = _select_top(older_pool, n_older, now_ts)

    logger.info(
        "appid=%d reviews: %d recent (from %d candidates), "
        "%d older (from %d candidates)",
        appid, len(recent_reviews), len(recent_pool), 
        len(older_reviews), len(older_pool),
    )

    return recent_reviews, older_reviews
