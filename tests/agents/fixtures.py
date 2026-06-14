"""
tests/agents/fixtures.py
-------------------------
Shared mock inputs for agent tests, including the hollow major update case that
motivated the triangulation redesign: a type-13/14 announcement with zero
build content, posted by a game whose ML-derived l1_state still looks
"active" because an event was posted recently.
"""

from agents.states import AnnouncementInput


# ---------------------------------------------------------------------------
# Forensic Agent fixtures
# ---------------------------------------------------------------------------

HOLLOW_MAJOR_ANNOUNCEMENT: AnnouncementInput = {
    "event_type": 14,  # "Major update" — but watch the content
    "title": "Raise the Dead... Your Undead Army Now Comes",
    "body_stripped": (
        "The wait is over! Something big is coming to our world. "
        "Stay tuned for more details soon. Thank you all for your continued support "
        "on this journey with us!"
    ),
    "word_count": 28,
    "days_ago": 5,
}

# A second, older announcement — also hollow, reinforcing the pattern
HOLLOW_PRIOR_ANNOUNCEMENT: AnnouncementInput = {
    "event_type": 13,
    "title": "Big things ahead",
    "body_stripped": "We've been hard at work behind the scenes. More news soon!",
    "word_count": 11,
    "days_ago": 35,
}

HOLLOW_ANNOUNCEMENTS = [HOLLOW_MAJOR_ANNOUNCEMENT, HOLLOW_PRIOR_ANNOUNCEMENT]


# A substantive update — should score high, no mismatch
SUBSTANTIVE_ANNOUNCEMENT: AnnouncementInput = {
    "event_type": 13,
    "title": "Patch 0.7.4 — Bug Fixes & AI Improvements",
    "body_stripped": (
        "Version 0.7.4 patch notes:\n"
        "- Fixed inventory desync when dropping items during combat state transition\n"
        "- Reworked enemy patrol AI to reduce CPU overhead by ~18% on large maps\n"
        "- Added 3 new ambient sound regions to the Northern Wastes biome\n"
        "- Resolved crash on save when player has >500 items in storage\n"
        "- Crafting station now correctly unlocks recipe tier 3 after skill threshold"
    ),
    "word_count": 58,
    "days_ago": 3,
}
SUBSTANTIVE_ANNOUNCEMENTS = [SUBSTANTIVE_ANNOUNCEMENT]


# Multiple small hotfixes — momentum should read "consistent_progress"
HOTFIX_SERIES = [
    {
        "event_type": 12,
        "title": "Hotfix 0.8.2c",
        "body_stripped": "Fixed crash when alt-tabbing during the loading screen.",
        "word_count": 9,
        "days_ago": 2,
    },
    {
        "event_type": 12,
        "title": "Hotfix 0.8.2b",
        "body_stripped": "Fixed a bug where boss health bars would not reset between attempts.",
        "word_count": 12,
        "days_ago": 9,
    },
    {
        "event_type": 13,
        "title": "Patch 0.8.2",
        "body_stripped": (
            "Rebalanced tier-2 weapon damage values, fixed three quest-blocking bugs "
            "in the Eastern Marsh region, improved load times by ~12%."
        ),
        "word_count": 23,
        "days_ago": 16,
    },
]


# Empty announcement — fast path
EMPTY_ANNOUNCEMENT: AnnouncementInput = {
    "event_type": 13,
    "title": "",
    "body_stripped": "",
    "word_count": 0,
    "days_ago": 1,
}
EMPTY_ANNOUNCEMENTS = [EMPTY_ANNOUNCEMENT]


# ---------------------------------------------------------------------------
# Sentiment Auditor fixtures
# ---------------------------------------------------------------------------

# Reviews that CONFLICT with l1_state="Healthy" — abandonment language
CONFLICTING_REVIEWS_RECENT = [
    {"text": "Dev literally hasn't patched the inventory bug in 3 months. Unplayable now.", "voted_up": False},
    {"text": "Roadmap items from 2023 still aren't in the game. Feels abandoned.", "voted_up": False},
    {"text": "No response from devs on the forum in months. Sad to see this die.", "voted_up": False},
    {"text": "Performance has gotten worse with every patch, not better.", "voted_up": False},
    {"text": "Used to love this game but development clearly stopped a while ago.", "voted_up": False},
]
CONFLICTING_REVIEWS_OLDER = [
    {"text": "Really promising early build, dev was super active and responsive.", "voted_up": True},
    {"text": "Update pace is great for a solo dev, lots of communication.", "voted_up": True},
]

# Reviews that AGREE with l1_state="Healthy" — positive, active development
ALIGNED_REVIEWS_RECENT = [
    {"text": "New zone added last month is stunning, dev keeps delivering.", "voted_up": True},
    {"text": "Devs replied to my bug report within a day, great support.", "voted_up": True},
    {"text": "Solid foundation, updates are frequent and meaningful.", "voted_up": True},
    {"text": "Performance improved a lot since the last patch, very playable now.", "voted_up": True},
]
ALIGNED_REVIEWS_OLDER = [
    {"text": "Promising early build, excited to see where this goes.", "voted_up": True},
]


# ---------------------------------------------------------------------------
# Review-quality edge cases (for reviews.py scoring, if testing separately)
# ---------------------------------------------------------------------------

MEME_REVIEW_RAW = {
    "recommendationid": "1",
    "review": "uninstalled immediately, my PC caught fire and now my cat won't talk to me 10/10",
    "voted_up": False,
    "votes_up": 240,
    "votes_funny": 230,
    "timestamp_created": 1_700_000_000,
}

SOBER_NEGATIVE_REVIEW_RAW = {
    "recommendationid": "2",
    "review": (
        "After 200 hours I have to say the core gameplay loop is solid but the "
        "progression system is fundamentally broken past level 40. XP gains "
        "flatten out and the only viable strategy becomes grinding the same "
        "three nodes repeatedly. Devs acknowledged this six months ago, no fix yet."
    ),
    "voted_up": False,
    "votes_up": 180,
    "votes_funny": 2,
    "timestamp_created": 1_700_000_000,
}

WALL_OF_TEXT_REVIEW_RAW = {
    "recommendationid": "3",
    "review": "good game good game good game good game " * 30,
    "voted_up": True,
    "votes_up": 1,
    "votes_funny": 0,
    "timestamp_created": 1_700_000_000,
}
