"""
scorecard_config.py
-------------------
All tunable parameters for the L1 scorecard.
Edit this file to recalibrate — do not edit scorecard.py directly.

Increment CONFIG_VERSION on every calibration pass so history is traceable.
"""

CONFIG_VERSION = "v1.0"

# ---------------------------------------------------------------------------
# Dimension weights (must sum to 1.0)
# ---------------------------------------------------------------------------

DIMENSION_WEIGHTS = {
    "update_health":    0.25,
    "player_retention": 0.25,
    "dev_engagement":   0.20,
    "sentiment":        0.20,
    "price_market":     0.10,
}

assert abs(sum(DIMENSION_WEIGHTS.values()) - 1.0) < 1e-9, \
    "DIMENSION_WEIGHTS must sum to 1.0"

# ---------------------------------------------------------------------------
# Momentum learning rates (per dimension)
#
# Momentum is now centered at 0 (range [-1, +1]).
# Positive = accelerating, negative = decelerating.
# final_score = base_score + lr * momentum_delta   (clamped to [0, 1])
# LR controls the maximum shift momentum can apply.
# ---------------------------------------------------------------------------

MOMENTUM_LR = {
    "update_health":    0.15,
    "player_retention": 0.15,
    "dev_engagement":   0.15,
    "sentiment":        0.15,
    "price_market":     0.05,
}

# ---------------------------------------------------------------------------
# Feature normalisation scales
# All backbone features normalise to [0.0, 1.0]; 1.0 = best health signal.
# Momentum features normalise to [-1.0, +1.0]; 0.0 = neutral/steady.
#
# Types (backbone):
#   inverted_cap     : higher value = worse. score = max(0, 1 - value/cap)
#   log_cap          : log transform, capped. score = log1p(v) / log1p(cap)
#   clamp            : linear scale between min/max → [0, 1]
#   binary           : 0 or 1 pass-through (1 = good)
#   binary_inverted  : 0 or 1 inverted (0 = good, i.e. flag not set = good)
#   inverted_distance: penalises distance from anchor, asymmetric by side
#   symlog_norm      : symmetric log for absolute slopes.
#                      v = sign(x) * log1p(abs(x)) / log1p(cap)
#                      score = (v + 1) / 2  → maps to [0, 1] (0.5 = steady)
#
# Types (momentum — produce [-1, +1]):
#   centered_ratio   : ratio feature centered at 1.0.
#                      delta = clamp((value - 1.0) / half_range, -1, 1)
#                      half_range: how far from 1.0 maps to ±1.
#   centered_clamp   : already-centered feature (e.g. slope through 0).
#                      delta = clamp(value / half_range, -1, 1)
# ---------------------------------------------------------------------------

FEATURE_SCALES = {

    # ------------------------------------------------------------------
    # Update Health — backbone
    # ------------------------------------------------------------------

    "days_since_last_build_update": {
        "type": "inverted_cap",
        "cap":  365,        # 0d → 1.0, 365d+ → 0.0
    },
    "build_update_count_last_90d": {
        "type": "log_cap",
        "cap":  10,         # 0 → 0.0, 10+ → 1.0
    },
    "max_hiatus_ever_days": {
        "type": "inverted_cap",
        "cap":  365,
    },
    "hiatus_recovery_count": {
        "type": "log_cap",
        "cap":  5,
    },
    "avg_changelog_word_count": {
        # Substance of build updates. log-scale: 0 → 0, 500+ words → ~1.
        # 500 words is a detailed patch note; image-only / one-liners score low.
        "type": "log_cap",
        "cap":  500,
    },
    "ea_age_days": {
        "type": "inverted_cap",
        "cap": 730,   # 0d → 1.0 (brand new, full benefit of doubt), 730d+ → 0.0
    },

    # ------------------------------------------------------------------
    # Update Health — momentum
    # ------------------------------------------------------------------

    "changelog_word_count_trend": {
        # Ratio: mean word count last_90d / prior_90d.
        # 1.0 = neutral, >1 = posts getting more detailed, <1 = getting thinner.
        "type":       "centered_ratio",
        "half_range":  0.5,
    },
    "update_frequency_trend": {
        # Ratio: last_90d / prior_90d. 1.0 = steady, 0.0 = stopped, 2.0 = doubled.
        # return [-1,1]
        "type":       "centered_ratio",
        "half_range":  1.0,
    },

    # ------------------------------------------------------------------
    # Player Retention — backbone
    # ------------------------------------------------------------------

    "ccu_vs_peak_ratio": {
        "type": "clamp",
        "min":  0.0,
        "max":  1.0,
    },
    "ccu_floor_established": {
        "type": "binary",
    },
    "ccu_vs_launch_ratio": {
        "type": "log_cap",
        "cap":  2.0,
    },
    "days_since_ccu_above_100": {
        "type": "inverted_cap",
        "cap":  365,
    },
    "ccu_avg_last_90d": {
        "type": "log_cap",
        "cap":  1000,
    },

    # ------------------------------------------------------------------
    # Player Retention — momentum
    # ------------------------------------------------------------------

    "ccu_trend_slope_30d": {
        # ccu_mom_ratio stored here for schema compat (see feature_builder).
        # Value is last_month_avg / prior_month_avg. 1.0 = neutral.
        # ex: half_range=0.3: ratio 1.3 → +1.0, ratio 0.7 → -1.0.
        "type":       "centered_ratio",
        "half_range":  1.0,
    },
    "ccu_trend_slope_90d": {
        "type": "symlog_norm",
        "cap":  1000,   # Map ±1000 absolute CCU change/month to 1.0/0.0
                        # OWNER_MULTIPLIER_TIERS for 50 reviews is 20x,
                        # so 1000 is used as the threshold
    },

    # ------------------------------------------------------------------
    # Developer Engagement — backbone
    # ------------------------------------------------------------------

    "build_to_post_ratio": {
        # how many build are made per post released?
        # more post per build = more communication
        # lower (below 1) = better
        "type": "linear",
        "min":  0.0,
        "max":  2.0,
        "inverse": True,
    },
    "days_since_dev_post": {
        "type": "inverted_cap",
        "cap":  365,
    },
    "dev_posts_last_90d": {
        "type": "log_cap",
        "cap":  10,
    },

    # ------------------------------------------------------------------
    # Developer Engagement — momentum
    # ------------------------------------------------------------------

    "dev_engagement_trend": {
        # Ratio: dev posts last_90d / prior_90d. 1.0 = neutral.
        # half_range=0.5: ratio 1.5 → +1.0, ratio 0.5 → -1.0.
        "type":       "centered_ratio",
        "half_range":  0.5,
    },

    # ------------------------------------------------------------------
    # Sentiment — backbone
    # ------------------------------------------------------------------

    "review_score_at_T": {
        "type": "linear",
        "min":  0.0,
        "max":  1.0,
    },
    "review_score_last_90d": {
        "type": "linear",
        "min":  0.0,
        "max":  1.0,
    },
    "review_count_at_T": {
        "type": "log_cap",
        "cap":  1000
    },


    # ------------------------------------------------------------------
    # Sentiment — momentum
    # ------------------------------------------------------------------

    "review_velocity_30d": {
        # 30d velocity vs 90d baseline: ratio centered at 1.0.
        # Replaces raw count; neutral = keeping pace with historical rate.
        # half_range=0.5: 1.5x velocity → +1.0, 0.5x → -1.0.
        "type":       "centered_ratio",
        "half_range":  0.5,
    },
    "review_score_delta_30d": {
        # Score change over 30d — backbone signal for recent trajectory.
        # Centered: 0 = neutral, ±0.3 = large shift.
        "type": "clamp",
        "min":  -0.3,
        "max":   0.3,
    },

    # ------------------------------------------------------------------
    # Price & Market — backbone
    # ------------------------------------------------------------------

    "price_vs_genre_median": {
        "type":         "inverted_distance",
        "anchor":        1.0,
        "penalty_side": "below",
    },
    "early_deep_discount_flag": {
        "type": "binary_inverted",
    },
    "discount_frequency": {
        "type": "inverted_cap",
        "cap":  1.0,
    },
}

# ---------------------------------------------------------------------------
# Dimension feature composition
#
# backbone : features → base_score [0, 1]   (90d anchored, stable)
# momentum : features → momentum_delta [-1, +1]  (30d anchored, centered at 0)
#
# scorecard.py formula:
#   final_score = clamp(base_score + MOMENTUM_LR[dim] * momentum_delta, 0, 1)
#
# Weights within each group must sum to 1.0.
# ---------------------------------------------------------------------------

DIMENSION_FEATURES = {

    "update_health": {
        "backbone": {
            "days_since_last_build_update": 0.20,
            "build_update_count_last_90d":  0.30,
            "avg_changelog_word_count":     0.30,   # substance of updates
            "max_hiatus_ever_days":         0.20,
            #"ea_age_days":                  0.20,
            #"hiatus_recovery_count":        0.20,
        },
        "momentum": {
            "update_frequency_trend":       0.60,
            "changelog_word_count_trend":   0.40,
        },
    },

    "player_retention": {
        "backbone": {
            "ccu_vs_peak_ratio":        0.15,
            "ccu_floor_established":    0.20,
            "ccu_avg_last_90d":         0.35,
            "ccu_vs_launch_ratio":      0.15,
            "days_since_ccu_above_100": 0.15,
        },
        "momentum": {
            "ccu_trend_slope_30d":      0.60,
            "ccu_trend_slope_90d":      0.40,
        },
    },

    "dev_engagement": {
        "backbone": {
            "build_to_post_ratio": 0.35,
            "days_since_dev_post": 0.35,
            "dev_posts_last_90d":  0.30,
        },
        "momentum": {
            "dev_engagement_trend": 1.00,
        },
    },

    "sentiment": {
        "backbone": {
            "review_score_at_T":      0.40,
            "review_score_last_90d":  0.40,
            "review_count_at_T":      0.20,
        },
        "momentum": {
            "review_velocity_30d":    0.40,
            "review_score_delta_30d": 0.60,
        },
    },

    "price_market": {
        "backbone": {
            "price_vs_genre_median":    0.30,
            "early_deep_discount_flag": 0.30,
            "discount_frequency":       0.40,
        },
        "momentum": {},     # no meaningful 30d price momentum
    },
}

# Validate sub-weights sum to 1.0
for _dim, _groups in DIMENSION_FEATURES.items():
    for _group, _weights in _groups.items():
        if _weights:
            _total = sum(_weights.values())
            assert (
                abs(_total - 1.0) < 1e-9
            ), (
                f"DIMENSION_FEATURES['{_dim}']['{_group}'] "
                f"weights sum to {_total}, expected 1.0"
            )

# ---------------------------------------------------------------------------
# State classification thresholds
# List of (min_score_inclusive, state_label), ordered high → low.
# First match wins.
# ---------------------------------------------------------------------------

STATE_THRESHOLDS = [
    (0.55, "Healthy"),
    (0.42, "Watch"),
    (0.00, "At Risk"),
]

# ---------------------------------------------------------------------------
# Hard override rules (checked before soft thresholds)
# ---------------------------------------------------------------------------

HARD_ABANDON_BUILD_GAP_DAYS = 365
HARD_ABANDON_MIN_EA_AGE     = 90

# ---------------------------------------------------------------------------
# ML eligibility
# ---------------------------------------------------------------------------

ML_ELIGIBLE_MIN_REVIEWS = 50
