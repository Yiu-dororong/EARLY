"""
scorecard_evaluate.py
---------------------
Evaluates L1 scorecard quality after a scorecard.py run.

Three evaluation layers:

    1. Distribution Analysis
       - State distribution across all snapshots
       - Composite score histogram per snapshot_pct group
       - Null feature rate per dimension
       - Flags Pareto skew symptoms

    2. Outcome Agreement
       - For each L1 state: what % of games ended EXIT_SUCCESS vs EXIT_ABANDONED?
       - Uses the FINAL snapshot per game (highest pct) for outcome matching
       - Confusion matrix: predicted state vs actual outcome
       - Prints calibration guidance if agreement is poor

    3. Dimension Health
       - Per-dimension score distributions
       - Features with high null rates (> 30%)
       - Dimensions where momentum is overriding backbone (delta > 0.3)

    4. Calibration Recommendations
       - Suggests threshold adjustments based on observed distribution
       - Suggests cap adjustments for features clustering at 0.0 or 1.0

Usage:
    python scorecard_evaluate.py [--snapshot-pct PCT] [--output-dir DIR]
"""

import argparse
import logging
import os
from dotenv import load_dotenv

load_dotenv()

import libsql
import pandas as pd

from training.scorecard_config import (
    CONFIG_VERSION,
    DIMENSION_FEATURES,
    FEATURE_SCALES,
    STATE_THRESHOLDS,
    HARD_ABANDON_BUILD_GAP_DAYS,
)

# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

DB_URL = os.getenv("TURSO_URL")
DB_AUTH = os.getenv("TURSO_AUTH_TOKEN")


def get_conn() -> libsql.Connection:
    if DB_URL and DB_AUTH:
        return libsql.connect(DB_URL, auth_token=DB_AUTH)
    return libsql.connect("early.db")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scorecard(conn: libsql.Connection) -> pd.DataFrame:
    rows = conn.execute("""
        SELECT
            sc.appid,
            sc.snapshot_date,
            sc.config_version,
            sc.l1_update_health_score,
            sc.l1_player_retention_score,
            sc.l1_dev_engagement_score,
            sc.l1_sentiment_score,
            sc.l1_price_market_score,
            sc.l1_composite_score,
            sc.l1_state,
            sc.ml_eligible,
            sc.null_feature_count,
            s.percentile_label,
            g.outcome
        FROM scorecard sc
        JOIN snapshots s
            ON sc.appid = s.appid AND sc.snapshot_date = s.snapshot_date
        JOIN games_v2 g
            ON sc.appid = g.appid
    """).fetchall()

    df = pd.DataFrame(rows, columns=[
        "appid", "snapshot_date", "config_version",
        "l1_update_health_score", "l1_player_retention_score",
        "l1_dev_engagement_score", "l1_sentiment_score",
        "l1_price_market_score", "l1_composite_score",
        "l1_state", "ml_eligible", "null_feature_count",
        "snapshot_pct", "outcome",
    ])
    log.info("Loaded %d scorecard rows.", len(df))
    return df


def load_raw_features(conn: libsql.Connection) -> pd.DataFrame:
    """Load raw feature values for null rate and distribution analysis."""
    all_features = list({
        feat
        for dim_cfg in DIMENSION_FEATURES.values()
        for group in ("backbone", "momentum")
        for feat in dim_cfg[group]
    })

    cols_sql = ", ".join(all_features)
    rows = conn.execute(
        f"SELECT appid, snapshot_date, {cols_sql} FROM snapshots"
    ).fetchall()

    return pd.DataFrame(rows, columns=["appid", "snapshot_date"] + all_features)


# ---------------------------------------------------------------------------
# 1. Distribution Analysis
# ---------------------------------------------------------------------------

def analyse_distribution(df: pd.DataFrame) -> None:
    log.info("=" * 60)
    log.info("1. STATE DISTRIBUTION  (config=%s)", CONFIG_VERSION)
    log.info("=" * 60)

    total = len(df)
    dist  = df["l1_state"].value_counts()

    for state, count in dist.items():
        pct = 100 * count / total
        bar = "█" * int(pct / 2)
        log.info("  %-20s %5d  (%5.1f%%)  %s", state, count, pct, bar)

    # Pareto skew warning
    at_risk_pct = 100 * dist.get("At Risk", 0) / total
    if at_risk_pct > 60:
        log.warning(
            "⚠ Pareto skew detected: %.1f%% of snapshots are At Risk. "
            "Consider raising STATE_THRESHOLDS or revisiting FEATURE_SCALES caps.",
            at_risk_pct,
        )

    # Hard abandon overrides
    hard_abandon_mask = (df["l1_state"] == "At Risk") & (df["l1_composite_score"] == 0.0)
    n_hard_snaps = hard_abandon_mask.sum()
    n_hard_games = df.loc[hard_abandon_mask, "appid"].nunique()
    log.info("")
    log.info("Hard abandon overrides (>= %d days without build): %d snapshots across %d games", 
             HARD_ABANDON_BUILD_GAP_DAYS, n_hard_snaps, n_hard_games)

    # Per snapshot_pct breakdown
    if "snapshot_pct" in df.columns:
        log.info("")
        log.info("State distribution by snapshot_pct:")
        pct_dist = (
            df.groupby(["snapshot_pct", "l1_state"])
            .size()
            .unstack(fill_value=0)
        )
        log.info("\n%s", pct_dist.to_string())

    # Composite score stats
    log.info("")
    log.info("Composite score statistics:")
    stats = df["l1_composite_score"].describe(percentiles=[.1, .25, .5, .75, .9])
    log.info("\n%s", stats.to_string())


# ---------------------------------------------------------------------------
# 2. Outcome Agreement
# ---------------------------------------------------------------------------

def analyse_outcome_agreement(df: pd.DataFrame) -> None:
    log.info("")
    log.info("=" * 60)
    log.info("2. OUTCOME AGREEMENT")
    log.info("=" * 60)

    # Use final snapshot per game
    final = (
        df.sort_values("snapshot_date")
        .groupby("appid")
        .last()
        .reset_index()
    )

    final = final[final["outcome"].notna() & (final["outcome"] != "STAYS_ACTIVE")]
    if final.empty:
        log.info("No graduated games (EXIT_SUCCESS / EXIT_ABANDONED) found yet.")
        return

    total_graduated = len(final)
    log.info("Graduated games: %d", total_graduated)

    # Per-state outcome breakdown
    log.info("")
    log.info("%-20s  %8s  %8s  %8s  %s",
             "L1 State", "n_games", "SUCCESS", "ABANDONED", "Agreement")
    log.info("-" * 70)

    state_order = [s for _, s in STATE_THRESHOLDS]
    calibration_warnings = []
    prev_pct_abandoned = -1.0
    prev_state = None

    for state in state_order:
        subset = final[final["l1_state"] == state]
        if subset.empty:
            continue

        n          = len(subset)
        n_success  = (subset["outcome"] == "EXIT_SUCCESS").sum()
        n_abandoned = (subset["outcome"] == "EXIT_ABANDONED").sum()
        pct_success  = 100 * n_success  / n
        pct_abandoned = 100 * n_abandoned / n

        # Agreement: Healthy/Watch should → SUCCESS, At Risk → ABANDONED
        if state in ("Healthy", "Watch"):
            agreement = pct_success
            expected  = "SUCCESS"
        else:
            agreement = pct_abandoned
            expected  = "ABANDONED"

        agreement_str = f"{agreement:.1f}% {expected}"
        log.info("  %-20s  %8d  %7.1f%%  %8.1f%%  %s",
                 state, n, pct_success, pct_abandoned, agreement_str)

        # Flag poor agreement
        if agreement < 60:
            calibration_warnings.append(
                f"  '{state}' has only {agreement:.1f}% agreement with expected "
                f"outcome '{expected}' — threshold may need adjustment."
            )
            
        if prev_state and pct_abandoned < prev_pct_abandoned:
            calibration_warnings.append(
                f"  Monotonicity violation: '{state}' has a lower abandoned rate ({pct_abandoned:.1f}%) "
                f"than '{prev_state}' ({prev_pct_abandoned:.1f}%)."
            )
            
        prev_pct_abandoned = pct_abandoned
        prev_state = state

    if calibration_warnings:
        log.warning("")
        log.warning("⚠ Calibration warnings:")
        for w in calibration_warnings:
            log.warning(w)

    # Confusion matrix style summary
    log.info("")
    log.info("Confusion summary (final snapshot, graduated games only):")
    confusion = pd.crosstab(
        final["l1_state"],
        final["outcome"],
        margins=True,
    )
    log.info("\n%s", confusion.to_string())

    # Generate Visualization (Decile Bins)
    try:
        import plotly.express as px
        import numpy as np
        
        bins = np.linspace(0, 1.0, 11)
        labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
        
        final_viz = final.copy()
        final_viz["score_bin"] = pd.cut(final_viz["l1_composite_score"], bins=bins, labels=labels, include_lowest=True)
        
        valid_outcomes = ["EXIT_SUCCESS", "EXIT_ABANDONED", "EXIT_SILENT"]
        final_viz = final_viz[final_viz["outcome"].isin(valid_outcomes)]
        
        bin_counts = final_viz.groupby(["score_bin", "outcome"], observed=False).size().reset_index(name="count")
        
        total_per_bin = bin_counts.groupby("score_bin", observed=False)["count"].transform("sum")
        bin_counts["Percentage"] = (bin_counts["count"] / total_per_bin * 100).fillna(0)
        bin_counts = bin_counts[bin_counts["Percentage"] > 0]
        
        bin_counts["Label"] = bin_counts.apply(lambda row: f"{row['Percentage']:.1f}% ({row['count']})", axis=1)
        
        fig = px.bar(
            bin_counts,
            x="score_bin",
            y="Percentage",
            color="outcome",
            title=f"Outcome Distribution by Composite Score Decile ({CONFIG_VERSION})",
            barmode="stack",
            text="Label",
            color_discrete_map={
                "EXIT_SUCCESS": "#2ca02c", 
                "EXIT_ABANDONED": "#d62728", 
                "EXIT_SILENT": "#ff7f0e"
            },
            labels={"score_bin": "L1 Composite Score", "outcome": "Outcome"}
        )
        fig.update_traces(texttemplate='%{text}', textposition='inside')
        
        os.makedirs("outputs", exist_ok=True)
        out_path = os.path.join("outputs", f"outcome_distribution_deciles_{CONFIG_VERSION}.html")
        fig.write_html(out_path)
        log.info("")
        log.info("Saved Outcome Distribution visualization to %s", out_path)
    except ImportError:
        log.warning("Plotly/numpy not installed. Skipping visualization. (Run: pip install plotly numpy)")

# ---------------------------------------------------------------------------
# 3. Dimension Health
# ---------------------------------------------------------------------------

def analyse_dimensions(sc_df: pd.DataFrame, raw_df: pd.DataFrame) -> None:
    log.info("")
    log.info("=" * 60)
    log.info("3. DIMENSION HEALTH")
    log.info("=" * 60)

    dim_score_cols = {
        "update_health":    "l1_update_health_score",
        "player_retention": "l1_player_retention_score",
        "dev_engagement":   "l1_dev_engagement_score",
        "sentiment":        "l1_sentiment_score",
        "price_market":     "l1_price_market_score",
        "composite":        "l1_composite_score",
    }

    # Per-dimension score distributions
    log.info("")
    log.info("%-25s  %6s  %6s  %6s  %6s  %6s",
             "Dimension", "mean", "p25", "p50", "p75", "null%")
    log.info("-" * 65)

    for dim, col in dim_score_cols.items():
        s     = sc_df[col].dropna()
        null_pct = 100 * sc_df[col].isna().mean()
        if s.empty:
            log.info("  %-23s  (all null)", dim)
            continue
        log.info("  %-23s  %6.3f  %6.3f  %6.3f  %6.3f  %5.1f%%",
                 dim,
                 s.mean(), s.quantile(0.25), s.median(), s.quantile(0.75),
                 null_pct)

    # Per-feature null rates
    log.info("")
    log.info("Feature null rates (> 20% flagged):")

    all_features = list({
        feat
        for dim_cfg in DIMENSION_FEATURES.values()
        for group in ("backbone", "momentum")
        for feat in dim_cfg[group]
    })

    high_null = []
    for feat in sorted(all_features):
        if feat not in raw_df.columns:
            continue
        null_rate = 100 * raw_df[feat].isna().mean()
        if null_rate > 20:
            high_null.append((feat, null_rate))
            log.warning("  %-40s  %.1f%% null", feat, null_rate)
        else:
            log.info("  %-40s  %.1f%% null", feat, null_rate)

    if not high_null:
        log.info("  No features with > 20%% null rate.")

    # Null feature count distribution
    log.info("")
    log.info("Null feature count per snapshot (p50/p90/max):")
    nc = sc_df["null_feature_count"]
    log.info("  p50=%.0f  p90=%.0f  max=%.0f", nc.median(), nc.quantile(0.9), nc.max())


# ---------------------------------------------------------------------------
# 4. Calibration Recommendations
# ---------------------------------------------------------------------------

def calibration_recommendations(sc_df: pd.DataFrame, raw_df: pd.DataFrame) -> None:
    log.info("")
    log.info("=" * 60)
    log.info("4. CALIBRATION RECOMMENDATIONS")
    log.info("=" * 60)

    recs = []

    # Check if composite scores cluster at extremes
    composite = sc_df["l1_composite_score"].dropna()
    pct_at_floor = (composite < 0.05).mean()
    pct_at_ceil  = (composite > 0.95).mean()
    if pct_at_floor > 0.10:
        recs.append(
            f"  {100*pct_at_floor:.1f}% of composites near 0.0 → "
            "consider raising FEATURE_SCALES caps to spread distribution."
        )
    if pct_at_ceil > 0.10:
        recs.append(
            f"  {100*pct_at_ceil:.1f}% of composites near 1.0 → "
            "consider lowering FEATURE_SCALES caps."
        )

    # Check per-feature floor/ceiling clustering
    all_features = [
        feat
        for dim_cfg in DIMENSION_FEATURES.values()
        for group in ("backbone", "momentum")
        for feat in dim_cfg[group]
        if feat in raw_df.columns
    ]

    for feat in all_features:
        scale = FEATURE_SCALES.get(feat)
        if not scale:
            continue
        col = raw_df[feat].dropna()
        if col.empty:
            continue

        if scale["type"] == "inverted_cap":
            cap = scale["cap"]
            pct_floor = (col >= cap).mean()
            if pct_floor > 0.30:
                recs.append(
                    f"  '{feat}': {100*pct_floor:.1f}% hit cap={cap} (score=0.0) → "
                    f"consider raising cap."
                )
        elif scale["type"] == "log_cap":
            cap = scale["cap"]
            pct_ceil = (col >= cap).mean()
            if pct_ceil > 0.30:
                recs.append(
                    f"  '{feat}': {100*pct_ceil:.1f}% hit cap={cap} (score=1.0) → "
                    f"consider raising cap."
                )

    # Threshold recommendations based on distribution quartiles
    log.info("")
    log.info("Threshold suggestion based on composite distribution:")
    q2 = composite.quantile(0.25)
    q3 = composite.quantile(0.50)
    current_thresholds_str = " / ".join(f"{t:.2f}" for t, _ in STATE_THRESHOLDS[:-1])
    log.info("  Current thresholds : %s", current_thresholds_str)
    log.info("  Suggested Q3/Q2    : %.2f / %.2f", q3, q2)
    log.info("  (Ensures ~25%% Healthy, ~25%% Watch, ~50%% At Risk)")
    log.info("  Update STATE_THRESHOLDS in scorecard_config.py if needed.")

    if recs:
        log.info("")
        log.info("Specific recommendations:")
        for r in recs:
            log.info(r)
    else:
        log.info("No specific cap adjustments flagged.")

    log.info("")
    log.info("To recalibrate: edit scorecard_config.py, bump CONFIG_VERSION, re-run scorecard.py")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(snapshot_pct: str | None) -> None:
    conn = get_conn()

    sc_df  = load_scorecard(conn)
    raw_df = load_raw_features(conn)

    if snapshot_pct:
        sc_df  = sc_df[sc_df["snapshot_pct"]  == snapshot_pct]
        raw_df = raw_df.merge(
            sc_df[["appid", "snapshot_date"]],
            on=["appid", "snapshot_date"],
        )
        log.info("Filtered to snapshot_pct='%s': %d rows", snapshot_pct, len(sc_df))

    if sc_df.empty:
        log.error("No scorecard rows found. Run scorecard.py first.")
        conn.close()
        return

    analyse_distribution(sc_df)
    analyse_outcome_agreement(sc_df)
    analyse_dimensions(sc_df, raw_df)
    calibration_recommendations(sc_df, raw_df)

    conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate L1 scorecard quality and generate calibration guidance."
    )
    parser.add_argument(
        "--snapshot-pct", type=str, default=None,
        help="Filter to a specific snapshot_pct value (e.g. 'pct_25')"
    )
    args = parser.parse_args()

    run(args.snapshot_pct)
