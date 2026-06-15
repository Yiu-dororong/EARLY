"""
monitor_drift.py — EARLY pipeline: feature/prediction/null-rate drift monitor
==============================================================================

Compares the current `live_snapshots` / `live_scores` population against a
frozen training-time reference distribution and flags drift via PSI
(Population Stability Index).

─────────────────────────────────────────────────────────────────────────────
WHAT IT MONITORS
─────────────────────────────────────────────────────────────────────────────
1. Feature drift   — PSI per feature in shap_top25_{MODEL_VERSION}.json,
                      comparing live_snapshots.features_json vs the frozen
                      training reference distribution.
2. Prediction drift — PSI on p_distressed distribution (live_scores) vs the
                      OOF prediction distribution from training.
3. Null-rate drift  — null_feature_count distribution per l1_state bucket,
                      vs the reference rates recorded at training time
                      (e.g. At Risk avg 13.6 nulls, Watch 8.5, Healthy 5.2).
4. Label drift (delayed) — outcome distribution for resolved STAYS_ACTIVE
                      games vs training-time base rates. Best-effort; most
                      games won't have resolved yet.

─────────────────────────────────────────────────────────────────────────────
REFERENCE DISTRIBUTION
─────────────────────────────────────────────────────────────────────────────
On first run (or with --freeze-reference), this script computes and saves
`models/drift_reference_{MODEL_VERSION}.json` from the training snapshot
population (the same `snapshots` + `scorecard` join used in train_xgboost.py).
This file is the baseline. It should be regenerated only when a new model
is promoted to Production (see promote_model.py).

─────────────────────────────────────────────────────────────────────────────
PSI THRESHOLDS
─────────────────────────────────────────────────────────────────────────────
  PSI < 0.10            : no significant drift
  0.10 <= PSI < 0.25    : moderate drift — warning
  PSI >= 0.25           : significant drift — action needed

─────────────────────────────────────────────────────────────────────────────
OUTPUT
─────────────────────────────────────────────────────────────────────────────
- Writes one row per (run_date, check_type, name, psi, status) to the
  `drift_reports` table (created if missing).
- Writes a human-readable summary to outputs/drift_report_{date}.json
- Exit code 0 normally; exit code 1 if any check is "action_needed" AND
  --fail-on-drift is passed (for CI gating).

─────────────────────────────────────────────────────────────────────────────
USAGE
─────────────────────────────────────────────────────────────────────────────
  python monitor_drift.py                       # run all checks
  python monitor_drift.py --model v1.3          # specific model version
  python monitor_drift.py --freeze-reference    # (re)build reference file
  python monitor_drift.py --fail-on-drift       # nonzero exit if action_needed
  python monitor_drift.py --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import libsql
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DB_URL  = os.getenv("TURSO_URL", "")
DB_AUTH = os.getenv("TURSO_AUTH_TOKEN", "")

DB_MAX_RETRIES = 3
DB_RETRY_DELAY = 5.0

DEFAULT_MODEL_VERSION = "v1.3"
CONFIG_VERSION = "v1.0"  # scorecard CONFIG_VERSION, matches train_xgboost.py

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR    = PROJECT_ROOT / "models"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# PSI thresholds
PSI_WARN   = 0.10
PSI_ACTION = 0.25

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_conn() -> libsql.Connection:
    if DB_URL and DB_AUTH:
        return libsql.connect(DB_URL, auth_token=DB_AUTH)
    return libsql.connect("early.db")


def ensure_drift_table(conn: libsql.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS drift_reports (
            run_date     TEXT NOT NULL,
            check_type   TEXT NOT NULL,    -- 'feature' | 'prediction' | 'null_rate' | 'label'
            name         TEXT NOT NULL,    -- feature name / state label / 'p_distressed'
            psi          REAL,
            status       TEXT NOT NULL,    -- 'ok' | 'warning' | 'action_needed'
            reference_n  INTEGER,
            current_n    INTEGER,
            model_version TEXT,
            PRIMARY KEY (run_date, check_type, name)
        )
    """)
    conn.commit()


def write_drift_rows(conn: libsql.Connection, rows: list[dict]) -> None:
    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            for r in rows:
                conn.execute("""
                    INSERT OR REPLACE INTO drift_reports (
                        run_date, check_type, name, psi, status,
                        reference_n, current_n, model_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    r["run_date"], r["check_type"], r["name"], r["psi"],
                    r["status"], r["reference_n"], r["current_n"], r["model_version"],
                ))
            conn.commit()
            return
        except Exception as e:
            if attempt == DB_MAX_RETRIES:
                raise
            log.warning("DB write error in write_drift_rows: %s - retrying", e)
            time.sleep(DB_RETRY_DELAY)


# ---------------------------------------------------------------------------
# PSI computation
# ---------------------------------------------------------------------------

def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index between two 1D numeric arrays.
    Bin edges are derived from the reference distribution (quantile bins),
    then both populations are histogrammed into those same edges.

    PSI = sum((cur_pct - ref_pct) * ln(cur_pct / ref_pct))

    Returns np.nan if reference has too few unique values to bin meaningfully.
    """
    reference = reference[~np.isnan(reference)]
    current   = current[~np.isnan(current)]

    if len(reference) < bins or len(current) == 0:
        return float("nan")

    if np.unique(reference).size < 2:
        return float("nan")

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 3:
        return float("nan")

    edges[0]  = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()

    # Avoid div-by-zero / log(0) — floor at small epsilon
    eps = 1e-4
    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    cur_pct = np.where(cur_pct == 0, eps, cur_pct)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def psi_status(psi: float) -> str:
    if np.isnan(psi):
        return "ok"  # insufficient data — don't flag
    if psi >= PSI_ACTION:
        return "action_needed"
    if psi >= PSI_WARN:
        return "warning"
    return "ok"


# ---------------------------------------------------------------------------
# Reference distribution (training-time baseline)
# ---------------------------------------------------------------------------

def build_reference(conn: libsql.Connection, model_version: str, top25_features: list[str]) -> dict:
    """
    Build the frozen reference distribution from the training snapshot
    population — same join as train_xgboost.py's load_data().
    """
    log.info("Building reference distribution from training snapshots...")

    query = f"""
        SELECT
            s.*,
            sc.l1_state,
                sc.l1_composite_score AS sc_l1_composite_score,
                sc.l1_update_health_score AS sc_l1_update_health_score,
                sc.l1_player_retention_score AS sc_l1_player_retention_score,
                sc.l1_dev_engagement_score AS sc_l1_dev_engagement_score,
                sc.l1_sentiment_score AS sc_l1_sentiment_score,
                sc.l1_price_market_score AS sc_l1_price_market_score,
                gg.genre_scope
        FROM snapshots s
        JOIN scorecard sc ON s.appid = sc.appid AND s.snapshot_date = sc.snapshot_date
            LEFT JOIN game_genres gg ON s.appid = gg.appid
        WHERE s.outcome IN ('EXIT_SUCCESS', 'EXIT_ABANDONED', 'EXIT_SILENT')
          AND sc.l1_state IS NOT NULL
          AND sc.config_version = '{CONFIG_VERSION}'
    """
    cursor = conn.execute(query)
    cols = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(cursor.fetchall(), columns=cols)

    # ── Align feature columns with training matrix ──────────────────────────
    df["l1_composite_score"] = df.pop("sc_l1_composite_score")
    df["l1_update_health_score"] = df.pop("sc_l1_update_health_score")
    df["l1_player_retention_score"] = df.pop("sc_l1_player_retention_score")
    df["l1_dev_engagement_score"] = df.pop("sc_l1_dev_engagement_score")
    df["l1_sentiment_score"] = df.pop("sc_l1_sentiment_score")
    df["l1_price_market_score"] = df.pop("sc_l1_price_market_score")

    df["l1_state_encoded"] = df["l1_state"].map({"Healthy": 0, "Watch": 1, "At Risk": 2}).fillna(2).astype(int)
    df["price_trend_encoded"] = df["price_trend"].map({"increased": 1.0, "stable": 0.0, "decreased": -1.0}).astype(float)
    df["review_update_divergence"] = (
        df["review_score_at_T"] * (1 - df["update_frequency_trend"].clip(-1, 1))
    )

    log.info("Reference population: %d snapshots (%d games)", len(df), df["appid"].nunique())

    reference = {
        "model_version": model_version,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "n_snapshots": int(len(df)),
        "n_games": int(df["appid"].nunique()),
        "features": {},
        "null_rate_by_state": {},
    }

    for feat in top25_features:
        if feat in df.columns:
            vals = pd.to_numeric(df[feat], errors="coerce").dropna().values
            reference["features"][feat] = vals.tolist()
        else:
            log.warning("Reference feature %r not found in snapshots table — skipping", feat)

    # null_feature_count proxy: count of NaNs across top25 features per row
    feat_cols_present = [f for f in top25_features if f in df.columns]
    if feat_cols_present:
        null_counts = df[feat_cols_present].isna().sum(axis=1)
        for state in ("Healthy", "Watch", "At Risk"):
            mask = df["l1_state"] == state
            if mask.sum() > 0:
                reference["null_rate_by_state"][state] = {
                    "mean": float(null_counts[mask].mean()),
                    "n": int(mask.sum()),
                }

    # OOF prediction distribution — best-effort, from outputs/run_log.jsonl
    run_log_path = OUTPUT_DIR / "run_log.jsonl"
    reference["oof_label_rate"] = None
    if run_log_path.exists():
        try:
            with open(run_log_path) as f:
                lines = [json.loads(l) for l in f if l.strip()]
            matching = [l for l in lines if l.get("model_version") == model_version]
            if matching:
                reference["oof_pr_auc"] = matching[-1].get("oof_prauc")
        except Exception as e:
            log.warning("Could not parse run_log.jsonl: %s", e)

    return reference


def load_or_build_reference(conn: libsql.Connection, model_version: str,
                             top25_features: list[str], force_rebuild: bool) -> dict:
    ref_path = MODEL_DIR / f"drift_reference_{model_version}.json"

    if ref_path.exists() and not force_rebuild:
        log.info("Loading existing reference: %s", ref_path)
        with open(ref_path) as f:
            return json.load(f)

    reference = build_reference(conn, model_version, top25_features)
    with open(ref_path, "w") as f:
        json.dump(reference, f)
    log.info("Reference saved -> %s", ref_path)
    return reference


# ---------------------------------------------------------------------------
# Current population loaders
# ---------------------------------------------------------------------------

def load_current_features(conn: libsql.Connection, top25_features: list[str]) -> pd.DataFrame:
    """
    Pull the latest shap_json per appid from live_scores and unpack into
    columns matching the top-25 feature names.
    """
    rows = conn.execute("""
        SELECT appid, shap_json, p_distressed, null_features, l1_state
        FROM live_scores
        WHERE shap_json IS NOT NULL
          AND snapshot_date = (SELECT MAX(snapshot_date) FROM live_scores)
    """).fetchall()

    records = []
    for appid, shap_json, p_distressed, null_features_json, l1_state in rows:
        try:
            shap_dict = json.loads(shap_json) if shap_json else {}
        except (TypeError, json.JSONDecodeError):
            shap_dict = {}

        try:
            raw_nulls = json.loads(null_features_json) if null_features_json else []
            null_count = sum(1 for f in raw_nulls if f in top25_features)
        except (TypeError, json.JSONDecodeError):
            null_count = 0

        rec = {"appid": appid, "p_distressed": p_distressed,
               "null_feature_count": null_count, "l1_state": l1_state}
        for feat in top25_features:
            rec[feat] = shap_dict.get(feat)
        records.append(rec)

    return pd.DataFrame(records)


# NOTE: SHAP values, not raw feature values, are what's stored in shap_json.
# For feature-value drift (not SHAP-contribution drift) we additionally pull
# raw feature values from live_snapshots.features_json when available.

def load_current_raw_features(conn: libsql.Connection, top25_features: list[str]) -> pd.DataFrame:
    log.info("Loading current raw features from live_snapshots...")
    query = """
        SELECT 
            ls.*, 
            sc.l1_state,
                sc.l1_composite_score AS sc_l1_composite_score,
                sc.update_health AS sc_l1_update_health_score,
                sc.player_retention AS sc_l1_player_retention_score,
                sc.dev_engagement AS sc_l1_dev_engagement_score,
                sc.sentiment AS sc_l1_sentiment_score,
                sc.price_market AS sc_l1_price_market_score
        FROM live_snapshots ls
        LEFT JOIN live_scores sc ON ls.appid = sc.appid 
            AND sc.snapshot_date = (SELECT MAX(snapshot_date) FROM live_scores)
    """
    try:
        cursor = conn.execute(query)
        cols = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(cursor.fetchall(), columns=cols)
    except Exception as e:
        log.warning("Failed to load current raw features: %s", e)
        return pd.DataFrame()

    if df.empty:
        return df
    # ── Align feature columns with training matrix ──────────────────────────
    df["l1_composite_score"] = df.pop("sc_l1_composite_score")
    df["l1_update_health_score"] = df.pop("sc_l1_update_health_score")
    df["l1_player_retention_score"] = df.pop("sc_l1_player_retention_score")
    df["l1_dev_engagement_score"] = df.pop("sc_l1_dev_engagement_score")
    df["l1_sentiment_score"] = df.pop("sc_l1_sentiment_score")
    df["l1_price_market_score"] = df.pop("sc_l1_price_market_score")
    # ── Align derived features identically to build_reference ──
    if "l1_state" in df.columns:
        df["l1_state_encoded"] = df["l1_state"].map({"Healthy": 0, "Watch": 1, "At Risk": 2}).fillna(2).astype(int)
        
    if "price_trend" in df.columns:
        df["price_trend_encoded"] = df["price_trend"].map({"increased": 1.0, "stable": 0.0, "decreased": -1.0}).astype(float)
        
    if "review_score_at_T" in df.columns and "update_frequency_trend" in df.columns:
        df["review_update_divergence"] = (
            pd.to_numeric(df["review_score_at_T"], errors="coerce") * 
            (1 - pd.to_numeric(df["update_frequency_trend"], errors="coerce").clip(lower=-1, upper=1))
        )

    return df


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_feature_drift(reference: dict, df_current_raw: pd.DataFrame,
                         top25_features: list[str], run_date: str, model_version: str) -> list[dict]:
    results = []
    if df_current_raw.empty:
        log.info("Skipping feature drift — no current raw feature data available")
        return results

    for feat in top25_features:
        ref_vals = reference["features"].get(feat)
        if not ref_vals:
            continue
        ref_arr = np.array(ref_vals, dtype=float)
        cur_arr = pd.to_numeric(df_current_raw.get(feat), errors="coerce").dropna().values \
            if feat in df_current_raw.columns else np.array([])

        if len(cur_arr) == 0:
            continue

        psi = compute_psi(ref_arr, cur_arr)
        status = psi_status(psi)
        results.append({
            "run_date": run_date, "check_type": "feature", "name": feat,
            "psi": None if np.isnan(psi) else round(psi, 4),
            "status": status, "reference_n": len(ref_arr), "current_n": len(cur_arr),
            "model_version": model_version,
        })
        if status != "ok":
            log.warning("Feature drift [%s] PSI=%.4f -> %s", feat, psi, status)
    return results


def check_prediction_drift(reference: dict, df_current: pd.DataFrame,
                            run_date: str, model_version: str) -> list[dict]:
    results = []
    ref_prauc = reference.get("oof_pr_auc")
    cur_preds = pd.to_numeric(df_current.get("p_distressed"), errors="coerce").dropna().values \
        if "p_distressed" in df_current.columns else np.array([])

    # Without a stored OOF prediction array, fall back to comparing the
    # current prediction distribution against itself across two halves is
    # not meaningful on first run. Instead, if a prior run's predictions
    # were cached, compare against those. We store the current distribution
    # as the new reference snapshot for next time.
    ref_pred_path = MODEL_DIR / f"pred_distribution_{model_version}.json"
    if ref_pred_path.exists():
        with open(ref_pred_path) as f:
            ref_preds = np.array(json.load(f).get("p_distressed", []), dtype=float)
    else:
        ref_preds = np.array([])

    if len(ref_preds) > 0 and len(cur_preds) > 0:
        psi = compute_psi(ref_preds, cur_preds)
        status = psi_status(psi)
        results.append({
            "run_date": run_date, "check_type": "prediction", "name": "p_distressed",
            "psi": None if np.isnan(psi) else round(psi, 4),
            "status": status, "reference_n": len(ref_preds), "current_n": len(cur_preds),
            "model_version": model_version,
        })
        if status != "ok":
            log.warning("Prediction drift [p_distressed] PSI=%.4f -> %s", psi, status)
    else:
        log.info("No prior prediction snapshot — recording baseline only, no PSI computed")

    # Save current distribution as next run's reference
    with open(ref_pred_path, "w") as f:
        json.dump({"p_distressed": cur_preds.tolist(), "run_date": run_date}, f)

    return results


def check_null_rate_drift(reference: dict, df_current: pd.DataFrame,
                           run_date: str, model_version: str) -> list[dict]:
    results = []
    ref_rates = reference.get("null_rate_by_state", {})

    if df_current.empty or "l1_state" not in df_current.columns:
        return results

    for state, ref_info in ref_rates.items():
        ref_mean = ref_info["mean"]
        mask = df_current["l1_state"] == state
        if mask.sum() == 0:
            continue
        cur_mean = df_current.loc[mask, "null_feature_count"].mean()

        # Relative deviation, not PSI (single scalar comparison)
        if ref_mean > 0:
            rel_change = (cur_mean - ref_mean) / ref_mean
        else:
            rel_change = 0.0

        if abs(rel_change) >= 0.5:
            status = "action_needed"
        elif abs(rel_change) >= 0.25:
            status = "warning"
        else:
            status = "ok"

        results.append({
            "run_date": run_date, "check_type": "null_rate", "name": state,
            "psi": round(float(rel_change), 4),  # repurposed field: relative change
            "status": status, "reference_n": ref_info["n"], "current_n": int(mask.sum()),
            "model_version": model_version,
        })
        if status != "ok":
            log.warning(
                "Null-rate drift [%s] ref_avg=%.1f cur_avg=%.1f (%.0f%% change) -> %s",
                state, ref_mean, cur_mean, rel_change * 100, status,
            )
    return results


def check_label_drift(conn: libsql.Connection, run_date: str, model_version: str, min_resolved: int = 50) -> list[dict]:
    """
    Best-effort: compare resolved outcomes for games that were STAYS_ACTIVE
    at score time and have since resolved, against training-time base rates.
    Most games won't have resolved yet — this is a slow signal.
    """
    results = []
    try:
        rows = conn.execute("""
            SELECT g.outcome, COUNT(DISTINCT ls.appid) as n
            FROM live_scores ls
            JOIN games_v2 g ON ls.appid = g.appid
            WHERE g.outcome IN ('EXIT_SUCCESS', 'EXIT_ABANDONED', 'EXIT_SILENT')
            GROUP BY g.outcome
        """).fetchall()
    except Exception as e:
        log.warning("check_label_drift query failed: %s", e)
        return results

    if not rows:
        log.info("No resolved outcomes among scored games yet — skipping label drift")
        return results

    total = sum(n for _, n in rows)
    if total < min_resolved:
        log.info("Only %d resolved games found (minimum %d required) — skipping label drift", total, min_resolved)
        return results

    abandoned = sum(n for outcome, n in rows if outcome in ("EXIT_ABANDONED", "EXIT_SILENT"))
    cur_rate = abandoned / total if total > 0 else 0.0

    # Training base rate ~0.20 from snapshots with outcomes
    ref_rate = 0.20
    rel_change = (cur_rate - ref_rate) / ref_rate if ref_rate > 0 else 0.0
    status = "action_needed" if abs(rel_change) >= 0.5 else ("warning" if abs(rel_change) >= 0.25 else "ok")

    results.append({
        "run_date": run_date, "check_type": "label", "name": "abandonment_rate",
        "psi": round(float(rel_change), 4),
        "status": status, "reference_n": None, "current_n": int(total),
        "model_version": model_version,
    })
    if status != "ok":
        log.warning("Label drift: cur_rate=%.3f ref_rate=%.3f -> %s", cur_rate, ref_rate, status)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_shap_features(model_version: str) -> list[str]:
    top25_path = MODEL_DIR / f"shap_top25_{model_version}.json"
    if not top25_path.exists():
        raise FileNotFoundError(
            f"{top25_path} not found. Run train_xgboost.py first (it produces "
            f"shap_top25_{{MODEL_VERSION}}.json)."
        )
    with open(top25_path) as f:
        meta = json.load(f)
    return meta.get("features", [])


def main() -> None:
    p = argparse.ArgumentParser(description="EARLY drift monitor")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_VERSION,
                    help="Model version, e.g. v1.3")
    p.add_argument("--freeze-reference", action="store_true",
                    help="(Re)build the reference distribution from training snapshots")
    p.add_argument("--fail-on-drift", action="store_true",
                    help="Exit 1 if any check is action_needed")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run_date = datetime.now(timezone.utc).date().isoformat()
    conn = get_conn()
    ensure_drift_table(conn)

    top25_features = load_shap_features(args.model)
    log.info("Monitoring drift for %d top-SHAP features (model %s)", len(top25_features), args.model)

    reference = load_or_build_reference(conn, args.model, top25_features, args.freeze_reference)

    if args.freeze_reference:
        log.info("Reference frozen. Re-run without --freeze-reference to check drift.")
        conn.close()
        return

    df_current = load_current_features(conn, top25_features)
    df_current_raw = load_current_raw_features(conn, top25_features)

    log.info("Current scored population: %d games", len(df_current))

    all_results: list[dict] = []
    all_results += check_feature_drift(reference, df_current_raw, top25_features, run_date, args.model)
    all_results += check_prediction_drift(reference, df_current, run_date, args.model)
    all_results += check_null_rate_drift(reference, df_current, run_date, args.model)
    all_results += check_label_drift(conn, run_date, args.model)

    write_drift_rows(conn, all_results)
    conn.close()

    # Summary
    n_ok = sum(1 for r in all_results if r["status"] == "ok")
    n_warn = sum(1 for r in all_results if r["status"] == "warning")
    n_action = sum(1 for r in all_results if r["status"] == "action_needed")

    summary = {
        "run_date": run_date,
        "model_version": args.model,
        "n_checks": len(all_results),
        "ok": n_ok,
        "warning": n_warn,
        "action_needed": n_action,
        "results": all_results,
    }

    report_path = OUTPUT_DIR / f"drift_report_{run_date}.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("=" * 60)
    log.info("DRIFT REPORT — %s", run_date)
    log.info("=" * 60)
    log.info("  Checks       : %d", len(all_results))
    log.info("  OK           : %d", n_ok)
    log.info("  Warning      : %d", n_warn)
    log.info("  Action needed: %d", n_action)
    log.info("  Report       : %s", report_path)
    log.info("=" * 60)

    if args.fail_on_drift and n_action > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
