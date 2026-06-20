"""
train_xgboost.py — EARLY Project  v1.0
========================================

Architecture:
  - All snapshots per game (GroupKFold by appid prevents leakage)
  - Temporal holdout: ea_start_year >= 2024 never seen during CV
  - Dynamic threshold from OOF PR curve (not hardcoded 0.5)
  - final_n_estimators = mean(cv_best_iterations) * 1.1
  - Lift comparison: Scorecard PR-AUC vs XGBoost PR-AUC

Target encoding:
  EXIT_SUCCESS   → 0
  EXIT_ABANDONED → 1
  EXIT_SILENT    → 1  (behaviorally abandoned)

Usage:
  python train_xgboost.py
  python train_xgboost.py --dry-run       # feature matrix only, no training
  python train_xgboost.py --no-shap       # skip SHAP (faster iteration)
  python train_xgboost.py --time-bounded-eval # time-bounded baseline evaluation
  python train_xgboost.py --tune --n-trials 50
                        # Initialize an Optuna study with your specified search space.
                        Automatically generate interactive Plotly HTML visualizations
                        for the Optimization History and Hyperparameter Importances,
                        saving them directly into your outputs/ directory!
                          --ignore-tuned
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import libsql
import numpy as np
import optuna
import optuna.visualization as vis
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from utils.mlflow_client import log_training_run, start_run


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SECTION 0 — Config & Constants
# ---------------------------------------------------------------------------

MODEL_VERSION = "v1.3"
RANDOM_SEED   = 42
N_FOLDS       = 5
TEMPORAL_HOLDOUT_YEAR = 2024
CONFIG_VERSION = "v1.0" #scorecard version

# XGBoost base params — scale_pos_weight set dynamically from class balance
XGB_PARAMS = {
    "objective":        "binary:logistic",
    "eval_metric":      "aucpr",          # PR-AUC as primary CV metric
    "learning_rate":    0.05,
    "max_depth":        6,
    "min_child_weight": 5,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "gamma":            0.1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "n_estimators":     2000,             # upper bound; early stopping controls actual
    "early_stopping_rounds": 50,
    "random_state":     RANDOM_SEED,
    "tree_method":      "hist",
    "verbosity":        0,
}

# L1 state ordinal (At Risk = highest risk = 2)
L1_STATE_MAP = {
    "Healthy":       0,
    "Watch":         1,
    "At Risk":       2,

}

# ------ Feature columns ------
# Explicit list — no SELECT *. Changing this list bumps MODEL_VERSION.

FEATURE_COLS = [
    # ── Update Health ──────────────────────────────────────────────────────
    "days_since_last_build_update",
    "build_update_count_last_30d",
    "build_update_count_last_90d",
    "build_update_count_last_180d",
    "mean_days_between_build_updates",
    "std_days_between_build_updates",
    "max_hiatus_ever_days",
    "hiatus_recovery_count",
    "update_frequency_trend",
    "build_gap_vs_allowable_ratio",
    "avg_changelog_word_count",
    "changelog_word_count_trend",

    # ── Player Retention ───────────────────────────────────────────────────
    "ccu_avg_last_30d",
    "ccu_avg_last_90d",
    "ccu_avg_last_180d",
    "ccu_median_all",
    "ccu_at_launch_30d",
    "peak_ccu_alltime",
    "ccu_vs_peak_ratio",
    "ccu_vs_launch_ratio",
    "ccu_trend_slope_30d",
    "ccu_trend_slope_90d",
    "ccu_trend_slope_180d",
    "ccu_floor_established",
    "days_since_ccu_above_100",
    "ccu_low_regime",
    #"ccu_unavailable",

    # ── Developer Engagement ───────────────────────────────────────────────
    "dev_posts_last_30d",
    "dev_posts_last_90d",
    "dev_posts_last_180d",
    "dev_engagement_trend",
    "days_since_dev_post",
    "build_to_post_ratio",
    "dev_previous_ea_count",
    "dev_has_prior_success",
    "dev_total_games_shipped",

    # ── Community Sentiment ────────────────────────────────────────────────
    "review_count_at_T",
    "review_score_at_T",
    "review_score_last_30d",
    "review_score_last_90d",
    "review_score_last_180d",
    "review_score_delta_30d",
    "review_velocity_30d",
    "review_velocity_90d",
    "review_velocity_180d",
    "review_velocity_trend",

    # ── Price & Market ─────────────────────────────────────────────────────
    "initial_price_usd",
    "current_price_at_T",
    "discount_count_to_date",
    "max_discount_ever_pct",
    "early_deep_discount_flag",
    "discount_frequency",
    "price_trend_encoded",
    "price_vs_genre_median",

    # ── Cross-dimension / Context ──────────────────────────────────────────
    "ea_age_days",
    "ea_age_lt_180d",
    "owner_estimate_at_T",
    "genre_scope",
    # "ccu_vs_genre_weighted_median",
    # "update_freq_vs_genre_median",
    # "review_score_vs_genre_median",
    "review_update_divergence",

    # ── L1 Scorecard ───────────────────────────────────────────────────────
    "l1_composite_score",
    "l1_update_health_score",
    "l1_player_retention_score",
    "l1_dev_engagement_score",
    "l1_sentiment_score",
    "l1_price_market_score",
    "l1_state_encoded",             # derived below from l1_state

    # ── Categorical (one-hot encoded inline) ───────────────────────────────
    # "primary_genre", #→ one-hot columns appended after loading
]

# Columns that exist in DB but must never enter training
EXCLUDED_COLS = {
    "dev_is_solo",
    "dev_avg_ea_duration_prior",
    "owner_estimate_current_lower",
    "owner_estimate_current_upper",
    "current_price_live_reference",
    "substance_score_latest",       # stub
    "fake_heartbeat_flag",          # stub
    "ccu_recovery_per_update_avg",  # stub
    "ccu_recovery_trend",           # stub
}

OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)
TUNED_PARAMS_PATH = OUTPUT_DIR / f"optuna_best_params_{MODEL_VERSION}.json"


# ---------------------------------------------------------------------------
# SECTION 1 — Data Loading
# ---------------------------------------------------------------------------

def get_conn():
    url  = os.getenv("TURSO_URL", "")
    auth = os.getenv("TURSO_AUTH_TOKEN", "")
    if url and auth:
        return libsql.connect(url, auth_token=auth)
    return libsql.connect("early.db")


def load_data(conn) -> pd.DataFrame:
    log.info("Loading snapshot + outcome data...")

    config_version = CONFIG_VERSION

    df = pd.read_sql(f"""
        SELECT
            s.*,
            g.ea_start_date,
            g.graduation_date,
            g.abandoned_date,
            g.outcome_date,
            strftime('%Y', g.ea_start_date) AS ea_start_year,
            sc.l1_state,
            sc.l1_composite_score AS sc_l1_composite_score,
            sc.l1_update_health_score AS sc_l1_update_health_score,
            sc.l1_player_retention_score AS sc_l1_player_retention_score,
            sc.l1_dev_engagement_score AS sc_l1_dev_engagement_score,
            sc.l1_sentiment_score AS sc_l1_sentiment_score,
            sc.l1_price_market_score AS sc_l1_price_market_score,
            gg.genre_scope
        FROM snapshots s
        JOIN games_v2 g ON s.appid = g.appid
        JOIN scorecard sc ON s.appid = sc.appid AND s.snapshot_date = sc.snapshot_date
        LEFT JOIN game_genres gg ON s.appid = gg.appid
        WHERE s.outcome IN ('EXIT_SUCCESS', 'EXIT_ABANDONED', 'EXIT_SILENT')
          AND sc.l1_state IS NOT NULL
          AND sc.config_version = '{config_version}'
    """, conn)

    log.info("Loaded %d snapshot rows across %d games",
             len(df), df["appid"].nunique())

    # Overwrite stubs from snapshots table with populated scores from scorecard
    df["l1_composite_score"] = df.pop("sc_l1_composite_score")
    df["l1_update_health_score"] = df.pop("sc_l1_update_health_score")
    df["l1_player_retention_score"] = df.pop("sc_l1_player_retention_score")
    df["l1_dev_engagement_score"] = df.pop("sc_l1_dev_engagement_score")
    df["l1_sentiment_score"] = df.pop("sc_l1_sentiment_score")
    df["l1_price_market_score"] = df.pop("sc_l1_price_market_score")

    # ── Target ──────────────────────────────────────────────────────────────
    df["label"] = df["outcome"].map({
        "EXIT_SUCCESS":   0,
        "EXIT_ABANDONED": 1,
        "EXIT_SILENT":    1,
    }).astype(int)

    # ── EA Duration Calculation ──────────────────────────────────────────────
    df["ea_end_date"] = df["graduation_date"]
    mask_not_success = df["outcome"] != "EXIT_SUCCESS"
    df.loc[mask_not_success, "ea_end_date"] = (
        df.loc[mask_not_success, "abandoned_date"].fillna(
                                            df.loc[mask_not_success, "outcome_date"])
    )
    start_dt = pd.to_datetime(df["ea_start_date"].str[:10], errors="coerce")
    end_dt = pd.to_datetime(df["ea_end_date"].str[:10], errors="coerce")
    df["ea_duration_days"] = (end_dt - start_dt).dt.days

    # ── Ordinal encodings ────────────────────────────────────────────────────
    df["l1_state_encoded"] = df["l1_state"].map(L1_STATE_MAP).fillna(2).astype(int)
    df["ea_start_year"]    = df["ea_start_year"].astype(int)

    df["price_trend_encoded"] = df["price_trend"].map({
        "increased": 1.0,
        "stable":    0.0,
        "decreased": -1.0,
    }).astype(float)

    # Positive = review score holding up while updates slow down (deceptive health)
    # Negative = review score dropping alongside update slowdown (consistent signal)
    df["review_update_divergence"] = (
        df["review_score_at_T"] * (1 - df["update_frequency_trend"].clip(-1, 1))
    )

    # Force null stubs and missing integers to float
    # to prevent XGBoost object-type errors
    for col in [
        "dev_previous_ea_count", "dev_has_prior_success",
        "ccu_vs_genre_weighted_median", "update_freq_vs_genre_median",
        "review_score_vs_genre_median",
        "genre_scope"
    ]:
        df[col] = df[col].astype(float)

    # ── Class balance ────────────────────────────────────────────────────────
    n_neg = (df["label"] == 0).sum()
    n_pos = (df["label"] == 1).sum()
    log.info("Class balance — SUCCESS(0): %d  ABANDONED(1): %d  ratio: %.2f:1",
             n_neg, n_pos, n_neg / n_pos)

    return df


def build_feature_matrix(df: pd.DataFrame, genre_encoder=None):
    """
    Returns X (DataFrame), y (Series), groups (Series), genre_encoder.
    genre_encoder is fit on first call (train), then reused for val/test.
    """
    # One-hot encode primary_genre
    if genre_encoder is None:
        genre_encoder = LabelEncoder()
        genre_encoder.fit(df["primary_genre"].fillna("Unknown"))

    genre_encoded = genre_encoder.transform(df["primary_genre"].fillna("Unknown"))
    genre_dummies = pd.get_dummies(
        pd.Series(genre_encoded, index=df.index).map(
            dict(enumerate(genre_encoder.classes_))
        ),
        prefix="genre",
        dtype=float
    )

    # Build feature matrix from FEATURE_COLS (skip missing columns gracefully)
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        log.warning("Missing feature columns (will be skipped): %s", missing)

    X = pd.concat([df[available].copy(), genre_dummies], axis=1)

    # XGBoost handles NaN natively — no imputation needed
    y      = df["label"]
    groups = df["appid"]

    return X, y, groups, genre_encoder


# ---------------------------------------------------------------------------
# SECTION 2 — Temporal Split
# ---------------------------------------------------------------------------

def temporal_split(df: pd.DataFrame):
    test_mask     = df["ea_start_year"] >= TEMPORAL_HOLDOUT_YEAR
    train_val_mask = ~test_mask

    assert df.loc[test_mask, "appid"].nunique() + \
           df.loc[train_val_mask, "appid"].nunique() == df["appid"].nunique(), \
           "appid overlap between train_val and test — check split logic"

    log.info("Temporal split — train_val: %d snapshots (%d games) | "
             "holdout: %d snapshots (%d games)",
             train_val_mask.sum(), df.loc[train_val_mask, "appid"].nunique(),
             test_mask.sum(),      df.loc[test_mask, "appid"].nunique())

    if df.loc[test_mask, "appid"].nunique() < 100:
        log.warning("Holdout pool < 100 games — consider shifting cutoff to 2023")

    return df[train_val_mask].copy(), df[test_mask].copy()


# ---------------------------------------------------------------------------
# SECTION 3 — GroupKFold CV
# ---------------------------------------------------------------------------

def run_cv(df_train_val: pd.DataFrame, scale_pos_weight: float):
    log.info("Building feature matrix for train_val...")
    X_tv, y_tv, groups_tv, genre_enc = build_feature_matrix(df_train_val)

    feature_names = list(X_tv.columns)
    log.info("Feature count: %d", len(feature_names))

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_preds  = np.zeros(len(X_tv))
    fold_aucs  = []
    fold_prauc = []
    best_iters = []

    params = {**XGB_PARAMS, "scale_pos_weight": scale_pos_weight}

    for fold, (tr_idx, val_idx) in enumerate(
        gkf.split(X_tv, y_tv, groups=groups_tv), 1
    ):
        X_tr, X_val = X_tv.iloc[tr_idx], X_tv.iloc[val_idx]
        y_tr, y_val = y_tv.iloc[tr_idx], y_tv.iloc[val_idx]

        # ── ADD: fold diagnostic header ──────────────────────────────────
        val_games = groups_tv.iloc[val_idx].nunique()
        val_pos_rate = y_val.mean()
        tr_pos_rate  = y_tr.mean()
        log.info("  Fold %d — val_games: %d  val_pos: %.3f  tr_pos: %.3f",
                fold, val_games, val_pos_rate, tr_pos_rate)
        # ─────────────────────────────────────────────────────────────────

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        # ─────────────────────────────────────────────────────────────────

        preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = preds

        auc   = roc_auc_score(y_val, preds)
        prauc = average_precision_score(y_val, preds)
        best  = model.best_iteration

        fold_aucs.append(auc)
        fold_prauc.append(prauc)
        best_iters.append(best)

        log.info("  Fold %d — AUC-ROC: %.4f  PR-AUC: %.4f  best_iter: %d",
                 fold, auc, prauc, best)

    log.info("CV complete — AUC-ROC: %.4f ± %.4f | PR-AUC: %.4f ± %.4f",
             np.mean(fold_aucs),  np.std(fold_aucs),
             np.mean(fold_prauc), np.std(fold_prauc))

    return oof_preds, y_tv, feature_names, genre_enc, best_iters


# ---------------------------------------------------------------------------
# SECTION 4 — Final Model
# ---------------------------------------------------------------------------

def train_final_model(
    df_train_val: pd.DataFrame,
    best_iters: list[int],
    scale_pos_weight: float,
    genre_enc,
    feature_names: list[str],
    threshold: float,
):
    # n_estimators: mean of CV folds + 10% data-volume buffer
    # final_n = int(np.mean(best_iters) * 1.1)
    final_n = int(np.percentile(best_iters, 75) * 1.1)
    log.info("CV best_iterations: %s → final_n_estimators: %d",
             best_iters, final_n)

    X_tv, y_tv, _, _ = build_feature_matrix(df_train_val, genre_enc)

    # Align columns to CV feature order (safety check)
    X_tv = X_tv.reindex(columns=feature_names, fill_value=np.nan)

    params = {
        **XGB_PARAMS,
        "scale_pos_weight":    scale_pos_weight,
        "n_estimators":        final_n,
        "early_stopping_rounds": None,   # no early stopping on final model
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_tv, y_tv, verbose=False)

    model_path   = OUTPUT_DIR / f"xgb_{MODEL_VERSION}.json"
    feature_path = OUTPUT_DIR / f"xgb_{MODEL_VERSION}_features.json"

    genre_cols = [c for c in feature_names if c.startswith("genre_")]
    base_features = [c for c in feature_names if not c.startswith("genre_")]

    model.save_model(str(model_path))
    with open(feature_path, "w") as f:
        json.dump({
            "features": base_features,
            "genre_columns": genre_cols,
            "threshold": threshold,
        }, f, indent=2)

    log.info("Model saved → %s", model_path)
    log.info("Features saved → %s", feature_path)

    return model


# ---------------------------------------------------------------------------
# SECTION 5 — Evaluation
# ---------------------------------------------------------------------------

def analyze_scorecard_thresholds(df, thresholds=[0.60, 0.52, 0.44, 0.32]):
    df = df.copy()
    # pd.cut requires strictly increasing bins
    bins = [0.0] + thresholds[::-1] + [1.0]
    # Labels must match the ascending bins (lowest scores = Abandoned)
    labels = ["Abandoned", "High Risk", "Stalled", "Slow but Honest", "Healthy"]

    df['state'] = pd.cut(df['composite_score'],
                        bins=bins,
                        labels=labels,
                        include_lowest=True)

    summary = df.groupby('state', observed=False).agg(
        count=('composite_score', 'size'),
        abandonment_rate=('is_distressed', 'mean'),
        avg_composite=('composite_score', 'mean')
    ).round(4)

    # Reorder index to display Healthy at the top
    summary = summary.reindex(["Healthy", "Slow but Honest",
                               "Stalled", "High Risk", "Abandoned"])

    log.info("\n%s", summary.to_string())
    return summary


def find_optimal_threshold(y_true: np.ndarray, oof_probs: np.ndarray) -> float:
    """F1-maximising threshold from OOF predictions."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, oof_probs)
    # precision_recall_curve returns one more value than thresholds — align
    f1 = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
        0.0,
    )
    best_idx = np.argmax(f1)
    threshold = float(thresholds[best_idx])
    log.info("Optimal OOF threshold: %.4f  (F1=%.4f at that threshold)",
             threshold, f1[best_idx])
    return threshold


def evaluate(
    model,
    df_test: pd.DataFrame,
    oof_preds: np.ndarray,
    y_train_val: pd.Series,
    df_train_val: pd.DataFrame,
    genre_enc,
    feature_names: list[str],
    threshold: float,
    time_bounded_eval: bool = False,
):
    log.info("=" * 60)
    log.info("EVALUATION")
    log.info("=" * 60)

    # ── OOF metrics (train_val) ─────────────────────────────────────────────
    oof_auc   = roc_auc_score(y_train_val, oof_preds)
    oof_prauc = average_precision_score(y_train_val, oof_preds)
    log.info("OOF (train_val) — AUC-ROC: %.4f  PR-AUC: %.4f", oof_auc, oof_prauc)

    log.info("OOF Scorecard Threshold Analysis:")
    df_oof_scorecard = pd.DataFrame({
        'composite_score': df_train_val['l1_composite_score'].values,
        'is_distressed': y_train_val.values
    })
    analyze_scorecard_thresholds(df_oof_scorecard)

    # ── Holdout metrics (temporal test set) ────────────────────────────────
    X_test, y_test, _, _ = build_feature_matrix(df_test, genre_enc)
    X_test = X_test.reindex(columns=feature_names, fill_value=np.nan)

    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= threshold).astype(int)

    test_auc   = roc_auc_score(y_test, test_probs)
    test_prauc = average_precision_score(y_test, test_probs)
    log.info("Holdout (2024+)  — AUC-ROC: %.4f  PR-AUC: %.4f", test_auc, test_prauc)

    log.info("Holdout Scorecard Threshold Analysis:")
    df_holdout_scorecard = pd.DataFrame({
        'composite_score': df_test['l1_composite_score'].values,
        'is_distressed': y_test.values
    })
    analyze_scorecard_thresholds(df_holdout_scorecard)

    drift = oof_auc - test_auc
    if drift > 0.05:
        log.warning("Temporal drift detected — OOF vs holdout AUC gap: %.4f", drift)

    # ── Confusion matrix at dynamic threshold ──────────────────────────────
    cm = confusion_matrix(y_test, test_preds)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)

    log.info("Confusion matrix (threshold=%.4f):", threshold)
    log.info("  TP: %d  FP: %d  TN: %d  FN: %d", tp, fp, tn, fn)
    log.info("  Precision: %.4f  Recall: %.4f  F1: %.4f", precision, recall, f1)

    # ── LIFT vs Scorecard baseline ─────────────────────────────────────────
    # Invert composite score:
    # higher score = healthier → invert so higher = more abandoned
    scorecard_signal = 1.0 - df_train_val["l1_composite_score"].values
    baseline_prauc   = average_precision_score(y_train_val, scorecard_signal)

    scorecard_signal_test = 1.0 - df_test["l1_composite_score"].values
    baseline_prauc_test   = average_precision_score(y_test, scorecard_signal_test)

    log.info("-" * 60)
    log.info("LIFT COMPARISON (PR-AUC):")
    log.info("  Scorecard baseline (train_val) : %.4f",
             baseline_prauc)
    log.info("  XGBoost OOF        (train_val) : %.4f",
             oof_prauc)
    log.info("  Data-Driven Lift   (train_val) : %+.4f",
             oof_prauc - baseline_prauc)
    log.info("  Scorecard baseline (holdout)   : %.4f",
             baseline_prauc_test)
    log.info("  XGBoost            (holdout)   : %.4f",
             test_prauc)
    log.info("  Data-Driven Lift   (holdout)   : %+.4f",
             test_prauc - baseline_prauc_test)

    res = {
        "oof_auc":    float(oof_auc),
        "oof_prauc":  float(oof_prauc),
        "test_auc":   float(test_auc),
        "test_prauc": float(test_prauc),
        "threshold":  float(threshold),
        "lift_oof":   float(oof_prauc  - baseline_prauc),
        "lift_test":  float(test_prauc - baseline_prauc_test),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
    }

    if time_bounded_eval:
        log.info("-" * 60)
        log.info("TIME-BOUNDED BASELINE EVALUATION")
        test_durations = df_test.groupby("appid")["ea_duration_days"].first().dropna()
        holdout_horizon = test_durations.quantile(0.95)
        log.info("Holdout horizon (95th percentile of 2024 durations): %.1f days",
                 holdout_horizon)

        fast_mask = ((df_train_val["ea_duration_days"] <= holdout_horizon)
                     & (df_train_val["ea_duration_days"].notna()))
        y_train_val_fast = y_train_val[fast_mask]
        oof_preds_fast = oof_preds[fast_mask]

        oof_auc_fast = roc_auc_score(y_train_val_fast, oof_preds_fast)
        oof_prauc_fast = average_precision_score(y_train_val_fast, oof_preds_fast)
        log.info("Time-Bounded OOF (fast games only) — AUC-ROC: %.4f  PR-AUC: %.4f",
                 oof_auc_fast, oof_prauc_fast)

        scorecard_signal_fast = 1.0 - (
            df_train_val.loc[fast_mask, "l1_composite_score"].values
        )
        baseline_prauc_fast = average_precision_score(y_train_val_fast,
                                                      scorecard_signal_fast)
        log.info("Time-Bounded Scorecard baseline        : %.4f",
                 baseline_prauc_fast)
        log.info("Time-Bounded Data-Driven Lift        : %+.4f",
                 oof_prauc_fast - baseline_prauc_fast)

        res["oof_auc_fast"] = float(oof_auc_fast)
        res["oof_prauc_fast"] = float(oof_prauc_fast)
        res["lift_oof_fast"] = float(oof_prauc_fast - baseline_prauc_fast)
        res["holdout_horizon_days"] = float(holdout_horizon)

    tier_agreement = {}
    for tier, target_label in [("Healthy", 0), ("Watch", 1), ("At Risk", 1)]:
        mask = df_test["l1_state"] == tier
        if mask.sum() > 0:
            tier_agreement[tier] = float((y_test[mask] == target_label).mean())
        else:
            tier_agreement[tier] = None

    if tier_agreement["Healthy"] is not None:
        res["healthy_outcome_agreement"] = tier_agreement["Healthy"]
    if tier_agreement["Watch"] is not None:
        res["watch_outcome_agreement"] = tier_agreement["Watch"]
    if tier_agreement["At Risk"] is not None:
        res["at_risk_outcome_agreement"] = tier_agreement["At Risk"]

    log.info("Per-tier outcome agreement (holdout):")
    for tier, val in tier_agreement.items():
        if val is not None:
            log.info("  %-10s %.4f", tier, val)

    log.info("-" * 60)

    return res


# ---------------------------------------------------------------------------
# SECTION 6 — SHAP
# ---------------------------------------------------------------------------

def run_shap(model, X_sample: pd.DataFrame, feature_names: list[str]):
    log.info("Computing native XGBoost SHAP values (sample size=%d)...", len(X_sample))

    # 1. Cast to float to destroy booleans, then load into native DMatrix
    X_float = X_sample.astype(float)
    dtest = xgb.DMatrix(X_float)

    # 2. THE BYPASS: Ask XGBoost to calculate its own SHAP values
    # pred_contribs=True returns a matrix of shape (n_samples, n_features + 1)
    # The final column is the expected value (base margin/bias)
    booster = model.get_booster()
    contributions = booster.predict(dtest, pred_contribs=True)

    # 3. Slice the matrix to separate the SHAP values from the base bias
    shap_values = contributions[:, :-1]
    base_value = contributions[0, -1]

    # 4. Calculate Mean Absolute SHAP for the logger
    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=feature_names
    ).sort_values(ascending=False)

    log.info("Top 25 features by mean |SHAP|:")
    cumulative_variance = mean_abs_shap.cumsum() / mean_abs_shap.sum()
    for i, (feat, val) in enumerate(mean_abs_shap.head(25).items()):
        log.info(
            "  %2d. %-45s %.5f  (cumulative variance: %.1f%%)",
            i + 1, feat, val, cumulative_variance.iloc[i] * 100
        )

    # 5. Export top 25 feature names — canonical order for Zilliz vectors
    # inference.py loads this to know which features to extract and in what order
    top25 = mean_abs_shap.head(25).index.tolist()
    top25_path = OUTPUT_DIR / f"shap_top25_{MODEL_VERSION}.json"
    with open(top25_path, "w") as f:
        json.dump({
            "model_version": MODEL_VERSION,
            "feature_count": 25,
            "cumulative_variance_pct": float(round(
                cumulative_variance.iloc[24] * 100, 2
            )),
            "features": top25,         # ordered by mean |SHAP|, index = vector position
        }, f, indent=2)
    log.info("Top 25 features exported → %s", top25_path)

    # 5. Save the exact payload the SHAP plotting library will need later
    shap_path = OUTPUT_DIR / f"shap_{MODEL_VERSION}.pkl"
    with open(shap_path, "wb") as f:
        pickle.dump({
            "shap_values": shap_values,
            "base_value": base_value,
            "data": X_float.values,
            "feature_names": feature_names
        }, f)
    log.info("SHAP saved → %s", shap_path)

    return mean_abs_shap


# ---------------------------------------------------------------------------
# SECTION 6.5 — Error Analysis
# ---------------------------------------------------------------------------

def analyze_errors_shap(
        model,
        df_test: pd.DataFrame,
        genre_enc,
        feature_names: list[str],
        threshold: float
        ):
    log.info("Computing SHAP for Error Analysis (Holdout)...")

    X_test, y_test, _, _ = build_feature_matrix(df_test, genre_enc)
    X_test = X_test.reindex(columns=feature_names, fill_value=np.nan)

    # Get FN and FP indices from holdout confusion matrix
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= threshold).astype(int)

    fn_mask = (y_test.values == 1) & (test_preds == 0)  # missed abandonments
    fp_mask = (y_test.values == 0) & (test_preds == 1)  # false alarms

    X_test_float = X_test.astype(float)
    dtest = xgb.DMatrix(X_test_float)
    contributions = model.get_booster().predict(dtest, pred_contribs=True)
    shap_values = contributions[:, :-1]

    # SHAP means on each error type
    shap_fn = (np.abs(shap_values[fn_mask]).mean(axis=0)
               if fn_mask.sum() > 0 else np.zeros(len(feature_names)))
    shap_fp = (np.abs(shap_values[fp_mask]).mean(axis=0)
               if fp_mask.sum() > 0 else np.zeros(len(feature_names)))

    # What features characterise each error type?
    error_df = pd.DataFrame({
        "fn_shap": shap_fn,
        "fp_shap": shap_fp
    }, index=feature_names).sort_values("fn_shap", ascending=False).head(15)

    log.info("Top features characterising holdout errors:")
    log.info("\n%s", error_df.to_string())

    return error_df

# ---------------------------------------------------------------------------
# SECTION 6.5 — Hyperparameter Tuning (Optuna)
# ---------------------------------------------------------------------------

def run_tuning(df_train_val: pd.DataFrame, scale_pos_weight: float, n_trials: int):
    log.info("Building feature matrix for tuning...")
    X_tv, y_tv, groups_tv, _ = build_feature_matrix(df_train_val)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3,
                                                     log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0,
                                                     log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0,
                                                     log=True),
            "scale_pos_weight":  scale_pos_weight,
            "objective":         "binary:logistic",
            "eval_metric":       "aucpr",
            "random_state":      RANDOM_SEED,
            "tree_method":       "hist",
            "n_jobs":            1, # Prevent nested parallelism
        }

        clf = xgb.XGBClassifier(**params)

        cv = StratifiedGroupKFold(n_splits=N_FOLDS,
                                  shuffle=True,
                                  random_state=RANDOM_SEED)

        scores = cross_val_score(
            clf,
            X_tv,
            y_tv,
            groups=groups_tv,
            cv=cv,
            scoring="average_precision",
            n_jobs=-1
        )

        return scores.mean()

    study = optuna.create_study(direction="maximize")
    log.info("Starting Optuna optimization (%d trials)...", n_trials)
    study.optimize(objective, n_trials=n_trials, timeout=3600)

    log.info("Best trial:")
    trial = study.best_trial
    log.info("  Value (PR-AUC): %f", trial.value)
    log.info("  Params: ")
    for key, value in trial.params.items():
        log.info("    %s: %s", key, value)

    with open(TUNED_PARAMS_PATH, "w") as f:
        json.dump(trial.params, f, indent=4)
    log.info("Saved Optuna tuned parameters to %s", TUNED_PARAMS_PATH)

    # Visualizations
    try:
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)

        fig_history = vis.plot_optimization_history(study)
        fig_importances = vis.plot_param_importances(study)

        history_path = out_dir / f"optuna_history_{MODEL_VERSION}.html"
        importances_path = out_dir / f"optuna_importances_{MODEL_VERSION}.html"

        fig_history.write_html(str(history_path))
        fig_importances.write_html(str(importances_path))

        log.info("Saved Optuna visualizations to %s and %s",
                 history_path, importances_path)
    except Exception as e:
        log.warning("Failed to save visualizations. "
                    "Do you have plotly installed? Error: %s", e)

    return study

# ---------------------------------------------------------------------------
# SECTION 7 — Artefact Summary
# ---------------------------------------------------------------------------

def write_run_log(metrics: dict, df_train_val, df_test, feature_names, best_iters):
    record = {
        "timestamp":       datetime.utcnow().isoformat(),
        "model_version":   MODEL_VERSION,
        "n_features":      len(feature_names),
        "train_val_snapshots": len(df_train_val),
        "train_val_games": df_train_val["appid"].nunique(),
        "test_snapshots":  len(df_test),
        "test_games":      df_test["appid"].nunique(),
        "cv_best_iters":   [int(i) for i in best_iters],
        **metrics,
    }

    log_path = Path("outputs") / "run_log.jsonl"
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    log.info("=" * 60)
    log.info("RUN SUMMARY — %s", MODEL_VERSION)
    log.info("=" * 60)
    log.info("  Features          : %d", len(feature_names))
    log.info("  Train/val games   : %d (%d snapshots)",
             df_train_val["appid"].nunique(), len(df_train_val))
    log.info("  Holdout games     : %d (%d snapshots)",
             df_test["appid"].nunique(), len(df_test))
    log.info("  OOF  AUC-ROC      : %.4f", metrics["oof_auc"])
    log.info("  OOF  PR-AUC       : %.4f", metrics["oof_prauc"])
    log.info("  Test AUC-ROC      : %.4f", metrics["test_auc"])
    log.info("  Test PR-AUC       : %.4f", metrics["test_prauc"])
    log.info("  Optimal threshold : %.4f", metrics["threshold"])
    if "oof_prauc_fast" in metrics:
        log.info("  Time-Bounded OOF PR-AUC : %.4f", metrics["oof_prauc_fast"])
        log.info("  Time-Bounded Lift       : %+.4f", metrics["lift_oof_fast"])
    log.info("  Lift (holdout)    : %+.4f", metrics["lift_test"])
    log.info("  Model path        : models/xgb_%s.json", MODEL_VERSION)
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run",  action="store_true",
                   help="Build feature matrix only, no training")
    p.add_argument("--no-shap",  action="store_true",
                   help="Skip SHAP computation")
    p.add_argument("--time-bounded-eval",  action="store_true",
                   help="Evaluate temporal holdout against right-censored OOF set")
    p.add_argument("--tune",  action="store_true",
                   help="Run Optuna hyperparameter tuning instead of training")
    p.add_argument("--n-trials", type=int, default=50,
                   help="Number of Optuna trials")
    p.add_argument("--ignore-tuned", action="store_true",
                   help="Ignore saved Optuna tuned parameters and use defaults")
    args = p.parse_args()

    conn = get_conn()
    df   = load_data(conn)
    conn.close()

    # ── Temporal split ───────────────────────────────────────────────────────
    df_train_val, df_test = temporal_split(df)

    # ── Class balance for scale_pos_weight ──────────────────────────────────
    n_neg = (df_train_val["label"] == 0).sum()
    n_pos = (df_train_val["label"] == 1).sum()

    if n_pos == 0:
        raise ValueError("No positive labels (EXIT_ABANDONED/EXIT_SILENT) found "
                         "in the train/val set!")

    scale_pos_weight = round(n_neg / n_pos, 2)
    log.info("scale_pos_weight: %.2f", scale_pos_weight)

    log.info("Train/val positive label rate: %.4f  "
             "(expected ~0.25 given 3:1 ratio)",
             df_train_val["label"].mean())
    log.info("Holdout positive label rate:   %.4f  "
             "(if much higher, the holdout is easier)",
             df_test["label"].mean())

    if args.dry_run:
        X, y, groups, enc = build_feature_matrix(df_train_val)
        log.info("DRY RUN — feature matrix shape: %s", X.shape)
        log.info("Columns: %s", list(X.columns))
        log.info("Null rates > 20%%:")
        null_rates = X.isnull().mean().sort_values(ascending=False)
        log.info("\n%s", null_rates[null_rates > 0.20].to_string())
        return

    if args.tune:
        run_tuning(df_train_val, scale_pos_weight, args.n_trials)
        return

    # ── Load Tuned Parameters ────────────────────────────────────────────────
    if not args.ignore_tuned and TUNED_PARAMS_PATH.exists():
        log.info("Loading tuned parameters from %s", TUNED_PARAMS_PATH)
        with open(TUNED_PARAMS_PATH) as f:
            tuned_params = json.load(f)
        XGB_PARAMS.update(tuned_params)
    elif not args.ignore_tuned:
        log.info("No tuned parameters found at %s. Using default XGB_PARAMS.",
                 TUNED_PARAMS_PATH)
    else:
        log.info("Ignoring tuned parameters (--ignore-tuned). "
                 "Using default XGB_PARAMS.")

    with start_run(model_version=MODEL_VERSION) as mlflow_run:

        # ── Section 3: CV ────────────────────────────────────────────────
        oof_preds, y_tv, feature_names, genre_enc, best_iters = run_cv(
            df_train_val, scale_pos_weight
        )

        # ── Dynamic threshold from OOF ───────────────────────────────────
        threshold = find_optimal_threshold(y_tv.values, oof_preds)

        # ── Section 4: Final model ───────────────────────────────────────
        model = train_final_model(
            df_train_val, best_iters, scale_pos_weight,
            genre_enc, feature_names, threshold
        )

        # ── Section 5: Evaluation + Lift ─────────────────────────────────
        metrics = evaluate(
            model, df_test, oof_preds, y_tv,
            df_train_val, genre_enc, feature_names, threshold,
            time_bounded_eval=args.time_bounded_eval
        )
        # ──  Unified Feature Matrix Creation for Signatures & SHAP ──
        X_tv, _, _, _ = build_feature_matrix(df_train_val, genre_enc)
        X_tv = X_tv.reindex(columns=feature_names, fill_value=np.nan)

        # ── Section 6: SHAP ───────────────────────────────────────────────
        shap_top25_path = OUTPUT_DIR / f"shap_top25_{MODEL_VERSION}.json"
        if not args.no_shap:
            shap_sample = X_tv.sample(min(2000, len(X_tv)), random_state=RANDOM_SEED)
            run_shap(model, shap_sample, feature_names)
            analyze_errors_shap(model, df_test, genre_enc, feature_names, threshold)

        # ── Section 7: Log ────────────────────────────────────────────────
        write_run_log(metrics, df_train_val, df_test, feature_names, best_iters)

        # ── MLflow: log run + register model ─────────────────────────────
        input_example = X_tv.astype(float).iloc[[0]]
        log_training_run(
            run=mlflow_run,
            params=XGB_PARAMS,
            metrics=metrics,
            model_path=OUTPUT_DIR / f"xgb_{MODEL_VERSION}.json",
            features_path=OUTPUT_DIR / f"xgb_{MODEL_VERSION}_features.json",
            shap_top25_path=shap_top25_path if not args.no_shap else None,
            model_version=MODEL_VERSION,
            scorecard_config_version=CONFIG_VERSION,
            training_cohort={
                "train_val_games": df_train_val["appid"].nunique(),
                "holdout_games": df_test["appid"].nunique(),
            },
            input_example=input_example,
        )


if __name__ == "__main__":
    main()
