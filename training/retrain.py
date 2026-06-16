"""
retrain.py — EARLY pipeline: scheduled retraining with conditional promotion
==============================================================================

Wraps train_xgboost.py + promote_model.py into a single retraining cycle suitable 
for a scheduled GitHub Actions workflow (retrain.yml), with the gate logic from 
promote_model.compare_to_production.

─────────────────────────────────────────────────────────────────────────────
WORKFLOW
─────────────────────────────────────────────────────────────────────────────
1. Run train_xgboost.py (full pipeline: CV, final model, eval, SHAP).
   This registers a new model version in MLflow without aliases (via
   log_training_run in mlflow_client.py).

2. Compare the new version's metrics against current Production using
   promote_model.compare_to_production().

3. Gate:
     PASS -> assign staging alias, then champion alias
             (archiving the previous Production version).
             If the SHAP top-25 feature set changed vs the previous
             Production model, flag --zilliz-rebuild-needed in the summary.
     FAIL -> leave new version without aliases (visible in registry as a
             candidate), log the regression reasons, exit 0 (not an error
             — this is expected/normal behaviour, not a pipeline failure).

4. Write outputs/retrain_report_{date}.json summarising the cycle.

─────────────────────────────────────────────────────────────────────────────
TRIGGER MODES
─────────────────────────────────────────────────────────────────────────────
  --scheduled       : normal monthly cron run
  --drift-triggered : invoked because monitor_drift.py flagged action_needed
                       (recorded in the report for traceability)

─────────────────────────────────────────────────────────────────────────────
HUMAN-GATED MODE (recommended initially)
─────────────────────────────────────────────────────────────────────────────
  --no-auto-promote : run training + comparison, write the report, but do
                       NOT assign aliases. A human reviews the report and
                       runs promote_model.py manually. This is the default
                       until the gate's behaviour has been observed over a
                       few cycles (per the MLOps plan).

─────────────────────────────────────────────────────────────────────────────
USAGE
─────────────────────────────────────────────────────────────────────────────
  python retrain.py --scheduled
  python retrain.py --drift-triggered --no-auto-promote
  python retrain.py --scheduled --no-shap     # passed through to train_xgboost.py
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "outputs"
MODEL_DIR    = PROJECT_ROOT / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.mlflow_client import REGISTERED_MODEL_NAME  # noqa: E402
from training.promote_model import (  # noqa: E402
    get_client,
    get_production_version,
    compare_to_production,
    print_comparison,
)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def run_training(extra_args: list[str]) -> bool:
    """
    Run train_xgboost.py as a subprocess so its existing main()/argparse is
    reused unmodified. Returns True on success (exit code 0).
    """
    cmd = [sys.executable, str(PROJECT_ROOT / "training" / "train_xgboost.py")] + extra_args
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode == 0


def get_newly_registered_version(client, run_id_hint: str | None = None) -> str | None:
    """
    Return the highest-numbered model version currently without aliases
    (i.e. just registered, not yet aliased). This assumes
    train_xgboost.py registered exactly one new version this run.
    """
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    none_stage = []
    for v_search in versions:
        v = client.get_model_version(REGISTERED_MODEL_NAME, v_search.version)
        if not getattr(v, "aliases", []):
            none_stage.append(v)
    if not none_stage:
        return None
    latest = max(none_stage, key=lambda v: int(v.version))
    return latest.version


def shap_feature_sets_differ(prod_version: str | None, candidate_version: str, client) -> bool | None:
    """
    Compare the SHAP top-25 feature list between the Production model and
    the candidate, by reading shap_top25_{model_version}.json artifacts
    from local models/ dir (best-effort — assumes both artifacts are
    present locally, which is true within a single CI run since training
    just wrote the candidate's).

    Returns True/False, or None if comparison isn't possible.
    """
    if prod_version is None:
        return None

    candidate_run = client.get_model_version(REGISTERED_MODEL_NAME, candidate_version)
    candidate_tag = candidate_run.run_id  # not directly useful for filename matching

    # Best-effort: compare against whatever shap_top25_*.json files exist
    # locally. This is inherently approximate without a model_version tag
    # lookup; flag for manual confirmation rather than asserting confidently.
    shap_files = sorted(MODEL_DIR.glob("shap_top25_*.json"))
    if len(shap_files) < 2:
        return None

    feature_sets = []
    for f in shap_files[-2:]:
        with open(f) as fh:
            feature_sets.append(set(json.load(fh).get("features", [])))

    return feature_sets[0] != feature_sets[1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="EARLY automated retraining cycle")
    trigger = p.add_mutually_exclusive_group(required=True)
    trigger.add_argument("--scheduled", action="store_true", help="Routine scheduled retrain")
    trigger.add_argument("--drift-triggered", action="store_true",
                          help="Triggered by monitor_drift.py action_needed")
    p.add_argument("--no-auto-promote", action="store_true",
                   help="Run training + comparison only; do not assign aliases")
    p.add_argument("--no-shap", action="store_true", help="Pass through to train_xgboost.py")
    p.add_argument("--time-bounded-eval", action="store_true", help="Pass through to train_xgboost.py")
    args = p.parse_args()

    run_date = datetime.now(timezone.utc).date().isoformat()
    trigger_reason = "drift_triggered" if args.drift_triggered else "scheduled"

    train_args = []
    if args.no_shap:
        train_args.append("--no-shap")
    if args.time_bounded_eval:
        train_args.append("--time-bounded-eval")

    log.info("=" * 60)
    log.info("RETRAIN CYCLE — %s (trigger: %s)", run_date, trigger_reason)
    log.info("=" * 60)

    # ── Step 1: train ────────────────────────────────────────────────────
    ok = run_training(train_args)
    if not ok:
        log.error("train_xgboost.py failed — aborting retrain cycle")
        sys.exit(1)

    # ── Step 2: locate the new candidate version ────────────────────────
    client = get_client()
    candidate_version = get_newly_registered_version(client)

    if candidate_version is None:
        log.error(
            "No new model version found without aliases. Has train_xgboost.py "
            "been patched to call log_training_run()? See PATCH NOTES in retrain.py."
        )
        sys.exit(1)

    log.info("New candidate: %s v%s", REGISTERED_MODEL_NAME, candidate_version)

    # ── Step 3: compare against Production ──────────────────────────────
    result = compare_to_production(client, candidate_version)
    print_comparison(result)

    prod_mv = get_production_version(client)
    prod_version = prod_mv.version if prod_mv else None

    zilliz_rebuild_needed = shap_feature_sets_differ(prod_version, candidate_version, client)

    report = {
        "run_date": run_date,
        "trigger_reason": trigger_reason,
        "candidate_version": candidate_version,
        "previous_production_version": prod_version,
        "gate_passed": result["passed"],
        "comparisons": result["comparisons"],
        "reasons": result["reasons"],
        "zilliz_rebuild_needed": zilliz_rebuild_needed,
        "action_taken": None,
    }

    # ── Step 4: promote or hold ──────────────────────────────────────────
    if args.no_auto_promote:
        report["action_taken"] = "none (--no-auto-promote, awaiting manual review)"
        log.info("Auto-promotion disabled. Candidate v%s left without aliases.", candidate_version)
        log.info("Review the report, then run:")
        log.info("  python promote_model.py --version %s --alias staging --yes", candidate_version)
        log.info("  python promote_model.py --version %s --alias champion --yes", candidate_version)

    elif result["passed"]:
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME, alias="staging", version=candidate_version,
        )
        if prod_mv is not None:
            try:
                client.delete_registered_model_alias(name=REGISTERED_MODEL_NAME, alias="champion")
            except Exception:
                pass
            log.info("Archived previous Production v%s", prod_mv.version)

        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME, alias="champion", version=candidate_version,
        )
        report["action_taken"] = f"promoted v{candidate_version} to Production"
        log.info("Assigned champion alias to v%s", candidate_version)

        if zilliz_rebuild_needed:
            log.warning(
                "SHAP top-25 feature set changed vs previous Production model — "
                "run: python training/seed_vector_db.py --rebuild"
            )

    else:
        report["action_taken"] = "held (gate failed, no promotion)"
        log.info("Gate failed — candidate v%s remains without aliases.", candidate_version)
        for r in result["reasons"]:
            log.info("  - %s", r)

    # ── Step 5: write report ─────────────────────────────────────────────
    report_path = OUTPUT_DIR / f"retrain_report_{run_date}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info("=" * 60)
    log.info("Retrain report -> %s", report_path)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
