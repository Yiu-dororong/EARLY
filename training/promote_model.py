"""
promote_model.py — EARLY pipeline: MLflow model registry promotion
====================================================================

Transitions a registered model version between
Staging / Champion / Archived, with a metric comparison report
against the current Production model before promoting.

This script does NOT make the promotion decision automatically — it shows
the comparison and requires --yes (or interactive confirmation) to act.
retrain.py calls the comparison logic here programmatically for
the automated gate.

─────────────────────────────────────────────────────────────────────────────
USAGE
─────────────────────────────────────────────────────────────────────────────
  python promote_model.py --list
      List all versions of early_xgb_classifier with alias + key metrics.

  python promote_model.py --version 4 --alias staging
      Assign version 4 the staging alias (no comparison needed for staging).

  python promote_model.py --version 4 --alias champion
      Compare version 4's metrics against current Production, show the
      diff, and prompt for confirmation before promoting. Archives the
      previous Production version.

  python promote_model.py --version 4 --alias champion --yes
      Same, but skip confirmation (for CI).

  python promote_model.py --version 4 --alias champion --force
      Promote even if the comparison gate fails (manual override).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.mlflow_client import REGISTERED_MODEL_NAME, get_run_metrics  # noqa: E402


try:
    import mlflow
    from mlflow.tracking import MlflowClient
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


# ---------------------------------------------------------------------------
# Promotion gate
#
# A candidate is "safe to promote" if:
#   1. oof_prauc does not regress more than TOLERANCE_PRAUC vs current
#      Production, AND
#   2. per-tier outcome agreement (if logged) does not regress more than
#      TOLERANCE_TIER_PP percentage points for any tier.
#
# Metrics referenced here must be logged by train_xgboost.py via
# log_training_run() (see mlflow_client.py). Tier agreement metrics are
# optional — if absent, only the PR-AUC check applies.
# ---------------------------------------------------------------------------

TOLERANCE_PRAUC   = 0.005   # allow up to -0.005 PR-AUC vs current Production
TOLERANCE_TIER_PP = 0.02    # allow up to -2pp outcome agreement per tier

TIER_METRIC_KEYS = [
    "healthy_outcome_agreement",
    "watch_outcome_agreement",
    "at_risk_outcome_agreement",
]


def get_client() -> MlflowClient:
    if not _MLFLOW_AVAILABLE:
        raise RuntimeError("mlflow is not installed. pip install mlflow")
    tracking_uri = ("databricks" if os.getenv("DATABRICKS_HOST")
                                else os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
    mlflow.set_tracking_uri(tracking_uri)
    if tracking_uri == "databricks":
        registry_uri = os.getenv("MLFLOW_REGISTRY_URI", "databricks-uc")
        mlflow.set_registry_uri(registry_uri)
    return MlflowClient()


def list_versions(client: MlflowClient) -> None:
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        log.info("No versions registered for %s", REGISTERED_MODEL_NAME)
        return

    log.info("%-8s %-15s %-10s %-10s %s", "version", "aliases",
             "oof_prauc", "test_prauc", "run_id")
    for mv_search in sorted(versions, key=lambda v: int(v.version), reverse=True):
        mv = client.get_model_version(REGISTERED_MODEL_NAME, mv_search.version)
        metrics = get_run_metrics(mv.run_id)
        aliases_raw = getattr(mv, "aliases", [])
        aliases_str = ",".join(aliases_raw) if aliases_raw else "—"
        log.info(
            "%-8s %-15s %-10s %-10s %s",
            mv.version, aliases_str,
            f"{metrics.get('oof_prauc', float('nan')):.4f}"
            if "oof_prauc" in metrics else "—",
            f"{metrics.get('test_prauc', float('nan')):.4f}"
            if "test_prauc" in metrics else "—",
            mv.run_id,
        )


def get_production_version(client: MlflowClient):
    try:
        return client.get_model_version_by_alias(name=REGISTERED_MODEL_NAME,
                                                 alias="champion")
    except Exception:
        return None


def compare_to_production(client: MlflowClient, candidate_version: str) -> dict:
    """
    Compare candidate metrics against current Production metrics.

    Returns dict:
      {
        "passed": bool,
        "current_production_version": str | None,
        "comparisons": {metric_name: {"candidate": x,
                                    "production": y,
                                    "delta": d,
                                    "ok": bool}},
        "reasons": [list of failure reasons, empty if passed]
      }

    If no Production model exists yet, passes by default (first promotion).
    """
    candidate_mv = client.get_model_version(REGISTERED_MODEL_NAME, candidate_version)
    candidate_metrics = get_run_metrics(candidate_mv.run_id)

    prod_mv = get_production_version(client)
    if prod_mv is None:
        return {
            "passed": True,
            "current_production_version": None,
            "comparisons": {},
            "reasons": [],
            "note": "No current Production model — first promotion auto-passes.",
        }

    prod_metrics = get_run_metrics(prod_mv.run_id)
    comparisons: dict = {}
    reasons: list[str] = []

    # 1. Primary metric — PR-AUC
    cand_prauc = candidate_metrics.get("oof_prauc")
    prod_prauc = prod_metrics.get("oof_prauc")
    if cand_prauc is not None and prod_prauc is not None:
        delta = cand_prauc - prod_prauc
        ok = delta >= -TOLERANCE_PRAUC
        comparisons["oof_prauc"] = {
            "candidate": cand_prauc, "production": prod_prauc, "delta": delta, "ok": ok,
        }
        if not ok:
            reasons.append(
                f"oof_prauc regressed by {delta:.4f} "
                f"(tolerance: -{TOLERANCE_PRAUC})"
            )
    else:
        comparisons["oof_prauc"] = {"candidate": cand_prauc, "production": prod_prauc,
                                     "delta": None, "ok": True}

    # 2. Per-tier outcome agreement (if logged)
    for key in TIER_METRIC_KEYS:
        cand_v = candidate_metrics.get(key)
        prod_v = prod_metrics.get(key)
        if cand_v is None or prod_v is None:
            continue
        delta = cand_v - prod_v
        ok = delta >= -TOLERANCE_TIER_PP
        comparisons[key] = {"candidate": cand_v,
                            "production": prod_v,
                            "delta": delta,
                            "ok": ok}
        if not ok:
            reasons.append(
                f"{key} regressed by {delta:.4f} "
                f"(tolerance: -{TOLERANCE_TIER_PP})"
            )

    return {
        "passed": len(reasons) == 0,
        "current_production_version": prod_mv.version,
        "comparisons": comparisons,
        "reasons": reasons,
    }


def print_comparison(result: dict) -> None:
    log.info("-" * 60)
    if result.get("current_production_version") is None:
        log.info(result.get("note", "No current Production model."))
    else:
        log.info("Comparison vs Production v%s:", result["current_production_version"])
        for name, c in result["comparisons"].items():
            if c["delta"] is None:
                log.info("  %-32s candidate=%s  production=%s  (not comparable)",
                         name, c["candidate"], c["production"])
                continue
            flag = "OK" if c["ok"] else "FAIL"
            log.info(
                "  %-32s candidate=%.4f  production=%.4f  delta=%+.4f  [%s]",
                name, c["candidate"], c["production"], c["delta"], flag,
            )
        if result["passed"]:
            log.info("Gate: PASS")
        else:
            log.info("Gate: FAIL")
            for r in result["reasons"]:
                log.info("  - %s", r)
    log.info("-" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="EARLY model registry promotion")
    p.add_argument("--list", action="store_true", help="List registered model versions")
    p.add_argument("--version", type=str, help="Model version to transition")
    p.add_argument("--alias", type=str,
                   choices=["staging", "champion", "challenger", "archived"],
                   help="Target alias")
    p.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    p.add_argument("--force", action="store_true",
                   help="Promote to Production even if comparison gate fails")
    args = p.parse_args()

    client = get_client()

    if args.list:
        list_versions(client)
        return

    if not args.version or not args.alias:
        log.error("--version and --alias are required (or use --list)")
        sys.exit(2)

    if args.alias == "champion":
        result = compare_to_production(client, args.version)
        print_comparison(result)

        if not result["passed"] and not args.force:
            log.error("Promotion gate failed. Use --force to override, "
                      "or address regressions.")
            sys.exit(1)

        if not args.yes:
            resp = input(
                f"Promote v{args.version} to Production? [y/N] "
                ).strip().lower()
            if resp != "y":
                log.info("Aborted.")
                return

        # Archive current production before promoting new one
        prod_mv = get_production_version(client)
        if prod_mv is not None and prod_mv.version != args.version:
            try:
                client.delete_registered_model_alias(name=REGISTERED_MODEL_NAME,
                                                     alias="champion")
            except Exception:
                pass
            log.info("Removed champion alias from previous Production v%s",
                     prod_mv.version)

    elif not args.yes:
        resp = input(
            f"Assign alias @{args.alias} to v{args.version}? [y/N] "
            ).strip().lower()
        if resp != "y":
            log.info("Aborted.")
            return

    if args.alias == "archived":
        # Remove all aliases to effectively "archive" the version
        mv = client.get_model_version(REGISTERED_MODEL_NAME, args.version)
        aliases_raw = getattr(mv, "aliases", [])
        if aliases_raw:
            for al in aliases_raw:
                client.delete_registered_model_alias(REGISTERED_MODEL_NAME, al)
        log.info("Removed aliases from v%s (Archived)", args.version)
    else:
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME, alias=args.alias, version=args.version,
        )
        log.info("v%s -> @%s", args.version, args.alias)


if __name__ == "__main__":
    main()
