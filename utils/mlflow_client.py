"""
mlflow_client.py — EARLY pipeline: thin MLflow wrapper
=======================================================

Provides a small, consistent interface for logging training runs and resolving
the current Production model, with a no-op fallback so training/inference work
standalone if MLflow isn't configured (mirrors the langfuse_client.py pattern).

─────────────────────────────────────────────────────────────────────────────
SETUP
─────────────────────────────────────────────────────────────────────────────
Local file-store tracking (default, free, no server):
  MLFLOW_TRACKING_URI not set -> defaults to ./mlruns (local file store)

Optional remote tracking server:
    DATABRICKS_TOKEN="dapixxxx"
    DATABRICKS_HOST="https://dbc-xxx.com/"
    MLFLOW_REGISTRY_URI="databricks-uc"

Registry model name is fixed: "early_xgb_classifier"

─────────────────────────────────────────────────────────────────────────────
USAGE (train_xgboost.py)
─────────────────────────────────────────────────────────────────────────────
    from training.mlflow_client import start_run, log_training_run

    with start_run(model_version=MODEL_VERSION) as run:
        ... train ...
        log_training_run(
            run=run,
            params=XGB_PARAMS,
            metrics=metrics,
            model_path=model_path,
            features_path=feature_path,
            shap_top25_path=top25_path,
            model_version=MODEL_VERSION,
            scorecard_config_version=CONFIG_VERSION,
            training_cohort={"train_val_games": ..., "holdout_games": ...},
            input_example=input_example,
        )

─────────────────────────────────────────────────────────────────────────────
USAGE (inference.py / populate_zilliz.py)
─────────────────────────────────────────────────────────────────────────────
    from utils.mlflow_client import get_production_model_uri, download_artifact

    uri = get_production_model_uri()   # None if registry unavailable/empty
    if uri:
        local_path = download_artifact(uri, "xgb_model.json")
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


load_dotenv()

log = logging.getLogger(__name__)

REGISTERED_MODEL_NAME = "workspace.default.early_xgb_classifier"

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    log.info("mlflow not installed — MLflow logging disabled (no-op mode)")


# ---------------------------------------------------------------------------
# No-op fallbacks
# ---------------------------------------------------------------------------

class _NoOpRun:
    """Stand-in for an mlflow.ActiveRun when mlflow is unavailable."""
    info = type("info", (), {"run_id": "no-op"})()


@contextmanager
def _noop_run(*_args, **_kwargs):
    yield _NoOpRun()


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------

@contextmanager
def start_run(model_version: str, experiment_name: str = "early_xgb_training"):
    """
    Context manager wrapping mlflow.start_run(). No-op if mlflow unavailable.
    """
    if not _MLFLOW_AVAILABLE:
        with _noop_run() as run:
            yield run
        return

    default_uri = "databricks" if os.getenv("DATABRICKS_HOST") else "./mlruns"
    mlflow.set_tracking_uri(default_uri)

    if default_uri == "databricks" and not experiment_name.startswith("/"):
        user_email = os.getenv("DATABRICKS_USER_EMAIL")

        if user_email:
            experiment_name = f"/Users/{user_email}/{experiment_name}"
        else:
            experiment_name = f"/Shared/{experiment_name}"

    if default_uri == "databricks":
        registry_uri = os.getenv("MLFLOW_REGISTRY_URI", "databricks-uc")
        mlflow.set_registry_uri(registry_uri)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"train_{model_version}") as run:
        mlflow.set_tag("model_version", model_version)
        yield run


def log_training_run(
    run,
    params: dict,
    metrics: dict,
    model_path: Path,
    features_path: Path,
    model_version: str,
    scorecard_config_version: str,
    training_cohort: dict,
    shap_top25_path: Path | None = None,
    input_example: pd.DataFrame | None = None,
) -> str | None:
    """
    Log params/metrics/artifacts for a completed training run, and register
    the model under REGISTERED_MODEL_NAME in stage 'None' (unstaged).

    Returns the registered model version string, or None if mlflow unavailable
    or registration failed.
    """
    if not _MLFLOW_AVAILABLE:
        log.info(
            "[no-op] Would log training run: model_version=%s metrics=%s",
            model_version, {k: round(v, 4)
                            if isinstance(v, float)
                            else v for k, v in metrics.items()},
        )
        return None

    # Params — XGBoost params + cohort metadata + config versions
    loggable_params = {
        **{f"xgb_{k}": v for k, v in params.items()
           if not isinstance(v, (dict | list))},
        "scorecard_config_version": scorecard_config_version,
        **{f"cohort_{k}": v for k, v in training_cohort.items()},
    }
    mlflow.log_params(loggable_params)

    # Metrics — only scalar numeric values
    for k, v in metrics.items():
        if isinstance(v, (int | float)) and not isinstance(v, bool):
            mlflow.log_metric(k, v)

    # Artifacts
    if model_path.exists():
        import xgboost as xgb
        bst = xgb.Booster()
        bst.load_model(str(model_path))
        mlflow.xgboost.log_model(xgb_model=bst, name="model",
                                 input_example=input_example)
    if features_path.exists():
        mlflow.log_artifact(str(features_path), artifact_path="features")
    if shap_top25_path and shap_top25_path.exists():
        mlflow.log_artifact(str(shap_top25_path), artifact_path="shap")

    # Register model — version is created without aliases; promote_model.py
    # handles assigning staging/champion aliases.
    registered_version = None
    try:
        run_id = run.info.run_id
        artifact_path = "model"
        model_uri = f"runs:/{run_id}/{artifact_path}"
        mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
        registered_version = mv.version
        log.info(
            "Registered %s version %s (run_id=%s)",
            REGISTERED_MODEL_NAME, registered_version, run_id,
        )
    except Exception as e:
        log.warning("Model registration failed: %s", e)

    return registered_version


# ---------------------------------------------------------------------------
# Registry resolution (for inference / Zilliz pipeline)
# ---------------------------------------------------------------------------

def _setup_mlflow_uris():
    """Ensure tracking and registry URIs are set consistently."""
    tracking_uri = ("databricks" if os.getenv("DATABRICKS_HOST")
                    else os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
    mlflow.set_tracking_uri(tracking_uri)
    if tracking_uri == "databricks":
        registry_uri = os.getenv("MLFLOW_REGISTRY_URI", "databricks-uc")
        mlflow.set_registry_uri(registry_uri)


def get_production_model_uri() -> str | None:
    """
    Return the model URI for the current champion registered model,
    or None if mlflow unavailable / no Production model exists.
    """
    if not _MLFLOW_AVAILABLE:
        return None

    _setup_mlflow_uris()
    client = MlflowClient()

    try:
        mv = client.get_model_version_by_alias(name=REGISTERED_MODEL_NAME,
                                               alias="champion")
    except Exception as e:
        log.warning("Could not query registry: %s", e)
        return None

    log.info("Production model: %s v%s (run_id=%s)", REGISTERED_MODEL_NAME,
             mv.version, mv.run_id)
    return f"models:/{REGISTERED_MODEL_NAME}@champion"


def get_run_metrics(run_id: str) -> dict:
    """Fetch logged metrics for a run_id. Empty dict if unavailable."""
    if not _MLFLOW_AVAILABLE:
        return {}
    _setup_mlflow_uris()
    client = MlflowClient()
    try:
        run = client.get_run(run_id)
        return dict(run.data.metrics)
    except Exception as e:
        log.warning("Could not fetch metrics for run %s: %s", run_id, e)
        return {}


def download_artifact(model_uri: str,
                      artifact_relpath: str,
                      dst_dir: str | Path = "models") -> Path | None:
    """
    Download a single artifact (e.g. 'model/xgb_v1.3.json') from a model
    version's artifact store to dst_dir. Returns local path or None.
    """
    if not _MLFLOW_AVAILABLE:
        return None
    _setup_mlflow_uris()
    try:
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"{model_uri}/{artifact_relpath}",
            dst_path=str(dst_dir),
        )
        return Path(local_path)
    except Exception as e:
        log.warning("Artifact download failed (%s / %s): %s",
                    model_uri, artifact_relpath, e)
        return None
