# MLOps

## Overview

Four-stage MLOps plan. Stages 1–3 are implemented. Stage 4 is roadmap.

The design philosophy throughout: **no new paid infrastructure, no complexity beyond what the problem requires.** Every tool chosen has a free tier that covers EARLY's scale. Every component degrades gracefully if not configured.

---

## Stage 1 — Drift Monitoring

**Script:** `training/monitor_drift.py`
**Trigger:** `score.yml`, after weekly inference, before agent triggers

Four checks, each producing a row in the `drift_reports` table:

| Check | What it measures | Method |
|---|---|---|
| Feature drift | Distribution shift in top-25 SHAP features (raw values from `live_snapshots`) vs training baseline | PSI |
| Prediction drift | Distribution of `p_distressed` across the scored population vs prior week | PSI |
| Null-rate drift | `null_feature_count` distribution per `l1_state` tier vs training-time rates | Relative deviation |
| Label drift | Abandonment rate among resolved games vs training-time base rate (~25%) | Relative deviation |

**PSI thresholds:**
```
PSI < 0.10   : ok
PSI < 0.25   : warning
PSI ≥ 0.25   : action_needed
```

**Reference distribution** (`models/drift_reference_{MODEL_VERSION}.json`) is frozen from the training snapshot population the first time `--freeze-reference` is run, and regenerated only on model promotion. It is the baseline — everything is measured against it.

**Prediction drift** is compared against last week's distribution (stored as `models/pred_distribution_{MODEL_VERSION}.json`), not the training distribution. This catches regime changes in scoring output even if features haven't drifted.

**Null-rate drift** is the most operationally important check. The key finding from the first production run — At Risk games have 2.6× the null rate of Healthy games — means any shift in this distribution signals either a data pipeline degradation or a population shift in the monitored games.

Output is written to both `drift_reports` (queryable) and `outputs/drift_report_{date}.json` (human-readable). `--fail-on-drift` exits with code 1 if any check is `action_needed`, enabling CI gating.

### Note on False Positives

When auditing the pipeline, it is critical to contextualize feature drift spikes rather than blindly treating every alert as a data corruption event. Certain feature channels are inherently volatile due to external marketplace rhythms. For example, during major platform events (such as Steam Seasonal Sales), some features will register temporary PSI spikes. 

Because the training baseline spans a multi-year historical epoch, a sudden, synchronized movement during a summer sale can skews the active distribution away from the historical mean. This should be treated as a predictable, event-driven anomaly rather than real systemic feature drift.

---

## Stage 2 — MLflow Model Registry

**Scripts:** `training/mlflow_client.py`, `training/promote_model.py`
**Tracking URI:** Local file store (`./mlruns`) by default; remote server via `MLFLOW_TRACKING_URI`

Every `train_xgboost.py` run logs:
- XGBoost hyperparameters
- OOF PR-AUC, test PR-AUC, per-tier outcome agreement
- `scorecard_config_version` and training cohort size
- Artifacts: model file, feature list, `shap_top25_{MODEL_VERSION}.json`

The `shap_top25` artifact is logged alongside the model because it is the contract between the model and the Zilliz pipeline — if the top-25 feature set changes, the vector database must be rebuilt. Version-coupling this to the model run makes that dependency explicit and traceable.

**Registry stages:** `None → Staging/Challenger → Champion(Production) → Archived`

`inference.py` and `populate_zilliz.py` resolve the Production model via MLflow client at startup, falling back to filesystem paths if the registry is unavailable. The no-op stub in `mlflow_client.py` means training works standalone without MLflow installed.

### Promotion gate

`promote_model.py --version N --stage Champion` compares the candidate's metrics against the current Production model before transitioning:

```
oof_prauc regression       > 0.005pp  → FAIL
per-tier agreement regression > 2pp   → FAIL (any tier)
```

The tier check prevents a model that improves aggregate AUC while quietly making At Risk classification worse — which is exactly the failure mode that matters most operationally.

If no Production model exists yet, the gate auto-passes (first promotion).

---

## Stage 3 — Conditional Retraining

**Script:** `training/retrain.py`
**Workflow:** `.github/workflows/retrain.yml`
**Trigger:** Manual (`workflow_dispatch`) or monthly cron (commented out until gate behaviour is confirmed over a few cycles)

The retraining cycle:

```
1. Run train_xgboost.py (full CV, final model, SHAP)
        │
        ▼
2. Locate new candidate version in MLflow registry
        │
        ▼
3. compare_to_production() → gate result
        │
     PASS                    FAIL
        │                      │
        ▼                      ▼
4. Staging → Production    Leave at 'None'
   Archive previous         Log reasons
   Flag Zilliz rebuild      Exit 0 (not an error)
   if SHAP set changed
        │
        ▼
5. Write outputs/retrain_report_{date}.json
   Post summary to GitHub Actions job summary
```

**`--no-auto-promote` is the default** for manual runs. Every cycle lands for human review until the gate's behaviour has been observed across multiple cycles. `--no-auto-promote` can be explicitly disabled once confidence is established.

**Zilliz rebuild flag** — if the SHAP top-25 feature set changed between the previous Production model and the new one, the report flags `zilliz_rebuild_needed: true`. The rebuild is not automatic — it requires a manual `python training/seed_vector_db.py --rebuild` because rebuilding the vector database is destructive and irreversible.

---

## Stage 4 — XGBoost AFT Survival Analysis (Roadmap)

The biggest available modeling upgrade. Reframes the problem from binary classification at a fixed snapshot to **time-to-event prediction** — not "will this game be abandoned" but "how much runway does it have."

XGBoost's `survival:aft` objective models `ea_age_days` as a survival/censoring problem. Games still in Early Access with unknown outcomes are right-censored (`STAYS_ACTIVE`) — AFT handles this honestly rather than treating them as a positive class. The output would be a predicted survival curve per game, from which both a distress probability and an estimated time-to-abandonment can be derived.

This aligns naturally with several existing design decisions: the mid-lifecycle snapshot timing, the developer-relative abandonment threshold, and the Watch tier's intentional ambiguity (a survival curve makes "ambiguous" concrete — it means the confidence interval is wide).

Requires: switching objective to `survival:aft`, redefining labels as `(time, event/censoring_indicator)` pairs, re-deriving scorecard tier thresholds against predicted survival curves rather than point probabilities. Significant enough to warrant its own design document before implementation.

---

## Testing

Agent behaviour is tested with DeepEval. `tests/agents/fixtures.py` includes:
- The Never Mourn case (hollow announcement, high-confidence fake heartbeat)
- A hotfix series (genuine development momentum)
- Edge cases (zero reviews, single announcement, maximum null features)

**Deterministic tests** (no LLM) cover `compute_signal_alignment` — the pure Python alignment node in the Critic Agent. These run without any API key.

**Live tests** (`@pytest.mark.live`) test full agent behaviour end-to-end. Auto-skipped if `CEREBRAS_API_KEY` is unset, so CI doesn't fail in environments without credentials.

```bash
# Deterministic tests (no API key needed)
python tests/run_tests.py -m not_live

# Full agent tests (requires CEREBRAS_API_KEY)
python tests/run_tests.py -m live
```
