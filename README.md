# EARLY — Steam Early Access Intelligence System

*THIS README IS NOT THE FINAL VERSION. ALL INFORMATION IS SUBJECT TO CHANGE*

**Detects early warning signs that a Steam Early Access game may be abandoned.**

EARLY monitors more than 1,000 active Early Access titles and draws on patterns from roughly 1,600 historically completed or abandoned games. It produces a distress risk score and a three-tier health label (**Healthy / Watch / At Risk**) to highlight games that are losing momentum — often well before the review scores reflect the decline.

<!-- INSERT: screenshot or short GIF of the Streamlit dashboard (dark theme, risk meters, health labels) -->

🔴 **[Live Demo](https://early-system.streamlit.app/)** &nbsp;&nbsp;|&nbsp;&nbsp; 📄 **[Technical Documentation](docs/technical.md)**

> **⚠️ Disclaimer**  
> EARLY is an independent, unofficial tool. It is not affiliated with Valve, Steam, or any game developers. <br/> All risk scores are statistical estimates which can be wrong. Use this for informational purposes only.

---

## The Problem

Our analysis of historical Early Access titles shows that roughly 30–40% never reach a full release. When development slows or stops, players are often left with an unfinished product they paid for.


Current systems make early detection difficult:

- Steam provides no public API to confirm whether an announced update actually shipped executable code.
- Important signals — player counts, changelog substance, developer response rates — are scattered across multiple endpoints with no unified view.
- Steam’s official warning only appears after 12 months of inactivity, long after the game has already lost momentum.
- Games falling below 10 monthly reviews lose their "Recent Reviews" metric, pushing Steam to show stale all-time averages that mask ongoing decline.
- Standard machine-learning approaches on time-series game data easily leak future information, inflating performance metrics and hiding real-world reliability issues.

EARLY was built to address these exact challenges.

---

## What It Does

EARLY ingests the full Steam catalog (160,000+ apps) through official APIs, filters to Early Access titles with sufficient history, and runs a weekly pipeline that generates a **distress risk score** and a three-tier health label (**Healthy / Watch / At Risk**) for each game.

For titles flagged as Watch or At Risk, an on-demand LangGraph agent layer cross-checks three independent sources: the ML model output, recent review sentiment, and the actual substance of developer announcements. This triangulation surfaces conflicts — for example, when a “major update” announcement contains little real progress while reviews indicate months of stagnation.


```mermaid
block-beta
    columns 5
    
B["Data Pipeline<br/>GitHub Actions"] space C["XGBoost + L1 Scorecard<br/>Weekly Inference<br/>~1000 Apps"] space D["FastAPI<br/>Turso + Zilliz (Milvus)"]
    
    space space space space space 

    G["LangGraph Agents<br/>Forensic • Auditor • Critic<br/>(On-demand)"] space E{"Watch / At Risk?"} space  F["Vector Search<br/>Similar Snapshots Lookup<br/>(On-demand)"] 

    space space space space space 
    
     I["Signal Triangulation + AI Verdict"] space H["Streamlit Frontend"]  space  space





    B --> C
    C --> D
    D -- "Scorecard Label" --> E
    D -.-> F
    E -- "Yes" --> G
    E -- "No" --> H
    G --> I
    F --> H
    I --> H
    style D fill:#60a5fa
    style H fill:#be5bf0,
```

---

## Quick Start

```bash
git clone https://github.com/Yiu-dororong/EARLY.git
cd early
cp .env.example .env

# [Optional] Only fill in GROQ_API_KEY if you want to run live AI analysis.
# All other cloud integrations (Turso, Zilliz, etc.) are bypassable out-of-the-box using the pre-seeded local fallback data.

docker compose up
```

API: `http://localhost:8000` &nbsp;|&nbsp; UI: `http://localhost:8501`

**Running Tests**

```bash
# Deterministic tests (no API key needed)
python tests/run_tests.py -m not_live
# Full agent tests (requires GROQ_API_KEY)
python tests/run_tests.py -m live
```

Agent tests use DeepEval. `fixtures.py` includes a fake heartbeat test case, a hotfix series, and edge cases. Deterministic tests cover `compute_signal_alignment` without any LLM call; live tests are auto-skipped if `GROQ_API_KEY` is unset.

---

## Key Insights & Results

**The hardest games to predict are also the ones with the least reliable data.** Games labeled At Risk average 13.6 missing features per snapshot, compared with 5.2 for Healthy games. This gap is surfaced directly in the API through a `data_quality` field (high/medium/low).

**Announcement signals can be misleading.** Even when Steam’s event API returns `build_id` or `build_branch`, these fields are optional. During testing we discovered cases where a standard update announcement (Type-13) was posted with no corresponding build. This finding drove a major redesign of the agent layer, shifting its role from explanation to active conflict detection.
<!-- INSERT: screenshot of Never Mourn game in the UI — Forensic Agent flagging event_state_mismatch, Critic verdict showing signal conflict -->

**Signal triangulation improves reliability.** Before any LLM call, the Critic Agent runs a deterministic check across three signals: ML state, review sentiment direction, and forensic substance score. When the signals disagree, the verdict explicitly states the conflict rather than forcing a single conclusion.
<!-- INSERT: screenshot or diagram of triangulation output — three signals, alignment result, verdict -->

**Model Performance & Validation Metrics (v1.3)**


| Evaluation Framework | AUC-ROC | PR-AUC | Lift Over Scorecard Baseline |
| --- | --- | --- | --- |
| **Standard Holdout Set** | 0.9096 | 0.7378 | **+0.2271** |
| **Time-Bounded Cohort**  | 0.8761 | 0.6533 | **+0.1577** |



**Risk Tier Classification Integrity**

The baseline heuristic scorecard is calibrated by tracking the final lifecycle snapshot of each game against its final outcome. This evaluation shows that while the system is highly reliable when identifying clearly Healthy titles, it requires extra scrutiny precisely where heuristic confidence is lowest:

| Risk Tier          | Final-Snapshot Agreement | Operational Takeaway |
|--------------------|--------------------------------------------------|----------------------|
| 🟢 **Healthy**     | 98.3%                                            | High-confidence identification of stable, actively progressing titles; false positives are minimal. |
| 🟡 **Watch**        | 75.7%                                            | Captures transitionary phases; represents an elevated risk profile with a lower probability of reaching distress. |
| 🔴 **At Risk**      | 51.3%                                            | Identifies titles experiencing communication gaps or abandonment; functions as a low-confidence heuristic triage step. |

*Note: Scorecard calibration is evaluated at a game’s terminal checkpoint to calculate the precise outcome agreement rate. For Healthy and Watch tiers, agreement measures successful full release. For the At Risk tier, agreement measures meeting the definition of distressed.*

<!-- INSERT: screenshot of Never Mourn case showing Forensic Agent flagging event_state_mismatch and Critic verdict highlighting signal conflict -->

---
## Tech Stack

| Layer | Tools |
|---|---|
| Data collection | Python, Steam Web API, ITAD API, Requests |
| ML model | XGBoost, scikit-learn, SHAP |
| Scorecard | Custom weighted engine |
| Agent layer | LangGraph, Groq (Llama 4 Scout + Llama 3.3 70B), Langfuse |
| Vector search | Zilliz (Milvus), cosine similarity, 25-dim SHAP vectors |
| API | FastAPI, Turso (libSQL) |
| Frontend | Streamlit |
| MLOps | MLflow model registry, PSI drift monitoring, DeepEval |
| Infrastructure | Docker, Docker Compose, GitHub Actions |

---

## Technical Deep Dive

**ML Design Decisions**

- **GroupKFold by `appid`**: All snapshots from the same game stay in one fold. This prevents temporal leakage where the model would see both early and late snapshots of the same title.
- **Dynamic threshold from OOF PR curve**: Instead of a fixed 0.5 cutoff, the classification threshold is chosen based on the precision-recall curve from out-of-fold predictions. This handles the class imbalance more effectively.
- **Cosine similarity on SHAP vectors**: Used instead of L2 distance so games failing for the same reasons cluster together regardless of score magnitude.
- **No imputation for missing values**: XGBoost natively handles nulls. We return dense SHAP vectors with `pred_contribs=True`; earlier mean-imputation attempts were removed as they added distortion.

**Agent Layer**

The agent system runs **on-demand only** for Watch and At Risk games and caches results until the `l1_state` changes or 14 days pass (to respect Groq rate limits).

- **Forensic Agent**: Analyzes the last 5 developer announcements for actual substance. Detects “fake heartbeat” updates that lack corresponding build changes and outputs `event_state_mismatch` flags.
- **Sentiment Auditor**: Compares recent versus historical reviews to produce a `sentiment_alignment` score that checks consistency with the current ML state.
- **Critic Agent**: First runs a fast deterministic alignment check across ML score, review sentiment, and forensic signals, then produces two plain-language verdicts: `consumer_verdict` and `developer_brief`.

**Review Quality Adjustments**

Reviews are processed with three corrections before sentiment scoring:
- Meme discount (high funny/helpful ratio on negative reviews reduces weight)
- CJK-aware length scoring (CJK characters weighted 2.5× due to information density)
- “Great Wall of Text” guard (deduplication and smart truncation at 300 characters)


---

## MLOps & Reliability

EARLY treats production ML reliability as a first-class concern. A four-stage maturity plan is in place, with Stages 1–3 currently implemented.

| Stage | Goal | Implementation |
|-------|------|----------------|
| **Stage 1 – Drift Monitoring** | Detect degradation before it impacts predictions | `monitor_drift.py` tracks PSI on the top-25 SHAP features, prediction distribution shift, null-rate by risk tier, and delayed label drift. Runs automatically after each inference in `score.yml` and writes results to the `drift_reports` table. |
| **Stage 2 – Model Registry** | Controlled, auditable promotion | Every training run logs parameters, metrics, and artifacts to MLflow. Promotion to Production is gated: new models must improve PR-AUC *and* per-tier outcome agreement compared to the current production model. When MLflow is not configured, the promotion step becomes a no-op stub. |
| **Stage 3 – Conditional Retraining** | Safe, drift-triggered updates | `retrain.py` can be triggered by schedule or drift thresholds. It trains a new model, runs it through the promotion gate, and only promotes if metrics hold. Defaults to `--no-auto-promote` (human review strongly recommended for early cycles). Automatically flags `--zilliz-rebuild-needed` if the SHAP top-25 feature contract changes. |
| **Stage 4 – Survival Modeling (Roadmap)** | Move beyond binary classification | Replace the current distress flag with XGBoost AFT (Accelerated Failure Time) survival analysis to predict *how much runway* an Early Access title has left. |


---

## Project Structure

```
early/
├── .github/workflows/     # discover.yml · collect.yml · score.yml (daily/weekly cron)
├── agents/                # LangGraph: orchestrator, forensic, auditor, critic
├── api/                   # FastAPI routers, services, schema
├── core/                  # Feature builder, inference engine (shared)
├── data/                  # Collection scripts (Steam API) + processing
├── frontend/              # Streamlit app
├── models/                # ML artifacts, SHAP top-25 contract, drift reference
├── tests/                 # DeepEval agent tests + fixtures
├── training/              # train_xgboost.py, scorecard, drift monitor
└── utils/                 # Langfuse client, ITAD client, MLflow client
```

---
## Important Notes

### Disclaimer
>* EARLY is an independent, unofficial analytical tool. It is not affiliated with, endorsed by, or connected to Valve, Steam, or any game developers.
>* All risk scores, predictions, and tier labels are statistical estimates based on historical patterns and publicly available data. Past performance does not guarantee future outcomes.
>* Predictions can be incorrect due to data limitations, concept drift, changes in developer behavior, or unforeseen events.
>* Self-fulfilling prophecy risk: Public visibility of risk signals could potentially influence developer or player behavior in ways that affect the predicted outcome.
>* EARLY should be used for informational and exploratory purposes only. It is not financial, investment, or purchasing advice.
>* Always verify information directly on Steam and perform your own due diligence. Do not base major decisions solely on this tool.
>* The authors assume no responsibility or liability for any decisions made based on EARLY's outputs.


### What This Is Not

>EARLY does not predict whether a game will be *good*. It detects signals that a developer has slowed or stopped meaningful development. A game can score Healthy and still be mediocre. A game can score At Risk and recover — the system flags it, not sentences it.

<!-- INSERT: example of a Watch game that recovered — score history chart showing trajectory reversal -->

---

*Built by <!-- INSERT: your name/handle --> · <!-- INSERT: year -->*
