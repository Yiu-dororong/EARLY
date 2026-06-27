# EARLY — Steam Early Access Intelligence System

**AI decision support system for Steam Early Access game abandonment risk.**

![CI](https://github.com/Yiu-dororong/EARLY/actions/workflows/score.yml/badge.svg) [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/) [![Docker](https://img.shields.io/badge/Infra-Docker-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/) [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<sup>Decision support tool — not financial or purchasing advice. Not affiliated with Valve or Steam. All scores are statistical estimates. [Full disclaimer ↓](#disclaimer)</sup>

<p align="center">
    <kbd>
        <img width="1280" height="720" alt="EARLY dashboard demo" src="https://github.com/user-attachments/assets/2119a420-a810-4606-91cc-a13c4632e847" />
    </kbd>
</p>

🔴 **[Live Demo](https://early-system.streamlit.app/)** &nbsp;&nbsp;|&nbsp;&nbsp; 📄 **[Technical Documentation](docs/technical.md)**

---

## 🔥 Why This Matters

Roughly **30–40% of Early Access games never reach a full release**, leaving players with unfinished products they paid for. **The developers go quiet. The games sit on storefronts, still purchasable, indefinitely.** 

Existing discovery tools make early detection of this lifecycle decay incredibly difficult:

- **Delayed Alerts:** Steam's official warning only appears after **12 months of total inactivity** — long after project momentum has already collapsed.
- **Hidden Decline:** Games falling below 10 monthly reviews lose their "Recent Reviews" metric, hiding active player abandonment behind stale, historical all-time averages.
- **Ghost Changelogs:** Public Steam storefront APIs do not expose whether an announced text update shipped *actual code* or just empty words.
- **Siloed Intelligence:** Real-time player counts, changelog substance, and community engagement remain highly scattered with no unified structural view.

EARLY addresses these gaps by deploying a weekly predictive ML pipeline coupled with an on-demand agent layer, cross-checking complex behavioral signals that a single score or a traditional storefront metric would miss.

---

## ⚙️ Tech Stack

| Layer | Tools |
|---|---|
| Data collection | Python, Steam Web API, ITAD API, Requests |
| ML model | XGBoost, scikit-learn, SHAP, Optuna |
| Agent layer | LangGraph, Cerebras (`gpt-oss-120b` + `zai-glm-4.7`), Langfuse |
| Vector search | Zilliz (Milvus), cosine similarity on 25-dim SHAP vectors |
| API | FastAPI, Turso (libSQL) |
| Frontend | Streamlit |
| MLOps | MLflow model registry, PSI drift monitoring, DeepEval |
| Infrastructure | Docker, GitHub Actions, Render, Streamlit Cloud |

---

## 🔮 How It Works

EARLY ingests the full Steam catalog (160,000+ apps), filters to Early Access titles with sufficient history, and runs a weekly pipeline that produces a **distress risk score** and a three-tier health label — **Healthy / Watch / At Risk** — for ~1,000 active games.

For titles flagged **Watch** or **At Risk**, an on-demand LangGraph agent layer triangulates three independent sources — ML score, recent review sentiment, and developer announcement substance — to surface conflicts a score alone would miss.

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
    style H fill:#be5bf0
```

---

## 📈 Results

**Model Performance (v1.4)**

| Evaluation Set | AUC-ROC | PR-AUC | Lift Over Baseline |
|---|---|---|---|
| **Test Set (Temporal Holdout, Post-2024)** | 0.9133 | 0.7341 | +0.2148 |
| **Validation Set (Time-Bounded Cohort)** | 0.8659 | 0.6239 | +0.1187 |

**Risk Tier Accuracy (v1.1)**

| Tier | Final Snapshot Agreement | Notes |
|---|---|---|
| 🟢 Healthy | 98.8% | High confidence; false positives are minimal |
| 🟡 Watch | 82.8% | Captures transitionary phases |
| 🔴 At Risk | 52.7% | Low-confidence triage — flags for closer review, not a verdict |

Of the [20 most recently resolved titles](docs/ea_exit_validation.csv) in our tracked universe, 6 of 7 **At Risk** games were distressed and 3 of 4 **Watch** games reached full release. Two **Healthy**-labeled games (`Wool at the Gates`, `Kebab Chefs! - Restaurant Simulator`) exited successfully despite elevated scores — a reminder the model sees developer activity signals, not the full picture.

Hence, outputs from EARLY should be interpreted together as a cohesive decision support system rather than in isolation.

📄 *Validation methodology, leakage controls, prediction interpretation, and error analysis are in the [ML Model Documentation](docs/ml-model.md).*

---

## 🧠 Key Design Decisions

**Temporal holdout, not random split.** All games released after 2024 are held out entirely from training. Random splitting would let the model see future snapshots of the same game — inflating metrics without reflecting real-world reliability.

**GroupKFold by `appid`.** All snapshots from one game stay in one fold, preventing the model from learning both early and late states of the same title across folds.

**No imputation for missing values.** XGBoost handles nulls natively. Imputing missing values added distortion and was removed. Missing feature counts are surfaced in the API as a `data_quality` field — the hardest games to predict also have the least reliable data (avg. 13.6 missing features vs. 5.2 for Healthy games).

**Cosine similarity on SHAP vectors, not L2.** Games failing for the same underlying reasons cluster together regardless of absolute score magnitude — more meaningful than distance-based lookup.

**Deterministic pre-check before any LLM call.** The Critic Agent runs a fast signal alignment check across ML score, review sentiment, and forensic substance before invoking the LLM. When signals conflict, the verdict states the conflict explicitly rather than forcing a single conclusion. This also keeps operating cost at zero — LLM calls only fire when genuinely needed.

**Production-grade architecture on a bounded dataset.** EARLY runs on ~1,600 historical and ~1,000 active titles — a dataset that could be handled with SQLite and a flat array. The architecture reflects what this problem demands at scale: decoupled services, independent deployability, and replaceable components — every layer can be upgraded in isolation without cascading changes elsewhere.

*The hardest engineering problems were not model training, but preventing silent data leakage, detecting unreliable signals, and making uncertainty legible to users.*

---

## ⚠️ Limitations & Future Work

**Current limitations:**
- At Risk tier agreement is 52.7% — useful as a triage signal, not a reliable individual prediction
- Data sparsity: games in a declining stage have significantly more null values, lowering data quality and model confidence
- Announcement signals can be gamed; a developer posting "fake heartbeat" updates is detectable but not foolproof
- Model sees developer activity, not game quality — a healthy-labeled game can still be mediocre

**On the roadmap:**
- **Survival modeling:** Replace binary classification with XGBoost AFT (Accelerated Failure Time) to predict *how much runway* a title has remaining, not just whether it's at risk
- **Online learning:** Drift-triggered retraining is implemented (Stage 3); closing the loop with automated promotion requires more historical cycles to validate safely
- **Explainability:** Surface per-game SHAP breakdowns in the UI, not just the aggregate score

---

## 🚀 Quick Start

```bash
git clone https://github.com/Yiu-dororong/EARLY.git
cd early
cp .env.example .env
# Add CEREBRAS_API_KEY to enable live AI analysis (optional)
# All cloud integrations are bypassable with pre-seeded local data
docker compose up
```

API: `http://localhost:8000` &nbsp;|&nbsp; UI: `http://localhost:8501`

**Zero operating cost** — every service runs on a free tier (Cerebras, Zilliz, MLflow/Databricks, Turso, Render\*, Streamlit Cloud, GitHub Actions).

\* *Render is free to use after a one-time $1 setup verification.*
<details>
<summary>Running tests</summary>

```bash
# Deterministic tests (no API key needed)
python tests/run_tests.py -m not_live

# Full agent tests (requires CEREBRAS_API_KEY)
python tests/run_tests.py -m live
```

Agent tests use DeepEval with fake heartbeat, hotfix series, and edge case fixtures. Live tests auto-skip if `CEREBRAS_API_KEY` is unset.
</details>

---

## <a name="disclaimer"></a>⚠️ Disclaimer

EARLY is an independent, unofficial analytical tool. It is not affiliated with, endorsed by, or connected to Valve, Steam, or any game developers.

All risk scores, predictions, and tier labels are statistical estimates based on historical patterns and publicly available data. Predictions can be incorrect due to data limitations, concept drift, or changes in developer behavior. 

EARLY is a decision support tool — it surfaces signals and context to help make more informed judgments. It does not constitute financial, investment, or purchasing advice, and the author assumes no responsibility for decisions made based on its outputs. Always verify directly on Steam.

---

*Built by Fox Yiu · 2026*
