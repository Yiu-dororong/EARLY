# ML Model

## Overview

A binary XGBoost classifier trained to distinguish games that will be abandoned from those that will reach a 1.0 release, using mid-lifecycle snapshot features. The model is one layer of a three-layer system — its output feeds the scorecard and triggers the agent layer, but no verdict is final until the agents have checked whether independent signals agree.

---

## The Core Design Constraint: Preventing Look-Ahead Leakage

This is the most important decision in the model design, and the one most likely to be wrong silently.

A game that eventually gets abandoned leaves signals throughout its history: update frequency slows, reviews shift, developer posts dry up. If you train on early snapshots and validate on later snapshots *from the same game*, the model learns from its own future. The metrics look fine — validation AUC is reasonable, the model generalises "well" — but it has actually learned the late-stage signals of games it already knows the outcome of. Deployed on genuinely unseen games, it would be miscalibrated.

**The fix:** `GroupKFold` with `groups=appid`. All snapshots from any one game stay in the same fold, always. The model never sees any snapshot from a game during training if that game appears in validation. This is non-negotiable and is the first thing checked in every training run.

### Subtle Leakage Risk: Null Features

While `GroupKFold` effectively isolates temporal leakage across titles, a second, subtler concern is whether the *distribution of missing features* itself leaks outcome information.

At Risk games average 13.6 null features per snapshot compared to 5.2 for Healthy games. If these nulls are a direct byproduct of abandonment—for instance, if third-party aggregators stop tracking a title as its metrics collapse—the model risks learning "high null density equals abandonment" rather than evaluating the actual behavioral telemetry. This risk is highly apparent in the `ccu_available` feature: the `UNAVAILABLE` cohort contains a disproportionately higher concentration of abandoned titles. This strongly suggests that platforms like Steam Charts cease data collection *because* a game’s active player count has already bottomed out—meaning the missing data itself is an artifact of a project that is already dead.

Three architectural factors contain and mitigate this leakage vector:

*   **Nulls reflect external platform constraints, not internal game decay.** The vast majority of nulls in the pipeline stem from structural platform limitations—such as low-volume data suppression by upstream APIs, unrecorded historical indices, or mathematically undefined derivations (e.g., division-by-zero errors during periods of flat activity)—rather than a direct consequence of a developer halting updates. 

*   **The null distribution is symmetric and actively monitored for drift.** Because both the historical bootstrap pipeline and the live weekly inference pipeline (`monitor_drift.py`) query the exact same downstream API logic, the missing data signatures remain structurally symmetric across training and production. The monitoring suite actively tracks null rates by health tier; any systematic shift in missing data signatures between the training baseline and live populations triggers an immediate drift alert, treating null-rate stability as a core data integrity signal.

*   **Sparsity uncertainty is surfaced directly to consumers.** Rather than masking the issue via synthetic imputation or suppressing the signal entirely, every API response exposes a calculated `data_quality` metric (High / Medium / Low) derived straight from the active snapshot's null count. A prediction generated over a highly sparse feature vector is explicitly flagged as less reliable at the point of consumption.

This layout does not pretend the problem is completely eliminated; the structural behavior is documented in [Signals, Limitations & Roadmap](signals-limitations.md) as a known characteristic of the underlying data. However, by combining an audit of null root causes, active drift monitoring, and explicit uncertainty surfacing, the system ensures it is managed rather than ignored.

---

## Feature Engineering

76 features are engineered for the final model, spanning five primary signal dimensions, contextual markers, and upstream heuristic metrics:

**Update Health** — `days_since_last_build_update`, `build_count_90d`, `max_hiatus_days`, `avg_changelog_word_count`. The changelog word count feature captures the *substance* of updates, not just frequency — a developer posting one-line patch notes every week scores lower than one posting detailed, descriptive monthly changelogs.

**Player Retention** — `ccu_vs_peak_ratio`, `ccu_floor_established` (binary), `ccu_avg_90d`, `days_since_ccu_above_100`. These features capture both the absolute magnitude of player activity and its downward or stabilizing trajectory.

**Developer Engagement** — `build_to_post_ratio` (builds per developer post — a lower ratio indicates highly communicative engineering loops), `days_since_last_dev_post`, `dev_posts_90d`. This isolates whether developers are maintaining an open dialogue or shipping silently.

**Sentiment** — `review_score_at_T`, `review_score_90d`, `total_review_count`. Evaluating both the long-run and short-run scores reveals localized deterioration — a game with an 80% all-time positive rating but a 40% recent score signals an immediate, acute rupture.

**Price & Market** — `price_vs_genre_median`, `early_deep_discount_flag`, `discount_frequency`, `price_trend_encoded`. Aggressive price dropping or erratic discounting early in the lifecycle serves as a strong economic proxy for development funding depletion.

**Cross-dimension & Context** — `ea_age_days`, `owner_estimate_at_T`, `genre_scope`, `review_update_divergence`. High-level structural anchors that condition how the raw telemetry should be interpreted relative to the game's lifecycle stage.

**Primary Genre Tracking¹** — One-hot encoded categorical vectors mapping the game's primary tags (e.g., *Action, RPG, Simulation, Strategy*). This allows the tree-based architecture to establish distinct baseline expectations by genre. This should be static and immutable, capturing the initial planned or dominant genre at launch.

<details>
  <summary> Difference from Genre Scope</summary>
  <p>
    <code>genre_scope</code> serves a separate structural purpose: it measures objective architectural complexity (e.g., an MMORPG carries a fundamentally higher baseline complexity than a Visual Novel), acting as a potential proxy to measure other signals such as expected update frequencies and relative pricing.
  </p>
</details>

**Scorecard (L1) Metrics** — The model ingests its own deterministic sibling metrics as advanced features, including the five individual dimensional scorecard tallies and the raw `composite_l1_score`. 

*Note: `owner_estimate_at_T` is derived via the **Boxleiter Method** (the industry-standard owner-to-review multiplier framework), which serves as the foundational data source for how our `OWNER_MULTIPLIER_TIERS` are defined.*

### The Ethics Exclusion

Developer cross-game features (track record, prior abandonment rate) were engineered and tested. They were excluded from the production model deliberately — not because they lacked signal, but because permanently encoding a developer's past failures as a penalty against new projects is a judgment call that the system shouldn't be making silently. The features are available; the decision not to use them is documented.

### Features That Didn't Make It

Two features showed high predictive power during early feature exploration but were completely stripped from the final architecture after structural auditing revealed obvious look-ahead leakage. 

#### 1. `ccu_unavailable`
*   **The Feature:** A binary flag tracking whether Concurrent User (CCU) data was missing from external tracking APIs at snapshot time $T$.
*   **The Leakage:** As noted in the structural null analysis, the *absence* of tracker coverage is heavily correlated with long-term project abandonment. Third-party trackers do not drop coverage randomly; they drop coverage because a game has hit zero active players and remained dead for months. Including this flag allowed the model to bypass behavioral modeling entirely—it simply read the external tracker's post-mortem drop behavior as a proxy for the game's final state, leaking the future outcome directly into mid-lifecycle training snapshots.

#### 2. `snapshot_pct`
*   **The Feature:** The percentage of time elapsed between a game's initial Early Access launch and the current snapshot date, relative to its total historical timeline ($T / \text{Total Lifespan}$).
*   **The Leakage:** This is a classic mathematical look-ahead trap. To calculate the denominator ($\text{Total Lifespan}$), the feature requires knowing the exact date the game *ended* its lifecycle—either by releasing as a 1.0 version or hitting its final abandonment point. In a live production setting, the total lifecycle length of an active, ongoing Early Access game is completely unknown. Training the model on `snapshot_pct` inadvertently anchors the snapshot relative to a future terminal date that the system could not possibly see at inference time, rendering live predictions completely invalid.
---

## Training Design

**GroupKFold Cross-Validation** — Evaluated using a 5-fold cross-validation scheme grouped strictly by `appid`. Out-of-fold (OOF) predictions are systematically compiled across all folds to serve as the baseline for downstream threshold calibration and error analysis.

**Dynamic threshold from OOF PR curve.** — The classification threshold is not 0.5. It is derived from the OOF precision-recall curve, optimising F1 on the training distribution. This correctly handles the 3:1 class imbalance — the threshold that maximises F1 under imbalance is almost never 0.5.

**Full Population Final Training** — The production model is trained on the complete, unified training and validation dataset (all folds combined). To eliminate overfitting, the tree count is structurally capped using the average `best_iteration` calculated across the cross-validation folds, maximizing data utilization while maintaining empirical regularization.

**Temporal Holdout Evaluation** — To enforce a rigorous final audit, a separate test set is completely isolated using an explicit temporal cutoff. This evaluates the system's generalization capacity not just across distinct titles (`GroupKFold`), but across forward-looking temporal shifts—the most demanding and realistic deployment condition. Given our baseline eligibility gate targets titles launched post-2022, a **2024 temporal cutoff** was established, maintaining an optimal sample volume while isolating a pure forward-looking test population.

**Hyperparameter Tuning** — Most hyperparameters are tuned by Optuna, the **75th percentile** stopping iteration of the folds is saved as an outlier-robust baseline (`base_trees`).
* **Production Stage (Full Retrain):** The final model is retrained on 100% of the data without early stopping. To prevent underfitting on the larger data volume, the final tree count is hardcoded using a $+10\%$ scaling buffer: 

**Hyperparameter Tuning** — Most structural hyperparameters are tuned via Optuna, while `n_estimators` is calculated by the 75th percentile (prevent skewness from early-stopping) of the cross-validation folds scaled by a $+10\%$ data-volume buffer.


---

## Metric Selection & Tiered Agreement 

**Primary Metric: OOF PR-AUC** — Precision-Recall AUC is utilized as the primary optimization metric rather than ROC-AUC. Because the dataset exhibits a 3:1 class imbalance, ROC-AUC is easily inflated by the massive True Negative population (since the majority of monitored games are not abandoned). PR-AUC forces the optimization loop to focus directly on precision and recall dynamics within the minority class, tracking performance where predictive failure carries real financial and operational costs.

**Per-Tier Outcome Agreement Contracts** — After the deterministic Scorecard maps snapshots into *Healthy / Watch / At Risk* classifications, the ML model's probabilities are cross-evaluated against true historical outcomes within each structural tier:
*   **Healthy Agreement:** $P(\text{Released} \mid \text{Scored Healthy})$ — Verifies the true negative retention rate.
*   **At Risk Agreement:** $P(\text{Abandoned} \mid \text{Scored At Risk})$ — Verifies minority class precision.
*   **Watch Agreement:** $P(\text{Abandoned} \mid \text{Scored Watch})$ — Monitored strictly as a baseline distress index, as the *Watch* tier is designed to isolate volatile, structurally ambiguous trajectories.

**System Promotion Gate:** These per-tier metrics are logged natively within MLflow during evaluation. To pass the production deployment gate, a challenger model cannot degrade any individual tier's historical agreement by more than 2 percentage points (2pp), even if its aggregate global PR-AUC increases. 
 
*Note: Unlike the Scorecard's internal agreement check—which evaluates only a game's final, terminal snapshot—the ML model evaluation ingests **all historical snapshots across a title's lifespan** to guarantee long-term risk calibration.*

---

## Explainability & Vector Core Contracts

**SHAP Feature Attribution** — Tree-SHAP analysis is executed over a representative 2,000-snapshot sample pulled from the training partition. The top 25 features ranked by mean absolute SHAP value ($|\text{SHAP}|$) are exported as `shap_top25_{MODEL_VERSION}.json`.

Rather than projecting unweighted raw data, the system extracts these top 25 SHAP values to construct optimized game profile vectors into the **Zilliz (Milvus)** Vector Search Space. Learn more in [Similarity Search](agents.md#similarity-search).

Our core attribution findings from the top 5 features indicate:

*   **Price Deterioration as a Terminal Proxy (`price_trend_encoded`):** Capturing 11.9% of total model variance, an aggressive downward price trend (frequent, deeper historical discounts) is the single most predictive feature. This aligns perfectly with a known industry pattern: developer teams frequently initiate desperate, high-frequency discount cycles as a final attempt to harvest liquidity before completely abandoning a project.

*   **Scorecard Synergy (`l1_update_health_score` & `l1_composite_score`):** Holding a combined ~17% of variance, the model gives substantial weights to deterministic scorecard (L1) aggregations. This proves that the engineered heuristic layer successfully wraps raw telemetry into high-value signals.

*   **Qualitative Footprints (`review_score_at_T` & `changelog_word_count_trend`):** Trailing review scores combined with declining substantive updates (shorter, less detailed changelogs) explain an additional 11% of variance. This confirms that long before a title explicitly halts updates, its operational footprint undergoes distinct, measurable structural compression.

---

## Reflection on Redundancy

Excluding the price trajectory, the top structural SHAP features heavily intersect with the operational domains monitored by downstream layers. This raises an immediate architectural question:

> *Are the ML model and the LangGraph agent layer looking into the exact same signals? Is this an expensive redundancy?*

**The answer is no. They execute entirely separate evaluation over identical raw data channels:**

- **ML Model measure quantitatively:** The XGBoost engine evaluates these features as unified mathematical signals across thousands of historical titles. It *calculates* that a changelog word count has dropped below an empirical boundary or that a sentiment vector is decaying at a velocity strongly correlated with past project failures. It identifies statistical patterns but remains completely blind to contextual nuance.

- **LangGraph Agents analyse qualitatively:** The multi-agent cluster is triggered *because* the machine learning engine flags its potential risk. The agents do not re-verify the metrics; instead, they audit the underlying content, substance, and sentiment of the raw announcements and players' reviews.

---

## Model Performance Metrics (v1.4)

| Evaluation Set | AUC-ROC | PR-AUC | Lift Over Heuristic Baseline¹ |
| :--- | :---: | :---: | :---: |
| **Test Set (Temporal Holdout, Post-2024)** | 0.9133 | 0.7341 | +0.2148 |
| **Validation Set (Time-Bounded Cohort²)** | 0.8659 | 0.6239 | +0.1187 |

***¹ Baseline Definition:** The standalone rule-based L1 Scorecard composite score PR-AUC. This explicitly measures the non-linear machine learning model's mathematical uplift over the static deterministic heuristic layers alone.*

***² Time-Bounded Cohort:** Outlier titles whose active Early Access lifespans exceed the 95th percentile of the baseline training distribution are excluded from cross-validation. This prevents long-tail entity distortions, ensuring stable, homogenous grouping metrics inside the `GroupKFold` cross-validation loops.*

---

## Error Analysis

To determine exactly why the model misclassifies, Tree-SHAP is executed exclusively over the temporal holdout error cohorts—segmenting feature attributions for False Negatives (FN) and False Positives (FP). 

The analysis reveals that the hierarchical order of error-driving features is almost perfectly symmetric with the global feature importance rankings. Rather than being tripped up by low-variance noise, the model's failure modes are localized entirely within its highest-gain signal channels. This statistical behavior highlights distinct operational blindspots:

| Feature Name | False Negative SHAP (`fn_shap`) | False Positive SHAP (`fp_shap`) | Potential Blindspot |
| :--- | :---: | :---: | :--- |
| `price_trend_encoded` | **0.2495** | **0.2113** | **FN:** Dead games that never drop prices (abandoned silently without a final cash-grab). <br>**FP:** High-velocity healthy games running aggressive, legitimate seasonal sale strategies. |
| `l1_update_health_score` <br><br> `changelog_word_count_trend` | **0.1976** <br><br> **0.1122** | **0.1609** <br><br> **0.1001** | **FN:** Games that maintain automated, empty "patch-bot" updates or shallow changelogs to mask true development death.<br><br>**FP:** Small teams undergoing lengthy, silent engine overhauls or deep development sprints where update frequency drops temporarily. |

Consolidating the insights from the [Reflection on Redundancy](#reflection-on-redundancy) section, the agent layer is explicitly designed to capture the qualitative nuance that numerical thresholds destroy, where these critical error vectors are trapped within our highest-variance feature channels in the model. 

---

## Wrong Turns

**L2 similarity metric (corrected to cosine).** The initial Zilliz collection used L2 (Euclidean) distance for nearest-neighbour search. This was wrong for SHAP vectors specifically: SHAP magnitude scales with `p_distressed` — a Watch game and an At Risk game driven by the *same* underlying causes would appear distant under L2 purely because the At Risk game's SHAP contributions are larger. Cosine similarity measures directional agreement — it clusters games failing for the same *reasons* regardless of how severely. The collection index was rebuilt with `COSINE`.

**Score Normalization Faults** — Early versions of the L1 Scorecard utilized an unstable normalization methodology. Individual dimensional scores frequently broke past their intended $[0, 1]$ bounds, and momentum metrics failed to center cleanly at $0$. 

During initial ablation studies, removing these raw L1 features actually *improved* the ML model's PR-AUC, which initially suggested the heuristic layer was introducing harmful noise. However, after revising the normalization algorithms, the L1 features earned their place. The validation of this fix is concrete: as shown in our SHAP analysis, the corrected `l1_composite_score` immediately jumped to become the 3rd most predictive feature in the entire machine learning model.