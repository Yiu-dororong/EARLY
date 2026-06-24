# Scorecard 

## Overview

The scorecard is the first layer(L1) of the system. It takes some of the features from snapshots that ML model used, applies a transparent weighted formula, and produces both a `l1_composite_score` (0–1) and a `l1_state` label (Healthy / Watch / At Risk).

Its outputs serve three roles: 
- Surface directly in the UI as primary labelling, lightweight and interpretable dimension bars 
- Triage Watch and At Risk titles for the agent layer  
- Act as a calibration check on the ML model — if the two systems consistently disagree on a game, that is a signal worth investigating.

---

## Architecture

```
Raw features
     │
     ▼
Backbone score [0, 1]          ← 90-day anchored, stable
     +
Momentum delta [-1, +1]        ← 30-day anchored, directional
     │
     ▼
dimension_score = backbone + lr × momentum × (backbone if positive momentum else 1.0 - backbone)*
     │                                        *for mathematical stability
     ▼ (× dimension weight)
l1_composite_score [0, 1]
     │
     ▼
STATE_THRESHOLDS → Healthy / Watch / At Risk
     │
     ▼
Hard override check
(>365 day build gap + game >90 days old → force At Risk)
```

**Five dimensions**, each with a backbone and a momentum component:

| Dimension | Weight | What it captures |
|---|---|---|
| Update Health | 0.25 | Build frequency, changelog substance, hiatus history |
| Player Retention | 0.25 | CCU levels, floor, trajectory |
| Developer Engagement | 0.20 | Post frequency, build-to-post ratio |
| Sentiment | 0.20 | Review scores, velocity, recent delta |
| Price & Market | 0.10 | Discount patterns, genre-relative pricing |

Update Health and Player Retention comprise half of the total weight because they are fundamental to the Early Access lifecycle; build frequency directly determines abandonment risk, while player retention drives the development feedback loop. Price holds the least weight because it contains the least information about active game development. The remaining dimensions measure responses from both sides of the feedback loop. Learn more about the meanings of each dimension at [Signals](signals-limitations.md).

**Backbone features** — 90-day anchored, stable signals. What the game looks like right now.

**Momentum features** — 30-day anchored, directional signals. Whether the game is improving or deteriorating. Normalised to `[-1, +1]` (0 = neutral/steady).

**Momentum learning rates** — each dimension has its own `lr` controlling how much the 30-day trajectory can shift the backbone score. The numbers are calibrated, Update Health and Player Retention have `lr=0.15` (trajectory matters a lot) for example. 

This can provide distinction between a game that looks bad but is improving from a game that looks bad and is getting worse, which attempts to answer the question how is this game performing *recently*?

---

## Normalisation

Every feature maps to `[0, 1]` (backbone) or `[-1, +1]` (momentum) via a documented scale type in `scorecard_config.py`. Key types:

- `inverted_cap` — higher value = worse (e.g. `days_since_last_build_update`: 0 days → 1.0, 365+ days → 0.0)
- `log_cap` — log-scaled, for count features with diminishing returns (e.g. `build_update_count_last_90d`)
- `centered_ratio` — ratio centered at 1.0, for momentum (e.g. `update_frequency_trend`: ratio of last 90d vs prior 90d activity)
- `symlog_norm` — symmetric log for features that can go positive or negative (e.g. CCU slope)
- `inverted_distance` — penalises distance from an anchor, asymmetrically by side (e.g. `price_vs_genre_median` penalises underpricing more than overpricing)

The full scale definitions are in `scorecard_config.py`. Editing calibration parameters there — not in `scorecard.py` — is the intended workflow. `CONFIG_VERSION` is incremented on every calibration pass so every scored game can be traced back to the exact config that produced its score.

---

## State Thresholds

```python
STATE_THRESHOLDS = [
    (0.60, "Healthy"),
    (0.45, "Watch"),
    (0.00, "At Risk"),
]
```

First match wins (high to low). A game scoring 0.54 is Watch, not Healthy.

### Hard override 

A game is forced to be At Risk and receive a zero composite score if:

- `days_since_last_build_update > 365` AND
- `ea_age_days > 90`

This catches games that the composite score might still rate as marginal due to strong historical signals, but which have demonstrably gone dark.

**The Operational Trade-Off**

This hard override prioritizes **immediate lifecycle state accuracy** over long-term historical variance, introducing a deliberate architectural trade-off:

* **The Goal:** Forces an honest evaluation of a title's *current* operational performance, ensuring that defunct projects cannot hide behind exceptional multi-year legacy metrics.

*  **The Risk:** This logic assumes developer silence equals permanent abandonment. However, a small percentage of developers undergo extended, unannounced development hiatuses before a massive 1.0 drop. By forcing these titles into *At Risk*, the system accepts a small, calculated volume of **False Positives** (labeling a hibernating game as dead) to guarantee it flags genuine ghost projects early.

The system treats these silent periods as active operational risks. While a game may theoretically wake up from hibernation later, the system classifies a 365-day total build absence as a critical, unmitigated threat vector until code deployment resumes.

<details>
     <summary>
          <b>
          View Outcome Distribution by Composite Score Decile
          </b>
     </summary>
          <img width="1600" height="1113" alt="Outcome Distribution by Composite Score Decile" src="https://github.com/user-attachments/assets/5bf18d97-54ba-48ab-8871-2a141022f4b9" />

</details>


---

## Calibration History

The current v1.1 thresholds (0.60 / 0.45) are the result of iterative calibration against outcome data — not chosen a priori.

**Original design** had five tiers. This was collapsed to three after early runs showed the two middle tiers had near-identical outcome distributions. Adding tiers implied precision the data couldn't support.

**Threshold iteration** used per-tier outcome agreement as the target metric — not the composite score distribution. The question was: "of all games we label Healthy, what fraction actually succeeded?" and "of all games we label At Risk, what fraction were actually abandoned?" The thresholds were adjusted until the agreement rates were meaningfully differentiated across tiers.

**The Watch tier** is intentionally ambiguous. A game in Watch has roughly even odds of recovering or deteriorating. This is by design — Watch is a "flag for monitoring" signal, not a verdict.

---

## Config Versioning

`CONFIG_VERSION` in `scorecard_config.py` is incremented on every calibration pass. Every row in `scorecard` and `live_scores` stores the `config_version` that produced it. This means:

- Historical scores are always traceable to their exact config
- Recalibration doesn't silently invalidate comparisons between old and new scores
- The drift monitor's reference distribution is keyed to a specific `CONFIG_VERSION`

This is a small discipline that prevents a class of silent bugs that are very hard to debug after the fact.

---

## Evaluation

The baseline heuristic scorecard is calibrated by tracking the final lifecycle snapshot of each game against its final outcome. This evaluation shows that while the system is highly reliable when identifying clearly Healthy titles, it requires extra scrutiny precisely where heuristic confidence is lowest:

| Risk Tier          | Final&nbsp;Snapshot&nbsp;Agreement | Operational Takeaway |
|--------------------|--------------------------------------------------|----------------------|
| 🟢&nbsp;**Healthy**     | 98.8%                                            | High-confidence identification of stable, actively progressing titles; false positives are minimal. |
| 🟡&nbsp;**Watch**        | 82.8%                                            | Captures transitionary phases; represents an elevated risk profile with a lower probability of reaching distress. |
| 🔴&nbsp;**At Risk**      | 52.7%                                            | Identifies titles experiencing communication gaps or abandonment; functions as a low-confidence heuristic triage step. |

*Note: Scorecard calibration is evaluated at a game’s terminal checkpoint to calculate the precise outcome agreement rate. For Healthy and Watch tiers, agreement measures successful full release. For the At Risk tier, agreement measures meeting the definition of distressed.*

---

## What the Scorecard Is Not

The scorecard is not a replacement for the ML model. It has no concept of interaction effects between features — each dimension is computed independently. It cannot learn that "low CCU *combined with* high changelog word count" is a different pattern from "low CCU combined with empty changelogs."

The ML model captures those interactions. The scorecard provides interpretability and a human-auditable check. When they disagree significantly, that disagreement is itself a signal.
