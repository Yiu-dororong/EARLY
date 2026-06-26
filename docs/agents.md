# Agent Layer

## Overview

The agent layer is not a narration system. It does not explain the ML score in plain language. It is a **triangulation system** — it checks whether three independent signal sources (the ML model, raw review sentiment, and developer announcement content) agree, and produces verdicts that explicitly state when they conflict.

This distinction matters. A system that narrates the score tells you *what* the model decided. A system that triangulates tells you *whether to trust it*.

---

## Why On-Demand, Not Scheduled

The weekly `score.yml` cron scores ~1,000 games. Approximately 400+ are in Watch or At Risk at any given time — the only games eligible for agent analysis. Running LLM calls across all of them weekly would exhaust free-tier Cerebras rate limits and produce stale analysis (a game's agent output is only meaningful relative to its current signals).

**Design decision:** agents run on-demand only, triggered by a user action in Streamlit (`POST /games/{appid}/analyse`). Results are cached in the `agent_analysis` table and stay valid until `l1_state` changes or 14 days pass. A "Refresh Analysis" button bypasses the staleness check (`force=true`) but still respects the hard eligibility gate — Healthy games never run agents.

This is the single biggest architectural difference from the original plan. The cron handles scoring; users opt into interpretation.

### Text Trimming vs. Completeness
To ensure token budget stability and prevent pipeline crashes on text-heavy games, developer announcements and reviews are trimmed. The clear trade-off is completeness for deterministic cost and speed: while this guarantees predictable inference performance, it carries the slight risk of omitting details hidden deep within long text logs.

### Future Scale Optimization

While the current on-demand model protects our API rate limits, it introduces **user-facing latency** during active LLM evaluation cycles. 

To mitigate this as traffic scales, we can introduce a **Hot-Cache Priority Queue**. By tracking the traffic and look-up frequency of individual titles in our database, a separate asynchronous worker can proactively trigger agent runs for the top 10% most frequently searched *Watch* or *At Risk* games. This allows the system to deliver instantaneous cache hits for high-visibility titles while strictly preserving on-demand lazy evaluation for long-tail, low-volume games.

---

## Orchestrator

Coordinates all three agents under one Langfuse trace per analysis run. Trigger conditions:

```
l1_state in ("Watch", "At Risk")           ← hard eligibility gate

Forensic Agent:   always runs 
Sentiment Auditor: always runs if there is at least 1 recent review
Critic Agent:     always runs (synthesises whatever is available)
```

Note: While the architecture is logically modeled with a parallel fan-out structure, it is currently executed sequentially due to free-tier API rate limits (RPM/TPM throttling). In a standard production environment with a higher tier, the Forensic Agent and Sentiment Auditor branches would run concurrently to minimize total latency.

---

## Forensic Agent

**Purpose:** Evaluate whether the developer's announcements represent real development activity.

**Input:** Last 3 announcements within last 365 days (sorted by date DESC, event type as tiebreaker), each with `days_ago` so the agent can reason about staleness itself.

**Outputs:**
- `substance_score` (0–10): quality of development evidence in the announcements
- `fake_heartbeat_flag`: announcement implies a build shipped but content doesn't support it
- `momentum`: pattern across posts (accelerating / decelerating / inconsistent / insufficient_data)
- `event_state_mismatch`: the Never Mourn flag — event type implies build, content contradicts it

**Fast paths:**
- Empty announcement body → skip LLM call entirely, score = 0
- Secondary heuristic: `score < 4 AND word_count < 20` → force `fake_heartbeat_flag = True`

**Model:** Cerebras `gpt-oss-120b`, temp=0.0

**Sample Outputs:**

*Hiatus announcemet*
<p align="center">
    <kbd>
        <img width="947" height="146" alt="fake heartbeat example" src="https://github.com/user-attachments/assets/90cdb04b-1ad8-4807-a49d-cc69e3211c56" />
    </kbd>
</p>

*Small hotfix*
<p align="center">
    <kbd>
        <img width="953" height="122" alt="hotfix example" src="https://github.com/user-attachments/assets/fd1811bd-d711-46f4-8db8-7aebe3494749" />
    </kbd>
</p>

---

## Sentiment Auditor

**Purpose:** Cluster recent vs older reviews into thematic signals and determine whether review sentiment agrees with `l1_state`.

**Input:** Recent reviews (requires `ml_eligible=True`, i.e. ≥50 reviews), plus the game's current `l1_state`.

**Outputs:**
- `sentiment_shift`: improving / declining / stable / mixed / insufficient_data
- `key_concerns`: developer-facing pain points extracted from reviews
- `sentiment_alignment`: explicitly checks whether review direction agrees or conflicts with `l1_state`
- Narrative summary

**Fast path:** Zero reviews available → skip LLM call entirely.

**Model:** Cerebras `gpt-oss-120b`, temp=0.0 

### Review quality adjustments

Reviews are not taken at face value before they reach the Auditor. Three adjustments are applied during collection:

- **Meme discount** — high funny/helpful ratio on negative reviews reduces their weight. A review with 200 "funny" votes and 5 "helpful" votes is not the same signal as a review with 200 "helpful" votes.
- **CJK-aware length scoring** — CJK characters weighted 2.5× vs Latin characters. A 50-character Chinese review contains roughly the same information as a 125-character English review.
- **Great Wall of Text guard** — line-level and token-level deduplication before scoring length; smart sentence-boundary truncation to 300 chars. Prevents copy-pasted walls of text from inflating substance scores.

Sample output please see below Critic Agent section.

---

## Critic Agent

**Purpose:** Synthesise all available signals into two verdicts — one for players, one for developers.

**The deterministic alignment node (no LLM):** Before any LLM call, the orchestrator runs `determine_alignment()` — a pure Python function that computes `signal_alignment` from structured fields:

```
ML state (Healthy/Watch/At Risk)
+ Sentiment alignment (agrees/conflicts with l1_state)
+ Forensic substance score + fake_heartbeat_flag
+ event_state_mismatch flag
→ signal_alignment: strong_positive / positive / neutral /
                    conflicted / strong_negative
```

Both LLM verdicts are given `signal_alignment` explicitly in their prompts. The Critic does not derive alignment from unstructured text — it is told the alignment result and asked to incorporate it into the verdict.

**Outputs:**
- `consumer_verdict`: player-facing, plain language distress assessment
- `developer_brief`: actionable, developer-facing recommendations
- `confidence_note`: explicit caveats about data quality, ML eligibility, signal staleness

**Two separate LLM calls** (consumer and developer verdicts), each its own Langfuse span — different audiences, different tones, different prompt structures.

**Model:** Cerebras `zai-glm-4.7`, temp=0.3 (slightly higher than Forensic — verdicts benefit from some variation in phrasing)

**Sample Outputs:** 

*Triangulation clarify Watch label*
<p align="center">
    <kbd>
        <img width="968" height="897" alt="Triangulation from critic agent" src="https://github.com/user-attachments/assets/6e7d8db0-df66-4cb7-a284-cd20ca91dd90" />
    </kbd>
</p>


---

## Error Handling

- **Structured JSON Schemas via Pydantic**

All agent outputs are strictly bound using LangGraph `.with_structured_output` following Pydantic schemas. This guarantees that when a node executes successfully, the downstream payload is perfectly formed.

- **LangGraph Exception Catching & Visibility**

If an LLM node encounters a validation anomaly, rate limit, or generation failure, the graph does not return an empty state or crash. Instead, LangGraph explicitly catches the exception at the node boundary. The error payload is captured and rendered directly to the user interface. The user sees an honest error log rather than a blank screen. The Critic Agent will also aware the upstream failure by reading `forensic_ran` and `auditor_ran`.

- **Current Free-Tier Constraints**

**No Silent Degradation:** We intentionally avoid hiding failures behind safe, hardcoded fallback metrics. If an LLM node fails, the data stream explicitly reflects it. It is because the primary objective of the agent layer is to inject net-new signals from external sources—rather than merely repackaging our existing metrics—allowing a failed agent to silently fall back to the original baseline metrics adds zero structural value to the system. 

**No Auto-Retry (Yet):** Due to strict free-tier rate limits (RPM/TPM throttling), an automated retry loop is intentionally omitted to prevent immediate API lockouts. This constraint forces the pipeline to be a clean, single-pass evaluation framework where errors are documented rather than looped.

---

## Observability

Every `run_analysis()` call produces one top-level Langfuse trace tagged by `appid`. Each LLM call (Forensic, Auditor, Critic×2) is a `generation` span with input/output/token usage.

**The v4 setup:** The entire client can be set up simply by using `CallbackHandler()` from `langfuse.langchain`, passed via LangGraph's `config={"callbacks": [handler]}`. The no-op stub pattern is preserved — agents work standalone without Langfuse configured.

---

## Streamlit Integration

- Watch/At Risk games show an "Analyse" trigger button
- Polls `GET /games/{appid}/analysis` every 3 seconds (max 12 polls / ~36s)
- Status enum: `not_eligible | never_run | ready | error`
- Healthy games show an explanatory message instead of the analysis section — the hard gate is visible to the user, not just an invisible filter

---

## Similarity Search

Alongside the agent layer, each game detail view shows 5 historically similar games — Early Access games with known outcomes that failed (or succeeded) for similar reasons.

**Vector:** 25-dim SHAP contribution vector (top-25 by mean |SHAP|, covering >80% of model variance).

**Metric:** Cosine similarity. SHAP vector magnitude scales with `p_distressed` — a Watch game and an At Risk game driven by the same underlying causes would appear distant under L2. Cosine measures directional agreement, clustering games failing for the same *reasons* regardless of severity.

**Filter relaxation (3-pass):**
```
Pass 1: ea_age ±90d  + same primary_genre
Pass 2: ea_age ±180d + same primary_genre
Pass 3: ea_age ±180d, no genre filter
```
Deduplicates to 5 unique games (closest snapshot per appid). The progressive relaxation ensures results are returned even for niche genres, while preferring age- and genre-matched comparisons when they exist.

---

## Wrong Turns

**Hard date cutoffs in the Forensic Agent:** An early design fetched only announcements within a 60-day window. This had a critical flaw: At Risk games — the ones the agent is most needed for — often have *no* recent announcements, so the agent was never running for the games that mattered most. The fix was to relax the date cutoff to 365 days and always fetch the last N announcements, with `days_ago` per post passed to the agent so it can reason about staleness itself.

**Trigger condition (corrected from thresholds to labels).** The orchestrator originally triggered agent analysis when `composite < 0.50 OR any_dimension < 0.20`. These thresholds were arbitrary — re-deriving conditions that the scorecard had already computed and labelled. The correct trigger is `l1_state in ("Watch", "At Risk")`. Using the scorecard's own label avoids threshold drift and keeps the trigger semantically consistent with everything else in the system.

**Omission of Raw Review Translations** — When auditing community sentiment, the agent extracts and quotes user reviews to justify its conclusions. In early testing, many of these quotes were fetched in their native, non-English scripts, severely degrading the legibility and utility of the final report for end-users. The ingestion prompt was updated to enforce an inline translation contract: the agent is now instructed to detect non-English raw quotes and provide a clean, English translation alongside the original text, ensures evidence remains universally interpretable.

