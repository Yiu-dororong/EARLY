# Data Pipeline

## Overview

The pipeline ingests the full Steam catalog, filters to Early Access games with sufficient history, builds a weekly snapshot per game, and produces labelled training data from resolved outcomes. It runs entirely on GitHub Actions — no persistent compute, no managed infrastructure.

```
        │ pipeline_discovery.py
Steam catalog (160k+ apps)
        │
        ▼ collect_ccu_history.py (also visit Steam Reviews API)  
Filter: EA games with ≥ 90 days history & ≥ 50 Review & Paid Games -- discover.yml
        │ 
        ▼ collect_review_history.py, collect_events.py, collect_genres.py
Collect CCU history · build events · review history 
        │
        │ label_outcomes.py (EXIT_SUCCESS / EXIT_ABANDONED / EXIT_SILENT / STAY_ACTIVE)
        ▼ compute_genre_price_medians.py
Derived data: label → aggregate
genres · price history (ITAD)
        │
        ▼ collect_pre2022_ea_games.py, compute_dev_features.py
Cross-game developer aggregates -- collect.yml
        │
        ▼ build_snapshots.py
One row per game per snapshot date
        │
        ├──► Scorecard · XGBoost (training data)
        │         
        │ inference.py
        └──► live_scores · live_snapshots (live scoring)
                  
```

*Note: `collect_ccu_history.py` is mixed with review threshold checking, which is not ideal for a clean structure. Future refactoring is planned.*

---

## Collection

**Steam catalog** — collected through Steam `IStoreService/GetAppList` API, batch response available.

*After that, all data have to be collect game by game*

**Gerneral Game Metadata** — collected through Steam `appdetails` API, information includes current review count, initial price.
> For Genre, `appdetails` is the primary source, fallback to community voted genre on storefront page if the first one fails.

**CCU history**  — collected through Steam Charts, a third party website, however, a substantial amount of games has **no available CCU history**.

**Event history** — collected via Steam event API endpoint (undocumented). Includes Steam event feed, filtered to types 12/13/14(build), 28(update announcements). Captures event title, body text, word count, and timestamp. This is the proxy for "did the developer ship something" — with the important caveat that event types are announcement categories, not depot-verified signals (see [Case Study](case-study.md)).

**Review history** — collected via Steam histogram API endpoint (undocumented). Includes weekly histogram of review scores and counts. Produces the rolling sentiment signals. Exact review can be collected by `appreviews` API.

**Price history** (ITAD API) — discount frequency and depth. Early deep discounting is a weak but real abandonment signal — developers sometimes discount aggressively when player interest is falling.


### Early Access Catalog Reconstruction

Graduated Early Access (EA) titles are exceptionally difficult to locate historically. Because the Steam API only exposes the *current* state of an application, a game's EA structural flags, storefront tags, and original EA release dates completely disappear from the live metadata once it graduates to a 1.0 release.

Third-party data alternatives present severe operational bottlenecks:

* **SteamDB:** Scraping their database represents a deliberate violation of their Terms of Service (ToS).
* **SteamSpy:** The underlying data models suffer from systemic reliability and API consistency risks.
* **Static Datasets (e.g., Kaggle):** Frozen, historical open-source dumps are architecturally impossible to maintain or sync with a live, production-grade tracking pipeline.

**The Review Histogram Solution** 

While the standard Steam Review API natively lacks chronological indexing (preventing direct queries for the oldest reviews, and looks for the `written_during_early_access` flag), the **Review History Histogram API** provides a reliable historical anchor. This endpoint exposes daily review counts beginning exactly on the date of the first user submission. By cross-referencing this timeline against the current storefront release date, a clear indicator emerges: if a game currently lacks an Early Access tag but contains a verified review history predating its official release date, it is mathematically proven to be a graduated EA title.

**Engineering Constraints & Mitigations**

* **Historical Boundary Offsets:** For legacy titles, the histogram API can occasionally round the initialization date down to the first day of that calendar month. This introduces zero distortion into our tracking universe, as the data pipeline strictly enforces a **post-2022 release filter**.
* **Zero-Volume Sparsity Noise:** For titles with exceptionally low engagement, a prolonged lack of reviews during the initial launch window can artificially push the histogram start date forward to the day the first review actually lands. To prevent this noise from corrupting historical release anchors, the **50+ review eligibility gate** implemented at the problem framing stage, filtered out extreme low-volume cases before they enter the dataset.

---

## Labelling

Games are labelled at exit from Early Access, or at observation cutoff. Three *outcome* classes: `EXIT_SUCCESS`, `EXIT_ABANDONED`, `EXIT_SILENT`.

>`STAYS_ACTIVE` is the progress state, not the *outcome* state, learn more about labels from the definition of the problem in [Premises](premises.md#what-abandoned-means-here)

The binary training label is: `1 = abandoned (EXIT_ABANDONED + EXIT_SILENT)`, `0 = EXIT_SUCCESS`.

**Class balance** is approximately 3:1 active vs abandoned across the ~1,600 labelled snapshots. `scale_pos_weight` in XGBoost is set to reflect this ratio. `EXIT_SILENT` is collapsed into EXIT_ABANDONED due to even more severe sample sparsity.

---

## Snapshot 

**Snapshot timing** — snapshots are mostly taken at the mid-lifecycle (not at launch or exit) in each game's Early Access lifecycle. This avoids the trivial case where a game looks healthy at launch and abandoned just before exit. Prediction near graduation is also trivial because of the potential existence of "1.0 release is coming soon" announcement.

To achieve this, snapshot timestamps are evenly sliced from the mid-lifecycle window (defaulting to percentires $P_{25} \text{ to } P_{70}$ of a game's lifespan), where longer-running Early Access games dynamically scale their snapshot volume: initializing at a default of $4$, increasing by $+1$ for every additional 180 days once a project passes 360 days of age, and capped at $8$ to prevent long-tail games from skewing model weights.

**Look-ahead Leakage** — To simulate a live production inference pipeline, features are engineered strictly using historical logs trailing behind the specific snapshot date.

---

## Developer Cross-Game Features

`compute_dev_features.py` builds aggregate signals at the developer level — e.g. how many of a developer's other EA games have been abandoned, average update cadence across their portfolio.

These features exist because a developer's track record is a real signal. A first-time developer abandoning a game is different from a studio with three prior successful releases.

**The ethical decision:** Not all developer identity features are adpoted. Cross-game developer penalisation — where a developer's past failures directly lower scores on a new game — was considered and deliberately excluded. A developer who abandoned one game under difficult circumstances shouldn't have that permanently encoded as a penalty against future projects. 

---

## API Serving Layer

All pipeline stages—from time-series retrieval to multi-agent critique—are exposed via a **FastAPI** backend using **Pydantic** validation contracts to ensure end-to-end schema safety.

To prevent automated scraper discovery and protect LLM token resources, the interactive OpenAPI specifications (`/docs`, `/redoc`) are completely disabled in production builds.

Core endpoints:

**System Integrity & Ingestion Health**

* `GET /health` — Retrieves global pipeline telemetry and system orchestration health metrics. 

*Standard container lifecycle probes (`/livez`, `/readyz`) are also available.*

**Catalog & Telemetry Retrieval**

* `GET /games` — Returns a paginated list of live monitored Early Access titles, supporting dynamic structural filtering.

* `GET /games/{appid}/features` — Pulls the latest raw, vectorized feature payload directly from the active `live_snapshots` table.

* `GET /games/{appid}/history` — Pulls the full historical time-series score array, tracking how a title's telemetry metrics have evolved across consecutive evaluation windows.

* `GET /games/{appid}/score` — Pulls the pre-computed full composite risk score paired with a detailed five-dimension mathematical breakdown.

**Agent Analysis**

* `POST /games/{appid}/analyse` — Triggers a multi-agent evaluation via **LangGraph** as a background task. The result will be cached in the database.

* `GET /games/{appid}/analysis` — Retrieves the latest cached analytical verdict and plain-language diagnostic summary generated by the agent layer.

**Vector Search Spaces**

* `POST /search/similar` — Projects the target title's (latest snapshot) feature profile into **Zilliz (Milvus)**, executing an Approximate Nearest Neighbor (ANN) cosine similarity search to surface historically similar anchor games' snapshots.

Here is a sharp, simplified version that retains all the defensive engineering details without the wordy preamble:

### Production Hardening

Downstream multi-agent evaluation invokes live LLM APIs, the system implements gateway security to prevent resource abuse:

* **Compound-Key Rate Limiting:** Traffic is throttled using a multi-factor key (`API-Key` + `Session-ID` + `Client-IP`). This blocks proxy-rotation bypasses and automated scripts while ensuring fair usage for real users.
* **Internal Token Authentication:** All frontend-to-backend communication requires strict internal token verification at the gateway level to reject direct third-party requests.
