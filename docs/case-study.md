# Case Study — Never Mourn 

> *A game posted a build announcement. The model believed it. The agent didn't.*

---

## The Discovery

During the initial production inference run (`xgb_v1.3`), the pipeline flagged a title in the *At Risk* tier with an anomalously high distress score. While a high risk metric was standard, manual validation of the downstream forensic agent's output exposed a critical structural edge case in our quantitative data ingestion layer.

The game's Steam storefront displayed a native platform warning indicating the last deployment was over 14 months ago. However, the game had just published a fresh Steam announcement flagged as **Event Type 13**—a categorical marker that Steam's API explicitly maps to a "Build Update." Because our engineering pipeline parsed this event flag blindly, the model’s update-recency feature (`days_since_last_build_update`) was prematurely recalculated from that announcement, tricking the machine learning layer into evaluating the title as an actively sustained project.

The forensic agent extracted the raw text of the announcement. The title read: "**Raise the Dead… Your Undead Army Now Comes with Trading Cards!**"

The payload analysis revealed a mismatch: the announcement body contained zero patch notes, no version increments, no deployment logs, and no engineering changelogs. 

The game was not actively developed. The model thought it was.

---

## What This Broke

The model's update-health features are built on Steam event types as a proxy for build activity. This is the best available signal — Steam exposes no public API for direct depot verification, so event types are the standard telemetry source. 

### Wait, can we just track Steam Depot Numbers?

One can technically track changes to a game's manifest and depot numbers via public scraping methods to verify if files were actually changed. However, implementing this as a primary feature channel introduces massive architectural and infrastructure bottlenecks: 

- **Severe API & Infrastructure Bottlenecks:** Tracking *real-time* demands high-frequency, aggressive polling against Steam's content delivery networks. Building and maintaining this tracking infrastructure at scale introduces severe platform rate-limiting risks.

- **Weak Signal Measurement:** Relying purely on depot changes exposes the system to direct exploitation. A developer looking to artificially inflate their operational health metrics can easily game this feature by pushing empty or low-substance modifications—such as uploading a token 1MB asset file. This will also capture noises from updating a localized text file, or swapping out storefront graphic banners. It completely fails to answer the core qualitative question: *Is this a genuine game update?*



Relying on Steam Event as a proxy for build health remains the only viable strategy. 

The assumption embedded in that proxy is: *developers posting build-type (12/13/14) announcements are, on average, actually shipping builds.* This assumption is mostly true. But "mostly true" means the failure mode exists, and when it fails, it fails in the direction of making a stalled game look active.

The *Never Mourn* case study perfectly illustrates this systemic vulnerability. The developers had unwittingly exploited this structural loophole—not maliciously, probably, but the downstream architectural impact was identical. A hollow announcement posted in the right category was enough to move the model's features in the false "Healthy" direction, completely masking the project's operational decay.

This was not a bug in the model. The model was doing exactly what it was trained to do. The bug was in the assumption that event types and build activity are the same thing.

---

## What It Did Not Break

Before deciding what to change, it was important to be precise about what was actually wrong.

- **The training data is internally consistent.** Both the training labels and the training features derive from the same event-based definition of "active." A game labelled `STAYS_ACTIVE` because it was posting events is evaluated on features that count those events. The model's learned relationship between events and outcomes is consistent — it's just that the underlying proxy has a failure mode.

- **The failure mode is the exception, not the norm.** Developers who are genuinely making progress post events. The correlation is real. The Never Mourn pattern — hollow announcements with no real development — is real but rare.

- **Retraining would not fix it.** Even with a larger dataset, the model cannot distinguish a real build announcement from a hollow one using the features available at the ML layer. The text of the announcement is not in the feature set. This is not something a better model can solve — it requires a different kind of signal.

---

## The Architectural Response

The insight that came from the Never Mourn case was not "fix the model." It was: **this is exactly what the agent layer is for.**

Before Never Mourn, the Forensic Agent's role was loosely framed as "interpret the build announcements." After, it was reframed precisely: **detect discrepancies between what the event type implies and what the announcement content actually contains.**

This reframing changed three things:

**1. The Forensic Agent gained new output fields.**

`event_state_mismatch` was added — a flag for exactly the Never Mourn pattern: announcement type implies a build shipped, but content analysis doesn't support it. `momentum` was added to capture the pattern across multiple posts, not just the most recent one.

**2. The Critic Agent gained a deterministic alignment node.**

`determine_alignment()` — a pure Python function, no LLM — now runs before any verdict is written. It takes structured fields from all three agents (ML state, sentiment alignment, forensic substance score, fake heartbeat flag, event state mismatch) and computes `signal_alignment`. Both LLM verdict calls are given this alignment result explicitly.

If a game released a hollow announcement recently, run through this system, would produce:

- Distress state: Watch (moderate distress)

- Forensic: substance_score = 2/10, event_state_mismatch = True

- Sentiment: declining, conflicts with Watch (closer to At Risk)

- Alignment: **conflicted**

- Verdict: explicit acknowledgment that the build announcement is not supported by announcement content, with a confidence note flagging the conflict

**3. The agent layer's purpose was restated.**

The original framing: "agents interpret the ML score." <br/> The revised framing: "agents detect when independent signals disagree."

When all three layers agree, the verdict is high-confidence. When they conflict, the verdict says so — and says why. This is the difference between a system that produces outputs and one that knows when not to trust itself.

---

## The Broader Implication

Never Mourn exposed a class of games, not just one game: developers who are aware (consciously or not) that posting Steam announcements in the right category keeps engagement metrics alive. These games are exactly the ones where the ML model is most likely to be wrong, and exactly the ones where players most need accurate information.

The `data_quality` field added to the API response (high/medium/low, derived from null feature count) was also partly motivated by this case — making the model's confidence visible to downstream consumers, so a Watch verdict with `data_quality: low` and `signal_alignment: conflicted` reads very differently from a Watch verdict with `data_quality: high` and `signal_alignment: positive`.

---

## What's Still Not Solved

**Depot verification** remains impossible with the public Steam API. The system cannot confirm that a build was actually pushed. It can flag hollow announcements through forensic analysis, but a developer who posts detailed, plausible-sounding patch notes for a build that didn't ship would still fool the Forensic Agent.

**The training data still contains Never Mourn-style games.** Some fraction of the ~1,600 labelled training snapshots include games with hollow announcements counted as active. The model's learned relationship between events and outcomes includes this noise. Cleaning it would require manually reviewing announcements at scale — a project in itself.

Both of these are documented in [Signals, Limitations & Roadmap](signals-limitations.md).
