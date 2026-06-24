# The Problem & Design Premises

> This page states the argument EARLY rests on — 
> what the problem is, why it's worth solving, what assumptions the solution makes, 
> and what would have to be false for the whole thing to fall apart.

---

## The Problem 

Steam Early Access lets developers sell a game before it is finished — here is the exact quote in [Valve's own guidelines](https://partner.steamgames.com/doc/store/earlyaccess?language=english) *(Living Document, accessed June 2026).*
> **What Is Early Access?** <br/>
> Steam Early Access enables you to sell your game on Steam while it is still being developed, and provides context to customers that a product should be considered "unfinished." Early Access is a place for games that are in a playable alpha or beta state, are worth the current value of the playable build, and that you plan to continue to develop for release. 
>Releasing a game in Early Access helps set context for prospective customers and provides them with information about your plans and goals before a "final" release.

This implied development will continue toward a full release. In practice, a meaningful fraction of Early Access games stop being developed without formally saying so. Updates slow. The developer goes quiet. The game sits on storefronts, still purchasable, indefinitely.

<img width="972" height="413" alt="image" src="https://github.com/user-attachments/assets/72c00435-683a-454d-81c0-3fed0b4992d4" />

<sub id="steamdb-attribution"><i>Source: SteamDB, retrieved June 2026.</i></sub>

According to [SteamDB](https://steamdb.info/stats/releases/?tagid=493), Early Access releases have surged dramatically, climbing from roughly 1000 annual titles in 2020 to more than 2500 in 2025. However, there is no official signal for this. Steam does not "flag" stalled games, there is only a line of warning appears after 12 months of inactivity. Review scores lag — players often don't update reviews when a game goes quiet, and a 70% positive score from 18 months ago says nothing about whether the developer shipped anything last quarter. By the time a game appears obviously abandoned, the signal has been there for months.

EARLY is an attempt to read that signal earlier.

---

## Theoretical Framework

### Model Design

<details>
    <summary> 
        <b>
            Premises
        </b>
    </summary>

The model's design is grounded in five structural premises about Steam Early Access:

**Premise 1 — Decentralized Risk Assessment**

The platform's operational framework decentralizes product validation. By explicitly informing buyers that "**Games in Early Access are not complete and may or may not change further.**" the platform establishes a boundary where development outcomes are unpoliced. Consequently, the burden of evaluating a project’s operational trajectory and long-term viability is placed entirely on the external observer.
<img width="791" height="139" alt="Steam's Note on Early Access Game" src="https://github.com/user-attachments/assets/72205353-ef76-4b18-a047-9d0c0f3388a6" />

**Premise 2 — The Developer Commitment**

The term "Early Access" is itself a semantic contract. **The word "early" is only meaningful if a later state — a finished 1.0 — is implied as the destination.** Steam's developer guidelines reinforce this by establishing an expectation of completion, making the commitment implicit in the product's own name before any policy is read.

**Premise 3 — The Consumer Expectation**

Consumers purchase Early Access titles based on the financial and emotional expectation that the developer's intent to reach 1.0 will be fulfilled, even though no such guarantee exists contractually.

**Premise 4 — The Operational Infrastructure**

Steam provides a CI/CD ecosystem — public depots, patch notes, and event hubs — specifically designed to facilitate, track, and prove iterative development. This infrastructure produces a continuous, objective, machine-readable record of developer activity.

**Premise 5 — The Economic Enforcement**

[Steam's Update Visibility Rounds](https://partner.steamgames.com/doc/marketing/visibility/update_rounds?language=english) provide front-page placement specifically tied to shipping actual game updates, making CI/CD activity the primary mechanism for re-engaging an existing audience. Each product receives only five rounds total, shared across EA and full release. A game that stops shipping updates loses access to this re-engagement tool — and without new revenue spikes to trigger algorithmic featuring, organic discovery collapses. Developmental silence therefore mechanically guarantees commercial death.
</details>

<u>
    <b>
        Model's Thesis
    </b>
</u>

Since risk assessment is structurally decentralized while the market retains an expectation of an eventual 1.0 release, and because a prolonged stagnation in developmental activity mechanically guarantees product death via platform economics, an objective audit of operational telemetry provides the most accurate window into an Early Access game's survival.

**Therefore, EARLY does not attempt to measure developer intent, capability, or morality. It mathematically audits operational momentum to predict structural failure.**


### Eligibility

<details>
    <summary> 
        <b>
            Foundation
        </b> 
    </summary>

**The Feedback Loop** (See our [model](signals-limitations.md))

Steam's own [FAQ](https://store.steampowered.com/earlyaccessfaq/?l=en) defines Early Access as an interactive feedback loop between developers and players. EARLY's eligibility gates are grounded in this definition: **a game without sufficient review count has no evidence that the feedback loop ever formed.** This is the theoretical foundation for the **50-review** ml_eligible gate and the 10-review discovery gate — not a statistical argument.

Low-review games are excluded because Steam's algorithm and psychological popularity bias ensure that only games with sufficient engagement traction enter the feedback loop the model is designed to measure. EARLY's valid domain is games with enough community presence to have something to lose.

**Maturity Gate**

A functioning feedback loop possesses inherent operational latency: the players must try the game, a developer must read community reviews, synthesize the feedback, implement code changes, and deploy an update. This cycle takes significant real-world time. Within the first 90 days of an Early Access launch, observed activity is almost entirely non-reactive—consisting of pre-planned launch patches and immediate structural hotfixes. **Enforcing a 90-day aging gate ensures the game has existed long enough to execute a genuine iteration of this loop.** 

Short EA examples include `Killer Klowns From Outer Space: The Game`, having a 7-day Early Access period on Steam, which is more likely to be a premium as a pre-order incentive. Hence, it does not meet our defined eligibility.

**Exclusion of Free-to-Play Titles**

Free-to-play (FTP) titles are completely removed from the model's scope. The economic and psychological feedback loops governing free games are structurally distinct from premium, paid titles. Without an upfront financial transaction, player acquisition costs, community retention behavior, and monetization strategies (like microtransactions) follow a different set of rules. Crucially, the developer's commercial risk and incentive structures are asymmetric compared to premium titles, rendering their operational telemetry incompatible with a unified baseline risk model.

**Pandemic Telemetry Anomalies**

The machine learning model strictly trains on and evaluates cohorts launched from 2022 onward to insulate it from severe macroeconomic and behavioral data distortions. The 2020–2021 pandemic era introduced unprecedented "black-swan" anomalies into the Steam ecosystem: home lockdowns triggered artificially inflated concurrent player peaks, hyper-extended or disrupted development timelines, and highly erratic consumer spending patterns. Training on this volatile historical period would permanently skew the model’s understanding of structural decay. **Restricting the dataset to post-2022 titles ensures the model evaluates behavior under the stabilized, modern baseline reality of the marketplace.**
</details>
<u>
    <b>
        Summary
    </b>
</u>


In short, to qualify for active predictive scoring within the EARLY engine, a title must concurrently meet four baseline criteria:

| Criterion | Required Threshold | Primary Core Rationale |
| --- | --- | --- |
| **Engagement Threshold** | 50+ user reviews | Verifies that a structural player-developer feedback loop has stabilized. |
| **Operational Maturity** | 90+ days in Early Access | Accounts for feedback loop latency; filters out brief pre-order marketing windows. |
| **Monetization Model** | Paid titles only | Excludes Free-to-Play titles due to highly asymmetric commercial risk profiles. |
| **Temporal Cohort** | Launch date of 2022 or later | Insulates model weights from volatile 2020–2021 pandemic telemetry distortions. |


---

## What "Abandoned" Means Here

Operationally, EARLY defines three exit states:

| Label | Meaning |
|---|---|
| `EXIT_SUCCESS` | Game graduated to 1.0 — confirmed via appdetails graduation_date |
| `STAYS_ACTIVE` | Build activity within allowable gap¹ — open/healthy label, excluded from training |
| `EXIT_ABANDONED` | No build AND no dev post for >allowable gap — total silence |
| `EXIT_SILENT` | No build for >allowable gap, but dev has posted within 365 days — collapsed into EXIT_ABANDONED for training |




`EXIT_SILENT` is collapsed to `EXIT_ABANDONED` at training time. The distinction matters for analysis; it doesn't change the binary classification task (will this game finish, or won't it).

This definition is deliberately observable. "Abandoned" here means *demonstrably stopped* — not "failed commercially," not "bad game," not "developer gave up privately." A game the developer has mentally abandoned but is still occasionally patching is not abandoned under this definition. That is a limitation, not an oversight.

*¹ Adaptive Developer-Relative Build Gap: a dynamic threshold by calulated by developers' historical update frequency.* *(Limitations see [here](signals-limitations.md))*<br/> 

<details>
    <summary> 
        Formula of allowable build gap
    </summary>
    
    MIN_EVENTS_FOR_PERSONAL_THRESHOLD = 5
    FLOOR_DAYS = 365
    TOLERANCE_MULTIPLIER = 1.5

    def compute_allowable_build_gap(historical_build_gaps: list[int]) -> int:
        if len(historical_build_gaps) < MIN_EVENTS_FOR_PERSONAL_THRESHOLD:
            return FLOOR_DAYS
        return max(FLOOR_DAYS, int(median(historical_build_gaps) * TOLERANCE_MULTIPLIER))
</details>

### The Long Hiatus Problem

One edge case this definition does not resolve cleanly: a game that goes dark for an extended period and then ships a 1.0 release. Under the current scheme it is labelled `EXIT_SUCCESS`, but any snapshot taken during the hiatus looks indistinguishable from an abandoned game — no updates, no communication, declining player counts.

This creates a genuine **labelling ambiguity** in the ML model: the training data contains hiatus-period snapshots labelled as successes, which weakens the model's ability to call At Risk confidently.

The scorecard is partly a structural response to this. Because it is calibrated against **final-snapshot** outcome agreement — not intermediate snapshots — it measures current development momentum rather than long-run trajectory. A game actively in hiatus scores poorly on the scorecard regardless of what ultimately happens. A game that returned from hiatus and shipped will score well at its final snapshot.

This produces a meaningful division of responsibility between the two layers:

- **Scorecard** → *is this game in trouble right now?* 
    Current momentum, last 30–90 days. Correctly flags an active hiatus as risky regardless of the eventual outcome.
- **ML model** → *do games with this profile tend to succeed long-term?*
  Historical pattern similarity, trained on resolved outcomes.

When they agree, confidence is high. When they disagree — a game the scorecard flags as At Risk but the ML model rates as low-distress, or vice versa — that disagreement is itself a signal, and is exactly what the agent layer surfaces. See [Agents](agents.md).

The long hiatus case is not fully solved. The 2×2 outcome matrix in the roadmap addresses it more directly by treating long-gap-success as a distinct class rather than forcing it into either label. See [Signals, Limitations & Roadmap](signals-limitations.md).

### Observability Gap for Build Activity

The core data constraint shaping this system is the lack of public access to Steam **depot changes history** (the actual build files). Instead, the data pipeline must rely on **announcement event types** (self-assigned categories like Type 12/13/14 for "patch notes") as a proxy for developer activity.

A developer can publish a "Build Update" announcement without pushing actual code. At the feature level, the ML model cannot distinguish a real release from a hollow text post. This is not a gap that can be closed with more data or a better model. It is a fundamental limitation of what Steam makes publicly observable.

The agent layer exists partly because of this constraint. The Forensic Agent reads announcement *content* to check whether it supports the implied event type — a check the ML model cannot perform. See [Never Mourn](never-mourn.md) for the case that made this explicit.

---

## What EARLY Does Not Claim

Stated once, here, before anything else:

- EARLY does not predict whether a game will be **good**.
- EARLY does not predict **commercial success** or sales performance.
- EARLY does not predict developer **intent** — only observable behaviour.
- EARLY does not have access to **private** developer communications, internal builds, or financial runway.
- A game can score **Healthy** and still be abandoned next month.
- A game can score **At Risk** and ship a full release tomorrow.

The system produces probability estimates from observable signals. It is a monitoring tool. The agent layer's `confidence_note` and the `signal_alignment` field exist specifically to make this uncertainty visible at the point of consumption, not buried in documentation.
