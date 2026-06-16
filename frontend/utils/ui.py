"""
frontend/utils/ui.py
--------------------
Reusable HTML/CSS components and the global stylesheet.
All render functions return HTML strings for st.markdown(..., unsafe_allow_html=True).
"""

from __future__ import annotations

STYLESHEET = """
<style>
/* ── Base ─────────────────────────────────────────────────────────────── */
.stApp { background-color: #0d1117; }
.block-container { padding-top: 1.5rem; max-width: 1100px; }
h1, h2, h3 { color: #e6edf3; }

/* ── Cards ────────────────────────────────────────────────────────────── */
.card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
}
.card-title {
    font-size: 0.7em;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #7d8590;
    margin-bottom: 0.5rem;
}

/* ── State badges ─────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 2px 12px;
    border-radius: 20px;
    font-size: 0.78em;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.badge-healthy  { background: #238636; color: #fff; }
.badge-watch    { background: #9e6a03; color: #fff; }
.badge-at-risk  { background: #da3633; color: #fff; }
.badge-unknown  { background: #30363d; color: #7d8590; }

/* ── Pulse dot (At Risk signature element) ────────────────────────────── */
@keyframes pulse { 0%,100% { opacity:1; box-shadow:0 0 0 0 #da363344; }
                   50%      { opacity:0.7; box-shadow:0 0 0 5px transparent; } }
.pulse-dot {
    display: inline-block; width: 9px; height: 9px;
    background: #da3633; border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
    margin-right: 7px; vertical-align: middle;
}

/* ── Game header ──────────────────────────────────────────────────────── */
.game-title {
    font-size: 1.7em; font-weight: 700;
    color: #e6edf3; margin: 0; line-height: 1.2;
}
.game-meta { font-size: 0.85em; color: #7d8590; margin-top: 4px; }

/* ── Signal meter bars (signature element) ───────────────────────────── */
.dim-row {
    display: flex; align-items: center;
    gap: 10px; margin: 8px 0;
}
.dim-label {
    width: 145px; font-size: 0.72em; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.07em;
    color: #7d8590; flex-shrink: 0;
}
.dim-bar-bg {
    flex: 1; height: 5px;
    background: #21262d; border-radius: 3px; overflow: hidden;
}
.dim-bar-fill {
    height: 100%; border-radius: 3px;
    transition: width 0.4s ease;
}
.dim-bar-fill.green  { background:#238636; box-shadow:0 0 6px #23863688; }
.dim-bar-fill.amber  { background:#d29922; box-shadow:0 0 6px #d2992244; }
.dim-bar-fill.red    { background:#da3633; box-shadow:0 0 6px #da363344; }
.dim-value {
    width: 38px; text-align: right;
    font-size: 0.82em; font-family: monospace;
    color: #c9d1d9;
}
.dim-na { font-size: 0.78em; color: #30363d; }

/* ── Verdict / brief cards ────────────────────────────────────────────── */
.verdict-card {
    background: #161b22;
    border-left: 3px solid #388bfd;
    padding: 0.9rem 1.1rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.93em; color: #c9d1d9; line-height: 1.65;
}
.brief-card {
    background: #161b22;
    border-left: 3px solid #d29922;
    padding: 0.9rem 1.1rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.93em; color: #c9d1d9; line-height: 1.65;
}

/* ── Forensic badge ───────────────────────────────────────────────────── */
.forensic-score {
    font-size: 1.6em; font-weight: 700;
    font-family: monospace; color: #e6edf3;
}
.forensic-max { font-size: 0.9em; color: #7d8590; }

/* ── Sentiment cluster ────────────────────────────────────────────────── */
.cluster-row {
    display: flex; align-items: flex-start;
    gap: 10px; padding: 8px 0;
    border-bottom: 1px solid #21262d;
}
.cluster-label {
    font-size: 0.82em; font-weight: 600; color: #c9d1d9;
    width: 160px; flex-shrink: 0;
}
.cluster-freq {
    font-size: 0.72em; text-transform: uppercase;
    letter-spacing: 0.06em; color: #7d8590; padding-top: 1px;
}
.cluster-quote {
    font-size: 0.8em; color: #7d8590;
    font-style: italic; margin-top: 2px;
}
.valence-pos  { color: #238636; }
.valence-neg  { color: #da3633; }
.valence-mix  { color: #d29922; }

/* ── Concern list ─────────────────────────────────────────────────────── */
.concern-item {
    display: flex; align-items: flex-start;
    gap: 8px; padding: 5px 0;
    font-size: 0.85em; color: #c9d1d9;
    border-bottom: 1px solid #21262d;
}
.concern-dot { color: #da3633; flex-shrink: 0; }

/* ── Similar game rows ────────────────────────────────────────────────── */
.similar-row {
    display: flex; align-items: center;
    justify-content: space-between;
    padding: 9px 12px;
    background: #161b22; border: 1px solid #21262d;
    border-radius: 6px; margin-bottom: 5px;
    font-size: 0.85em;
}
.similar-name { color: #c9d1d9; font-weight: 500; }
.similar-meta { color: #7d8590; font-size: 0.9em; }
.outcome-success  { color: #238636; font-weight: 600; }
.outcome-abandoned { color: #da3633; font-weight: 600; }
.match-high   { color: #238636; }
.match-medium { color: #d29922; }
.match-low    { color: #7d8590; }

/* ── Confidence note ──────────────────────────────────────────────────── */
.conf-note {
    font-size: 0.78em; color: #7d8590;
    padding: 6px 0; border-top: 1px solid #21262d;
    margin-top: 8px;
}

/* ── Sidebar ──────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
}

/* ── Streamlit overrides ──────────────────────────────────────────────── */
div[data-testid="stMetricValue"] { color: #e6edf3; }
div[data-testid="stMetricLabel"] { color: #7d8590; }
.stTabs [data-baseweb="tab"] { color: #7d8590; }
.stTabs [aria-selected="true"] { color: #e6edf3; }
</style>
"""


# ---------------------------------------------------------------------------
# Component renderers
# ---------------------------------------------------------------------------

def state_badge(state: str | None) -> str:
    cls = {
        "Healthy": "badge-healthy",
        "Watch":   "badge-watch",
        "At Risk": "badge-at-risk",
    }.get(state or "", "badge-unknown")
    label = state or "Unknown"
    pulse = '<span class="pulse-dot"></span>' if state == "At Risk" else ""
    return f'<span class="badge {cls}">{pulse}{label}</span>'


def data_quality_badge(quality: str | None) -> str:
    colors = {"high": "#238636", "medium": "#d29922", "low": "#da3633"}
    labels = {"high": "High confidence", "medium": "Medium confidence", "low": "Low confidence"}
    c = colors.get(quality or "", "#30363d")
    l = labels.get(quality or "", "Unknown confidence")
    return (
        f'<span style="background:{c}18;color:{c};border:1px solid {c}44;'
        f'padding:2px 10px;border-radius:20px;font-size:0.75em;font-weight:600;">{l}</span>'
    )


def signal_bar(label: str, value: float | None) -> str:
    if value is None:
        return (
            f'<div class="dim-row">'
            f'<span class="dim-label">{label}</span>'
            f'<div class="dim-bar-bg"></div>'
            f'<span class="dim-na">N/A</span>'
            f'</div>'
        )
    pct   = value * 100
    cls   = "green" if pct >= 60 else "amber" if pct >= 35 else "red"
    return (
        f'<div class="dim-row">'
        f'<span class="dim-label">{label}</span>'
        f'<div class="dim-bar-bg">'
        f'<div class="dim-bar-fill {cls}" style="width:{pct:.0f}%"></div>'
        f'</div>'
        f'<span class="dim-value">{pct:.0f}%</span>'
        f'</div>'
    )


def verdict_card(text: str) -> str:
    return f'<div class="verdict-card">{text}</div>'


def brief_card(text: str) -> str:
    return f'<div class="brief-card">{text}</div>'


def cluster_row(cluster: dict) -> str:
    valence_cls = {
        "positive": "valence-pos",
        "negative": "valence-neg",
        "mixed":    "valence-mix",
    }.get(cluster.get("valence", ""), "")
    valence_sym = {"positive": "▲", "negative": "▼", "mixed": "◆"}.get(
        cluster.get("valence", ""), "●"
    )
    quote_html = ""
    if cluster.get("representative_quote"):
        quote_html = f'<div class="cluster-quote">"{cluster["representative_quote"]}"</div>'

    return (
        f'<div class="cluster-row">'
        f'<span class="{valence_cls}" style="font-size:0.9em;flex-shrink:0;">{valence_sym}</span>'
        f'<div>'
        f'<div class="cluster-label">{cluster.get("theme","")}</div>'
        f'<div class="cluster-freq">{cluster.get("frequency","")}</div>'
        f'{quote_html}'
        f'</div>'
        f'</div>'
    )


def concern_item(text: str) -> str:
    return (
        f'<div class="concern-item">'
        f'<span class="concern-dot">●</span>'
        f'<span>{text}</span>'
        f'</div>'
    )


def similar_game_row(game: dict) -> str:
    outcome     = game.get("outcome", "")
    outcome_cls = "outcome-success" if outcome == "SUCCESS" else "outcome-abandoned"
    quality     = game.get("match_quality", "")
    quality_cls = f"match-{quality}"
    age         = game.get("ea_age_days", 0)
    name        = game.get("name") or f"appid {game.get('appid')}"
    dist        = game.get("distance", 0)
    return (
        f'<div class="similar-row">'
        f'<div>'
        f'<div class="similar-name">{name}</div>'
        f'<div class="similar-meta">{age}d in EA · dist {dist:.3f}</div>'
        f'</div>'
        f'<div style="text-align:right">'
        f'<div class="{outcome_cls}">{outcome}</div>'
        f'<div class="{quality_cls}" style="font-size:0.75em;">{quality} match</div>'
        f'</div>'
        f'</div>'
    )
