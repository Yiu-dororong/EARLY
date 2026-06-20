"""
frontend/app.py
---------------
EARLY — Early Access Health Monitor

Run:
    streamlit run frontend/app.py

Env:
    API_BASE_URL   default: http://localhost:8000
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from frontend.utils import api, ui # noqa: E402, I001


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="EARLY",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(ui.STYLESHEET, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "current_appid":    None,
        "analysis_polling": False,
        "poll_count":       0,
        "similar_loaded":   False,
        "similar_results":  None,
        "browse_limit":     200,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCORECARD_STATS = None

def _get_scorecard_stats():
    global SCORECARD_STATS
    if SCORECARD_STATS is None:
        pattern = os.path.join(PROJECT_ROOT, "models", "scorecard_stats_*.json")
        files = glob.glob(pattern)
        if files:
            files.sort(reverse=True)
            try:
                with open(files[0]) as f:
                    SCORECARD_STATS = json.load(f)
            except Exception:
                pass
    return SCORECARD_STATS

def _map_dimension_score(dim_key: str, raw_value: float | None) -> float | None:
    if raw_value is None:
        return None
    stats = _get_scorecard_stats()
    if (not stats
        or "dimension_stats" not in stats
        or dim_key not in stats["dimension_stats"]):
        return raw_value  # fallback

    dim_data = stats["dimension_stats"][dim_key]
    p25, p50, p75 = dim_data["p25"], dim_data["p50"], dim_data["p75"]

    val = np.clip(raw_value, 0.0, 1.0)
    if val <= p25:
        mapped = np.interp(val, [0.0, p25], [0, 25])
    elif val <= p50:
        mapped = np.interp(val, [p25, p50], [25, 50])
    elif val <= p75:
        mapped = np.interp(val, [p50, p75], [50, 75])
    else:
        mapped = np.interp(val, [p75, 1.0], [75, 100])

    return float(mapped / 100.0)

DIMENSION_LABELS = {
    "update_health":    "Update Activity",
    "player_retention": "Player Retention",
    "dev_engagement":   "Dev Engagement",
    "sentiment":        "Sentiment",
    "price_market":     "Price / Market",
}

TOTAL_FEATURES  = 76
POLL_INTERVAL_S = 3
MAX_POLLS       = 12   # ~36 seconds before giving up


def _ts_to_date(ts: int | None) -> str:
    if not ts:
        return "—"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def _resolve_date_obj(date_str: str | None, ts: int | None):
    if date_str:
        try:
            return datetime.fromisoformat(date_str).date()
        except ValueError:
            pass
    if ts:
        return datetime.fromtimestamp(ts).date()
    return None


def _history_chart(snapshots: list[dict]) -> go.Figure:
    """Distress probability over time with threshold line."""
    dates = []
    for s in snapshots:
        dt = _resolve_date_obj(s.get("snap_date") or s.get("snapshot_date"),
                               s.get("scored_at"))
        dates.append(dt.strftime("%Y-%m-%d") if dt else "—")
    values = [s.get("p_distressed") for s in snapshots]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=values,
        mode="lines+markers",
        line=dict(color="#388bfd", width=2),
        marker=dict(size=5, color="#388bfd"),
        hovertemplate="%{x}<br>%{y:.1%}<extra></extra>",
        name="Distress Risk",
    ))
    fig.add_hline(
        y=0.5124,
        line_dash="dot", line_color="#852312",
        annotation_text="Threshold",
        annotation_font_color="#7d8590",
        annotation_position="bottom right",
    )
    fig.update_layout(
        plot_bgcolor="#161b22", paper_bgcolor="#0d1117",
        font=dict(color="#7d8590", size=11),
        yaxis=dict(range=[0, 1], tickformat=".0%",
                   gridcolor="#21262d", zeroline=False),
        xaxis=dict(gridcolor="#21262d"),
        margin=dict(l=0, r=0, t=10, b=0),
        height=220,
        showlegend=False,
    )
    return fig


def _navigate_to(appid: int):
    st.session_state.current_appid  = appid
    st.session_state.analysis_polling = False
    st.session_state.poll_count     = 0
    st.session_state.similar_loaded = False
    st.session_state.similar_results = None
    st.rerun()


def _go_back():
    st.session_state.current_appid = None
    st.rerun()


def _reset_limit():
    st.session_state.browse_limit = 200


@st.cache_data(ttl=600)
def _get_cached_health():
    return api.get_health()


@st.cache_data(ttl=600)
def _get_cached_games(state_filter: str,
                      min_reviews: int,
                      max_days_since_build: int,
                      search_name: str,
                      limit: int):
    kwargs: dict = {}
    if state_filter != "All":
        kwargs["l1_state"] = state_filter
    if min_reviews > 0:
        kwargs["min_reviews"] = min_reviews
    if max_days_since_build > 0:
        kwargs["max_days_since_build"] = max_days_since_build
    if search_name:
        kwargs["search_name"] = search_name
    return api.list_games(**kwargs, limit=limit)

@st.cache_data
def _build_dataframe(items: list[dict]):
    state_map = {
        "Healthy": "🟢 Healthy",
        "Watch":   "🟡 Watch",
        "At Risk": "🔴 At Risk"
    }
    rows = []
    for g in items:
        raw_state = g.get("l1_state")
        rows.append({
            "appid":        g.get("appid"),
            "Name":         g.get("name") or f"appid {g.get('appid')}",
            "State":        state_map.get(raw_state, raw_state or "—"),
            "EA Age (d)":   g.get("ea_age_days") or 0,
            "Days Since Build": g.get("days_since_last_build_update"),
            "Reviews":      g.get("review_count_at_T") or 0,
            "Scored":       _resolve_date_obj(
                            g.get("snap_date") or g.get("snapshot_date"),
                            g.get("scored_at")),
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Browse view
# ---------------------------------------------------------------------------

def render_browse():
    # Sidebar filters
    with st.sidebar:
        st.markdown("## 📡 EARLY")
        st.markdown(
            '<span style="font-size:0.8em;color:#7d8590;">'
            'Early Access Health Monitor</span>',
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("**Filters**")
        state_filter    = st.selectbox("State", ["All", "Healthy", "Watch", "At Risk"],
                                       on_change=_reset_limit)
        min_reviews     = st.number_input("Min Reviews",
                                          min_value=0, value=0, step=10,
                                          on_change=_reset_limit)
        max_days_since_build = st.number_input("Max Days Since Build",
                                               min_value=0, value=0, step=30,
                                               help="0 to ignore",
                                               on_change=_reset_limit)
        search_name     = st.text_input("Search by name",
                                        placeholder="e.g. Nightingale",
                                        on_change=_reset_limit)

    # Fetch games
    data = _get_cached_games(
        state_filter, min_reviews, max_days_since_build,
        search_name, st.session_state.browse_limit
    )

    total = data.get("total", 0) if data else 0

    with st.sidebar:
        st.divider()
        st.metric("Filtered Matches", total)

    if not data:
        st.error("Cannot reach the EARLY API. Is it running?")
        st.code(f"Expected: {api.API_BASE}")
        return

    items = data.get("items", [])

    # Header stats
    health = _get_cached_health()

    at_risk = health.get("at_risk_count", 0) if health else 0
    watch   = health.get("watch_count", 0) if health else 0
    healthy = health.get("healthy_count", 0) if health else 0
    games_total = health.get("games_total", 0) if health else 0

    st.markdown("## 📡 EARLY &nbsp; Early Access Health Monitor")

    st.markdown("""
    **Welcome to the EARLY Intelligence Dashboard.**

    EARLY analyzes public Steam data (update cadence, player retention, 
                community sentiment, developer behavior, and more) 
                to estimate the probability that an Early Access game 
                will **stop updating** before reaching full release (1.0).
    """)

    with st.expander("📜 Why This Tool Exists", expanded=False):
        st.markdown(
                    """
                Steam Early Access is a high-risk, high-reward environment. While many
                games succeed, many others go silent and never ship.

                EARLY was built to audit a game's **operational momentum** — what
                developers are actually doing, not what they promise to do. The system
                breaks down its analysis into four straightforward steps:
                - **Activity Scorecard:** Measures the game's current momentum. It
                tracks how often does developers update and flags if a developer's
                real-world update habits are slowing down.
                - **Risk Classifier:** Evaluates the long-run probability of failure.
                It looks at the game's overall footprint to calculate the statistical
                risk ($p_{\text{distressed}}$) of the project stalling or being
                abandoned before launch.
                - **AI Forensic Agents:** Deep-dives into the text. When requested, our
                specialized AI agents cross-reference recent developer announcements
                and player reviews, translating raw community text and patch notes
                into clear, actionable intelligence.
                - **Similarity Search:** Looks for lookalike projects. It instantly
                scans our database of historical games to find past projects with the
                exact same data footprint, identifying how similar cases turned out.
                """
                )
    # Prominent Disclaimer
    st.error(
                """
            **⚠️ Important Disclaimer**:

            EARLY is an independent analytical tool, **not affiliated with Valve,
            Steam, or any game developers**.

            - All predictions are statistical estimates only. Past performance does
            not guarantee future results.
            - Predictions can be wrong. The model may suffer from data limitations
            and concept drift over time.
            - A self-fulfilling prophecy effect is possible — public awareness of
            risk can influence outcomes.
            - **Always verify information directly on Steam**, use EARLY as one
            data point among many — always do your own research.
            """
            )

    if health:
        status = health.get("status", "unknown")
        color  = "#238636" if status == "ok" else "#da3633"
        last_scored_date = (health.get("snapshot_date")
                            or _ts_to_date(health.get("last_scored_at")))
        st.markdown(
            f'<span style="color:{color};font-size:0.8em;">● '
            f'Pipeline: {status}</span> &nbsp;&nbsp; '
            f'<span style="font-size:0.8em;color:#7d8590;">'
            f'Last scored: {last_scored_date}</span>',
            unsafe_allow_html=True,
        )
        st.markdown("")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Games Tracked", games_total)
    c2.metric("🔴 At Risk", at_risk)
    c3.metric("🟡 Watch",   watch)
    c4.metric("🟢 Healthy", healthy)

    st.divider()

    if not items:
        st.info("No games match the current filters.")
        return

    # Build dataframe
    df = _build_dataframe(items)

    event = st.dataframe(
        df.drop(columns=["appid"]),
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Days Since Build": st.column_config.NumberColumn("Days Since Build",
                                                              format="%d"),
            "State": st.column_config.TextColumn("State",
                                                 width="small"),
            "Scored": st.column_config.DateColumn("Scored",
                                                  format="YYYY-MM-DD"),
        },
    )

    # Navigate on row selection
    sel = event.selection.get("rows", []) if hasattr(event, "selection") else []
    if sel:
        selected_appid = int(df.iloc[sel[0]]["appid"])
        _navigate_to(selected_appid)

    st.caption(f"Showing {len(items)} games · Click a row to view details")

    if len(items) < total:
        if st.button("Show more", use_container_width=True):
            st.session_state.browse_limit += 200
            st.rerun()


# ---------------------------------------------------------------------------
# Game detail view — shared header
# ---------------------------------------------------------------------------

def render_game_header(score: dict):
    name    = score.get("name") or f"appid {score['appid']}"
    state   = score.get("l1_state")
    ea_age  = score.get("ea_age_days") or 0
    scored_dt = _resolve_date_obj(score.get("snap_date") or score.get("snapshot_date"),
                                  score.get("scored_at"))
    scored  = scored_dt.strftime("%Y-%m-%d") if scored_dt else "—"
    quality = score.get("data_quality", "medium")
    appid   = score["appid"]

    days_since = score.get("days_since_last_build_update")
    last_build = "Unknown"
    if scored_dt and days_since is not None:
        last_build = (scored_dt - timedelta(days=days_since)).strftime("%Y-%m-%d")

    badge   = ui.state_badge(state)
    qbadge  = ui.data_quality_badge(quality)

    st.markdown(
        f'<div class="game-title">{name}</div>'
        f'<div class="game-meta">'
        f'{badge} &nbsp; {qbadge} &nbsp;'
        f'<span>In Early Access for <strong>{ea_age}</strong> days &nbsp;·&nbsp; '
        f'Last scored {scored} &nbsp;·&nbsp; '
        f'Last build: <strong>{last_build}</strong></span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    btn1, btn2, btn3, _ = st.columns([1.5, 1.5, 2.5, 4])
    with btn1:
        st.link_button("Steam Storefront", f"https://store.steampowered.com/app/{appid}",
                       use_container_width=True)
    with btn2:
        st.link_button("Steam Community", f"https://steamcommunity.com/app/{appid}",
                       use_container_width=True)
    with btn3:
        st.link_button("SteamDB (Third-party)", f"https://steamdb.info/app/{appid}",
                       use_container_width=True)
    st.markdown("")
    _render_how_to_read()


# ---------------------------------------------------------------------------
# Player tab
# ---------------------------------------------------------------------------

def render_player_tab(score: dict, history: dict | None):
    p_dist  = score.get("p_distressed")
    rev     = score.get("review_count_at_T") or 0
    quality = score.get("data_quality", "medium")
    dims    = score.get("dimensions") or {}

    # Key metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        val = f"{p_dist:.1%}" if p_dist is not None else "N/A"
        st.metric("Distress Risk", val,
                  help="Overall likelihood that the game will fail to "
                  "reach a successful full release based on "
                  "its current trajectory and historical patterns")
    with c2:
        st.metric("Reviews", f"{rev:,}",
                  help="Total review count at last scoring snapshot.")
    with c3:
        st.metric("Data Confidence", quality.title(),
                  help="Based on number of null features in the model input.")

    st.markdown("")

    # Dimension signal meters
    st.markdown("**Health Dimensions**",
                help="Percentages are relative to historical training data.")
    bars_html = "".join(
        ui.signal_bar(label, _map_dimension_score(key, dims.get(key)))
        for key, label in DIMENSION_LABELS.items()
    )
    st.markdown(bars_html, unsafe_allow_html=True)

    # Score history
    if history and history.get("snapshots"):
        st.markdown("")
        st.markdown("**Distress Risk Over Time**")
        st.plotly_chart(
            _history_chart(history["snapshots"]),
            use_container_width=True,
        )

    st.divider()
    _render_analysis_section(score, audience="player")
    st.markdown("<br><br>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Developer tab
# ---------------------------------------------------------------------------

def render_developer_tab(score: dict):
    dims   = score.get("dimensions") or {}
    nulls  = score.get("null_features") or []
    p_dist = score.get("p_distressed")
    state  = score.get("l1_state")

    # Score breakdown
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**Signal Breakdown**")
        bars_html = "".join(
            ui.signal_bar(label, _map_dimension_score(key, dims.get(key)))
            for key, label in DIMENSION_LABELS.items()
        )
        st.markdown(bars_html, unsafe_allow_html=True)

    with col2:
        st.markdown("**Model Output**")
        val = f"{p_dist:.1%}" if p_dist is not None else "N/A"
        st.metric("P(distressed)", val)

        state_help = None
        stats = _get_scorecard_stats()
        if stats and "state_agreement" in stats and state in stats["state_agreement"]:
            sa = stats["state_agreement"][state]
            state_help = (f"Historically, {sa['agreement_rate']:.1%} of games "
            f"in this state ended up as {sa['expected_outcome']}.")

        st.metric("State", state or "—", help=state_help)
        st.metric("Model", score.get("model_version") or "—")

    if nulls:
        with st.expander(f"⚠ {len(nulls)} / {TOTAL_FEATURES} features null"):
            st.caption(
                "ℹ️ *Null features typically occur when a data source is unavailable "
                "or a metric cannot be mathematically derived (e.g. division by zero).*"
            )
            st.caption(", ".join(nulls))

    st.divider()
    _render_analysis_section(score, audience="developer")


# ---------------------------------------------------------------------------
# AI Analysis section (shared, audience-aware)
# ---------------------------------------------------------------------------

def _render_analysis_section(score: dict, audience: str):
    appid   = score.get("appid")
    state   = score.get("l1_state")
    eligible = state in ("Watch", "At Risk")

    st.markdown("**AI Analysis**")

    if not eligible:
        st.info(
            "AI Analysis is available for Watch and At Risk games only. "
            "This game is currently Healthy — the scorecard signal is sufficient."
        )
        return

    # Poll for in-progress analysis
    if st.session_state.analysis_polling:
        if st.session_state.poll_count < MAX_POLLS:
            with st.spinner("Running analysis… this takes 15–30 seconds"):
                time.sleep(POLL_INTERVAL_S)
                result = api.get_analysis(appid)
                if result and result.get("status") in ("ready", "error"):
                    st.session_state.analysis_polling = False
                    st.session_state.poll_count = 0
                else:
                    st.session_state.poll_count += 1
                st.rerun()
        else:
            st.session_state.analysis_polling = False
            st.session_state.poll_count = 0
            st.warning("Analysis is taking longer than expected. "
                       "Try refreshing the page.")
            return

    # Load cached analysis
    analysis = api.get_analysis(appid)
    status   = analysis.get("status") if analysis else "never_run"

    # Trigger buttons
    btn_col1, btn_col2 = st.columns([1, 3])
    with btn_col1:
        force = status in ("ready", "error")
        label = "🔄 Refresh Analysis" if force else "▶ Run Analysis"
        if st.button(label, key=f"run_analysis_{audience}"):
            resp = api.trigger_analysis(appid, force=force)
            if resp and resp.get("status") == "queued":
                st.session_state.analysis_polling = True
                st.session_state.poll_count = 0
                st.rerun()
            elif resp and resp.get("status") == "not_eligible":
                st.info(resp.get("message"))
            else:
                st.error("Failed to queue analysis. Is the API running?")

    if status == "never_run" or not analysis:
        st.caption("No analysis run yet. Click Run Analysis to generate an AI verdict.")
        return

    if status == "error":
        st.error(f"Analysis encountered an error: {analysis.get('error')}")

    # Render based on audience
    if audience == "player":
        _render_player_analysis(analysis, appid)
    else:
        _render_developer_analysis(analysis, appid)


def _render_player_analysis(analysis: dict, appid: int):
    critic  = analysis.get("critic") or {}
    forensic = analysis.get("forensic") or {}
    verdict = critic.get("consumer_verdict")
    conf    = critic.get("confidence_note")
    analysed = _ts_to_date(analysis.get("analysed_at"))

    if verdict:
        st.markdown("**Verdict**")
        st.markdown(ui.verdict_card(verdict), unsafe_allow_html=True)

    if forensic.get("ran"):
        st.markdown("")
        st.markdown("**Latest Update Analysis**")
        score_val = forensic.get("update_substance_score")
        flag      = forensic.get("fake_heartbeat_flag")
        reason    = forensic.get("reasoning")

        fc1, fc2 = st.columns([1, 3])
        with fc1:
            st.markdown(
                f'<div style="text-align:center">'
                f'<div class="forensic-score">{score_val:.1f}</div>'
                f'<div class="forensic-max">&nbsp;/ 10</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if flag:
                st.markdown(
                    '<div style="text-align:center;margin-top:6px">'
                    '<span style="background:#da363322;'
                    'color:#da3633;border:1px solid #da363344;'
                    'padding:2px 8px;border-radius:4px;'
                    'font-size:0.75em;font-weight:600;">'
                    '🚩 Fake Heartbeat</span></div>',
                    unsafe_allow_html=True,
                )
        with fc2:
            if reason:
                st.caption(reason)

    if conf:
        st.markdown(f'<div class="conf-note">ⓘ {conf}</div>', unsafe_allow_html=True)
    if analysed:
        st.caption(f"Analysis from {analysed}")

    st.markdown("")
    _render_similar_section(appid, "player")


def _render_developer_analysis(analysis: dict, appid: int):
    critic   = analysis.get("critic") or {}
    forensic = analysis.get("forensic") or {}
    auditor  = analysis.get("auditor") or {}
    conf     = critic.get("confidence_note")
    analysed = _ts_to_date(analysis.get("analysed_at"))

    # Developer brief
    brief = critic.get("developer_brief")
    if brief:
        st.markdown("**Developer Brief**")
        st.markdown(ui.brief_card(brief), unsafe_allow_html=True)
        st.markdown("")

    # Sentiment clusters
    if auditor.get("ran"):
        clusters  = auditor.get("theme_clusters") or []
        concerns  = auditor.get("key_concerns") or []
        shift     = auditor.get("sentiment_shift")
        summary   = auditor.get("summary")

        st.markdown("**What Players Are Saying**")
        if shift:
            shift_color = {
                "improving": "#238636", "declining": "#da3633",
                "stable": "#7d8590",    "mixed": "#d29922",
            }.get(shift, "#7d8590")
            st.markdown(
                f'<span style="color:{shift_color};font-size:0.85em;font-weight:600;">'
                f'Sentiment trend: {shift} →</span>',
                unsafe_allow_html=True,
            )

        if clusters:
            for c in clusters:
                trans = c.get("quote_translation")
                if trans and str(trans).strip().lower() not in ("null", "none", ""):
                    orig = c.get("representative_quote") or ""
                    c["representative_quote"] = (
                        f"{orig}<br><span style='font-size:0.9em; "
                        f"color:#8b949e;'><i>Translation:</i> {trans}</span>"
                        )
            clusters_html = "".join(ui.cluster_row(c) for c in clusters)
            st.markdown(clusters_html, unsafe_allow_html=True)

        if concerns:
            st.markdown("")
            st.markdown("**Key Player Concerns**")
            concerns_html = "".join(ui.concern_item(c) for c in concerns)
            st.markdown(concerns_html, unsafe_allow_html=True)

        if summary:
            st.caption(summary)
        st.markdown("")

    # Forensic details
    if forensic.get("ran"):
        score_val = forensic.get("update_substance_score")
        flag      = forensic.get("fake_heartbeat_flag")
        reason    = forensic.get("reasoning")

        with st.expander(
            f"🔍 Latest Update — Substance {score_val:.1f}/10"
            + (" 🚩" if flag else "")
        ):
            if flag:
                st.warning("This update was flagged as a potential fake heartbeat "
                           "— minimal content with no real development substance.")
            if reason:
                st.write(reason)
    elif analysis.get("forensic_ran") is False:
        st.caption("No build update found in the last 30 days at snapshot time.")

    if conf:
        st.markdown(f'<div class="conf-note">ⓘ {conf}</div>', unsafe_allow_html=True)
    if analysed:
        st.caption(f"Analysis from {analysed}")

    st.markdown("")
    _render_similar_section(appid, "developer")


# ---------------------------------------------------------------------------
# Similar games section (shared)
# ---------------------------------------------------------------------------

def _render_similar_section(appid: int, audience: str):
    st.markdown("**Historically Similar Games**")

    if not st.session_state.similar_loaded:
        if st.button("🔍 Find Similar Games", key=f"similar_{appid}_{audience}"):
            with st.spinner("Searching historical anchors…"):
                result = api.get_similar(appid)
                st.session_state.similar_results = result
                st.session_state.similar_loaded  = True
                st.rerun()
        else:
            st.caption("Find games with a similar health profile "
                       "that have already resolved.")
            return

    result = st.session_state.similar_results
    if not result:
        st.warning("Similarity search unavailable — Zilliz may not be configured.")
        return

    games = result.get("results", [])
    msg   = result.get("message")

    if not games:
        st.info(msg or "No similar games found.")
        return

    # Summary stat
    abandoned = sum(1 for g in games if g.get("outcome") == "ABANDONED")
    success   = sum(1 for g in games if g.get("outcome") == "SUCCESS")
    st.caption(
        f"{len(games)} similar games found — "
        f"{abandoned} abandoned, {success} succeeded")
    st.caption(
        "_Note: `dist` is cosine similarity where 1 = complete match, "
        "-1 = complete opposite_"
    )

    rows_html = "".join(ui.similar_game_row(g) for g in games)
    st.markdown(rows_html, unsafe_allow_html=True)

    snap_date = result.get("query_snap_date")
    if snap_date:
        st.caption(f"Based on snapshot from {snap_date}")


def _render_how_to_read():
    st.markdown("📖 How to Read the Score")

    with st.expander("🔍 Understanding EARLY Scores", expanded=False):
        st.markdown(
            """
        ### Risk Tiers (Layer 1 Scorecard)

        EARLY classifies games into three risk tiers based on a **weighted
        scorecard** across five health dimensions:

        - **🟢 Healthy** Strong momentum across most dimensions.
          **Historical outcome**: High likelihood of a successful 1.0 full
          launch.

        - **🟡 Watch** Mixed or weakening signals. Worth monitoring. Acts as a
          transitional boundary.
          **Historical outcome**: Unstable middle tier—some games stabilize
          and launch, while others degrade further.

        - **🔴 At Risk** Clear signs of stagnation or structural decline.
          **Historical outcome**: High probability of developer abandonment
          or permanent radio silence.
        """
        )

        st.markdown("""
        ### Distress Probability (Layer 2 Machine Learning)

        This is our **machine learning model's** estimated probability
        (0–100%) that a game is on a distressed or failing trajectory.

        - **&lt; 45%** → Generally healthy footprint
        - **45–65%** → Gray zone / Watch territory
        - **&gt; 65%** → Elevated risk / High operational distress

        The ML model evaluates the whole ecosystem simultaneously, meaning it
        often catches subtle, compounding warning signs that simple checkboxes miss.

        Please note that the probability **does not refer to an instantaneous
        risk of abandonment**, but rather the overall likelihood that the game
        will fail to reach a successful full release based on
        its current trajectory and historical patterns.
        """)

        st.markdown(

                "### Dimension Scores (0–100 Rating)\n\n"
                "To ensure fairness, these 5 key indicators are **normalized "
                "against our entire historical training database**. A score of "
                '"50" means the game is performing exactly at the industry '
                'average for that metric, while a "75+" means it is '
                "outperforming 75% of early access history:\n\n"
                "| Dimension | What it measures | Why it matters |\n"
                "| :--- | :--- | :--- |\n"
                "| **Update Health** | Cadence + substance of build updates | "
                "Core signal of developer coding momentum |\n"
                "| **Player Retention** | Player count trends and engagement | "
                "Real consumer interest over time |\n"
                "| **Dev Engagement** | Community posts & developer "
                "responsiveness | Verifies the studio is still actively "
                "involved |\n"
                "| **Sentiment** | Review scores & recent review velocity | "
                "Player happiness & community momentum |\n"
                "| **Price & Market** | Pricing trends, discounts, and genre "
                "context | Long-term commercial viability |\n\n"
                "Each dimension factors in a **backbone** (long-term history) "
                "and a **momentum** (recent change) component."

        )

        st.markdown("""
        ### Data Quality & Missing Features

        Stalled projects often stop generating clean signals,
        which can cause data gaps (null values) in our trackers.

        If a game has a high number of missing features,
        treat its predictive scores with extra caution.
        A sudden drop in data density is frequently
        an early indicator that a project's operational wheels have stopped turning.
        """)

    with st.expander("🧠 AI Analysis & Similar Games"):
        st.markdown("""
        **For complex or borderline games**,
        you can trigger an **on-demand AI Deep Analysis**:

        - **Forensic Agent**: Evaluates the technical substance of
        recent patches against actual code deployment gaps.
        - **Sentiment Auditor**: Reads between the lines of text reviews
        to surface core player complaints.
        - **Critic Agent**: Synthesizes the data into a clear, unified verdict.

        **Lookalike Projects (Similarity Search):** Instantly scans
        our historical database to find past games with the exact same data footprint.
        This allows you to see how previous projects
        with identical patterns ultimately turned out.
        """)

    st.caption("All scores are data-driven estimates based on "
               "public Steam tracking metrics. No statistical model is perfect.")

# ---------------------------------------------------------------------------
# Game detail view — orchestrator
# ---------------------------------------------------------------------------

def render_game_detail(appid: int):
    st.markdown("<div style='padding-top: 2rem;'></div>", unsafe_allow_html=True)
    if st.button("← Back to games"):
        _go_back()

    score = api.get_score(appid)
    if not score:
        st.error(f"Could not load score for appid {appid}.")
        return

    render_game_header(score)

    player_tab, dev_tab = st.tabs(["👤  Player View", "🛠  Developer View"])

    history = api.get_history(appid)

    with player_tab:
        render_player_tab(score, history)

    with dev_tab:
        render_developer_tab(score)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if st.session_state.current_appid:
    render_game_detail(st.session_state.current_appid)
else:
    render_browse()


st.markdown("---")
st.header("About EARLY")

tab1, tab2, tab3 = st.tabs(["Overview", "Technical Details", "Performance"])

with tab1:
    st.markdown("""
    **EARLY** is a solo-built hybrid ML system designed to give
    consumers and developers early warning signals about Early Access game health.

    It combines rule-based scoring with machine learning
    and LLM agents to deliver transparent, explainable risk assessments.
    """)

with tab2:
    st.markdown(
            """
        ### Architecture

        - **Layer 1 — Scorecard**: Weighted deterministic evaluation across
          5 dimensions (Update Health, Player Retention, Developer Engagement,
          Community Sentiment, Price & Market Signals)
        - **Layer 2 — ML Model**: XGBoost binary classifier
          (`P(IS_DISTRESSED)`)
        - **Layer 3 — AI Agents**: Forensic update substance scoring +
          Sentiment Auditor + Critic synthesis (triggered on-demand for Watch /
          At Risk games)
        - **Similarity Search**: Finds historically similar games using SHAP
          vector embeddings (Zilliz)

        **Tech Stack**: Python, XGBoost, FastAPI, LangGraph, Groq, Streamlit,
        Turso (libSQL), Zilliz Cloud
        """
        )

    with st.expander("Key Design Principles"):
        st.markdown("""
        - Developer-relative abandonment thresholds
        - Strict look-ahead discipline during training
        - Free-tier first design
        - On-demand agents (cost control)
        - Transparency over black-box predictions
        """)

with tab3:
    st.markdown("""
    ### Current Model Performance (v1.3)

    - **Holdout AUC-ROC**: 0.9127
    - **Holdout PR-AUC**: 0.7382
    - **Lift over Scorecard**: +0.262
    """)

    st.info("""
    Three risk tiers:
    - **Healthy** — 98.2% success rate
    - **Watch** — 74.9% success rate
    - **At Risk** — ~48% success rate
    """)

    st.caption("Metrics are for resolved games (2022+). "
               "Performance can vary on new titles.")
