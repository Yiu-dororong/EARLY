"""
frontend/utils/api.py
---------------------
Thin API client for the EARLY FastAPI backend.
All functions return parsed JSON or None on failure.
"""

from __future__ import annotations

import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("RENDER_URL") or os.getenv("API_BASE_URL", "http://localhost:8000")
TIMEOUT  = 10
IS_RENDER = bool(os.getenv("RENDER_URL"))


def _get_headers() -> dict:
    headers = {}
    if IS_RENDER:
        token = os.getenv("INTERNAL_API_TOKEN")
        if token:
            headers["X-API-Token"] = token
        else:
            st.error("Configuration Error: Production API Token missing from environment.")
    return headers


def _get(path: str, params: dict | None = None) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, headers=_get_headers(), timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _post(path: str, params: dict | None = None) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", params=params, headers=_get_headers(), timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def get_health() -> dict | None:
    return _get("/health")


def list_games(
    l1_state:    str | None = None,
    ml_eligible: int | None = None,
    min_reviews: int | None = None,
    search_name: str | None = None,
    offset: int = 0,
    limit:  int = 100,
) -> dict | None:
    params: dict = {"offset": offset, "limit": limit}
    if l1_state    is not None: params["l1_state"]    = l1_state
    if ml_eligible is not None: params["ml_eligible"] = ml_eligible
    if min_reviews is not None: params["min_reviews"] = min_reviews
    if search_name is not None: params["search_name"] = search_name
    return _get("/games", params)


def get_score(appid: int) -> dict | None:
    return _get(f"/games/{appid}/score")


def get_history(appid: int) -> dict | None:
    return _get(f"/games/{appid}/history")


def get_analysis(appid: int) -> dict | None:
    return _get(f"/games/{appid}/analysis")


def trigger_analysis(appid: int, force: bool = False) -> dict | None:
    return _post(f"/games/{appid}/analyse", params={"force": str(force).lower()})


def get_similar(appid: int, n_results: int = 5) -> dict | None:
    return _post("/search/similar", params={"appid": appid, "n_results": n_results})
