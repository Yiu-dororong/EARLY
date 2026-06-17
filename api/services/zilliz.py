"""
api/services/zilliz.py
-----------------------
Zilliz Cloud (Milvus) client for EARLY similarity search.

Collection: early_historical_anchors
  — Populated once from ML training set labeled snapshots
  — Each row = one historical snapshot with known outcome
  — Queried at runtime with live game SHAP vector + metadata filters

Vector: 25-dim SHAP float vector (mean-imputed for nulls)
Dedup:  query returns k=50, deduplicate to 5 unique games server-side

Filter relaxation strategy:
  Pass 1: ea_age ±90d  + same primary_genre
  Pass 2: ea_age ±180d + same primary_genre   (if < 5 unique games)
  Pass 3: ea_age ±180d, no genre filter       (last resort)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COLLECTION_NAME  = "early_historical_anchors"
VECTOR_DIM       = 25
TOP_K_QUERY      = 50     # fetch this many, deduplicate down to TARGET_UNIQUE
TARGET_UNIQUE    = 5      # unique games to return after dedup


# ---------------------------------------------------------------------------
# Zilliz client
# ---------------------------------------------------------------------------

_client = None
_reconnect_lock = threading.Lock()


class ResilientZilliz:
    def _reconnect(self) -> None:
        global _client
        with _reconnect_lock:
            try:
                if _client is not None:
                    _client.close()
            except Exception:
                pass

            use_local = os.getenv("USE_LOCAL_DB", "false").lower() == "true"
            
            if use_local:
                db_path = "./demo_data/early_vector.db"
                try:
                    from pymilvus import MilvusClient
                    _client = MilvusClient(db_path)
                    if _client.has_collection(COLLECTION_NAME):
                        _client.load_collection(COLLECTION_NAME)
                        logger.info("Local collection '%s' loaded.", COLLECTION_NAME)
                    logger.info("Zilliz client initialised to local Milvus Lite (%s)", db_path)
                except ImportError:
                    logger.warning("pymilvus[milvus_lite] not installed — Local Zilliz disabled")
                    _client = None
                except Exception as e:
                    logger.error("Local Zilliz connection failed: %s", e)
                    _client = None
                return

            uri   = os.getenv("ZILLIZ_URI")
            token = os.getenv("ZILLIZ_TOKEN")

            if not uri or not token:
                logger.warning("ZILLIZ_URI/ZILLIZ_TOKEN not set — Zilliz disabled")
                _client = None
                return

            try:
                from pymilvus import MilvusClient
                _client = MilvusClient(uri=uri, token=token)
                logger.info("Zilliz client initialised/reconnected (uri=%s)", uri)
            except ImportError:
                logger.warning("pymilvus not installed — Zilliz disabled")
                _client = None
            except Exception as e:
                logger.error("Zilliz connection failed: %s", e)
                _client = None

    def _execute_with_retry(self, operation: str, *args, **kwargs):
        global _client
        for attempt in range(1, 4):
            try:
                if _client is None:
                    self._reconnect()
                if _client is None:
                    raise RuntimeError("Zilliz client not available")
                
                method = getattr(_client, operation)
                return method(*args, **kwargs)
            except Exception as e:
                if attempt == 3:
                    raise
                logger.warning("Zilliz '%s' error (attempt %d): %s — reconnecting", operation, attempt, e)
                time.sleep(1)
                self._reconnect()

    def search(self, *args, **kwargs): return self._execute_with_retry("search", *args, **kwargs)
    def upsert(self, *args, **kwargs): return self._execute_with_retry("upsert", *args, **kwargs)
    def has_collection(self, *args, **kwargs): return self._execute_with_retry("has_collection", *args, **kwargs)
    def create_schema(self, *args, **kwargs): return self._execute_with_retry("create_schema", *args, **kwargs)
    def prepare_index_params(self, *args, **kwargs): return self._execute_with_retry("prepare_index_params", *args, **kwargs)
    def create_collection(self, *args, **kwargs): return self._execute_with_retry("create_collection", *args, **kwargs)


def get_client() -> ResilientZilliz | None:
    use_local = os.getenv("USE_LOCAL_DB", "false").lower() == "true"
    if not use_local and (not os.getenv("ZILLIZ_URI") or not os.getenv("ZILLIZ_TOKEN")):
        return None
    return ResilientZilliz()


# ---------------------------------------------------------------------------
# Collection schema + setup
# ---------------------------------------------------------------------------

def ensure_collection() -> bool:
    """
    Create the collection if it doesn't exist.
    Safe to call multiple times (idempotent).
    Returns True if collection is ready.
    """
    client = get_client()
    if client is None:
        return False

    try:
        from pymilvus import DataType

        if client.has_collection(COLLECTION_NAME):
            logger.info("Collection '%s' already exists", COLLECTION_NAME)
            return True

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id",                 DataType.VARCHAR,       max_length=64,  is_primary=True)
        schema.add_field("shap_vector",        DataType.FLOAT_VECTOR,  dim=VECTOR_DIM)
        schema.add_field("appid",              DataType.INT64)
        schema.add_field("snapshot_date",      DataType.VARCHAR,       max_length=16)
        schema.add_field("ea_age_days",        DataType.INT64)
        schema.add_field("primary_genre",      DataType.VARCHAR,       max_length=64)
        schema.add_field("l1_state",           DataType.VARCHAR,       max_length=16)
        schema.add_field("outcome",            DataType.VARCHAR,       max_length=32)
        schema.add_field("null_feature_count", DataType.INT64)
        schema.add_field("p_distressed",       DataType.FLOAT)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="shap_vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )

        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params,
        )
        logger.info("Collection '%s' created", COLLECTION_NAME)
        return True

    except Exception as e:
        logger.error("ensure_collection failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Upsert (called by inference.py at score time)
# ---------------------------------------------------------------------------

def upsert_snapshot(
    appid: int,
    snapshot_date: str,
    shap_dict: dict[str, float | None],
    feature_order: list[str],
    ea_age_days: int,
    primary_genre: str,
    l1_state: str,
    outcome: str,                    # "SUCCESS" | "ABANDONED" | "UNKNOWN"
    null_feature_count: int,
    p_distressed: float | None,
) -> bool:
    """
    Upsert a single snapshot into Zilliz.
    Called from inference.py after scoring historical labeled games.
    For live games: outcome="UNKNOWN", updated when game resolves.

    id = f"{appid}_{snapshot_date}" — unique per game per snapshot date.
    """
    client = get_client()
    if client is None:
        return False

    try:
        vector = [float(shap_dict[f]) for f in feature_order]

        if len(vector) != VECTOR_DIM:
            logger.error(
                "upsert_snapshot: vector dim %d != expected %d for appid=%d",
                len(vector), VECTOR_DIM, appid,
            )
            return False

        row = {
            "id":                 f"{appid}_{snapshot_date}",
            "shap_vector":        vector,
            "appid":              appid,
            "snapshot_date":      snapshot_date,
            "ea_age_days":        ea_age_days,
            "primary_genre":      primary_genre or "unknown",
            "l1_state":           l1_state or "unknown",
            "outcome":            outcome,
            "null_feature_count": null_feature_count,
            "p_distressed":       float(p_distressed) if p_distressed is not None else 0.0,
        }

        client.upsert(collection_name=COLLECTION_NAME, data=[row])
        return True

    except Exception as e:
        logger.error("upsert_snapshot failed appid=%d: %s", appid, e)
        return False


# ---------------------------------------------------------------------------
# Search result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SimilarSnapshot:
    appid:              int
    snapshot_date:      str
    ea_age_days:        int
    primary_genre:      str
    l1_state:           str
    outcome:            str
    p_distressed:       float
    null_feature_count: int
    distance:           float
    match_quality:      str     # "high" | "medium" | "low" based on null_feature_count


# ---------------------------------------------------------------------------
# ANN search with filter relaxation + dedup
# ---------------------------------------------------------------------------

def search_similar(
    query_shap_dict: dict[str, float | None],
    feature_order: list[str],
    query_appid: int,
    query_ea_age_days: int,
    query_primary_genre: str,
    n_results: int = TARGET_UNIQUE,
) -> list[SimilarSnapshot]:
    """
    Find the n_results most similar historical snapshots.

    Filter relaxation:
      Pass 1: ea_age ±90d  + same genre
      Pass 2: ea_age ±180d + same genre
      Pass 3: ea_age ±180d, no genre filter

    Deduplication: keep closest snapshot per unique appid.
    Excludes: the query game itself.
    """
    client = get_client()
    if client is None:
        return []

    try:
        vector = [float(query_shap_dict.get(f, 0.0)) for f in feature_order]

        passes = [
            _build_filter(query_appid, query_ea_age_days, query_primary_genre, age_window=90,  use_genre=True),
            _build_filter(query_appid, query_ea_age_days, query_primary_genre, age_window=180, use_genre=True),
            _build_filter(query_appid, query_ea_age_days, query_primary_genre, age_window=180, use_genre=False),
        ]

        for i, filter_expr in enumerate(passes, 1):
            results = _run_search(vector, filter_expr, limit=TOP_K_QUERY)
            unique  = _deduplicate(results, n_results)

            if len(unique) >= n_results:
                logger.info(
                    "search_similar: pass %d found %d unique results for appid=%d",
                    i, len(unique), query_appid,
                )
                return unique

            logger.info(
                "search_similar: pass %d returned only %d unique — relaxing filter",
                i, len(unique),
            )

        # Return whatever we have after all passes
        return _deduplicate(
            _run_search(vector, passes[-1], limit=TOP_K_QUERY),
            n_results,
        )

    except Exception as e:
        logger.error("search_similar failed: %s", e)
        return []


def _build_filter(
    exclude_appid: int,
    ea_age_days: int,
    primary_genre: str,
    age_window: int,
    use_genre: bool,
) -> str:
    age_min = ea_age_days - age_window
    age_max = ea_age_days + age_window
    parts = [
        f"appid != {exclude_appid}",
        f"ea_age_days >= {age_min}",
        f"ea_age_days <= {age_max}",
        "outcome in ['SUCCESS', 'ABANDONED']",   # labeled outcomes only
    ]
    if use_genre and primary_genre:
        parts.append(f'primary_genre == "{primary_genre}"')
    return " and ".join(parts)


def _run_search(vector: list[float], filter_expr: str, limit: int) -> list[dict]:
    client = get_client()
    if client is None:
        return []

    try:
        results = client.search(
            collection_name=COLLECTION_NAME,
            data=[vector],
            anns_field="shap_vector",
            search_params={"metric_type": "COSINE", "params": {"ef": 100}},
            filter=filter_expr,
            limit=limit,
            output_fields=[
                "appid", "snapshot_date", "ea_age_days", "primary_genre",
                "l1_state", "outcome", "p_distressed", "null_feature_count",
            ],
        )
        return results[0] if results else []
    except Exception as e:
        logger.error("_run_search failed: %s", e)
        return []


def _deduplicate(hits: list, n: int) -> list[SimilarSnapshot]:
    """Keep only the closest hit per appid, return top-n."""
    seen: dict[int, SimilarSnapshot] = {}

    for hit in hits:
        entity  = hit.get("entity", hit)  # pymilvus 2.x returns entity dict
        appid   = entity.get("appid")
        if appid is None:
            continue
        if appid in seen:
            continue   # already have the closest hit for this game

        null_count = entity.get("null_feature_count", 0)
        match_quality = (
            "high"   if null_count <= 5 else
            "medium" if null_count <= 15 else
            "low"
        )

        seen[appid] = SimilarSnapshot(
            appid=appid,
            snapshot_date=entity.get("snapshot_date", ""),
            ea_age_days=entity.get("ea_age_days", 0),
            primary_genre=entity.get("primary_genre", ""),
            l1_state=entity.get("l1_state", ""),
            outcome=entity.get("outcome", ""),
            p_distressed=entity.get("p_distressed", 0.0),
            null_feature_count=null_count,
            distance=hit.get("distance", 0.0),
            match_quality=match_quality,
        )

        if len(seen) >= n:
            break

    return list(seen.values())
