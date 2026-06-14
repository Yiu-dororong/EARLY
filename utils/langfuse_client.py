"""
utils/langfuse_client.py
--------------------------------
Shared Langfuse setup for LangGraph.

Usage in agents:
    from utils.langfuse_client import get_callback_handler
    handler = get_callback_handler(appid=...)
"""

from __future__ import annotations

import logging
import os

from langfuse.langchain import CallbackHandler

logger = logging.getLogger(__name__)

def get_callback_handler(
    appid: int,
    session_id: str | None = None,
    metadata: dict | None = None,
) -> CallbackHandler | None:
    """Create and return a Langfuse CallbackHandler. Returns None if disabled."""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    if not public_key:
        logger.warning("LANGFUSE_PUBLIC_KEY not set — tracing disabled")
        return None

    try:
        return CallbackHandler(
            session_id=session_id or str(appid),
            user_id=str(appid),
            metadata=metadata or {},
            tags=[f"appid:{appid}"],
        )
    except Exception as e:
        logger.debug("Langfuse CallbackHandler creation failed: %s", e)
        return None
