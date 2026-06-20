"""
utils/langfuse_client.py
--------------------------------
Shared Langfuse setup for LangGraph.

Usage in agents:
    from utils.langfuse_client import get_callback_handler
    handler = get_callback_handler()
"""
import os

from langfuse.langchain import CallbackHandler


def get_callback_handler() -> CallbackHandler | None:
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        return None
    
    return CallbackHandler()
