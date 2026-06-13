"""
api/services/langfuse_client.py
--------------------------------
Shared Langfuse setup and helper context managers.

Usage in agents:
    from api.services.langfuse_client import get_langfuse, generation_span

    lf = get_langfuse()
    with generation_span(lf, trace_id, name="forensic_llm", input=...) as span:
        response = llm.invoke(...)
        span.end(output=response.content, usage=...)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

logger = logging.getLogger(__name__)

_langfuse = None


def get_langfuse():
    """
    Return a singleton Langfuse client.
    Returns a no-op stub if env vars are not set (safe for local dev without tracing).
    """
    global _langfuse
    if _langfuse is not None:
        return _langfuse

    public_key  = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key  = os.getenv("LANGFUSE_SECRET_KEY")
    base_url    = os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")

    if not public_key or not secret_key:
        logger.warning("Langfuse env vars not set — tracing disabled")
        _langfuse = _NoOpLangfuse()
        return _langfuse

    try:
        from langfuse import Langfuse
        _langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=base_url,
        )
        logger.info("Langfuse client initialised (host=%s)", base_url)
    except ImportError:
        logger.warning("langfuse package not installed — tracing disabled")
        _langfuse = _NoOpLangfuse()

    return _langfuse


def flush():
    """Flush pending events — call at end of background tasks."""
    lf = get_langfuse()
    try:
        lf.flush()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Trace factory
# ---------------------------------------------------------------------------

def create_trace(
    name: str,
    appid: int,
    session_id: str | None = None,
    metadata: dict | None = None,
):
    """Create a top-level trace. Returns trace object (real or no-op)."""
    lf = get_langfuse()
    try:
        t = lf.trace(
            name=name,
            user_id=str(appid),
            session_id=session_id,
            metadata=metadata or {},
            tags=[f"appid:{appid}"],
        )
        return _SafeTraceWrapper(t)
    except Exception as e:
        logger.debug("Langfuse trace creation failed: %s", e)
        return _NoOpTrace()


class _SafeTraceWrapper:
    """Wraps a Langfuse trace to prevent LangGraph deepcopy from crashing on thread locks."""
    def __init__(self, trace):
        self._trace = trace
        
    def __deepcopy__(self, memo):
        return self
        
    def generation(self, **kwargs):
        return self._trace.generation(**kwargs)
        
    def span(self, **kwargs):
        return self._trace.span(**kwargs)
        
    def update(self, **kwargs):
        return self._trace.update(**kwargs)

# ---------------------------------------------------------------------------
# Generation span context manager
# ---------------------------------------------------------------------------

@contextmanager
def generation_span(
    trace,
    name: str,
    model: str,
    input_data: Any,
    metadata: dict | None = None,
):
    """
    Context manager that wraps an LLM call as a Langfuse generation span.

    Usage:
        with generation_span(trace, "forensic_llm", "llama-3.3-70b", prompt) as span:
            response = llm.invoke(messages)
            span.set_output(response.content)
            span.set_usage(input_tokens=..., output_tokens=...)
    """
    span = _GenerationSpanWrapper(trace, name, model, input_data, metadata)
    try:
        yield span
    except Exception as e:
        span._end_with_error(str(e))
        raise
    finally:
        span._finalise()


class _GenerationSpanWrapper:
    def __init__(self, trace, name, model, input_data, metadata):
        self._gen = None
        self._output = None
        self._input_tokens = None
        self._output_tokens = None
        try:
            self._gen = trace.generation(
                name=name,
                model=model,
                input=_safe_serialise(input_data),
                metadata=metadata or {},
            )
        except Exception as e:
            logger.debug("Langfuse generation span failed: %s", e)

    def set_output(self, output: Any) -> None:
        self._output = output

    def set_usage(self, input_tokens: int | None = None, output_tokens: int | None = None) -> None:
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens

    def _end_with_error(self, error: str) -> None:
        if self._gen:
            try:
                self._gen.end(level="ERROR", status_message=error)
                self._gen = None
            except Exception:
                pass

    def _finalise(self) -> None:
        if self._gen:
            try:
                kwargs: dict = {}
                if self._output is not None:
                    kwargs["output"] = _safe_serialise(self._output)
                if self._input_tokens or self._output_tokens:
                    kwargs["usage"] = {
                        "input": self._input_tokens,
                        "output": self._output_tokens,
                    }
                self._gen.end(**kwargs)
            except Exception as e:
                logger.debug("Langfuse span finalise failed: %s", e)


# ---------------------------------------------------------------------------
# Scorecard span (no LLM — just latency + metadata)
# ---------------------------------------------------------------------------

@contextmanager
def scorecard_span(trace, name: str, metadata: dict | None = None):
    """Lightweight span for non-LLM steps like scorecard computation."""
    span = None
    try:
        span = trace.span(name=name, metadata=metadata or {})
    except Exception:
        pass
    try:
        yield span
    finally:
        if span:
            try:
                span.end()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# No-op stubs (when Langfuse is disabled)
# ---------------------------------------------------------------------------

def _safe_serialise(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, (list, dict)):
        return obj
    return str(obj)


class _NoOpLangfuse:
    def trace(self, **kwargs):        return _NoOpTrace()
    def flush(self):                  pass


class _NoOpTrace:
    def __deepcopy__(self, memo):     return self
    def generation(self, **kwargs):   return _NoOpSpan()
    def span(self, **kwargs):         return _NoOpSpan()
    def update(self, **kwargs):       pass


class _NoOpSpan:
    def end(self, **kwargs):          pass
    def update(self, **kwargs):       pass
    def set_output(self, *a):         pass
    def set_usage(self, **kwargs):    pass
    def _end_with_error(self, *a):    pass
    def _finalise(self):              pass
    def __enter__(self):              return self
    def __exit__(self, *a):           pass
