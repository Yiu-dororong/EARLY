import os
from fastapi import Request
from slowapi import Limiter

IS_RENDER = bool(os.getenv("RENDER_URL"))

def rate_limiter_key(request: Request) -> str:
    """
    Compound key for rate limiting: api-key + session-id + user-ip.
    Ensures fair usage and prevents simple bypasses.
    """
    api_token = request.headers.get("api-key")
    session_id = request.headers.get("session-id")
    user_ip = request.headers.get("user-ip")

    if not user_ip:
        forwarded_for = request.headers.get("x-forwarded-for")
        user_ip = forwarded_for.split(",")[0].strip() if forwarded_for else (request.client.host if request.client else "unknown")

    if api_token and session_id:
        return f"token:{api_token}:session:{session_id}:ip:{user_ip}"

    return f"ip:{user_ip}"

limiter = Limiter(key_func=rate_limiter_key)

# Stricter limit for LLM AI-heavy routes (triggering analysis)
ai_rate_limit = "5/minute" if IS_RENDER else "100/minute"

# Moderate limit for vector DB similarity searches
search_rate_limit = "30/minute" if IS_RENDER else "200/minute"

# General-purpose limit for all other GET requests to prevent basic DoS
general_rate_limit = "100/minute" if IS_RENDER else "500/minute"