import os

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


IS_RENDER = bool(os.getenv("RENDER_URL"))

def get_real_ip(request: Request) -> str:
    """
    Extract the real user IP, respecting proxies (e.g., Render load balancers).
    """
    user_ip = request.headers.get("user-ip")
    if not user_ip:
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            user_ip = forwarded_for.split(",")[0].strip()
        else:
            user_ip = get_remote_address(request)
    return user_ip

def get_compound_key(request: Request) -> str:
    """
    Compound key for rate limiting: api-key + session-id + user-ip.
    Ensures fair usage and prevents simple bypasses.
    """
    api_token = request.headers.get("api-key")
    session_id = (request.headers.get("session-id") 
                  or request.cookies.get("session_id", "anon_session"))
    user_ip = get_real_ip(request)

    if api_token and session_id != "anon_session":
        return f"token:{api_token}:session:{session_id}:ip:{user_ip}"

    return f"token:{session_id}:ip:{user_ip}"

limiter = Limiter(key_func=get_compound_key)

# Stricter limit for LLM AI-heavy routes (triggering analysis)
ai_rate_limit = "5/minute" if IS_RENDER else "100/minute"
ai_ip_rate_limit = "10/minute" if IS_RENDER else "200/minute"

# Moderate limit for vector DB similarity searches
search_rate_limit = "25/minute" if IS_RENDER else "200/minute"
search_ip_rate_limit = "50/minute" if IS_RENDER else "400/minute"

# General-purpose limit for all other GET requests to prevent basic DoS
general_rate_limit = "100/minute" if IS_RENDER else "500/minute"
general_ip_rate_limit = "200/minute" if IS_RENDER else "1000/minute"