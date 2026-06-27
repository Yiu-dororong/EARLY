"""
api/main.py
-----------
FastAPI application entry point.

Run locally:
    fastapi dev api/main.py

Production:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
"""

import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from slowapi.errors import RateLimitExceeded

from api.db import close_db, get_db, init_db
from api.rate_limit import (
    IS_RENDER,
    general_ip_rate_limit,
    general_rate_limit,
    get_real_ip,
    limiter,
)
from api.routers import games, health
from api.routers.search import router as search_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await close_db()


app = FastAPI(
    title="EARLY API",
    description="EA game health scoring and distress prediction.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None if IS_RENDER else "/docs",
    redoc_url=None if IS_RENDER else "/redoc",
    openapi_url=None if IS_RENDER else "/openapi.json",
)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "error": "Rate Limit Exceeded",
            "detail": f"Request volume threshold breached. Limit rule: {exc.detail}",
            "hint": "Please wait before retrying.",
        },
    )

# ── 1. Configure Dynamic CORS ──────────────────────────────────────────────────
if IS_RENDER:
    # Production: Restrict to your specific Frontend deployment URL on Render
    frontend_url = os.getenv("STREAMLIT_APP_URL")
    if not frontend_url:
        # This will be logged by FastAPI on startup if empty.
        print("WARNING: STREAMLIT_APP_URL environment variable not set. CORS may fail.")
    allowed_origins = [frontend_url] if frontend_url else []
else:
    # Local Development: Allow your local Streamlit instance to connect freely
    allowed_origins = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 2. Configure Dynamic API Token Enforcement ──────────────────────────────
API_KEY_NAME = "api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_token(api_key: str = Depends(api_key_header)):
    """
    Enforces token validation ONLY when running on Render cloud architecture.
    """
    if IS_RENDER:
        expected_token = os.getenv("INTERNAL_API_TOKEN")
        if not expected_token:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="API token configuration missing on production host server."
            )
        if api_key != expected_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid or missing {API_KEY_NAME} authentication header."
            )
    return api_key

# ── 3. Apply the Security Dependency to Protected Routes ──────────────────────
app.include_router(health.router, dependencies=[Depends(verify_api_token)])
app.include_router(
    games.router,
    prefix="/games",
    dependencies=[Depends(verify_api_token)]
)
app.include_router(
    search_router,
    prefix="/search",
    dependencies=[Depends(verify_api_token)]
)

@app.get("/")
@app.head("/")
@limiter.limit(general_rate_limit)
@limiter.limit(general_ip_rate_limit, key_func=get_real_ip)
async def root(request: Request):
    return {
        "api_name": "EARLY API",
        "version": "1.0.0",
    }


# ── Liveness Endpoint ────────────────────────────────────────────────────
@app.get("/livez", status_code=status.HTTP_200_OK)
@app.head("/livez")
@limiter.limit(general_rate_limit)
@limiter.limit(general_ip_rate_limit, key_func=get_real_ip)
async def liveness(request: Request):
    """
    Indicates whether the container process is running.
    Should remain lightweight; avoid blocking operations or external calls.
    """
    return {"status": "OK"}


# ── Readiness Endpoint ───────────────────────────────────────────────────
@app.get("/readyz")
@app.head("/readyz")
@limiter.limit(general_rate_limit)
@limiter.limit(general_ip_rate_limit, key_func=get_real_ip)
async def readiness(request: Request, response: Response):
    """
    Indicates whether the application is ready to process queries.
    Verifies that backing services like the database are connected.
    """
    try:
        db = get_db()
        # A simple, fast query to check if the DB is responsive.
        db.execute("SELECT 1")
        return {"status": "READY"}
    except Exception as e:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "UNREADY", "reason": f"Database connection failed: {e}"}
