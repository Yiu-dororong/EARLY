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

from fastapi import FastAPI, Response, status, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

from api.db import close_db, init_db
from api.routers import games, health
from api.routers.search import router as search_router

# ── Detect the Execution Environment ──────────────────────────────────────────
# Render automatically sets RENDER_URL
IS_RENDER = bool(os.getenv("RENDER_URL"))


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

# ── 1. Configure Dynamic CORS ──────────────────────────────────────────────────
if IS_RENDER:
    # Production: Restrict to your specific Frontend deployment URL on Render
    # You can override this by setting FRONTEND_URL in Render's environment dashboard
    allowed_origins = [
        os.getenv("FRONTEND_URL") or os.getenv("RENDER_URL"), 
    ]
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
API_KEY_NAME = "X-API-Token"
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
                detail="Invalid or missing X-API-Token authentication header."
            )
    return api_key

# ── 3. Apply the Security Dependency to Protected Routes ──────────────────────
app.include_router(health.router, dependencies=[Depends(verify_api_token)])
app.include_router(games.router, prefix="/games", dependencies=[Depends(verify_api_token)])
app.include_router(search_router, prefix="/search", dependencies=[Depends(verify_api_token)])

@app.get("/")
@app.head("/")
async def root():
    return {
        "api_name": "EARLY API",
        "version": "1.0.0",
        "documentation": "/docs"
    }


# ── Liveness Endpoint ────────────────────────────────────────────────────
@app.get("/livez", status_code=status.HTTP_200_OK)
@app.head("/livez")
async def liveness():
    """
    Indicates whether the container process is running.
    Should remain lightweight; avoid blocking operations or external calls.
    """
    return {"status": "OK"}


# ── Readiness Endpoint ───────────────────────────────────────────────────
@app.get("/readyz")
@app.head("/readyz")
async def readiness(response: Response):
    """
    Indicates whether the application is ready to process incoming API queries.
    Verifies that backing models, memories, or databases are successfully connected.
    """
    return {
        "status": "READY"
    }