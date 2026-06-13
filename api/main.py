"""
api/main.py
-----------
FastAPI application entry point.

Run locally:
    fastapi dev api/main.py

Production:
    fastapi run api/main.py --port 8000 --workers 4
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.db import close_db, init_db
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
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(games.router, prefix="/games")
app.include_router(search_router, prefix="/search")

@app.get("/")
async def root():
    return {
        "api_name": "EARLY API",
        "version": "1.0.0",
        "documentation": "/docs"
    }