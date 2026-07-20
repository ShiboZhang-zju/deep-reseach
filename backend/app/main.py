"""FastAPI application entry point."""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db.session import engine, Base
from app.api.routes import tasks, papers, reports, ideas, experiments, events, traces
from app.api.auth import ApiKeyMiddleware
from app.agent.runner import recover_interrupted_tasks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Create tables
Base.metadata.create_all(engine)
logger.info("Database tables created")

# Recover tasks interrupted by process crash/restart
try:
    recover_interrupted_tasks()
except Exception as e:
    logger.error("Failed to recover interrupted tasks on startup: %s", e)

app = FastAPI(
    title="Deep Research API",
    description="AI Agent for automated research: multi-source paper search, scoring, report generation, and idea generation.",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key authentication (P0-4): protects POST/PUT/DELETE on /api/tasks if API_KEY is set
if settings.api_key:
    app.add_middleware(ApiKeyMiddleware, api_key=settings.api_key)
    logger.info("API Key authentication enabled")
else:
    logger.warning("API_KEY not set — authentication disabled (not recommended for production)")

# Register routers
app.include_router(tasks.router, prefix="/api", tags=["tasks"])
app.include_router(papers.router, prefix="/api", tags=["papers"])
app.include_router(reports.router, prefix="/api", tags=["reports"])
app.include_router(ideas.router, prefix="/api", tags=["ideas"])
app.include_router(experiments.router, prefix="/api", tags=["experiments"])
app.include_router(events.router, prefix="/api", tags=["events"])
app.include_router(traces.router, prefix="/api", tags=["traces"])

# Mount paper assets directory for serving extracted figures
_assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "paper_assets")
os.makedirs(_assets_dir, exist_ok=True)
app.mount("/paper_assets", StaticFiles(directory=_assets_dir), name="paper_assets")


@app.get("/")
def root():
    return {"name": "Deep Research API", "version": "1.0.0", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}
