"""FastAPI application entry point."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db.session import engine, Base
from app.api.routes import tasks, papers, reports, ideas, experiments, events, traces, contracts, gaps, interventions
from app.api.auth import ApiKeyMiddleware
from app.agent.runner import recover_interrupted_tasks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Create tables
# Phase 2.2: Alembic is the authoritative schema source.
# create_all() is disabled for production; set AUTO_CREATE_SCHEMA=true for dev/test.
if os.environ.get("AUTO_CREATE_SCHEMA", "false").lower() == "true":
    Base.metadata.create_all(engine)
    logger.info("Database tables created via create_all (AUTO_CREATE_SCHEMA=true)")
else:
    logger.info("Schema managed by Alembic. Run 'alembic upgrade head' to create/update tables.")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Recover tasks interrupted by a process crash/restart.

    This must run on server startup, NOT at import time: importing this module
    (which any test or CLI script doing `from app.main import app` does) would
    otherwise rewrite task rows — including marking a currently running task's
    phases as interrupted.
    """
    try:
        # Widen the default asyncio thread pool. asyncio.to_thread users include
        # slow NETWORK calls (embedding pre-filters, PDF downloads, OCR): at
        # multi-task concurrency they occupy every default worker (cpu_count+4)
        # and further to_thread calls — e.g. audit's _variant_is_invariant —
        # queue indefinitely, freezing whole phases while the event loop itself
        # sits idle (diagnosed 2026-09-07 via /api/debug/asyncio: the coroutine
        # was parked on threads.py:25 in to_thread with a pending FutureIter).
        from concurrent.futures import ThreadPoolExecutor

        loop = asyncio.get_running_loop()
        loop.set_default_executor(
            ThreadPoolExecutor(max_workers=64, thread_name_prefix="agent_t"))
        if settings.recover_tasks_on_startup:
            recover_interrupted_tasks()
        else:
            # Multi-service experiments share one DB: recovering here would
            # reset the OTHER arm's in-flight task (statuses include
            # `searching`), silently handing an arm's run to the wrong
            # configuration. Experiment services must start with
            # RECOVER_TASKS_ON_STARTUP=false.
            logger.warning(
                "Startup task recovery DISABLED (RECOVER_TASKS_ON_STARTUP=false)"
                " — in-flight tasks from other services are left untouched.")
    except Exception as e:
        logger.error("Failed to recover interrupted tasks on startup: %s", e)
    yield


app = FastAPI(
    title="Deep Research API",
    description="AI Agent for automated research: multi-source paper search, scoring, report generation, and idea generation.",
    version="1.0.0",
    lifespan=lifespan,
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
# Phase 1: New routes
app.include_router(contracts.router, prefix="/api", tags=["contracts"])
# Phase 3A: Gap routes (read-only)
app.include_router(gaps.router, tags=["gaps"])
# Mid-priority #6: Intervention routes (read-only)
app.include_router(interventions.router, tags=["interventions"])

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


@app.get("/experiment-config")
def experiment_config():
    """Report the experiment knobs THIS process actually resolved.

    Two-service experiments (arms on different ports) are only valid if each
    process really runs its configured arm. Reading the value back from the
    process is the difference between "we set the env var" and "the arm ran as
    specified": a mistyped var name, an unloaded .env or a stale process all
    look identical from the outside, and the resulting run would be silently
    attributed to the wrong arm.

    Read-only and derived from `settings` (already resolved at import), so it
    cannot itself change behaviour.
    """
    from app.agent.steps.audit_gaps import _audit_budget

    budget = _audit_budget()
    return {
        "audit_neighbors_override": int(
            getattr(settings, "audit_neighbors_override", 0) or 0),
        "effective_audit_budget": budget,
        "gap_audit_budgeted": bool(settings.gap_audit_budgeted),
        "gap_audit_disabled_ablation": bool(
            settings.gap_audit_disabled_ablation),
        "gap_audit_progressive": bool(settings.gap_audit_progressive),
        "gap_mining_legacy_v6_mode": bool(settings.gap_mining_legacy_v6_mode),
        "gap_relevance_min_neighbor": settings.gap_relevance_min_neighbor,
        "audit_corpus_recall_enabled": bool(settings.audit_corpus_recall_enabled),
        "audit_screen_before_cap": bool(settings.audit_screen_before_cap),
        "embedding_backend": settings.embedding_backend,
        "embedding_model": settings.embedding_model,
    }
